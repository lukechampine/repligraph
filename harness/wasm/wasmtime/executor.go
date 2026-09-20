package wasmtime

import (
	"errors"
	"fmt"

	wt "github.com/bytecodealliance/wasmtime-go/v38"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

const (
	maxWasmStack = 256 << 10

	clockEpochNanos   = 1577836800_000000000
	clockQuantumNanos = 1000
)

// Wat2Wasm converts WebAssembly text to binary format.
func Wat2Wasm(wat string) ([]byte, error) {
	return wt.Wat2Wasm(wat)
}

// Run is an wasm.Executor. It is safe for concurrent use. Calls share a bounded
// cache of compiled modules; each creates fresh guest state and resource budgets.
func Run(t tree.Tree, req wasm.Request) (response wasm.Response, result tree.Tree, runErr error) {
	defer func() {
		if p := recover(); p != nil {
			if abort, ok := p.(resourceAbort); ok {
				response, result, runErr = wasm.Response{}, t, fmt.Errorf("%s: %w", abort, wasm.ErrResourceLimit)
			} else {
				panic(p)
			}
		}
	}()
	fail := func(err error) (wasm.Response, tree.Tree, error) {
		return wasm.Response{}, t, err
	}
	if _, err := t.Get(artifactRoot); err == nil {
		return fail(errors.New("artifact root is a file"))
	}

	task, _ := t.Without("env")
	if uint64(task.Stats().Entries) > req.Limits.MaxTreeEntries {
		return fail(fmt.Errorf("initial tree entries: %w", wasm.ErrResourceLimit))
	} else if task.Stats().Bytes < 0 || uint64(task.Stats().Bytes) > min(req.Limits.MaxTreeBytes, uint64(^uint(0)>>1)) {
		return fail(fmt.Errorf("initial tree bytes: %w", wasm.ErrResourceLimit))
	}

	compiled, err := sharedModules.acquire(req.Module)
	if err != nil {
		return fail(err)
	}
	// Release after the store and linker are closed: eviction must not close a
	// module or its engine while a run is still using them.
	defer sharedModules.release(compiled)
	engine, mod := compiled.engine, compiled.module
	for _, export := range mod.Exports() {
		if export.Name() == privateMemory && export.Type().MemoryType().Minimum()*65536 > req.Limits.MemoryBytes {
			return fail(fmt.Errorf("initial memory: %w", wasm.ErrResourceLimit))
		}
	}

	store := wt.NewStore(engine)
	defer store.Close()
	// A wasm32 memory cannot exceed 4 GiB.
	store.Limiter(int64(min(req.Limits.MemoryBytes, 1<<32)), 1<<20, 1, 8, 1)
	// Wasmtime requires a positive reserve even when the final instruction
	// consumes exactly the measured fuel. This headroom is not guest budget.
	fuel := req.Limits.Fuel
	if fuel < ^uint64(0) {
		fuel++
	}
	if err := store.SetFuel(fuel); err != nil {
		return fail(err)
	}
	ov := newOverlay(t, req.Limits.MaxTreeEntries, req.Limits.MaxTreeBytes)
	r := newRun(ov, req)
	linker := wt.NewLinker(engine)
	defer linker.Close()
	if err := r.define(linker); err != nil {
		return fail(err)
	}

	// Resolve and validate imports separately. Failure after this point may
	// reflect native allocation or instance limits and must remain a host error.
	for _, imp := range mod.Imports() {
		name := imp.Name()
		if name == nil {
			return fail(wasm.ErrBadModule)
		}
		e := linker.Get(store, imp.Module(), *name)
		if e == nil {
			return fail(wasm.ErrBadModule)
		}
		expected, actual := imp.Type().FuncType(), e.Func().Type(store)
		valid := sameFuncType(expected, actual)
		actual.Close()
		e.Close()
		if !valid {
			return fail(wasm.ErrBadModule)
		}
	}

	// The instrumented start section is invoked only after setting the local
	// ceiling, so hidden memory and traps during startup remain observable.
	instance, callErr := linker.Instantiate(store, mod)
	if callErr != nil {
		err := fmt.Errorf("module instantiation failed: %s", callErr.Error())
		if closer, ok := callErr.(interface{ Close() }); ok {
			closer.Close()
		}
		return fail(err)
	}
	if e := instance.GetExport(store, privateLimit); e != nil {
		if err := e.Global().Set(store, wt.ValI64(int64(min(req.Limits.MemoryBytes/65536, 65536)))); err != nil {
			e.Close()
			return fail(err)
		}
		e.Close()
	}
	if start := instance.GetFunc(store, privateStart); start != nil {
		_, callErr = start.Call(store)
	}
	if callErr == nil {
		_, callErr = instance.GetFunc(store, "_start").Call(store)
	}
	if callErr != nil {
		if closer, ok := callErr.(interface{ Close() }); ok {
			defer closer.Close()
		}
	}
	if e := instance.GetExport(store, privateAbort); e != nil {
		aborted := e.Global().Get(store).I32() != 0
		e.Close()
		if aborted {
			return fail(fmt.Errorf("memory growth: %w", wasm.ErrResourceLimit))
		}
	}
	var memoryBytes uint64
	if e := instance.GetExport(store, privateMemory); e != nil {
		memoryBytes = uint64(e.Memory().DataSize(store))
		e.Close()
	}
	fuelLeft, err := store.GetFuel()
	if err != nil {
		return fail(err)
	}
	resp := wasm.Response{
		ExitCode: r.exitCode,
		Stdout:   r.stdout,
		Stderr:   r.stderr,
		Usage: wasm.Usage{
			Fuel:        fuel - fuelLeft,
			MemoryBytes: memoryBytes,
			WriteBytes:  r.written,
			StdoutBytes: uint64(len(r.stdout)),
			StderrBytes: uint64(len(r.stderr)),
			TreeEntries: ov.peakEntries,
			TreeBytes:   ov.peakBytes,
		},
	}
	if resp.Usage.Fuel > req.Limits.Fuel {
		return fail(fmt.Errorf("fuel: %w", wasm.ErrResourceLimit))
	}
	if !r.exited && callErr != nil {
		var trap *wt.Trap
		if !errors.As(callErr, &trap) {
			return fail(callErr)
		}
		if resp.Trap, err = trapOf(trap); err != nil {
			return fail(err)
		}
		return resp, t, nil
	}
	nt, err := ov.commit()
	if err != nil {
		return fail(fmt.Errorf("commit: %w", err))
	}
	return resp, nt, nil
}

func trapOf(trap *wt.Trap) (wasm.Trap, error) {
	code := trap.Code()
	if code == nil {
		return "", errors.New("unclassified Wasmtime trap")
	}
	switch *code {
	// The pinned Go binding omits ALWAYS_TRAP_ADAPTER (11); OUT_OF_FUEL is 12.
	case 12:
		return "", fmt.Errorf("fuel: %w", wasm.ErrResourceLimit)
	case wt.StackOverflow:
		// Native stack usage varies by architecture; this is not a replayable outcome.
		return "", fmt.Errorf("native Wasmtime stack: %w", wasm.ErrResourceLimit)
	case wt.MemoryOutOfBounds, wt.HeapMisaligned:
		return wasm.TrapMemoryOOB, nil
	case wt.TableOutOfBounds:
		return wasm.TrapTableOOB, nil
	case wt.IndirectCallToNull:
		return wasm.TrapIndirectNull, nil
	case wt.BadSignature:
		return wasm.TrapBadSignature, nil
	case wt.IntegerOverflow:
		return wasm.TrapIntegerOverflow, nil
	case wt.IntegerDivisionByZero:
		return wasm.TrapDivByZero, nil
	case wt.BadConversionToInteger:
		return wasm.TrapBadConversion, nil
	case wt.UnreachableCodeReached:
		return wasm.TrapUnreachable, nil
	default:
		return "", fmt.Errorf("unsupported Wasmtime trap code %d", *code)
	}
}

func sameFuncType(a, b *wt.FuncType) bool {
	if a == nil || b == nil {
		return false
	}
	for _, pair := range [][2][]*wt.ValType{{a.Params(), b.Params()}, {a.Results(), b.Results()}} {
		if len(pair[0]) != len(pair[1]) {
			return false
		}
		for i := range pair[0] {
			if pair[0][i].Kind() != pair[1][i].Kind() {
				return false
			}
		}
	}
	return true
}
