package wasmtime

import (
	"container/list"
	"fmt"
	"sync"

	wt "github.com/bytecodealliance/wasmtime-go/v38"

	"lukechampine.com/repligraph/blake3"
	"lukechampine.com/repligraph/harness/wasm"
)

// Retention is bounded by module count and source size. Source size is an
// admission weight, not a bound on Wasmtime's native allocations. Active runs
// retain their modules independently of the idle cache budget.
var sharedModules = moduleCache{maxEntries: 16, maxSourceBytes: 128 << 20}

type compiledModule struct {
	hash        blake3.Hash
	sourceBytes int
	engine      *wt.Engine
	module      *wt.Module
	err         error
	ready       chan struct{}
	refs        int
	idle        *list.Element
}

type moduleCache struct {
	mu                         sync.Mutex
	modules                    map[blake3.Hash]*compiledModule
	idle                       list.List // most recently released first
	sourceBytes                int
	maxEntries, maxSourceBytes int
}

func (c *moduleCache) acquire(bin []byte) (*compiledModule, error) {
	hash := wasm.ModuleHash(bin)
	c.mu.Lock()
	if m := c.modules[hash]; m != nil {
		m.refs++
		if m.idle != nil {
			c.idle.Remove(m.idle)
			m.idle = nil
			c.sourceBytes -= m.sourceBytes
		}
		c.mu.Unlock()
		<-m.ready
		if m.err != nil {
			return nil, m.err
		}
		return m, nil
	}
	if c.modules == nil {
		c.modules = make(map[blake3.Hash]*compiledModule)
	}
	m := &compiledModule{hash: hash, sourceBytes: len(bin), ready: make(chan struct{}), refs: 1}
	c.modules[hash] = m
	c.mu.Unlock()

	// Only callers waiting for this module block on compilation. Other cache
	// hits and compilations can proceed concurrently.
	m.engine, m.module, m.err = compileModule(bin)
	c.mu.Lock()
	if m.err != nil {
		delete(c.modules, hash)
	}
	close(m.ready)
	c.mu.Unlock()
	if m.err != nil {
		return nil, m.err
	}
	return m, nil
}

func (c *moduleCache) release(m *compiledModule) {
	c.mu.Lock()
	defer c.mu.Unlock()
	m.refs--
	if m.refs != 0 {
		return
	}
	if c.maxEntries <= 0 || m.sourceBytes > c.maxSourceBytes {
		delete(c.modules, m.hash)
		m.module.Close()
		m.engine.Close()
		return
	}
	m.idle = c.idle.PushFront(m)
	c.sourceBytes += m.sourceBytes
	for c.idle.Len() > c.maxEntries || c.sourceBytes > c.maxSourceBytes {
		old := c.idle.Back().Value.(*compiledModule)
		c.idle.Remove(old.idle)
		old.idle = nil
		c.sourceBytes -= old.sourceBytes
		delete(c.modules, old.hash)
		old.module.Close()
		old.engine.Close()
	}
}

func compileModule(bin []byte) (*wt.Engine, *wt.Module, error) {
	// The Wasmtime version and these settings pin compilation and fuel costs.
	cfg := wt.NewConfig()
	cfg.SetConsumeFuel(true)
	cfg.SetCraneliftNanCanonicalization(true)
	cfg.SetWasmRelaxedSIMDDeterministic(true)
	cfg.SetWasmThreads(false)
	cfg.SetWasmMemory64(false)
	cfg.SetWasmMultiMemory(false)
	cfg.SetMaxWasmStack(maxWasmStack)
	cfg.SetStrategy(wt.StrategyCranelift)
	cfg.SetCraneliftOptLevel(wt.OptLevelSpeed)
	engine := wt.NewEngineWithConfig(cfg)
	if err := wt.ModuleValidate(engine, bin); err != nil {
		if closer, ok := err.(interface{ Close() }); ok {
			closer.Close()
		}
		engine.Close()
		return nil, nil, fmt.Errorf("module does not validate: %w", wasm.ErrBadModule)
	}
	instrumented, err := instrumentMemory(bin)
	if err != nil {
		engine.Close()
		return nil, nil, fmt.Errorf("%v: %w", err, wasm.ErrBadModule)
	}
	mod, err := wt.NewModule(engine, instrumented)
	if err != nil {
		if closer, ok := err.(interface{ Close() }); ok {
			closer.Close()
		}
		engine.Close()
		return nil, nil, fmt.Errorf("module does not compile: %w", wasm.ErrBadModule)
	}
	for _, export := range mod.Exports() {
		if export.Name() == "_start" {
			ft := export.Type().FuncType()
			if ft != nil && len(ft.Params()) == 0 && len(ft.Results()) == 0 {
				return engine, mod, nil
			}
			break
		}
	}
	mod.Close()
	engine.Close()
	return nil, nil, fmt.Errorf("_start must be a function with no parameters or results: %w", wasm.ErrBadModule)
}
