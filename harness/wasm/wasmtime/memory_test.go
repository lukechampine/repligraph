package wasmtime

import (
	"bytes"
	"fmt"
	"testing"

	wt "github.com/bytecodealliance/wasmtime-go/v38"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func TestMemoryRanges(t *testing.T) {
	// Wasmtime reserves virtual memory without committing all 4 GiB. Inspect
	// only slice bounds here; no operation should touch the entire allocation.
	engine := wt.NewEngine()
	defer engine.Close()
	store := wt.NewStore(engine)
	defer store.Close()
	ty := wt.NewMemoryType(65536, true, 65536, false)
	defer ty.Close()
	memory, err := wt.NewMemory(store, ty)
	if err != nil {
		t.Fatal(err)
	}
	mem := memory.UnsafeData(store)
	for _, tc := range []struct {
		name   string
		ptr, n uint32
		ok     bool
	}{
		{"cross signed boundary", 1<<31 - 1, 4, true},
		{"high address", 1 << 31, 4, true},
		{"high length", 1, 1 << 31, true},
		{"last byte", 1<<32 - 1, 1, true},
		{"largest length", 0, 1<<32 - 1, true},
		{"exact end", 1, 1<<32 - 1, true},
		{"empty high address", 1<<32 - 1, 0, true},
		{"address wrap", 1<<32 - 1, 2, false},
		{"length wrap", 2, 1<<32 - 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, ok := memRange(mem, int32(tc.ptr), int32(tc.n))
			if ok != tc.ok {
				t.Fatalf("memRange(%d, %d) valid = %v, want %v", tc.ptr, tc.n, ok, tc.ok)
			}
			if ok && (uint64(len(b)) != uint64(tc.n) || len(b) != cap(b)) {
				t.Fatalf("slice length/capacity = %d/%d, want %d", len(b), cap(b), tc.n)
			}
		})
	}
	if _, ok := memBytes(mem, 1, ^uint64(0)); ok {
		t.Fatal("accepted overflowing widened length")
	}
	for _, tc := range []struct{ ptr, n int32 }{{65536, 1}, {-1, 1}, {1, -1}} {
		if _, ok := memRange(mem[:65536], tc.ptr, tc.n); ok {
			t.Fatalf("accepted range outside small memory: %+v", tc)
		}
	}
}

func TestGuestHighMemory(t *testing.T) {
	// Exercise direct host writes, high-address iovec tables/buffers, and argument
	// pointer generation through the actual WASI imports. Both ranges cross the
	// signed i32 boundary; the final argument's terminator is at the end of memory.
	bin := watModule(t, `(module
  (import "wasi_snapshot_preview1" "random_get" (func $random (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "args_get" (func $args (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 65536)
  (func $check (param $rc i32)
    (if (local.get $rc) (then (call $exit (local.get $rc)))))
  (func (export "_start")
    (call $check (call $random (i32.const 2147483647) (i32.const 4)))
    (call $check (call $args (i32.const 2147483664) (i32.const 4294967291)))
    (i32.store (i32.const 2147483680) (i32.const 2147483647))
    (i32.store (i32.const 2147483684) (i32.const 4))
    (i32.store (i32.const 2147483688) (i32.load (i32.const 2147483664)))
    (i32.store (i32.const 2147483692) (i32.const 5))
    (call $check (call $write (i32.const 1) (i32.const 2147483680) (i32.const 2) (i32.const 2147483700)))))`)
	resp, nt := runModule(t, bin, tree.Tree{}, func(req *wasm.Request) { req.Limits.MemoryBytes = 1 << 32 })
	if resp.Trap != wasm.TrapNone || resp.ExitCode != 0 || len(resp.Stdout) != 9 || !bytes.Equal(resp.Stdout[4:], []byte("prog\x00")) {
		t.Fatalf("high-memory response = %+v", resp)
	}
	if nt.Root() != (tree.Tree{}).Root() {
		t.Fatal("high-memory calls changed the tree")
	}
}

func TestGuestMemoryWrap(t *testing.T) {
	for _, tc := range []struct {
		ptr, n uint32
		errno  uint32
	}{
		{1<<32 - 1, 1, 0},
		{1<<32 - 1, 2, uint32(errFault)},
	} {
		t.Run(fmt.Sprintf("%d/%d", tc.ptr, tc.n), func(t *testing.T) {
			bin := watModule(t, fmt.Sprintf(`(module
  (import "wasi_snapshot_preview1" "random_get" (func $random (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 65536)
  (func (export "_start") (call $exit (call $random (i32.const %d) (i32.const %d)))))`, tc.ptr, tc.n))
			resp, _ := runModule(t, bin, tree.Tree{}, func(req *wasm.Request) { req.Limits.MemoryBytes = 1 << 32 })
			if resp.Trap != wasm.TrapNone || resp.ExitCode != tc.errno {
				t.Fatalf("memory boundary response = %+v, want errno %d", resp, tc.errno)
			}
		})
	}
}
