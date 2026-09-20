package wasmtime

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func TestFuelLimits(t *testing.T) {
	bin := watModule(t, `(module (func (export "_start")))`)
	resp, _ := runModule(t, bin, tree.Tree{}, nil)
	if resp.Usage.Fuel == 0 {
		t.Fatal("empty function used no fuel")
	}
	for _, fuel := range []uint64{0, resp.Usage.Fuel, resp.Usage.Fuel + 1, math.MaxUint64} {
		t.Run(fmt.Sprint(fuel), func(t *testing.T) {
			got, nt, err := Run(tree.Tree{}, wasm.Request{Module: bin, Limits: wasm.Limits{Fuel: fuel}})
			if fuel < resp.Usage.Fuel {
				checkResourceAbort(t, got, nt, err, tree.Tree{})
			} else if err != nil || got.Trap != wasm.TrapNone || got.Usage.Fuel != resp.Usage.Fuel {
				t.Fatalf("fuel %d: %+v, %v; want %d consumed", fuel, got, err, resp.Usage.Fuel)
			}
		})
	}
}

func TestMemoryLimits(t *testing.T) {
	bin := watModule(t, `(module
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (func (export "_start") (call $exit (memory.grow (i32.const 1)))))`)
	for _, limit := range []uint64{0, 65535, 65536, 131071, 131072, math.MaxUint64} {
		t.Run(fmt.Sprint(limit), func(t *testing.T) {
			req := wasm.Request{Module: bin, Limits: wasm.Limits{Fuel: 100, MemoryBytes: limit}}
			for range 2 {
				resp, nt, err := Run(tree.Tree{}, req)
				if nt.Root() != (tree.Tree{}).Root() {
					t.Fatal("memory limit changed tree")
				}
				if limit < 131072 {
					checkResourceAbort(t, resp, nt, err, tree.Tree{})
				} else if err != nil || resp.Trap != wasm.TrapNone || resp.ExitCode != 1 || resp.Usage.MemoryBytes != 131072 {
					t.Fatalf("memory.grow: %+v, %v", resp, err)
				}
			}
		})
	}
}

func TestStreamLimits(t *testing.T) {
	bin := watModule(t, `(module
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "abcde")
  (func $send (param $fd i32) (result i32)
    (call $write (local.get $fd) (i32.const 100) (i32.const 2) (i32.const 116)))
  (func (export "_start")
    (i32.store (i32.const 100) (i32.const 0))
    (i32.store (i32.const 104) (i32.const 3))
    (i32.store (i32.const 108) (i32.const 3))
    (i32.store (i32.const 112) (i32.const 2))
    (drop (call $send (i32.const 1)))
    (drop (call $send (i32.const 2)))
    (call $exit (i32.add (call $send (i32.const 1))
      (i32.mul (call $send (i32.const 2)) (i32.const 100))))))`)
	for _, tc := range []struct{ stdout, stderr uint64 }{
		{0, 0}, {3, 7}, {5, 5}, {10, 10}, {math.MaxUint64, math.MaxUint64},
	} {
		t.Run(fmt.Sprintf("%d/%d", tc.stdout, tc.stderr), func(t *testing.T) {
			req := wasm.Request{Module: bin, Limits: wasm.Limits{Fuel: 1000, MemoryBytes: 65536, StdoutBytes: tc.stdout, StderrBytes: tc.stderr}}
			var previous wasm.Response
			for i := range 2 {
				resp, nt, err := Run(tree.Tree{}, req)
				if tc.stdout < 10 || tc.stderr < 10 {
					checkResourceAbort(t, resp, nt, err, tree.Tree{})
					continue
				}
				if err != nil || resp.Trap != wasm.TrapNone || resp.ExitCode != 0 || string(resp.Stdout) != "abcdeabcde" || string(resp.Stderr) != "abcdeabcde" || resp.Usage.StdoutBytes != 10 || resp.Usage.StderrBytes != 10 {
					t.Fatalf("streams: %+v, %v", resp, err)
				}
				if i > 0 && !reflect.DeepEqual(resp, previous) {
					t.Fatal("stream limits did not reset")
				}
				previous = resp
			}
		})
	}
}

func TestWriteLimits(t *testing.T) {
	write := func(off uint64, n int) string {
		return fmt.Sprintf(`(call $write (i64.const %d) (i32.const %d))`, off, n)
	}
	resize := func(n int) string { return fmt.Sprintf(`(call $resize (i32.load (i32.const 100)) (i64.const %d))`, n) }
	checked := func(calls ...string) string {
		var out string
		for _, call := range calls {
			out += `(if ` + call + ` (then (call $exit (i32.const 100))))`
		}
		return out
	}
	for _, tc := range []struct {
		name      string
		limit     uint64
		setup     string
		call      string
		want      string
		wantErrno uint32
	}{
		{"zero write", 0, "", write(0, 1), "seed", 51},
		{"zero empty write", 0, "", write(math.MaxInt64, 0), "seed", 0},
		{"exact write", 1, "", write(0, 1), "Xeed", 0},
		{"oversized atomic write", 1, "", write(0, 2), "seed", 51},
		{"overwrite accounting", 1, checked(write(0, 1)), write(1, 1), "Xeed", 51},
		{"sparse exact", 3, "", write(6, 1), "seed\x00\x00X", 0},
		{"sparse too large", 2, "", write(6, 1), "seed", 51},
		{"resize exact", 2, "", resize(6), "seed\x00\x00", 0},
		{"resize too large", 1, "", resize(6), "seed", 51},
		{"resize cumulative", 2, checked(resize(5), resize(6)), resize(7), "seed\x00\x00", 51},
		{"shrink no refund", 2, checked(resize(6), resize(2)), resize(3), "se", 51},
		{"zero same size", 0, "", resize(4), "seed", 0},
		{"zero shrink", 0, "", resize(2), "se", 0},
		{"growth then write", 3, checked(resize(6), write(0, 1)), write(1, 1), "Xeed\x00\x00", 51},
		{"write then growth", 3, checked(write(0, 2)), resize(6), "XYed", 51},
		{"huge limit", math.MaxUint64, "", write(0, 2), "XYed", 0},
		{"unrepresentable end", math.MaxUint64, "", write(math.MaxInt64, 2), "seed", 51},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bin := watModule(t, fmt.Sprintf(`(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_pwrite" (func $pwrite (param i32 i32 i32 i64 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_filestat_set_size" (func $resize (param i32 i64) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/file")
  (data (i32.const 32) "XY")
  (func $write (param $off i64) (param $n i32) (result i32)
    (i32.store (i32.const 200) (i32.const 32))
    (i32.store (i32.const 204) (local.get $n))
    (call $pwrite (i32.load (i32.const 100)) (i32.const 200) (i32.const 1) (local.get $off) (i32.const 208)))
  (func (export "_start")
    (if (call $open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 13)
      (i32.const 0) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100))
      (then (call $exit (i32.const 101))))
    %s
    (call $exit %s)))`, tc.setup, tc.call))
			base := testTree(t, map[string]string{"/artifact/file": "seed"})
			req := wasm.Request{Module: bin, Limits: wasm.Limits{Fuel: 1000, MemoryBytes: 65536, WriteBytes: tc.limit, MaxTreeEntries: 2, MaxTreeBytes: 1 << 20}}
			for range 2 {
				resp, nt, err := Run(base, req)
				if tc.wantErrno != 0 {
					checkResourceAbort(t, resp, nt, err, base)
					continue
				}
				if err != nil || resp.Trap != wasm.TrapNone || resp.ExitCode != tc.wantErrno {
					t.Fatalf("Run: %+v, %v; want errno %d", resp, err, tc.wantErrno)
				}
				if f, err := nt.Get("/artifact/file"); err != nil || string(f.Content) != tc.want {
					t.Fatalf("file: %q, %v; want %q", f.Content, err, tc.want)
				}
			}
		})
	}
}

func TestLimitedWritesRollback(t *testing.T) {
	req := wasm.Request{Module: watModule(t, trapWAT), Limits: testLimits(10000)}
	req.Limits.WriteBytes = 4
	req.Limits.StdoutBytes = 2
	req.Limits.StderrBytes = 3
	resp, nt, err := Run(tree.Tree{}, req)
	checkResourceAbort(t, resp, nt, err, tree.Tree{})
}

func TestWriteOffsetOverflow(t *testing.T) {
	bin := watModule(t, strings.ReplaceAll(`(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_seek" (func $seek (param i32 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/file")
  (func (export "_start") (local $fd i32)
    (drop (call $open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 13)
      (i32.const 0) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100)))
    (local.set $fd (i32.load (i32.const 100)))
    (drop (call $seek (local.get $fd) (i64.const MAX) (i32.const 0) (i32.const 120)))
    (drop (call $seek (local.get $fd) (i64.const MAX) (i32.const 1) (i32.const 120)))
    (i32.store (i32.const 200) (i32.const 0))
    (i32.store (i32.const 204) (i32.const 4))
    (call $exit (call $write (local.get $fd) (i32.const 200) (i32.const 1) (i32.const 208)))))`, "MAX", fmt.Sprint(math.MaxInt64)))
	base := testTree(t, map[string]string{"/artifact/file": "seed"})
	req := wasm.Request{Module: bin, Limits: testLimits(10000)}
	req.Limits.WriteBytes = math.MaxUint64
	resp, nt, err := Run(base, req)
	checkResourceAbort(t, resp, nt, err, base)
}

func checkResourceAbort(t *testing.T, resp wasm.Response, nt tree.Tree, err error, initial tree.Tree) {
	t.Helper()
	if !errors.Is(err, wasm.ErrResourceLimit) || nt.Root() != initial.Root() || !reflect.DeepEqual(resp, wasm.Response{}) {
		t.Fatalf("resource abort returned a result or changed the tree: %+v, %v", resp, err)
	}
}
