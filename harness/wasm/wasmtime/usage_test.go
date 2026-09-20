package wasmtime

import (
	"fmt"
	"reflect"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func TestUsageAndExactReplay(t *testing.T) {
	for _, trap := range []bool{false, true} {
		t.Run(fmt.Sprint(trap), func(t *testing.T) {
			end := `(call $exit (i32.const 7))`
			if trap {
				end = `unreachable`
			}
			bin := watModule(t, fmt.Sprintf(`(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "path_create_directory" (func $mkdir (param i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "path_remove_directory" (func $rmdir (param i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_pwrite" (func $pwrite (param i32 i32 i32 i64 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/file")
  (data (i32.const 16) "artifact/temp")
  (data (i32.const 32) "abc")
  (func (export "_start")
    (drop (memory.grow (i32.const 1)))
    (drop (call $open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 13)
      (i32.const 1) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100)))
    (i32.store (i32.const 104) (i32.const 32))
    (i32.store (i32.const 108) (i32.const 3))
    (drop (call $write (i32.load (i32.const 100)) (i32.const 104) (i32.const 1) (i32.const 112)))
    (drop (call $pwrite (i32.load (i32.const 100)) (i32.const 104) (i32.const 1) (i64.const 0) (i32.const 112)))
    (drop (call $write (i32.const 1) (i32.const 104) (i32.const 1) (i32.const 112)))
    (drop (call $write (i32.const 2) (i32.const 104) (i32.const 1) (i32.const 112)))
    (drop (call $mkdir (i32.const 3) (i32.const 16) (i32.const 13)))
    (drop (call $rmdir (i32.const 3) (i32.const 16) (i32.const 13)))
    %s))`, end))
			req := wasm.Request{Module: bin, Limits: testLimits(100000)}
			baseline, final, err := Run(tree.Tree{}, req)
			if err != nil {
				t.Fatal(err)
			}
			wantUsage := wasm.Usage{Fuel: baseline.Usage.Fuel, MemoryBytes: 131072, WriteBytes: 6, StdoutBytes: 3, StderrBytes: 3, TreeEntries: 3, TreeBytes: 3}
			if baseline.Usage != wantUsage || baseline.Usage.Fuel == 0 || string(baseline.Stdout) != "abc" || string(baseline.Stderr) != "abc" {
				t.Fatalf("usage: %+v", baseline)
			}
			if trap {
				if baseline.Trap != wasm.TrapUnreachable || final.Root() != (tree.Tree{}).Root() {
					t.Fatalf("semantic trap: %+v", baseline)
				}
			} else if f, err := final.Get("/artifact/file"); err != nil || string(f.Content) != "abc" || baseline.ExitCode != 7 {
				t.Fatalf("normal exit: %+v, %v", baseline, err)
			}
			u := baseline.Usage
			exact := wasm.Limits{Fuel: u.Fuel, MemoryBytes: u.MemoryBytes, WriteBytes: u.WriteBytes, StdoutBytes: u.StdoutBytes, StderrBytes: u.StderrBytes, MaxTreeEntries: u.TreeEntries, MaxTreeBytes: u.TreeBytes}
			req.Limits = exact
			got, nt, err := Run(tree.Tree{}, req)
			if err != nil || !reflect.DeepEqual(got, baseline) || nt.Root() != final.Root() {
				t.Fatalf("exact usage replay: %+v, %v; want %+v", got, err, baseline)
			}
			for _, resource := range []string{"fuel", "memory", "write", "stdout", "stderr", "tree", "tree bytes"} {
				t.Run(resource, func(t *testing.T) {
					req.Limits = exact
					switch resource {
					case "fuel":
						req.Limits.Fuel--
					case "memory":
						req.Limits.MemoryBytes--
					case "write":
						req.Limits.WriteBytes--
					case "stdout":
						req.Limits.StdoutBytes--
					case "stderr":
						req.Limits.StderrBytes--
					case "tree":
						req.Limits.MaxTreeEntries--
					case "tree bytes":
						req.Limits.MaxTreeBytes--
					}
					resp, nt, err := Run(tree.Tree{}, req)
					checkResourceAbort(t, resp, nt, err, tree.Tree{})
				})
			}
		})
	}
}

func TestMemoryUsageInStartAndHiddenMemory(t *testing.T) {
	for _, trap := range []bool{false, true} {
		end := ""
		if trap {
			end = "unreachable"
		}
		bin := watModule(t, fmt.Sprintf(`(module (memory 1)
          (func $init (drop (memory.grow (i32.const 1))) %s) (start $init)
          (func (export "_start") (drop (memory.grow (i32.const 1)))))`, end))
		req := wasm.Request{Module: bin, Limits: testLimits(10000)}
		resp, nt, err := Run(tree.Tree{}, req)
		wantMemory := uint64(3 * 65536)
		wantTrap := wasm.TrapNone
		if trap {
			wantMemory, wantTrap = 2*65536, wasm.TrapUnreachable
		}
		if err != nil || resp.Usage.MemoryBytes != wantMemory || resp.Trap != wantTrap {
			t.Fatalf("startup memory usage: %+v, %v", resp, err)
		}
		req.Limits.MemoryBytes, req.Limits.Fuel = resp.Usage.MemoryBytes, resp.Usage.Fuel
		replayed, _, err := Run(tree.Tree{}, req)
		if err != nil || !reflect.DeepEqual(resp, replayed) {
			t.Fatalf("exact startup usage: %+v, %v", replayed, err)
		}
		req.Limits.MemoryBytes = 65536
		resp, nt, err = Run(tree.Tree{}, req)
		checkResourceAbort(t, resp, nt, err, tree.Tree{})
	}
}

func TestDeclaredMemoryMaximum(t *testing.T) {
	// Exceeding the program's own declared maximum has defined guest
	// semantics, regardless of whether the local ceiling equals that maximum.
	bin := watModule(t, `(module
      (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
      (memory 1 1)
      (func (export "_start") (call $exit (memory.grow (i32.const 1)))))`)
	for _, limit := range []uint64{65536, 131072} {
		req := wasm.Request{Module: bin, Limits: testLimits(10000)}
		req.Limits.MemoryBytes = limit
		resp, _, err := Run(tree.Tree{}, req)
		if err != nil || resp.ExitCode != ^uint32(0) || resp.Trap != wasm.TrapNone || resp.Usage.MemoryBytes != 65536 {
			t.Fatalf("declared maximum: %+v, %v", resp, err)
		}
	}
}

func TestTreeByteUsage(t *testing.T) {
	// Grow and shrink one file, sparsely grow another, replace the first by
	// rename, then delete and truncate. The final tree is smaller than the
	// initial tree, but replay must retain the highest intermediate size.
	for _, trap := range []bool{false, true} {
		t.Run(fmt.Sprint(trap), func(t *testing.T) {
			end := `(call $exit (i32.const 7))`
			if trap {
				end = `unreachable`
			}
			bin := watModule(t, fmt.Sprintf(`(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_pwrite" (func $write (param i32 i32 i32 i64 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_filestat_set_size" (func $resize (param i32 i64) (result i32)))
  (import "wasi_snapshot_preview1" "path_rename" (func $rename (param i32 i32 i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "path_unlink_file" (func $unlink (param i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/a")
  (data (i32.const 16) "artifact/b")
  (data (i32.const 32) "artifact/c")
  (data (i32.const 48) "XY")
  (func $check (param $rc i32) (if (local.get $rc) (then unreachable)))
  (func $openat (param $path i32) (result i32)
    (call $check (call $open (i32.const 3) (i32.const 0) (local.get $path) (i32.const 10)
      (i32.const 1) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100)))
    (i32.load (i32.const 100)))
  (func $put (param $fd i32) (param $offset i64) (param $n i32)
    (i32.store (i32.const 200) (i32.const 48))
    (i32.store (i32.const 204) (local.get $n))
    (call $check (call $write (local.get $fd) (i32.const 200) (i32.const 1) (local.get $offset) (i32.const 208))))
  (func (export "_start") (local $a i32) (local $b i32) (local $c i32)
    (local.set $a (call $openat (i32.const 0)))
    (local.set $b (call $openat (i32.const 16)))
    (call $check (call $resize (local.get $a) (i64.const 6)))
    (call $put (local.get $a) (i64.const 0) (i32.const 1))
    (call $check (call $resize (local.get $a) (i64.const 2)))
    (call $put (local.get $b) (i64.const 6) (i32.const 1))
    (call $check (call $rename (i32.const 3) (i32.const 16) (i32.const 10) (i32.const 3) (i32.const 0) (i32.const 10)))
    (local.set $c (call $openat (i32.const 32)))
    (call $put (local.get $c) (i64.const 0) (i32.const 2))
    (call $check (call $unlink (i32.const 3) (i32.const 0) (i32.const 10)))
    (call $check (call $resize (local.get $c) (i64.const 0)))
    %s))`, end))
			base := testTree(t, map[string]string{"/artifact/a": "abcd", "/artifact/b": "uv", "/input/keep": "k", "/env/large": "environment content does not count"})
			want := testTree(t, map[string]string{"/artifact/c": "", "/input/keep": "k", "/env/large": "environment content does not count"})
			if trap {
				want = base
			}
			req := wasm.Request{Module: bin, Limits: testLimits(10000)}
			var baseline wasm.Response
			for _, cap := range []uint64{0, 6, 9, 10, 100} {
				req.Limits.MaxTreeBytes = cap
				resp, nt, err := Run(base, req)
				if cap < 10 {
					checkResourceAbort(t, resp, nt, err, base)
					continue
				}
				if err != nil || resp.Usage.TreeBytes != 10 || resp.Usage.WriteBytes != 10 || nt.Root() != want.Root() {
					t.Fatalf("tree byte accounting: %+v, %v", resp, err)
				}
				if trap && resp.Trap != wasm.TrapUnreachable || !trap && (resp.Trap != wasm.TrapNone || resp.ExitCode != 7) {
					t.Fatalf("outcome: %+v", resp)
				}
				if cap == 10 {
					baseline = resp
				} else if !reflect.DeepEqual(resp, baseline) {
					t.Fatal("tree byte ceiling changed the recorded outcome")
				}
			}
			u := baseline.Usage
			req.Limits = wasm.Limits{Fuel: u.Fuel, MemoryBytes: u.MemoryBytes, WriteBytes: u.WriteBytes, StdoutBytes: u.StdoutBytes, StderrBytes: u.StderrBytes, MaxTreeEntries: u.TreeEntries, MaxTreeBytes: u.TreeBytes}
			resp, nt, err := Run(base, req)
			if err != nil || !reflect.DeepEqual(resp, baseline) || nt.Root() != want.Root() {
				t.Fatalf("exact peak replay: %+v, %v", resp, err)
			}
		})
	}
}
