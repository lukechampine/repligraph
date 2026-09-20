package wasmtime

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func watModule(t *testing.T, wat string) []byte {
	t.Helper()
	bin, err := Wat2Wasm(wat)
	if err != nil {
		t.Fatalf("wat2wasm: %v", err)
	}
	return bin
}

func testLimits(fuel uint64) wasm.Limits {
	return wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 10000, Fuel: fuel, MemoryBytes: 1 << 30, WriteBytes: 256 << 20, StdoutBytes: 8 << 20, StderrBytes: 8 << 20}
}

func runModule(t *testing.T, bin []byte, tr tree.Tree, mut func(*wasm.Request)) (wasm.Response, tree.Tree) {
	t.Helper()
	req := wasm.Request{Module: bin, Args: []string{"prog"}, Limits: testLimits(1 << 20)}
	if mut != nil {
		mut(&req)
	}
	resp, nt, err := Run(tr, req)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return resp, nt
}

func testTree(t *testing.T, files map[string]string) tree.Tree {
	t.Helper()
	var tr tree.Tree
	for path, content := range files {
		var err error
		tr, err = tr.Put(path, tree.File{Content: []byte(content)})
		if err != nil {
			t.Fatal(err)
		}
	}
	return tr
}

const helloWAT = `(module
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "hello\n")
  (func (export "_start")
    (i32.store (i32.const 16) (i32.const 0))
    (i32.store (i32.const 20) (i32.const 6))
    (drop (call $fd_write (i32.const 1) (i32.const 16) (i32.const 1) (i32.const 24)))))`

func TestHelloStdout(t *testing.T) {
	bin := watModule(t, helloWAT)
	resp, nt := runModule(t, bin, tree.Tree{}, nil)
	if resp.Trap != wasm.TrapNone || resp.ExitCode != 0 {
		t.Fatalf("resp: %+v", resp)
	}
	if string(resp.Stdout) != "hello\n" {
		t.Fatalf("stdout = %q", resp.Stdout)
	}
	if nt.Root() != (tree.Tree{}).Root() {
		t.Fatal("tree changed")
	}
	if resp.Usage.Fuel == 0 {
		t.Fatal("no fuel consumed")
	}
}

const copyWAT = `(module
  (import "wasi_snapshot_preview1" "path_open" (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_read" (func $fd_read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_close" (func $fd_close (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/in.txt")
  (data (i32.const 16) "artifact/out.txt")
  (func (export "_start")
    (if (i32.ne (call $path_open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 15)
        (i32.const 0) (i64.const 2) (i64.const 0) (i32.const 0) (i32.const 100)) (i32.const 0))
      (then (call $proc_exit (i32.const 10))))
    (i32.store (i32.const 200) (i32.const 1024))
    (i32.store (i32.const 204) (i32.const 4096))
    (if (i32.ne (call $fd_read (i32.load (i32.const 100)) (i32.const 200) (i32.const 1) (i32.const 208)) (i32.const 0))
      (then (call $proc_exit (i32.const 11))))
    (drop (call $fd_close (i32.load (i32.const 100))))
    (if (i32.ne (call $path_open (i32.const 3) (i32.const 0) (i32.const 16) (i32.const 16)
        (i32.const 9) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 104)) (i32.const 0))
      (then (call $proc_exit (i32.const 12))))
    (i32.store (i32.const 200) (i32.const 1024))
    (i32.store (i32.const 204) (i32.load (i32.const 208)))
    (if (i32.ne (call $fd_write (i32.load (i32.const 104)) (i32.const 200) (i32.const 1) (i32.const 212)) (i32.const 0))
      (then (call $proc_exit (i32.const 13))))
    (call $proc_exit (i32.const 7))))`

func TestCopyFileAndDeterminism(t *testing.T) {
	bin := watModule(t, copyWAT)
	base := testTree(t, map[string]string{"/artifact/in.txt": "workspace as a value\n"})

	resp1, t1 := runModule(t, bin, base, nil)
	if resp1.Trap != wasm.TrapNone || resp1.ExitCode != 7 {
		t.Fatalf("resp: %+v stderr=%q", resp1, resp1.Stderr)
	}
	f, err := t1.Get("/artifact/out.txt")
	if err != nil || string(f.Content) != "workspace as a value\n" {
		t.Fatalf("out.txt: %q, %v", f.Content, err)
	}
	if resp1.Usage.TreeBytes != 2*uint64(len(f.Content)) {
		t.Fatal("identical file contents at different paths must count separately")
	}

	resp2, t2 := runModule(t, bin, base, nil)
	if !reflect.DeepEqual(resp1, resp2) || t1.Root() != t2.Root() {
		t.Fatalf("nondeterministic: %+v vs %+v", resp1, resp2)
	}
	if _, err := t2.Get("/artifact/out.txt"); err != nil {
		t.Fatal("proc_exit discarded the write")
	}
}

const burnWAT = `(module
  (memory (export "memory") 1)
  (func (export "_start") (loop $l (br $l))))`

func TestFuelExhaustion(t *testing.T) {
	req := wasm.Request{Module: watModule(t, burnWAT), Limits: testLimits(10000)}
	for range 2 {
		resp, nt, err := Run(tree.Tree{}, req)
		checkResourceAbort(t, resp, nt, err, tree.Tree{})
	}
}

const trapWAT = `(module
  (import "wasi_snapshot_preview1" "path_open" (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/junk")
  (func (export "_start")
    (if (call $path_open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 13)
        (i32.const 9) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100))
      (then (call $exit (i32.const 10))))
    (i32.store (i32.const 200) (i32.const 0))
    (i32.store (i32.const 204) (i32.const 4))
    (if (call $fd_write (i32.load (i32.const 100)) (i32.const 200) (i32.const 1) (i32.const 208))
      (then (call $exit (i32.const 11))))
    (i32.store (i32.const 200) (i32.const 0))
    (i32.store (i32.const 204) (i32.const 4))
    (drop (call $fd_write (i32.const 1) (i32.const 200) (i32.const 1) (i32.const 208)))
    (drop (call $fd_write (i32.const 2) (i32.const 200) (i32.const 1) (i32.const 208)))
    unreachable))`

func TestTrapDiscardsWorkspaceKeepsStreams(t *testing.T) {
	bin := watModule(t, trapWAT)
	resp, nt := runModule(t, bin, tree.Tree{}, nil)
	if resp.Trap != wasm.TrapUnreachable {
		t.Fatalf("trap = %q, want unreachable", resp.Trap)
	}
	if nt.Root() != (tree.Tree{}).Root() {
		t.Fatal("trap committed workspace changes")
	}
	if string(resp.Stdout) != "arti" || string(resp.Stderr) != "arti" {
		t.Fatalf("stdout = %q, stderr = %q; want both streams retained", resp.Stdout, resp.Stderr)
	}
}

const clockWAT = `(module
  (import "wasi_snapshot_preview1" "clock_time_get" (func $clock (param i32 i64 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (drop (call $clock (i32.const 1) (i64.const 0) (i32.const 0)))
    (drop (call $clock (i32.const 1) (i64.const 0) (i32.const 8)))
    (drop (call $clock (i32.const 0) (i64.const 0) (i32.const 16)))
    (i32.store (i32.const 100) (i32.const 0))
    (i32.store (i32.const 104) (i32.const 24))
    (drop (call $fd_write (i32.const 1) (i32.const 100) (i32.const 1) (i32.const 108)))))`

func TestDeterministicClock(t *testing.T) {
	bin := watModule(t, clockWAT)
	resp, _ := runModule(t, bin, tree.Tree{}, nil)
	if len(resp.Stdout) != 24 {
		t.Fatalf("stdout len = %d", len(resp.Stdout))
	}
	le := func(off int) uint64 { return binary.LittleEndian.Uint64(resp.Stdout[off:]) }
	t1, t2, rt := le(0), le(8), le(16)
	if t1 != clockQuantumNanos || t2 != 2*clockQuantumNanos {
		t.Fatalf("monotonic reads %d, %d; want quantum steps", t1, t2)
	}
	if rt != clockEpochNanos+3*clockQuantumNanos {
		t.Fatalf("realtime read %d, want epoch + 3 quanta", rt)
	}
}

const randomWAT = `(module
  (import "wasi_snapshot_preview1" "random_get" (func $random (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (drop (call $random (i32.const 0) (i32.const 16)))
    (i32.store (i32.const 100) (i32.const 0))
    (i32.store (i32.const 104) (i32.const 16))
    (drop (call $fd_write (i32.const 1) (i32.const 100) (i32.const 1) (i32.const 108)))))`

func TestSeededRandom(t *testing.T) {
	bin := watModule(t, randomWAT)
	seedA := func(r *wasm.Request) { r.RandomSeed[0] = 1 }
	seedB := func(r *wasm.Request) { r.RandomSeed[0] = 2 }
	r1, _ := runModule(t, bin, tree.Tree{}, seedA)
	r2, _ := runModule(t, bin, tree.Tree{}, seedA)
	r3, _ := runModule(t, bin, tree.Tree{}, seedB)
	if !bytes.Equal(r1.Stdout, r2.Stdout) {
		t.Fatal("same seed, different stream")
	}
	if bytes.Equal(r1.Stdout, r3.Stdout) {
		t.Fatal("different seed, same stream")
	}
	if bytes.Equal(r1.Stdout, make([]byte, 16)) {
		t.Fatal("random stream is zero")
	}
}

const argsWAT = `(module
  (import "wasi_snapshot_preview1" "args_sizes_get" (func $sizes (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "args_get" (func $get (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (drop (call $sizes (i32.const 0) (i32.const 4)))
    (drop (call $get (i32.const 100) (i32.const 1000)))
    (i32.store (i32.const 200) (i32.const 1000))
    (i32.store (i32.const 204) (i32.load (i32.const 4)))
    (drop (call $fd_write (i32.const 1) (i32.const 200) (i32.const 1) (i32.const 208)))))`

func TestArgs(t *testing.T) {
	bin := watModule(t, argsWAT)
	resp, _ := runModule(t, bin, tree.Tree{}, func(r *wasm.Request) {
		r.Args = []string{"prog", "--flag", "value"}
	})
	if string(resp.Stdout) != "prog\x00--flag\x00value\x00" {
		t.Fatalf("args buffer = %q", resp.Stdout)
	}
}

func TestEnvironment(t *testing.T) {
	bin := watModule(t, strings.NewReplacer("args_sizes_get", "environ_sizes_get", "args_get", "environ_get").Replace(argsWAT))
	resp, _ := runModule(t, bin, tree.Tree{}, func(r *wasm.Request) {
		r.Env = []string{"Z=last", "A=first", "EMPTY="}
	})
	if resp.Trap != wasm.TrapNone || string(resp.Stdout) != "Z=last\x00A=first\x00EMPTY=\x00" {
		t.Fatalf("environment response = %+v", resp)
	}
}

func TestUnsignedExitCode(t *testing.T) {
	bin := watModule(t, `(module
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (func (export "_start") (call $exit (i32.const -1))))`)
	resp, _ := runModule(t, bin, tree.Tree{}, nil)
	if resp.ExitCode != ^uint32(0) || resp.Trap != wasm.TrapNone {
		t.Fatalf("response = %+v", resp)
	}
}

func TestBadModule(t *testing.T) {
	base := testTree(t, map[string]string{"/artifact/keep": "unchanged"})
	for _, tc := range []struct {
		name string
		wat  string
	}{
		{"invalid bytes", ""},
		{"missing start", `(module (memory (export "memory") 1))`},
		{"start is global", `(module (memory (export "memory") 1) (global (export "_start") i32 (i32.const 0)))`},
		{"start takes argument", `(module (memory (export "memory") 1) (func (export "_start") (param i32)))`},
		{"start returns value", `(module (memory (export "memory") 1) (func (export "_start") (result i32) (i32.const 7)))`},
		{"unknown import", `(module (import "host" "unknown" (func)) (memory (export "memory") 1) (func (export "_start")))`},
		{"wrong WASI signature", `(module (import "wasi_snapshot_preview1" "proc_exit" (func (param i64))) (memory (export "memory") 1) (func (export "_start")))`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bin := []byte("invalid wasm")
			if tc.wat != "" {
				bin = watModule(t, tc.wat)
			}
			resp, nt, err := Run(base, wasm.Request{Module: bin, Limits: testLimits(10000)})
			if !errors.Is(err, wasm.ErrBadModule) {
				t.Fatalf("error = %v, want ErrBadModule", err)
			}
			if nt.Root() != base.Root() || len(resp.Stdout) != 0 || len(resp.Stderr) != 0 {
				t.Fatal("invalid module had side effects")
			}
		})
	}
}

func TestStartSignatureValidatedBeforeInstantiation(t *testing.T) {
	for _, signature := range []string{"(param i32)", "(result i32) (i32.const 7)"} {
		t.Run(signature, func(t *testing.T) {
			wat := `(module (func $initialize unreachable) (start $initialize)
  (func (export "_start") ` + signature + `))`
			resp, nt, err := Run(tree.Tree{}, wasm.Request{Module: watModule(t, wat), Limits: testLimits(10000)})
			if !errors.Is(err, wasm.ErrBadModule) {
				t.Fatalf("error = %v, want ErrBadModule", err)
			}
			if resp.Usage.Fuel != 0 || len(resp.Stdout) != 0 || nt.Root() != (tree.Tree{}).Root() {
				t.Fatalf("module initialized before signature validation: %+v", resp)
			}
		})
	}
}

func TestModuleStartSection(t *testing.T) {
	for _, tc := range []struct {
		name string
		wat  string
		trap wasm.Trap
		exit uint32
	}{
		{"trap", strings.Replace(trapWAT, `(func (export "_start")`, `(func $initialize`, 1), wasm.TrapUnreachable, 0},
		{"exit", strings.Replace(copyWAT, `(func (export "_start")`, `(func $initialize`, 1), wasm.TrapNone, 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wat := tc.wat[:len(tc.wat)-1] + `(start $initialize) (func (export "_start") unreachable))`
			base := testTree(t, map[string]string{"/artifact/in.txt": "start section"})
			resp, nt := runModule(t, watModule(t, wat), base, nil)
			if resp.Trap != tc.trap || resp.ExitCode != tc.exit {
				t.Fatalf("response = %+v", resp)
			}
			if tc.trap != wasm.TrapNone {
				if nt.Root() != base.Root() || string(resp.Stdout) != "arti" || string(resp.Stderr) != "arti" {
					t.Fatalf("start trap failed to roll back with streams retained: %+v", resp)
				}
			} else if f, err := nt.Get("/artifact/out.txt"); err != nil || string(f.Content) != "start section" {
				t.Fatalf("start exit discarded writes: %q, %v", f.Content, err)
			}
		})
	}
}

func TestNativeStackExhaustion(t *testing.T) {
	bin := watModule(t, `(module (func $recurse (export "_start") (call $recurse)))`)
	base := testTree(t, map[string]string{"/artifact/keep": "unchanged"})
	resp, nt, err := Run(base, wasm.Request{Module: bin, Limits: testLimits(1 << 32)})
	checkResourceAbort(t, resp, nt, err, base)
}

func TestReadOnlyWrites(t *testing.T) {
	for _, path := range []string{"env/original", "env/new"} {
		t.Run(path, func(t *testing.T) {
			wat := `(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "` + path + `")
  (func (export "_start")
    (call $exit (call $open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const ` + fmt.Sprint(len(path)) + `)
      (i32.const 9) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100)))))`
			base := testTree(t, map[string]string{"/env/original": "keep", "/artifact/seed": "also keep"})
			resp, nt := runModule(t, watModule(t, wat), base, nil)
			if resp.Trap != wasm.TrapNone || resp.ExitCode != 69 || nt.Root() != base.Root() {
				t.Fatalf("read-only write: response = %+v, changed = %v", resp, nt.Root() != base.Root())
			}
		})
	}
}

const stdinWAT = `(module
  (import "wasi_snapshot_preview1" "fd_read" (func $fd_read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (i32.store (i32.const 100) (i32.const 1000))
    (i32.store (i32.const 104) (i32.const 4096))
    (drop (call $fd_read (i32.const 0) (i32.const 100) (i32.const 1) (i32.const 108)))
    (i32.store (i32.const 100) (i32.const 1000))
    (i32.store (i32.const 104) (i32.load (i32.const 108)))
    (drop (call $fd_write (i32.const 1) (i32.const 100) (i32.const 1) (i32.const 112)))))`

func TestStdinEcho(t *testing.T) {
	bin := watModule(t, stdinWAT)
	resp, _ := runModule(t, bin, tree.Tree{}, func(r *wasm.Request) {
		r.Stdin = []byte("piped input")
	})
	if string(resp.Stdout) != "piped input" {
		t.Fatalf("stdout = %q", resp.Stdout)
	}
}

const floodWAT = `(module
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (memory (export "memory") 2)
  (func (export "_start")
    (block $done
      (loop $l
        (i32.store (i32.const 70000) (i32.const 0))
        (i32.store (i32.const 70004) (i32.const 65536))
        (br_if $done (i32.ne (call $fd_write (i32.const 1) (i32.const 70000) (i32.const 1) (i32.const 70008)) (i32.const 0)))
        (br $l)))
    (call $proc_exit (i32.const 42))))`

func TestStdoutBudget(t *testing.T) {
	req := wasm.Request{Module: watModule(t, floodWAT), Limits: testLimits(1 << 24)}
	resp, nt, err := Run(tree.Tree{}, req)
	checkResourceAbort(t, resp, nt, err, tree.Tree{})
}
