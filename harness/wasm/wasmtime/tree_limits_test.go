package wasmtime

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func TestTreeEntryLimitGuest(t *testing.T) {
	// Try to create 1,000 distinct empty files, one-byte files, or empty
	// directories. The entry quota, rather than the write budget, must stop it.
	for _, kind := range []string{"empty files", "one-byte files", "empty directories"} {
		t.Run(kind, func(t *testing.T) {
			create := `(call $open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 14)
              (i32.const 1) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100))`
			after := `(drop (call $close (i32.load (i32.const 100))))`
			if kind == "one-byte files" {
				after = `(if (call $write (i32.load (i32.const 100)) (i32.const 104) (i32.const 1) (i32.const 112))
              (then (call $exit (i32.const 999999))))` + after
			} else if kind == "empty directories" {
				create = `(call $mkdir (i32.const 3) (i32.const 0) (i32.const 14))`
				after = ""
			}
			bin := watModule(t, fmt.Sprintf(`(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "path_create_directory" (func $mkdir (param i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/x0000")
  (func (export "_start") (local $n i32) (local $rc i32)
    (i32.store (i32.const 104) (i32.const 0))
    (i32.store (i32.const 108) (i32.const 1))
    (loop $again
      (i32.store8 (i32.const 10) (i32.add (i32.const 48) (i32.div_u (local.get $n) (i32.const 1000))))
      (i32.store8 (i32.const 11) (i32.add (i32.const 48) (i32.rem_u (i32.div_u (local.get $n) (i32.const 100)) (i32.const 10))))
      (i32.store8 (i32.const 12) (i32.add (i32.const 48) (i32.rem_u (i32.div_u (local.get $n) (i32.const 10)) (i32.const 10))))
      (i32.store8 (i32.const 13) (i32.add (i32.const 48) (i32.rem_u (local.get $n) (i32.const 10))))
      (local.set $rc %s)
      (if (local.get $rc) (then
        (call $exit (i32.add (local.get $n) (i32.mul (local.get $rc) (i32.const 10000))))))
      %s
      (local.set $n (i32.add (local.get $n) (i32.const 1)))
      (br_if $again (i32.lt_u (local.get $n) (i32.const 1000))))
    (call $exit (i32.const 0))))`, create, after))
			base := testTree(t, map[string]string{
				"/env/tools/a": "a", "/env/lib/nested/b": "b",
			})
			for _, limit := range []uint64{0, 1, 2, 10, 1001} {
				t.Run(fmt.Sprint(limit), func(t *testing.T) {
					req := wasm.Request{Module: bin, Limits: testLimits(100000)}
					req.Limits.MaxTreeEntries = limit
					req.Limits.WriteBytes = 0
					req.Limits.MaxTreeBytes = 0
					if kind == "one-byte files" {
						req.Limits.WriteBytes = 1000
						req.Limits.MaxTreeBytes = 1000
					}
					var count uint64
					if limit > 0 {
						count = limit - 1 // /artifact becomes a counted directory.
					}
					var previous wasm.Response
					for i := range 2 {
						resp, nt, err := Run(base, req)
						if limit < 1001 {
							checkResourceAbort(t, resp, nt, err, base)
							continue
						}
						if err != nil || resp.Trap != wasm.TrapNone || resp.ExitCode != 0 || resp.Usage.TreeEntries != 1001 {
							t.Fatalf("Run: %+v, %v; want peak of 1001 entries", resp, err)
						}
						if resp.Usage.TreeBytes != req.Limits.MaxTreeBytes {
							t.Fatalf("tree bytes: %d, want %d", resp.Usage.TreeBytes, req.Limits.MaxTreeBytes)
						}
						if i > 0 && !reflect.DeepEqual(resp, previous) {
							t.Fatal("entry budget did not reset deterministically")
						}
						previous = resp
						task, _ := nt.Without("env")
						if kind == "empty directories" {
							if nt.Root() != base.Root() {
								t.Fatal("empty directories must disappear at commit")
							}
						} else {
							var wantBytes uint64
							if kind == "one-byte files" {
								wantBytes = count
							}
							if got := task.Stats(); uint64(got.Files) != count || uint64(got.Entries) > limit || uint64(got.Bytes) != wantBytes {
								t.Fatalf("committed entries: %+v", got)
							}
						}
						for p, want := range map[string]string{"/env/tools/a": "a", "/env/lib/nested/b": "b"} {
							if f, err := nt.Get(p); err != nil || string(f.Content) != want {
								t.Fatalf("environment file %s changed", p)
							}
						}
					}
				})
			}
		})
	}
}

func TestTreeEntryLimitInitial(t *testing.T) {
	base := testTree(t, map[string]string{"/input/a/b": ""})
	req := wasm.Request{Module: []byte("not wasm"), Limits: wasm.Limits{MaxTreeEntries: 2}}
	resp, nt, err := Run(base, req)
	if !errors.Is(err, wasm.ErrResourceLimit) || nt.Root() != base.Root() || !reflect.DeepEqual(resp, wasm.Response{}) {
		t.Fatalf("oversized initial tree: %+v, %v", resp, err)
	}
	// Zero permits an empty task tree, even with a nonempty environment.
	base = testTree(t, map[string]string{"/env/a/b": ""})
	req.Module = watModule(t, `(module (func (export "_start")))`)
	req.Limits = wasm.Limits{Fuel: 100}
	if resp, nt, err := Run(base, req); err != nil || resp.Trap != wasm.TrapNone || nt.Root() != base.Root() {
		t.Fatalf("empty task tree with zero limit: %+v, %v", resp, err)
	}
}

func TestTreeEntryLimitRecovery(t *testing.T) {
	o := newOverlay(tree.Tree{}, 4, 1<<20)
	check := func(want errno, run func() errno, entries uint64) {
		t.Helper()
		got := errno(0)
		if want == errNospc {
			expectResourcePanic(t, func() { run() })
			got = errNospc
		} else {
			got = run()
		}
		if got != want || o.entries != entries {
			t.Fatalf("operation: errno %d, entries %d; want %d, %d", got, o.entries, want, entries)
		}
	}
	check(errSuccess, func() errno { return o.setFile("/artifact/a/file", nil) }, 3)
	check(errSuccess, func() errno { return o.mkdir("/artifact/empty") }, 4)
	check(errNospc, func() errno { return o.setFile("/artifact/full", nil) }, 4)
	check(errSuccess, func() errno { return o.setFile("/artifact/a/file", []byte("overwrite")) }, 4)
	check(errSuccess, func() errno { return o.unlink("/artifact/a/file") }, 3)
	if !o.dirExists("/artifact/a") {
		t.Fatal("unlink must preserve its empty parent directory")
	}
	check(errSuccess, func() errno { return o.setFile("/artifact/recovered", nil) }, 4)
	check(errSuccess, func() errno { return o.rmdir("/artifact/empty") }, 3)
	check(errSuccess, func() errno { return o.mkdir("/artifact/reused") }, 4)
	check(errNospc, func() errno { return o.setFile("/artifact/deep/new/file", nil) }, 4)
	if o.dirExists("/artifact/deep") || o.fileExists("/artifact/deep/new/file") {
		t.Fatal("failed creation left partial parents or a file")
	}
}

func TestTreeEntryLimitGuestRecovery(t *testing.T) {
	bin := watModule(t, `(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "path_unlink_file" (func $unlink (param i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "path_rename" (func $rename (param i32 i32 i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/a")
  (data (i32.const 16) "artifact/b")
  (data (i32.const 32) "artifact/c")
  (func $create (param $p i32) (param $flags i32) (result i32) (local $rc i32)
    (local.set $rc (call $open (i32.const 3) (i32.const 0) (local.get $p) (i32.const 10)
      (local.get $flags) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100)))
    (if (i32.eqz (local.get $rc)) (then (drop (call $close (i32.load (i32.const 100))))))
    (local.get $rc))
  (func $check (param $got i32) (param $want i32)
    (if (i32.ne (local.get $got) (local.get $want)) (then (call $exit (i32.const 100)))))
  (func (export "_start")
    (call $check (call $create (i32.const 0) (i32.const 9)) (i32.const 0))
    (call $check (call $unlink (i32.const 3) (i32.const 16) (i32.const 10)) (i32.const 0))
    (call $check (call $create (i32.const 32) (i32.const 1)) (i32.const 0))
    (call $check (call $rename (i32.const 3) (i32.const 0) (i32.const 10)
      (i32.const 3) (i32.const 32) (i32.const 10)) (i32.const 0))
    (call $check (call $create (i32.const 16) (i32.const 1)) (i32.const 0))))`)
	base := testTree(t, map[string]string{"/artifact/a": "old a", "/artifact/b": "old b"})
	resp, nt := runModule(t, bin, base, func(req *wasm.Request) { req.Limits.MaxTreeEntries = 3 })
	want := testTree(t, map[string]string{"/artifact/b": "", "/artifact/c": ""})
	if resp.Trap != wasm.TrapNone || resp.ExitCode != 0 || nt.Root() != want.Root() {
		t.Fatalf("overwrite/delete/rename at entry cap: %+v", resp)
	}
}

func TestTreeEntryLimitRename(t *testing.T) {
	for _, kind := range []string{"file", "directory"} {
		t.Run(kind, func(t *testing.T) {
			base := testTree(t, map[string]string{"/artifact/src": "source", "/artifact/target": "target"})
			if kind == "directory" {
				base = testTree(t, map[string]string{"/artifact/src/nested/file": "source", "/artifact/target/file": "target"})
			}
			o := newOverlay(base, uint64(base.Stats().Entries), 1<<20)
			if kind == "directory" {
				if rc := o.unlink("/artifact/target/file"); rc != errSuccess {
					t.Fatal(rc)
				}
				o.maxEntries-- // no spare quota after emptying the target
			}
			before := o.entries
			if rc := o.rename("/artifact/src", "/artifact/moved"); rc != errSuccess || o.entries != before {
				t.Fatalf("rename at cap: %d, entries %d; want %d", rc, o.entries, before)
			}
			expectResourcePanic(t, func() { o.rename("/artifact/moved", "/artifact/new/parent/child") })
			if o.entries != before {
				t.Fatal("failed rename changed usage")
			}
			if o.dirExists("/artifact/new") {
				t.Fatal("failed rename left parents")
			}
			if rc := o.rename("/artifact/moved", "/artifact/target"); rc != errSuccess || o.entries != before-1 {
				t.Fatalf("replace at cap: %d, entries %d; want %d", rc, o.entries, before-1)
			}
			if rc := o.setFile("/artifact/refunded", nil); rc != errSuccess || o.entries != before {
				t.Fatalf("replace did not release target entry: %d, %d", rc, o.entries)
			}
			nt, err := o.commit()
			path := "/artifact/target"
			if kind == "directory" {
				path += "/nested/file"
			}
			if f, getErr := nt.Get(path); err != nil || getErr != nil || string(f.Content) != "source" {
				t.Fatalf("moved content: %q, %v, %v", f.Content, err, getErr)
			}
		})
	}
}

func expectResourcePanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if _, ok := recover().(resourceAbort); !ok {
			t.Fatal("expected resource abort")
		}
	}()
	fn()
}
