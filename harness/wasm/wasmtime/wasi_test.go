package wasmtime

import (
	"encoding/binary"
	"fmt"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func TestGuestInputBounds(t *testing.T) {
	const openFile = `(if (call $open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 13)
    (i32.const 0) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100))
    (then (call $exit (i32.const 100))))`
	for _, tc := range []struct {
		name, imports, setup, call string
		errno                      uint32
		content                    string
	}{
		{
			name:    "negative read iovec count",
			imports: `(import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))`,
			call:    `(call $read (i32.const 0) (i32.const 200) (i32.const -1) (i32.const 208))`,
			errno:   21,
		},
		{
			name:    "huge read iovec count",
			imports: `(import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))`,
			call:    `(call $read (i32.const 0) (i32.const 200) (i32.const 536870912) (i32.const 208))`,
			errno:   21,
		},
		{
			name:  "negative write iovec count",
			call:  `(call $write (i32.const 1) (i32.const 200) (i32.const -1) (i32.const 208))`,
			errno: 21,
		},
		{
			name:  "huge write iovec count",
			call:  `(call $write (i32.const 1) (i32.const 200) (i32.const 536870912) (i32.const 208))`,
			errno: 21,
		},
		{
			name:    "negative directory cookie",
			imports: `(import "wasi_snapshot_preview1" "fd_readdir" (func $readdir (param i32 i32 i32 i64 i32) (result i32)))`,
			call:    `(call $readdir (i32.const 3) (i32.const 1024) (i32.const 1024) (i64.const -1) (i32.const 208))`,
			errno:   28,
		},
		{
			name:    "poll count overflow",
			imports: `(import "wasi_snapshot_preview1" "poll_oneoff" (func $poll (param i32 i32 i32 i32) (result i32)))`,
			call:    `(call $poll (i32.const 1024) (i32.const 4096) (i32.const 268435456) (i32.const 208))`,
			errno:   21,
		},
		{
			name:    "negative positional write",
			imports: `(import "wasi_snapshot_preview1" "fd_pwrite" (func $pwrite (param i32 i32 i32 i64 i32) (result i32)))`,
			setup:   openFile,
			call:    `(call $pwrite (i32.load (i32.const 100)) (i32.const 200) (i32.const 1) (i64.const -1) (i32.const 208))`,
			errno:   28,
		},
		{
			name:    "huge resize after write",
			imports: `(import "wasi_snapshot_preview1" "fd_filestat_set_size" (func $resize (param i32 i64) (result i32)))`,
			setup: openFile + `(if (call $write (i32.load (i32.const 100)) (i32.const 200) (i32.const 1) (i32.const 208))
    (then (call $exit (i32.const 101))))`,
			call:    `(call $resize (i32.load (i32.const 100)) (i64.const 9223372036854775807))`,
			errno:   51,
			content: "Xeed",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wat := fmt.Sprintf(`(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  %s
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/file")
  (data (i32.const 32) "X")
  (func (export "_start")
    (i32.store (i32.const 200) (i32.const 32))
    (i32.store (i32.const 204) (i32.const 1))
    %s
    (call $exit %s)))`, tc.imports, tc.setup, tc.call)
			base := testTree(t, map[string]string{"/artifact/file": "seed"})
			if tc.errno == 51 {
				req := wasm.Request{Module: watModule(t, wat), Stdin: []byte("input"), Limits: testLimits(10000)}
				resp, nt, err := Run(base, req)
				checkResourceAbort(t, resp, nt, err, base)
				return
			}
			resp, nt := runModule(t, watModule(t, wat), base, func(req *wasm.Request) { req.Stdin = []byte("input") })
			if resp.Trap != wasm.TrapNone || resp.ExitCode != tc.errno {
				t.Fatalf("response = %+v; want errno %d", resp, tc.errno)
			}
			if len(resp.Stdout) != 0 || len(resp.Stderr) != 0 {
				t.Fatalf("invalid input emitted streams: %+v", resp)
			}
			want := tc.content
			if want == "" {
				want = "seed"
				if nt.Root() != base.Root() {
					t.Fatal("invalid input changed the tree")
				}
			}
			f, err := nt.Get("/artifact/file")
			if err != nil || string(f.Content) != want {
				t.Fatalf("file = %q, %v; want %q", f.Content, err, want)
			}
		})
	}
}

func TestPoll(t *testing.T) {
	le := binary.LittleEndian
	clock := func(user, timeout uint64, absolute bool) []byte {
		sub := make([]byte, 48)
		le.PutUint64(sub, user)
		le.PutUint64(sub[24:], timeout)
		if absolute {
			le.PutUint16(sub[40:], 1)
		} else {
			le.PutUint32(sub[16:], 1)
		}
		return sub
	}
	stdin := make([]byte, 48)
	le.PutUint64(stdin, 30)
	stdin[8] = 1
	type event struct {
		user, nbytes uint64
		kind         byte
	}
	for _, tc := range []struct {
		name  string
		subs  []byte
		alias bool
		time  uint64
		want  []event
	}{
		{"earliest relative clock", append(clock(10, 7000, false), clock(20, 3000, false)...), false, 5000, []event{{user: 20}}},
		{"absolute realtime clock", clock(10, clockEpochNanos+9000, true), false, 10000, []event{{user: 10}}},
		{"ready descriptor", append(clock(10, 100000, false), stdin...), false, 2000, []event{{user: 30, kind: 1, nbytes: 5}}},
		{"aliased events", append(clock(10, 3000, false), clock(20, 3000, false)...), true, 5000, []event{{user: 10}, {user: 20}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var data strings.Builder
			for _, b := range tc.subs {
				fmt.Fprintf(&data, `\%02x`, b)
			}
			out := 1024
			if tc.alias {
				out = 128
			}
			wat := fmt.Sprintf(`(module
  (import "wasi_snapshot_preview1" "poll_oneoff" (func $poll (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "clock_time_get" (func $clock (param i32 i64 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 128) "%s")
  (func (export "_start")
    (drop (call $clock (i32.const 1) (i64.const 0) (i32.const 8)))
    (if (call $poll (i32.const 128) (i32.const %d) (i32.const %d) (i32.const 0))
      (then (call $exit (i32.const 1))))
    (drop (call $clock (i32.const 1) (i64.const 0) (i32.const 8)))
    (i32.store (i32.const 32) (i32.const 0))
    (i32.store (i32.const 36) (i32.const 16))
    (i32.store (i32.const 40) (i32.const %d))
    (i32.store (i32.const 44) (i32.const 64))
    (drop (call $write (i32.const 1) (i32.const 32) (i32.const 2) (i32.const 48)))))`, data.String(), out, len(tc.subs)/48, out)
			resp, _ := runModule(t, watModule(t, wat), tree.Tree{}, func(req *wasm.Request) { req.Stdin = []byte("ready") })
			if resp.Trap != wasm.TrapNone || resp.ExitCode != 0 || len(resp.Stdout) != 80 {
				t.Fatalf("response = %+v", resp)
			}
			if n := le.Uint32(resp.Stdout); n != uint32(len(tc.want)) {
				t.Fatalf("event count = %d, want %d", n, len(tc.want))
			}
			if got := le.Uint64(resp.Stdout[8:]); got != tc.time {
				t.Fatalf("clock = %d, want %d", got, tc.time)
			}
			for i, want := range tc.want {
				b := resp.Stdout[16+i*32:]
				got := event{user: le.Uint64(b), kind: b[10], nbytes: le.Uint64(b[16:])}
				if got != want || le.Uint16(b[8:]) != 0 {
					t.Fatalf("event %d = %+v (errno %d), want %+v", i, got, le.Uint16(b[8:]), want)
				}
			}
		})
	}
}

func TestIOVecTotalOverflow(t *testing.T) {
	bin := watModule(t, `(module
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 16)
  (func (export "_start") (local $p i32)
    (loop $fill
      (i32.store (local.get $p) (i32.const 0))
      (i32.store offset=4 (local.get $p) (i32.const 1048576))
      (local.set $p (i32.add (local.get $p) (i32.const 8)))
      (br_if $fill (i32.lt_u (local.get $p) (i32.const 32768))))
    (call $exit (call $write (i32.const 1) (i32.const 0) (i32.const 4096) (i32.const 32768)))))`)
	resp, nt := runModule(t, bin, tree.Tree{}, nil)
	if resp.Trap != wasm.TrapNone || resp.ExitCode != 28 || len(resp.Stdout) != 0 || nt.Root() != (tree.Tree{}).Root() {
		t.Fatalf("oversized aggregate iovec response = %+v", resp)
	}
}
