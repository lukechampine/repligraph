package tools_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/tools"
	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
	"lukechampine.com/repligraph/harness/wasm/wasmtime"
)

func runTree(t *testing.T, module []byte) tree.Tree {
	t.Helper()
	tr, err := (tree.Tree{}).Put("/env/program.wasm", tree.File{Content: module})
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

func TestRunRequestAndResult(t *testing.T) {
	module := []byte{0, 'a', 's', 'm', 0xff}
	initial := runTree(t, module)
	limits := wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, Fuel: 100, MemoryBytes: 1 << 30, WriteBytes: 256 << 20, StdoutBytes: 8 << 20, StderrBytes: 8 << 20}
	committed, err := initial.Put("/artifact/output", tree.File{Content: []byte("changed")})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, args string
		argv       []string
		stdin      string
		response   wasm.Response
		want       string
		commit     bool
	}{
		{"defaults", `{"module":"/env/program.wasm"}`, []string{"/env/program.wasm"}, "", wasm.Response{Usage: wasm.Usage{Fuel: 3}}, "<result ok>\nexit 0 (fuel used: 3)\n--- stdout ---\n\n--- stderr ---\n\n</result>", true},
		{"nonzero exit", `{"module":"/env/program.wasm","args":["a","","β"],"stdin":"input\n"}`, []string{"/env/program.wasm", "a", "", "β"}, "input\n", wasm.Response{ExitCode: 7, Usage: wasm.Usage{Fuel: 20, StdoutBytes: 3, StderrBytes: 3}, Stdout: []byte{0xff, 0, '\n'}, Stderr: []byte("err")}, "<result error>\nexit 7 (fuel used: 20)\n--- stdout ---\n\xff\x00\n\n--- stderr ---\nerr\n</result>", true},
		{"trap", `{"module":"/env/program.wasm","args":[]}`, []string{"/env/program.wasm"}, "", wasm.Response{Trap: wasm.TrapUnreachable, Usage: wasm.Usage{Fuel: 100, StdoutBytes: 11, TreeEntries: 5, TreeBytes: 11}, Stdout: []byte("before trap")}, "<result error>\ntrap: unreachable (fuel used: 100)\n--- stdout ---\nbefore trap\n--- stderr ---\n\n</result>", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			ex := wasm.Executor(func(tr tree.Tree, req wasm.Request) (wasm.Response, tree.Tree, error) {
				calls++
				want := wasm.Request{Module: module, Args: tc.argv, Stdin: []byte(tc.stdin), Limits: limits}
				if tr.Root() != initial.Root() || !reflect.DeepEqual(req, want) {
					t.Fatalf("unexpected execution request: %+v, want %+v", req, want)
				}
				if tc.commit {
					return tc.response, committed, nil
				}
				return tc.response, tr, nil
			})
			next, output, usage, err := tools.ApplyMounted(initial, ex, limits, "run", json.RawMessage(tc.args))
			if err != nil || string(output) != tc.want || calls != 1 {
				t.Fatalf("Apply = %q, %v (%d calls), want %q", output, err, calls, tc.want)
			}
			wantRoot := initial.Root()
			if tc.commit {
				wantRoot = committed.Root()
			}
			if next.Root() != wantRoot {
				t.Fatal("unexpected result tree")
			}
			wantUsage := tc.response.Usage
			if tc.commit {
				wantUsage.TreeEntries = max(wantUsage.TreeEntries, 2)
				wantUsage.TreeBytes = max(wantUsage.TreeBytes, 7)
			}
			if usage != wantUsage {
				t.Fatalf("usage = %+v, want %+v", usage, wantUsage)
			}
			// Enforce executor-reported transient peaks even if it rolled
			// its tree back after a semantic trap.
			fullLimit := limits.MaxTreeBytes
			limits.MaxTreeBytes = wantUsage.TreeBytes - 1
			next, output, usage, err = tools.ApplyMounted(initial, ex, limits, "run", json.RawMessage(tc.args))
			limits.MaxTreeBytes = fullLimit
			if !errors.Is(err, wasm.ErrResourceLimit) || next.Root() != initial.Root() || output != nil || usage != (wasm.Usage{}) {
				t.Fatalf("peak tree bytes bypassed limit: %s, %+v, %v", output, usage, err)
			}
		})
	}
}

func TestRunLimits(t *testing.T) {
	initial := runTree(t, []byte("module"))
	for _, limits := range []wasm.Limits{
		{},
		{Fuel: 1, MemoryBytes: 2, WriteBytes: 3, StdoutBytes: 4, StderrBytes: 5, MaxTreeBytes: 6},
		{Fuel: ^uint64(0), MemoryBytes: ^uint64(0) - 1, WriteBytes: ^uint64(0) - 2, StdoutBytes: ^uint64(0) - 3, StderrBytes: ^uint64(0) - 4, MaxTreeBytes: ^uint64(0) - 5},
	} {
		calls := 0
		ex := wasm.Executor(func(tr tree.Tree, req wasm.Request) (wasm.Response, tree.Tree, error) {
			calls++
			if req.Limits != limits {
				t.Fatalf("limits = %+v, want %+v", req.Limits, limits)
			}
			return wasm.Response{}, tr, nil
		})
		next, output, _, err := tools.ApplyMounted(initial, ex, limits, "run", json.RawMessage(`{"module":"/env/program.wasm"}`))
		want := "<result ok>\nexit 0 (fuel used: 0)\n--- stdout ---\n\n--- stderr ---\n\n</result>"
		if err != nil || string(output) != want || calls != 1 || next.Root() != initial.Root() {
			t.Fatalf("Apply = %q, %v (%d calls), want %q and unchanged tree", output, err, calls, want)
		}
	}
}

func TestRunErrors(t *testing.T) {
	initial := runTree(t, []byte("module"))
	for _, tc := range []struct{ name, args, message string }{
		{"wrong module type", `{"module":1}`, `bad_args]: run: argument "module" has the wrong type`},
		{"wrong args type", `{"module":"/env/program.wasm","args":"x"}`, `bad_args]: run: argument "args" has the wrong type`},
		{"wrong element type", `{"module":"/env/program.wasm","args":[1]}`, `bad_args]: run: argument "args" has the wrong type`},
		{"null element", `{"module":"/env/program.wasm","args":[null]}`, `bad_args]: run: argument "args" has the wrong type`},
		{"wrong stdin type", `{"module":"/env/program.wasm","stdin":false}`, `bad_args]: run: argument "stdin" has the wrong type`},
		{"relative path", `{"module":"program.wasm"}`, "bad_path]: path must be absolute: program.wasm"},
		{"missing module", `{"module":"/env/missing"}`, "not_found]: file not found: /env/missing"},
		{"directory", `{"module":"/env"}`, "is_dir]: path is a directory: /env"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := wasm.Executor(func(tree.Tree, wasm.Request) (wasm.Response, tree.Tree, error) {
				t.Fatal("executed invalid request")
				return wasm.Response{}, initial, nil
			})
			next, output, _, err := tools.ApplyMounted(initial, ex, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, Fuel: 100}, "run", json.RawMessage(tc.args))
			want := "<result error>\nerror[" + tc.message + "\n</result>"
			if err != nil || string(output) != want || next.Root() != initial.Root() {
				t.Fatalf("Apply = %q, %v, want %q and unchanged tree", output, err, want)
			}
		})
	}
	args := json.RawMessage(`{"module":"/env/program.wasm"}`)
	changed, err := initial.Put("/artifact/partial", tree.File{Content: []byte("partial")})
	if err != nil {
		t.Fatal(err)
	}
	for _, runErr := range []error{fmt.Errorf("compilation detail: %w", wasm.ErrBadModule), errors.New("host failure"), fmt.Errorf("fuel: %w", wasm.ErrResourceLimit)} {
		ex := wasm.Executor(func(tree.Tree, wasm.Request) (wasm.Response, tree.Tree, error) {
			return wasm.Response{Usage: wasm.Usage{Fuel: 100, WriteBytes: 7}, Stdout: []byte("partial")}, changed, runErr
		})
		next, output, usage, err := tools.ApplyMounted(initial, ex, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, Fuel: 100}, "run", args)
		if errors.Is(runErr, wasm.ErrBadModule) {
			want := "<result error>\nerror[bad_module]: not a runnable WASM module: /env/program.wasm\n</result>"
			if err != nil || string(output) != want {
				t.Fatalf("bad module: %q, %v", output, err)
			}
		} else if !errors.Is(err, runErr) || output != nil || usage != (wasm.Usage{}) {
			t.Fatalf("host failure became a tool result: %q, %v", output, err)
		}
		if next.Root() != initial.Root() {
			t.Fatal("execution error changed tree")
		}
	}
	if next, output, _, err := tools.ApplyMounted(initial, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, Fuel: 100}, "run", args); err == nil || output != nil || next.Root() != initial.Root() {
		t.Fatalf("missing executor: %q, %v", output, err)
	}
}

func TestRunTruncation(t *testing.T) {
	initial := runTree(t, []byte("module"))
	stdout := strings.Repeat("a", 32769)
	stderr := "\xff\x00end"
	body := "exit 0 (fuel used: 0)\n--- stdout ---\n" + stdout + "\n--- stderr ---\n" + stderr
	wantBody := body[:24576] + fmt.Sprintf("\n[truncated: %d bytes omitted]\n", len(body)-24576-4096) + body[len(body)-4096:]
	ex := wasm.Executor(func(tr tree.Tree, _ wasm.Request) (wasm.Response, tree.Tree, error) {
		return wasm.Response{Stdout: []byte(stdout), Stderr: []byte(stderr)}, tr, nil
	})
	_, output, _, err := tools.ApplyMounted(initial, ex, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, Fuel: 100}, "run", json.RawMessage(`{"module":"/env/program.wasm"}`))
	if err != nil || string(output) != "<result ok>\n"+wantBody+"\n</result>" {
		t.Fatalf("unexpected truncation: %d bytes, %v", len(output), err)
	}
}

func TestRunWasmtimeCommitAndRollback(t *testing.T) {
	limits := wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, Fuel: 1000000, MemoryBytes: 1 << 30, WriteBytes: 256 << 20, StdoutBytes: 8 << 20, StderrBytes: 8 << 20}
	const wat = `(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/output")
  (data (i32.const 32) "out")
  (func (export "_start")
    (if (call $open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 15)
        (i32.const 9) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100))
      (then (call $exit (i32.const 10))))
    (i32.store (i32.const 200) (i32.const 32))
    (i32.store (i32.const 204) (i32.const 3))
    (if (call $write (i32.load (i32.const 100)) (i32.const 200) (i32.const 1) (i32.const 208))
      (then (call $exit (i32.const 11))))
    (drop (call $write (i32.const 1) (i32.const 200) (i32.const 1) (i32.const 208)))
    (drop (call $write (i32.const 2) (i32.const 200) (i32.const 1) (i32.const 208)))
    %s))`
	for _, tc := range []struct {
		name, finish, head string
		commit             bool
	}{
		{"nonzero exit", "(call $exit (i32.const 7))", "exit 7", true},
		{"trap", "unreachable", "trap: unreachable", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			module, err := wasmtime.Wat2Wasm(fmt.Sprintf(wat, tc.finish))
			if err != nil {
				t.Fatal(err)
			}
			initial := runTree(t, module)
			var response wasm.Response
			ex := wasm.Executor(func(tr tree.Tree, req wasm.Request) (wasm.Response, tree.Tree, error) {
				r, next, err := wasmtime.Run(tr, req)
				response = r
				return r, next, err
			})
			args := json.RawMessage(`{"module":"/env/program.wasm"}`)
			next, output, _, err := tools.ApplyMounted(initial, ex, limits, "run", args)
			want := fmt.Sprintf("<result error>\n%s (fuel used: %d)\n--- stdout ---\nout\n--- stderr ---\nout\n</result>", tc.head, response.Usage.Fuel)
			if err != nil || response.Usage.Fuel == 0 || string(output) != want {
				t.Fatalf("Apply = %q, %v, want %q", output, err, want)
			}
			if tc.commit {
				if f, err := next.Get("/artifact/output"); err != nil || string(f.Content) != "out" {
					t.Fatalf("nonzero exit did not commit: %q, %v", f.Content, err)
				}
			} else if next.Root() != initial.Root() {
				t.Fatal("trap committed a file write")
			}
			if _, err := initial.Get("/artifact/output"); !errors.Is(err, tree.ErrNotFound) {
				t.Fatal("execution changed the initial snapshot")
			}
			replayed, replayOutput, _, err := tools.ApplyMounted(initial, ex, limits, "run", args)
			if err != nil || replayed.Root() != next.Root() || !bytes.Equal(replayOutput, output) {
				t.Fatalf("execution is not repeatable: %q, %v", replayOutput, err)
			}
		})
	}
}
