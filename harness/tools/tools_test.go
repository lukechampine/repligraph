package tools_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"lukechampine.com/repligraph/harness/tools"
	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
	"lukechampine.com/repligraph/harness/wasm/wasmtime"
)

func TestReadWrite(t *testing.T) {
	initial := tree.Tree{}
	initialRoot := initial.Root()
	args := json.RawMessage(`{"path":"/artifact/nested/file","content":"héllo\n"}`)
	written, output, _, err := tools.ApplyMounted(initial, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, WriteBytes: 1 << 20}, "write_file", args)
	if err != nil {
		t.Fatal(err)
	}
	if string(output) != "<result ok>\nwrote 7 bytes: /artifact/nested/file\n</result>" {
		t.Fatalf("unexpected write output: %q", output)
	} else if initial.Root() != initialRoot {
		t.Fatal("write changed the original tree")
	} else if _, err := initial.Get("/artifact/nested/file"); err == nil {
		t.Fatal("write added a file to the original tree")
	}
	replayed, replayOutput, _, err := tools.ApplyMounted(initial, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, WriteBytes: 1 << 20}, "write_file", args)
	if err != nil {
		t.Fatal(err)
	}
	if replayed.Root() != written.Root() || !bytes.Equal(replayOutput, output) {
		t.Fatal("replaying write produced a different result")
	}

	read, output, _, err := tools.ApplyMounted(written, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, WriteBytes: 1 << 20}, "read_file", json.RawMessage(`{"path":"/artifact/nested/file"}`))
	if err != nil {
		t.Fatal(err)
	}
	if string(output) != "<result ok>\nhéllo\n\n</result>" || read.Root() != written.Root() {
		t.Fatalf("unexpected read result: %q", output)
	}
	overwritten, output, _, err := tools.ApplyMounted(written, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, WriteBytes: 1 << 20}, "write_file", json.RawMessage(`{"path":"/artifact/nested/file","content":""}`))
	if err != nil {
		t.Fatal(err)
	}
	if string(output) != "<result ok>\nwrote 0 bytes: /artifact/nested/file\n</result>" {
		t.Fatalf("unexpected overwrite output: %q", output)
	}
	if f, err := overwritten.Get("/artifact/nested/file"); err != nil || len(f.Content) != 0 {
		t.Fatalf("overwrite did not empty file: %q, %v", f.Content, err)
	}
	if f, err := written.Get("/artifact/nested/file"); err != nil || string(f.Content) != "héllo\n" {
		t.Fatalf("overwrite changed previous snapshot: %q, %v", f.Content, err)
	}

	mounted, err := overwritten.Put("/source/file", tree.File{Content: []byte("mounted")})
	if err != nil {
		t.Fatal(err)
	}
	read, output, _, err = tools.ApplyMounted(mounted, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, WriteBytes: 1 << 20}, "read_file", json.RawMessage(`{"path":"/source/file"}`))
	if err != nil {
		t.Fatal(err)
	}
	if string(output) != "<result ok>\nmounted\n</result>" || read.Root() != mounted.Root() {
		t.Fatalf("read-only mount could not be read: %q", output)
	}
}

func TestToolErrors(t *testing.T) {
	initial, err := (tree.Tree{}).Put("/artifact/file", tree.File{Content: []byte{0xff}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, tool, args, error string
	}{
		{"unknown tool", "other", `{}`, "unknown_tool]: unknown tool: other"},
		{"missing arguments", "read_file", "", "bad_path]: path must be absolute: "},
		{"null object", "read_file", `null`, "bad_args]: read_file: arguments are not a JSON object"},
		{"array", "write_file", `[]`, "bad_args]: write_file: arguments are not a JSON object"},
		{"malformed", "read_file", `{`, "bad_args]: read_file: arguments are not a JSON object"},
		{"trailing object", "read_file", `{} {}`, "bad_args]: read_file: trailing data after arguments"},
		{"trailing junk", "read_file", `{} junk`, "bad_args]: read_file: trailing data after arguments"},
		{"unknown argument", "write_file", `{"z":0,"a":0}`, `bad_args]: write_file: unknown argument "a"`},
		{"wrong path type", "read_file", `{"path":1}`, `bad_args]: read_file: argument "path" has the wrong type`},
		{"null path", "read_file", `{"path":null}`, `bad_args]: read_file: argument "path" has the wrong type`},
		{"wrong content type", "write_file", `{"path":"/artifact/file","content":false}`, `bad_args]: write_file: argument "content" has the wrong type`},
		{"null content", "write_file", `{"path":"/artifact/file","content":null}`, `bad_args]: write_file: argument "content" has the wrong type`},
		{"relative path", "write_file", `{"path":"artifact/file"}`, "bad_path]: path must be absolute: artifact/file"},
		{"parent traversal", "write_file", `{"path":"/artifact/../file"}`, "bad_path]: path contains '.' or '..': /artifact/../file"},
		{"empty component", "write_file", `{"path":"/artifact//file"}`, "bad_path]: path contains an empty component: /artifact//file"},
		{"readonly", "write_file", `{"path":"/source/file"}`, "read_only]: read-only path: /source/file"},
		{"artifact root", "write_file", `{"path":"/artifact"}`, "read_only]: read-only path: /artifact"},
		{"artifact prefix", "write_file", `{"path":"/artifact-other/file"}`, "read_only]: read-only path: /artifact-other/file"},
		{"missing file", "read_file", `{"path":"/artifact/missing"}`, "not_found]: file not found: /artifact/missing"},
		{"directory", "read_file", `{"path":"/artifact"}`, "is_dir]: path is a directory: /artifact"},
		{"file parent", "write_file", `{"path":"/artifact/file/child"}`, "not_dir]: path component is a file: /artifact/file/child"},
		{"binary", "read_file", `{"path":"/artifact/file"}`, "binary]: file is not valid UTF-8: /artifact/file"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := initial.Root()
			next, output, _, err := tools.ApplyMounted(initial, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, WriteBytes: 1 << 20}, tc.tool, json.RawMessage(tc.args))
			if err != nil {
				t.Fatal(err)
			}
			want := "<result error>\nerror[" + tc.error + "\n</result>"
			if string(output) != want {
				t.Fatalf("got %q, want %q", output, want)
			} else if next.Root() != root || initial.Root() != root {
				t.Fatal("failed operation changed the tree")
			}
			replayed, replayOutput, _, err := tools.ApplyMounted(initial, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, WriteBytes: 1 << 20}, tc.tool, json.RawMessage(tc.args))
			if err != nil {
				t.Fatal(err)
			}
			if replayed.Root() != next.Root() || !bytes.Equal(replayOutput, output) {
				t.Fatal("replaying failed operation produced a different result")
			}
		})
	}
}

func TestReadTruncation(t *testing.T) {
	for _, tc := range []struct {
		name, content, want string
	}{
		{"at limit", strings.Repeat("a", 32768), strings.Repeat("a", 32768)},
		{"above limit", strings.Repeat("a", 32769), strings.Repeat("a", 24576) + "\n[truncated: 4097 bytes omitted]\n" + strings.Repeat("a", 4096)},
		{"UTF-8 boundaries", strings.Repeat("a", 24575) + "🙂" + strings.Repeat("b", 8192) + "🙂" + strings.Repeat("c", 4095), strings.Repeat("a", 24575) + "\n[truncated: 8200 bytes omitted]\n" + strings.Repeat("c", 4095)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			initial, err := (tree.Tree{}).Put("/artifact/file", tree.File{Content: []byte(tc.content)})
			if err != nil {
				t.Fatal(err)
			}
			next, output, _, err := tools.ApplyMounted(initial, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1024, WriteBytes: 1 << 20}, "read_file", json.RawMessage(`{"path":"/artifact/file"}`))
			if err != nil {
				t.Fatal(err)
			}
			want := "<result ok>\n" + tc.want + "\n</result>"
			if string(output) != want {
				t.Fatalf("unexpected truncated output: got %d bytes, want %d bytes", len(output), len(want))
			} else if !utf8.Valid(output) {
				t.Fatal("truncation split a UTF-8 sequence")
			} else if next.Root() != initial.Root() {
				t.Fatal("read changed the tree")
			}
		})
	}
}

func TestApplyEnvironment(t *testing.T) {
	// The module requires the task input, then writes a task file and stdout.
	module, err := wasmtime.Wat2Wasm(`(module
  (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "artifact/result")
  (data (i32.const 16) "input")
  (data (i32.const 32) "result")
  (data (i32.const 48) "done")
  (func (export "_start")
    (if (call $open (i32.const 3) (i32.const 0) (i32.const 16) (i32.const 5)
        (i32.const 0) (i64.const 2) (i64.const 0) (i32.const 0) (i32.const 100))
      (then (call $exit (i32.const 10))))
    (if (call $open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 15)
        (i32.const 9) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 100))
      (then (call $exit (i32.const 11))))
    (i32.store (i32.const 200) (i32.const 32))
    (i32.store (i32.const 204) (i32.const 6))
    (if (call $write (i32.load (i32.const 100)) (i32.const 200) (i32.const 1) (i32.const 208))
      (then (call $exit (i32.const 12))))
    (i32.store (i32.const 200) (i32.const 48))
    (i32.store (i32.const 204) (i32.const 4))
    (drop (call $write (i32.const 1) (i32.const 200) (i32.const 1) (i32.const 208)))))`)
	if err != nil {
		t.Fatal(err)
	}
	files, err := (tree.Tree{}).Put("/tool.wasm", tree.File{Content: module})
	if err != nil {
		t.Fatal(err)
	}
	root := files.Root()
	initial, err := (tree.Tree{}).Put("/input", tree.File{Content: []byte("input")})
	if err != nil {
		t.Fatal(err)
	}
	initialRoot := initial.Root()
	limits := wasm.Limits{MaxTreeBytes: 1 << 20, MaxTreeEntries: 1024, Fuel: 100000, MemoryBytes: 1 << 20, WriteBytes: 1 << 20, StdoutBytes: 16}
	args := json.RawMessage(`{"module":"/env/tool.wasm"}`)

	next, output, usage, err := tools.Apply(initial, files, limits, "run", args)
	if err != nil {
		t.Fatalf("run: %v", err)
	} else if f, err := next.Get("/artifact/result"); err != nil || string(f.Content) != "result" || !bytes.Contains(output, []byte("exit 0")) || !bytes.Contains(output, []byte("done")) {
		t.Fatalf("result: %q, %q, %v", f.Content, output, err)
	} else if usage.Fuel == 0 || usage.WriteBytes != 6 || usage.StdoutBytes != 4 {
		t.Fatalf("usage changed across environment mounting: %+v", usage)
	} else if _, err := next.Get("/env"); !errors.Is(err, tree.ErrNotFound) {
		t.Fatal("returned task tree contains the environment")
	}

	limits.Fuel = 1
	next, output, usage, err = tools.Apply(initial, files, limits, "run", args)
	if !errors.Is(err, wasm.ErrResourceLimit) {
		t.Fatalf("run without fuel: %v", err)
	} else if next.Root() != initialRoot || output != nil || usage != (wasm.Usage{}) {
		t.Fatal("resource failure changed task state or emitted a result/usage")
	}
	if initial.Root() != initialRoot || files.Root() != root {
		t.Fatal("input trees changed")
	}
}

func TestApplyEmptyEnvironment(t *testing.T) {
	limits := wasm.Limits{MaxTreeBytes: 1 << 20, MaxTreeEntries: 1024, WriteBytes: 1 << 20}
	next, output, usage, err := tools.Apply(tree.Tree{}, tree.Tree{}, limits, "write_file", json.RawMessage(`{"path":"/artifact/a","content":"a"}`))
	if err != nil || !bytes.Contains(output, []byte("wrote 1 bytes")) {
		t.Fatalf("write with empty environment: %q, %v", output, err)
	} else if f, err := next.Get("/artifact/a"); err != nil || string(f.Content) != "a" {
		t.Fatalf("task output: %q, %v", f.Content, err)
	} else if usage != (wasm.Usage{WriteBytes: 1, TreeEntries: 2, TreeBytes: 1}) {
		t.Fatalf("builtin usage with empty environment: %+v", usage)
	}
	for _, path := range []string{"/env", "/env/tool"} {
		initial, err := (tree.Tree{}).Put(path, tree.File{})
		if err != nil {
			t.Fatal(err)
		}
		if next, output, _, err := tools.Apply(initial, tree.Tree{}, limits, "read_file", json.RawMessage(`{"path":"/env"}`)); err == nil || output != nil || next.Root() != initial.Root() {
			t.Fatal("accepted task input at /env")
		}
	}
}
