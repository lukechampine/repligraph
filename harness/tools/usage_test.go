package tools_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	"lukechampine.com/repligraph/harness/tools"
	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func TestBuiltinUsage(t *testing.T) {
	initial, err := (tree.Tree{}).Put("/artifact/a", tree.File{Content: []byte("prefix old suffix")})
	if err != nil {
		t.Fatal(err)
	}
	initial, err = initial.Put("/env/tool", tree.File{Content: []byte("environment")})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		tool, args string
		want       wasm.Usage
	}{
		{"read_file", `{"path":"/artifact/a"}`, wasm.Usage{TreeEntries: 2, TreeBytes: 17}},
		{"list_files", `{}`, wasm.Usage{TreeEntries: 2, TreeBytes: 17}},
		{"search_files", `{"query":"old"}`, wasm.Usage{TreeEntries: 2, TreeBytes: 17}},
		{"write_file", `{"path":"/artifact/a","content":"é"}`, wasm.Usage{WriteBytes: 2, TreeEntries: 2, TreeBytes: 17}},
		{"write_file", `{"path":"/artifact/a","content":"é","append":true}`, wasm.Usage{WriteBytes: 2, TreeEntries: 2, TreeBytes: 19}},
		{"write_file", `{"path":"/artifact/a","content":"/wAB","encoding":"base64"}`, wasm.Usage{WriteBytes: 3, TreeEntries: 2, TreeBytes: 17}},
		{"write_file", `{"path":"/artifact/new","content":"/wAB","encoding":"base64"}`, wasm.Usage{WriteBytes: 3, TreeEntries: 3, TreeBytes: 20}},
		{"write_file", `{"path":"/artifact/a","content":""}`, wasm.Usage{TreeEntries: 2, TreeBytes: 17}},
		{"write_file", `{"path":"/artifact/sub/b","content":"new"}`, wasm.Usage{WriteBytes: 3, TreeEntries: 4, TreeBytes: 20}},
		{"edit_file", `{"path":"/artifact/a","old":"old","new":"new"}`, wasm.Usage{WriteBytes: 17, TreeEntries: 2, TreeBytes: 17}},
		{"edit_file", `{"path":"/artifact/a","old":"old","new":"longer"}`, wasm.Usage{WriteBytes: 20, TreeEntries: 2, TreeBytes: 20}},
		{"copy_file", `{"source":"/artifact/a","destination":"/artifact/copy"}`, wasm.Usage{TreeEntries: 3, TreeBytes: 34}},
		{"copy_file", `{"source":"/artifact/a","destination":"/artifact/a"}`, wasm.Usage{TreeEntries: 2, TreeBytes: 17}},
		{"copy_file", `{"source":"/env/tool","destination":"/artifact/copy"}`, wasm.Usage{TreeEntries: 3, TreeBytes: 28}},
		{"rename_file", `{"source":"/artifact/a","destination":"/artifact/b"}`, wasm.Usage{TreeEntries: 2, TreeBytes: 17}},
		{"delete_file", `{"path":"/artifact/a"}`, wasm.Usage{TreeEntries: 2, TreeBytes: 17}},
	} {
		t.Run(tc.tool+tc.args, func(t *testing.T) {
			// Exactly the reported write/tree budget suffices; all guest limits
			// remain zero for builtin operations.
			limits := wasm.Limits{WriteBytes: tc.want.WriteBytes, MaxTreeEntries: tc.want.TreeEntries, MaxTreeBytes: tc.want.TreeBytes}
			next, out, got, err := tools.ApplyMounted(initial, nil, limits, tc.tool, json.RawMessage(tc.args))
			if err != nil || !bytes.HasPrefix(out, []byte("<result ok>")) || got != tc.want {
				t.Fatalf("got %s, %+v, %v; want %+v", out, got, err, tc.want)
			}
			if f, err := initial.Get("/artifact/a"); err != nil || string(f.Content) != "prefix old suffix" {
				t.Fatal("operation changed its initial snapshot")
			}
			// A one-byte-smaller peak allowance rejects either the initial
			// tree or growth, even for copies that write no new bytes.
			limits.MaxTreeBytes--
			next, out, got, err = tools.ApplyMounted(initial, nil, limits, tc.tool, json.RawMessage(tc.args))
			if !errors.Is(err, wasm.ErrResourceLimit) || out != nil || got != (wasm.Usage{}) || next.Root() != initial.Root() {
				t.Fatalf("tree byte exhaustion did not abort atomically: %s, %+v, %v", out, got, err)
			}
			limits.MaxTreeBytes++
			if tc.want.WriteBytes > 0 {
				limits.WriteBytes--
				next, out, got, err = tools.ApplyMounted(initial, nil, limits, tc.tool, json.RawMessage(tc.args))
				if !errors.Is(err, wasm.ErrResourceLimit) || out != nil || got != (wasm.Usage{}) || next.Root() != initial.Root() {
					t.Fatalf("write exhaustion did not abort atomically: %s, %+v, %v", out, got, err)
				}
			}
		})
	}
}

func TestBuiltinErrorsDoNotConsumeWrites(t *testing.T) {
	initial, err := (tree.Tree{}).Put("/artifact/a", tree.File{Content: []byte("original")})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ tool, args string }{
		{"unknown", `{}`},
		{"write_file", `{"path":"/input/a","content":"new"}`},
		{"write_file", `{"path":"/artifact","content":"new"}`},
		{"write_file", `{"path":"/artifact/a/child","content":"new"}`},
		{"write_file", `{"path":"/artifact/a","encoding":"invalid","content":"new"}`},
		{"write_file", `{"path":"/artifact/a","encoding":"base64","content":"!"}`},
		{"edit_file", `{"path":"/artifact/a","old":"missing","new":"new"}`},
	} {
		next, out, got, err := tools.ApplyMounted(initial, nil, wasm.Limits{MaxTreeEntries: 2, MaxTreeBytes: 8}, tc.tool, json.RawMessage(tc.args))
		if err != nil || !bytes.HasPrefix(out, []byte("<result error>")) || got != (wasm.Usage{TreeEntries: 2, TreeBytes: 8}) || next.Root() != initial.Root() {
			t.Fatalf("ordinary error became resource exhaustion: %s, %+v, %v", out, got, err)
		}
	}
}

func TestBuiltinTreeBytesLifecycle(t *testing.T) {
	tr := tree.Tree{}
	for path, content := range map[string]string{"/input": "data", "/artifact/a": "data", "/env/large": "environment is excluded"} {
		var err error
		tr, err = tr.Put(path, tree.File{Content: []byte(content)})
		if err != nil {
			t.Fatal(err)
		}
	}
	limits := wasm.Limits{MaxTreeEntries: 16, MaxTreeBytes: 12, WriteBytes: 8}
	for _, step := range []struct {
		tool, args    string
		bytes, writes uint64
	}{
		{"copy_file", `{"source":"/artifact/a","destination":"/artifact/b"}`, 12, 0},
		{"write_file", `{"path":"/artifact/b","content":"b"}`, 9, 1},
		{"copy_file", `{"source":"/input","destination":"/artifact/b"}`, 12, 0},
		{"delete_file", `{"path":"/artifact/a"}`, 8, 0},
		{"copy_file", `{"source":"/input","destination":"/artifact/c"}`, 12, 0},
		{"rename_file", `{"source":"/artifact/b","destination":"/artifact/c"}`, 8, 0},
		{"write_file", `{"path":"/artifact/c","content":"abcd","append":true}`, 12, 4},
		{"edit_file", `{"path":"/artifact/c","old":"dataabcd","new":"x"}`, 5, 1},
		{"delete_file", `{"path":"/artifact/c"}`, 4, 0},
	} {
		task, _ := tr.Without("env")
		before := task.Stats().Bytes
		beforeRoot := tr.Root()
		next, out, usage, err := tools.ApplyMounted(tr, nil, limits, step.tool, json.RawMessage(step.args))
		if err != nil || !bytes.HasPrefix(out, []byte("<result ok>")) {
			t.Fatalf("%s: %s, %v", step.tool, out, err)
		}
		task, _ = next.Without("env")
		if uint64(task.Stats().Bytes) != step.bytes || usage.TreeBytes != max(uint64(before), step.bytes) || usage.WriteBytes != step.writes {
			t.Fatalf("%s: got %d task bytes, %+v; want %d bytes, peak %d, writes %d", step.tool, task.Stats().Bytes, usage, step.bytes, max(uint64(before), step.bytes), step.writes)
		} else if tr.Root() != beforeRoot {
			t.Fatal("mutation changed its initial snapshot")
		}
		tr = next
	}
}
