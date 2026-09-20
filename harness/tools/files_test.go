package tools_test

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/tools"
	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func fileTool(t *testing.T, tr tree.Tree, cap uint64, tool string, args any) (tree.Tree, string) {
	t.Helper()
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	// Builtins consume no guest fuel, memory, stdout, or stderr budget.
	next, out, _, err := tools.ApplyMounted(tr, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: cap, WriteBytes: 1 << 20}, tool, raw)
	if err != nil {
		t.Fatal(err)
	}
	return next, string(out)
}

func TestFileMutations(t *testing.T) {
	var tr tree.Tree
	var out string
	tr, out = fileTool(t, tr, 16, "write_file", map[string]any{"path": "/artifact/note", "content": "first\nsecond\n"})
	if !strings.HasPrefix(out, "<result ok>") {
		t.Fatal(out)
	}
	before := tr
	tr, out = fileTool(t, tr, 16, "edit_file", map[string]string{"path": "/artifact/note", "old": "second", "new": "third"})
	if !strings.HasPrefix(out, "<result ok>") {
		t.Fatal(out)
	}
	tr, out = fileTool(t, tr, 16, "write_file", map[string]any{"path": "/artifact/note", "content": "fourth\n", "append": true})
	if f, err := tr.Get("/artifact/note"); err != nil || string(f.Content) != "first\nthird\nfourth\n" {
		t.Fatalf("edit/append: %q, %v, %s", f.Content, err, out)
	}
	if f, _ := before.Get("/artifact/note"); string(f.Content) != "first\nsecond\n" {
		t.Fatal("edit changed the old snapshot")
	}
	data := []byte{0xff, 0, 1, 2, 3}
	tr, out = fileTool(t, tr, 16, "write_file", map[string]string{"path": "/artifact/binary", "encoding": "base64", "content": base64.StdEncoding.EncodeToString(data)})
	if !strings.HasPrefix(out, "<result ok>") {
		t.Fatal(out)
	}
	tr, out = fileTool(t, tr, 16, "copy_file", map[string]string{"source": "/artifact/binary", "destination": "/artifact/nested/copy"})
	if f, err := tr.Get("/artifact/nested/copy"); err != nil || !bytes.Equal(f.Content, data) {
		t.Fatalf("binary copy: %v, %s", err, out)
	}
	tr, out = fileTool(t, tr, 16, "rename_file", map[string]string{"source": "/artifact/nested/copy", "destination": "/artifact/note"})
	if f, err := tr.Get("/artifact/note"); err != nil || !bytes.Equal(f.Content, data) {
		t.Fatalf("rename replacement: %v, %s", err, out)
	} else if _, err := tr.Get("/artifact/nested"); err != tree.ErrNotFound {
		t.Fatal("rename left the source or an empty parent")
	}
	tr, out = fileTool(t, tr, 16, "delete_file", map[string]string{"path": "/artifact/binary"})
	if !strings.HasPrefix(out, "<result ok>") || tr.Stats().Entries != 2 {
		t.Fatalf("delete: %s, %+v", out, tr.Stats())
	}
	tr, out = fileTool(t, tr, 16, "delete_file", map[string]string{"path": "/artifact/note"})
	if tr.Stats().Entries != 0 || !strings.HasPrefix(out, "<result ok>") {
		t.Fatalf("delete did not prune last parent: %s", out)
	}
}

func TestFileMutationFailuresAreAtomic(t *testing.T) {
	tr, err := (tree.Tree{}).Put("/artifact/file", tree.File{Content: []byte("aaa")})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		tool, args, want string
	}{
		{"edit_file", `{"path":"/artifact/file","old":"aa","new":"b"}`, "multiple_matches"},
		{"edit_file", `{"path":"/artifact/file","old":"missing","new":"b"}`, "no_match"},
		{"edit_file", `{"path":"/artifact/file","old":"","new":"b"}`, "bad_args"},
		{"edit_file", `{"path":"/input/file","old":"a","new":"b"}`, "read_only"},
		{"write_file", `{"path":"/artifact/file","encoding":"base64","content":"bad"}`, "bad_args"},
		{"write_file", `{"path":"/artifact/file","encoding":"wrong"}`, "bad_args"},
		{"write_file", `{"path":"/artifact","append":true}`, "read_only"},
		{"copy_file", `{"source":"/artifact/missing","destination":"/artifact/copy"}`, "not_found"},
		{"copy_file", `{"source":"/artifact/file","destination":"/env/copy"}`, "read_only"},
		{"rename_file", `{"source":"/input/file","destination":"/artifact/copy"}`, "read_only"},
		{"rename_file", `{"source":"/artifact/file","destination":"/artifact/file/child"}`, "not_dir"},
		{"delete_file", `{"path":"/artifact/missing"}`, "not_found"},
		{"delete_file", `{"path":"/input/file"}`, "read_only"},
	} {
		t.Run(tc.tool+tc.args, func(t *testing.T) {
			next, out, _, err := tools.ApplyMounted(tr, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 16, WriteBytes: 1 << 20}, tc.tool, json.RawMessage(tc.args))
			if err != nil || !strings.Contains(string(out), "error["+tc.want+"]") || next.Root() != tr.Root() {
				t.Fatalf("failed mutation: %s, %v", out, err)
			}
		})
	}
}

func TestBuiltinEntryQuota(t *testing.T) {
	var tr tree.Tree
	for _, limit := range []uint64{0, 1} {
		next, out, usage, err := tools.ApplyMounted(tr, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: limit}, "write_file", json.RawMessage(`{"path":"/artifact/a","content":""}`))
		if !errors.Is(err, wasm.ErrResourceLimit) || out != nil || usage != (wasm.Usage{}) || next.Root() != tr.Root() {
			t.Fatalf("entry limit %d failed to abort empty-file creation: %s, %+v, %v", limit, out, usage, err)
		}
	}
	tr, out := fileTool(t, tr, 2, "write_file", map[string]string{"path": "/artifact/a", "content": "a"})
	if !strings.HasPrefix(out, "<result ok>") || tr.Stats().Entries != 2 {
		t.Fatal(out)
	}
	tr, out = fileTool(t, tr, 2, "write_file", map[string]string{"path": "/artifact/a", "content": "overwrite"})
	if !strings.HasPrefix(out, "<result ok>") {
		t.Fatal("overwrite at capacity failed: " + out)
	}
	for _, tc := range []struct{ tool, args string }{
		{"write_file", `{"path":"/artifact/b"}`},
		{"copy_file", `{"source":"/artifact/a","destination":"/artifact/b"}`},
		{"rename_file", `{"source":"/artifact/a","destination":"/artifact/sub/b"}`},
	} {
		next, out, usage, err := tools.ApplyMounted(tr, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 2}, tc.tool, json.RawMessage(tc.args))
		if !errors.Is(err, wasm.ErrResourceLimit) || out != nil || usage != (wasm.Usage{}) || next.Root() != tr.Root() {
			t.Fatalf("entry limit was not atomic for %s: %s, %v", tc.tool, out, err)
		}
	}
	tr, out = fileTool(t, tr, 2, "rename_file", map[string]string{"source": "/artifact/a", "destination": "/artifact/b"})
	if !strings.HasPrefix(out, "<result ok>") || tr.Stats().Entries != 2 {
		t.Fatal("rename charged a transient extra entry: " + out)
	}
	// A large immutable environment consumes none of the task entry allowance.
	for _, p := range []string{"/env/a", "/env/nested/b", "/env/nested/c"} {
		var err error
		tr, err = tr.Put(p, tree.File{Content: []byte("env")})
		if err != nil {
			t.Fatal(err)
		}
	}
	tr, out = fileTool(t, tr, 2, "write_file", map[string]string{"path": "/artifact/b", "content": "still works"})
	if !strings.HasPrefix(out, "<result ok>") {
		t.Fatal("environment charged to task: " + out)
	}
	if next, result, usage, err := tools.ApplyMounted(tr, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1}, "read_file", json.RawMessage(`{"path":"/artifact/b"}`)); !errors.Is(err, wasm.ErrResourceLimit) || result != nil || usage != (wasm.Usage{}) || next.Root() != tr.Root() {
		t.Fatal("accepted an initial task tree above its local limit")
	}
}

func TestReadUTF8RangeInBinaryFile(t *testing.T) {
	tr, err := (tree.Tree{}).Put("/file", tree.File{Content: []byte{'A', 0x80}})
	if err != nil {
		t.Fatal(err)
	}
	_, out := fileTool(t, tr, 1, "read_file", map[string]any{"path": "/file", "offset": 0, "length": 1})
	if !strings.HasPrefix(out, "<result ok>") || !strings.Contains(out, `"content":"A"`) {
		t.Fatalf("valid range depended on bytes outside it: %s", out)
	}
}

func TestPagedRead(t *testing.T) {
	for _, content := range [][]byte{
		bytes.Repeat([]byte("🙂\x00\n\""), 9000),
		bytes.Repeat([]byte{0, 0xff, 0xfe, 1, 2}, 9000),
		{},
	} {
		tr, err := (tree.Tree{}).Put("/file", tree.File{Content: content})
		if err != nil {
			t.Fatal(err)
		}
		for _, encoding := range []string{"utf8", "base64"} {
			if encoding == "utf8" && bytes.Contains(content, []byte{0xff}) {
				continue
			}
			var joined []byte
			var offset uint64
			for {
				_, out := fileTool(t, tr, 1, "read_file", map[string]any{"path": "/file", "offset": offset, "encoding": encoding})
				body := strings.TrimSuffix(strings.TrimPrefix(out, "<result ok>\n"), "\n</result>")
				var page struct {
					Offset     uint64 `json:"offset"`
					NextOffset uint64 `json:"next_offset"`
					Size       uint64 `json:"size"`
					Content    string `json:"content"`
				}
				if err := json.Unmarshal([]byte(body), &page); err != nil {
					t.Fatalf("invalid page: %v, %s", err, out)
				}
				data := []byte(page.Content)
				if encoding == "base64" {
					data, err = base64.StdEncoding.DecodeString(page.Content)
					if err != nil {
						t.Fatal(err)
					}
				}
				if page.Offset != offset || page.Size != uint64(len(content)) || page.NextOffset != offset+uint64(len(data)) {
					t.Fatalf("incorrect page offsets: %+v", page)
				}
				joined = append(joined, data...)
				if page.NextOffset == page.Size {
					break
				} else if page.NextOffset <= offset {
					t.Fatal("page made no progress")
				}
				offset = page.NextOffset
			}
			if !bytes.Equal(joined, content) {
				t.Fatal("paged reads changed or omitted bytes")
			}
		}
	}
}

func TestReadRangeErrors(t *testing.T) {
	tr, err := (tree.Tree{}).Put("/file", tree.File{Content: []byte("🙂end")})
	if err != nil {
		t.Fatal(err)
	}
	for _, args := range []string{
		`{"path":"/file","offset":999}`, `{"path":"/file","length":0}`,
		`{"path":"/file","offset":-1}`, `{"path":"/file","length":4097}`,
		`{"path":"/file","offset":1}`, `{"path":"/file","length":1}`,
		`{"path":"/file","encoding":"wrong"}`, `{"path":"/file","offset":null}`,
	} {
		_, out, _, err := tools.ApplyMounted(tr, nil, wasm.Limits{MaxTreeBytes: 256 << 20, MaxTreeEntries: 1}, "read_file", json.RawMessage(args))
		if err != nil || !bytes.HasPrefix(out, []byte("<result error>")) {
			t.Fatalf("invalid range accepted: %s, %s, %v", args, out, err)
		}
	}
}
