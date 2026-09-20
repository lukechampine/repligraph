package tools

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
)

func browseTree(t *testing.T, files map[string][]byte) tree.Tree {
	t.Helper()
	var tr tree.Tree
	for path, content := range files {
		var err error
		tr, err = tr.Put(path, tree.File{Content: content})
		if err != nil {
			t.Fatal(err)
		}
	}
	return tr
}

func decodeBrowsePage(t *testing.T, r result, v any) {
	t.Helper()
	if r.status != "ok" {
		t.Fatal(r.output)
	} else if len(r.output) > maxOutputBytes {
		t.Fatalf("page exceeds output cap: %d bytes", len(r.output))
	} else if err := json.Unmarshal([]byte(r.output), v); err != nil {
		t.Fatal(err)
	} else if strings.Contains(string(render(r)), "[truncated:") {
		t.Fatal("render truncated a page")
	}
}

func TestListFilesPagination(t *testing.T) {
	tr := browseTree(t, map[string][]byte{
		"/input/a/item": []byte("three"),
		"/input/a.txt":  []byte("one"),
		"/input/a0":     nil,
		"/input/é":      []byte("🙂"),
		"/other/file":   nil,
	})
	var got []listedFile
	args := listArgs{Path: "/input", Limit: 1}
	for i := 0; ; i++ {
		if i > 10 {
			t.Fatal("pagination did not finish")
		}
		var page listPage
		decodeBrowsePage(t, applyList(tr, browseJSON(args)), &page)
		got = append(got, page.Files...)
		if page.NextAfter == "" {
			break
		} else if page.NextAfter <= args.After {
			t.Fatal("cursor did not advance")
		}
		args.After = page.NextAfter
	}
	want := []listedFile{{"/input/a.txt", 3}, {"/input/a/item", 5}, {"/input/a0", 0}, {"/input/é", 4}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	var page listPage
	decodeBrowsePage(t, applyList(tr, []byte(`{"path":"/input/a.txt"}`)), &page)
	if len(page.Files) != 1 || page.Files[0] != want[0] || page.NextAfter != "" {
		t.Fatalf("listing one file: %+v", page)
	}
	if r := applyList(tree.Tree{}, nil); r.status != "ok" || r.output != `{"files":[]}` {
		t.Fatalf("empty tree: %+v", r)
	}
}

func TestBrowseOutputCaps(t *testing.T) {
	// Quotes need escaping, while '<' must not expand to a six-byte JSON escape.
	prefix := strings.Repeat("/"+strings.Repeat(`"<`, 127), 15)
	files := make(map[string][]byte)
	for i := 0; i < 8; i++ {
		files[fmt.Sprintf("%s/%d", prefix, i)] = bytes.Repeat([]byte("x"), 20)
	}
	tr := browseTree(t, files)
	var listed int
	a := listArgs{Path: "/", Limit: 1000}
	for i := 0; ; i++ {
		if i > 10 {
			t.Fatal("list pagination did not finish")
		}
		var page listPage
		decodeBrowsePage(t, applyList(tr, browseJSON(a)), &page)
		listed += len(page.Files)
		if page.NextAfter == "" {
			break
		}
		a.After = page.NextAfter
	}
	if listed != len(files) {
		t.Fatalf("listed %d files, want %d", listed, len(files))
	}
	var found int
	s := searchArgs{Path: "/", Query: "x", Limit: 200}
	for i := 0; ; i++ {
		if i > 100 {
			t.Fatal("search pagination did not finish")
		}
		var page searchPage
		// Omit the absent optional cursor: explicit null is intentionally invalid.
		raw := map[string]any{"path": s.Path, "query": s.Query, "limit": s.Limit}
		if len(s.Cursor) != 0 {
			raw["cursor"] = s.Cursor
		}
		decodeBrowsePage(t, applySearch(tr, browseJSON(raw)), &page)
		found += len(page.Matches)
		if page.Next == nil {
			break
		}
		s.Cursor = browseJSON(page.Next)
	}
	if found != len(files)*20 {
		t.Fatalf("found %d matches, want %d", found, len(files)*20)
	}
}

func TestSearchBytesAndContinuation(t *testing.T) {
	tr := browseTree(t, map[string][]byte{
		"/input/binary": {0xff, 'a', 'a', 'a', 'a', 0},
		"/input/text":   []byte("aaaaa"),
		"/input/utf8":   []byte("éaa🙂aa"),
	})
	var got []searchCursor
	var cursor *searchCursor
	for i := 0; ; i++ {
		if i > 10 {
			t.Fatal("pagination did not finish")
		}
		a := map[string]any{"query": "aa", "limit": 1}
		if cursor != nil {
			a["cursor"] = cursor
		}
		var page searchPage
		decodeBrowsePage(t, applySearch(tr, browseJSON(a)), &page)
		got = append(got, page.Matches...)
		if page.Next == nil {
			break
		} else if cursor != nil && (page.Next.Path < cursor.Path || page.Next.Path == cursor.Path && page.Next.Offset <= cursor.Offset) {
			t.Fatal("cursor did not advance")
		}
		cursor = page.Next
	}
	want := []searchCursor{
		{"/input/binary", 1}, {"/input/binary", 3},
		{"/input/text", 0}, {"/input/text", 2},
		{"/input/utf8", 2}, {"/input/utf8", 8},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestSearchBoundedScan(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content []byte
		query   string
		match   uint64
	}{
		{"unicode across boundary", append(bytes.Repeat([]byte("a"), maxSearchBytes-1), []byte("éxyz")...), "éx", maxSearchBytes - 1},
		{"after empty page", append(bytes.Repeat([]byte("a"), maxSearchBytes+10), []byte("needle")...), "needle", maxSearchBytes + 10},
		{"maximum query across boundary", append(bytes.Repeat([]byte("a"), maxSearchBytes-1), bytes.Repeat([]byte("b"), maxQueryBytes)...), strings.Repeat("b", maxQueryBytes), maxSearchBytes - 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := browseTree(t, map[string][]byte{"/input/long": tc.content})
			args := map[string]any{"query": tc.query}
			var got []searchCursor
			for i := 0; ; i++ {
				if i > 3 {
					t.Fatal("scan did not finish")
				}
				var page searchPage
				decodeBrowsePage(t, applySearch(tr, browseJSON(args)), &page)
				got = append(got, page.Matches...)
				if i == 0 && tc.name == "after empty page" && (len(page.Matches) != 0 || page.Next == nil || page.Next.Offset != maxSearchBytes) {
					t.Fatalf("expected empty page with progress, got %+v", page)
				}
				if page.Next == nil {
					break
				}
				args["cursor"] = page.Next
			}
			if len(got) != 1 || got[0] != (searchCursor{"/input/long", tc.match}) {
				t.Fatalf("wrong matches: %+v", got)
			}
		})
	}
	files := make(map[string][]byte)
	for i := 0; i <= maxSearchFiles; i++ {
		files[fmt.Sprintf("/input/%04d", i)] = nil
	}
	tr := browseTree(t, files)
	var page searchPage
	decodeBrowsePage(t, applySearch(tr, []byte(`{"query":"none"}`)), &page)
	if len(page.Matches) != 0 || page.Next == nil || *page.Next != (searchCursor{"/input/1000", 0}) {
		t.Fatalf("file scan budget was not enforced: %+v", page)
	}
	next := page.Next
	page = searchPage{}
	decodeBrowsePage(t, applySearch(tr, browseJSON(map[string]any{"query": "none", "cursor": next})), &page)
	if page.Next != nil || len(page.Matches) != 0 {
		t.Fatalf("empty-file continuation did not finish: %+v", page)
	}
}

func TestBrowseErrors(t *testing.T) {
	tr := browseTree(t, map[string][]byte{"/input/file": []byte("abc")})
	for _, tc := range []struct {
		tool, args, want string
	}{
		{"list_files", `{"limit":0}`, "bad_args]: list_files: limit must be between 1 and 1000"},
		{"list_files", `{"limit":1001}`, "bad_args]: list_files: limit must be between 1 and 1000"},
		{"list_files", `{"limit":-1}`, `bad_args]: list_files: argument "limit" has the wrong type`},
		{"list_files", `{"path":"/missing"}`, "not_found]: file not found: /missing"},
		{"list_files", `{"path":"/input","after":"/other/a"}`, "bad_args]: list_files: after must be at or below path"},
		{"list_files", `{"after":"relative"}`, "bad_path]: path must be absolute: relative"},
		{"search_files", `{}`, "bad_args]: search_files: query must be 1 to 4096 UTF-8 bytes"},
		{"search_files", `{"query":"x","limit":201}`, "bad_args]: search_files: limit must be between 1 and 200"},
		{"search_files", `{"query":"x","cursor":{"path":"/input/file","offset":4}}`, "bad_args]: search_files: cursor offset exceeds file size"},
		{"search_files", `{"path":"/input","query":"x","cursor":{"path":"/other/file"}}`, "bad_args]: search_files: cursor must be at or below path"},
		{"search_files", `{"query":"x","cursor":{"path":"/input/missing"}}`, "not_found]: file not found: /input/missing"},
		{"search_files", `{"query":"x","cursor":{"path":"/input/file","extra":0}}`, `bad_args]: search_files cursor: unknown argument "extra"`},
		{"search_files", `{"query":"x","cursor":{"path":"/input/file","offset":-1}}`, `bad_args]: search_files cursor: argument "offset" has the wrong type`},
	} {
		t.Run(tc.tool+tc.args, func(t *testing.T) {
			f := applyList
			if tc.tool == "search_files" {
				f = applySearch
			}
			r := f(tr, []byte(tc.args))
			if r.status != "error" || r.output != "error["+tc.want {
				t.Fatalf("got %+v, want error[%s", r, tc.want)
			}
		})
	}
}
