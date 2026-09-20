package tree

import (
	"bytes"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
)

func mustPut(t *testing.T, tr Tree, path, content string) Tree {
	t.Helper()
	nt, err := tr.Put(path, File{Content: []byte(content)})
	if err != nil {
		t.Fatalf("Put(%s): %v", path, err)
	}
	return nt
}

func TestOrderIndependence(t *testing.T) {
	paths := []string{"/src/main.go", "/src/lib/a", "/README.md", "/src/lib/b"}
	var forward, reverse Tree
	for i, path := range paths {
		forward = mustPut(t, forward, path, path)
		path = paths[len(paths)-1-i]
		reverse = mustPut(t, reverse, path, path)
	}
	if forward.Root() != reverse.Root() {
		t.Fatal("insertion order changed the root")
	}
}

func TestRootSensitivity(t *testing.T) {
	base := mustPut(t, Tree{}, "/a", "x")
	for _, other := range []Tree{
		{},
		mustPut(t, Tree{}, "/a", "y"),
		mustPut(t, Tree{}, "/b", "x"),
		mustPut(t, Tree{}, "/a/b", "x"),
	} {
		if base.Root() == other.Root() {
			t.Fatal("different tree has the same root")
		}
	}
	if base.Root() != mustPut(t, base, "/a", "x").Root() {
		t.Fatal("rewriting identical content changed root")
	}
}

func TestPersistence(t *testing.T) {
	content := []byte("one")
	first, err := (Tree{}).Put("/a", File{Content: content})
	if err != nil {
		t.Fatal(err)
	}
	root := first.Root()
	second := mustPut(t, first, "/b", "two")
	third := mustPut(t, second, "/a", "three")
	content[0] = 'X'
	for _, tr := range []Tree{first, second} {
		if file, err := tr.Get("/a"); err != nil || string(file.Content) != "one" {
			t.Fatalf("old snapshot mutated: %q, %v", file.Content, err)
		}
	}
	if first.Root() != root {
		t.Fatal("old root mutated")
	}
	if _, err := first.Get("/b"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("old snapshot sees new file: %v", err)
	}
	if file, err := third.Get("/a"); err != nil || string(file.Content) != "three" {
		t.Fatalf("replacement failed: %q, %v", file.Content, err)
	}
	if file, err := third.Get("/b"); err != nil || string(file.Content) != "two" {
		t.Fatalf("sibling changed: %q, %v", file.Content, err)
	}
}

func TestPrefixConflicts(t *testing.T) {
	for _, test := range []struct {
		tree Tree
		path string
		err  error
	}{
		{mustPut(t, Tree{}, "/a", "file"), "/a/b", ErrNotDir},
		{mustPut(t, Tree{}, "/a/b", "file"), "/a", ErrIsDir},
		{Tree{}, "/", ErrIsDir},
	} {
		if _, err := test.tree.Put(test.path, File{}); !errors.Is(err, test.err) {
			t.Errorf("Put(%s): want %v, got %v", test.path, test.err, err)
		}
		if _, err := test.tree.Get(test.path); !errors.Is(err, test.err) {
			t.Errorf("Get(%s): want %v, got %v", test.path, test.err, err)
		}
	}
}

func TestPathValidation(t *testing.T) {
	longest := strings.Repeat("/"+strings.Repeat("a", 255), 16)
	bad := map[string]string{
		"":                             "path must be absolute",
		"a":                            "path must be absolute",
		"/a//b":                        "path contains an empty component",
		"/a/":                          "path contains an empty component",
		"/a/../b":                      "path contains '.' or '..'",
		"/a/./b":                       "path contains '.' or '..'",
		"/a\x00b":                      "path contains control characters",
		"/a\tb":                        "path contains control characters",
		"/a\x7fb":                      "path contains control characters",
		"/cafe\u0301":                  "path is not NFC-normalized",
		"/\xff":                        "path is not valid UTF-8",
		"/" + strings.Repeat("a", 256): "path component is too long",
		"/" + strings.Repeat("é", 128): "path component is too long",
		longest + "a":                  "path is too long",
	}
	for path, reason := range bad {
		_, getErr := (Tree{}).Get(path)
		_, putErr := (Tree{}).Put(path, File{})
		for _, err := range []error{getErr, putErr, ValidPath(path)} {
			var pe *PathError
			if !errors.As(err, &pe) || pe.Reason != reason {
				t.Errorf("path %q: want PathError %q, got %v", path, reason, err)
			}
		}
	}
	for _, path := range []string{"/café", "/" + strings.Repeat("a", 255), longest} {
		if err := ValidPath(path); err != nil {
			t.Errorf("ValidPath(%q): %v", path, err)
		}
		if _, err := mustPut(t, Tree{}, path, "x").Get(path); err != nil {
			t.Errorf("Get(%q): %v", path, err)
		}
	}
	if err := ValidPath("/"); err != nil {
		t.Errorf("ValidPath(/): %v", err)
	}
}

func TestDelete(t *testing.T) {
	base := mustPut(t, Tree{}, "/a/b/file", "x")
	base = mustPut(t, base, "/keep", "y")
	next, err := base.Delete("/a/b/file")
	if err != nil {
		t.Fatal(err)
	}
	if want := mustPut(t, Tree{}, "/keep", "y"); next.Root() != want.Root() {
		t.Fatal("Delete did not prune empty parents")
	}
	if f, err := base.Get("/a/b/file"); err != nil || string(f.Content) != "x" {
		t.Fatalf("Delete mutated original snapshot: %v", err)
	}
	next, err = next.Delete("/keep")
	if err != nil || next.Root() != (Tree{}).Root() {
		t.Fatalf("Delete last file: %v", err)
	}
	for _, tc := range []struct {
		path string
		err  error
	}{
		{"/", ErrIsDir},
		{"/a", ErrIsDir},
		{"/absent", ErrNotFound},
		{"/keep/child", ErrNotDir},
	} {
		if _, err := base.Delete(tc.path); !errors.Is(err, tc.err) {
			t.Errorf("Delete(%q): got %v, want %v", tc.path, err, tc.err)
		}
	}
	if _, err := base.Delete("/bad/../path"); err == nil {
		t.Fatal("Delete accepted an invalid path")
	}
}

func TestWalkFiles(t *testing.T) {
	var base Tree
	want := []string{"/a.txt", "/a/z", "/aa", "/b"}
	for _, p := range want {
		base = mustPut(t, base, p, p)
	}
	var got []string
	if err := base.WalkFiles("/", func(p string, f File) error {
		if string(f.Content) != p {
			t.Errorf("WalkFiles(%q) returned wrong content", p)
		}
		got = append(got, p)
		return nil
	}); err != nil || strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("WalkFiles order: %v, %v", got, err)
	}
	for _, path := range []string{"/a", "/a/z"} {
		var got []string
		err := base.WalkFiles(path, func(p string, _ File) error {
			got = append(got, p)
			return nil
		})
		if err != nil || len(got) != 1 || got[0] != "/a/z" {
			t.Fatalf("WalkFiles(%q): %v, %v", path, got, err)
		}
	}
	stop := errors.New("stop")
	calls := 0
	err := base.WalkFiles("/", func(string, File) error { calls++; return stop })
	if err != stop || calls != 1 {
		t.Fatalf("WalkFiles did not stop: %v, %d calls", err, calls)
	}
	if err := (Tree{}).WalkFiles("/", func(string, File) error { t.Fatal("empty tree callback"); return nil }); err != nil {
		t.Fatal(err)
	}
	if err := base.WalkFiles("/absent", func(string, File) error { return nil }); !errors.Is(err, ErrNotFound) {
		t.Fatalf("WalkFiles(absent): %v", err)
	}
}

func TestStatsSnapshots(t *testing.T) {
	type snapshot struct {
		name string
		tree Tree
		want Stats
	}
	var snapshots []snapshot
	check := func(name string, tr Tree, want Stats) Tree {
		t.Helper()
		snapshots = append(snapshots, snapshot{name, tr, want})
		// Changes must update the new snapshot's caches without changing any
		// previously captured snapshot, including after subtree removal.
		for _, s := range snapshots {
			if got := s.tree.Stats(); got != s.want {
				t.Fatalf("after %s, %s Stats: got %+v, want %+v", name, s.name, got, s.want)
			}
		}
		return tr
	}
	tr := check("empty", Tree{}, Stats{})
	tr = check("empty file", mustPut(t, tr, "/empty", ""), Stats{Entries: 1, Files: 1})
	tr = check("parent", mustPut(t, tr, "/d/a", "abc"), Stats{Entries: 3, Files: 2, Bytes: 3, MaxFileBytes: 3})
	tr = check("nested parent", mustPut(t, tr, "/d/deep/b", "1234567"), Stats{Entries: 5, Files: 3, Bytes: 10, MaxFileBytes: 7})
	tr = check("sibling", mustPut(t, tr, "/other", "12345"), Stats{Entries: 6, Files: 4, Bytes: 15, MaxFileBytes: 7})
	tr = check("replace largest", mustPut(t, tr, "/d/deep/b", "x"), Stats{Entries: 6, Files: 4, Bytes: 9, MaxFileBytes: 5})
	var err error
	tr, err = tr.Delete("/other")
	if err != nil {
		t.Fatal(err)
	}
	tr = check("delete largest", tr, Stats{Entries: 5, Files: 3, Bytes: 4, MaxFileBytes: 3})
	sub := mustPut(t, Tree{}, "/inside/a", "12345678901")
	sub = mustPut(t, sub, "/b", "12")
	check("graft source", sub, Stats{Entries: 3, Files: 2, Bytes: 13, MaxFileBytes: 11})
	tr, err = tr.Graft("mounted", sub)
	if err != nil {
		t.Fatal(err)
	}
	tr = check("graft", tr, Stats{Entries: 9, Files: 5, Bytes: 17, MaxFileBytes: 11})
	tr, err = tr.Without("d")
	if err != nil {
		t.Fatal(err)
	}
	tr = check("without subtree", tr, Stats{Entries: 5, Files: 3, Bytes: 13, MaxFileBytes: 11})
	tr, err = tr.Without("mounted")
	if err != nil {
		t.Fatal(err)
	}
	tr = check("without largest subtree", tr, Stats{Entries: 1, Files: 1})
	tr, err = tr.Delete("/empty")
	if err != nil {
		t.Fatal(err)
	}
	check("delete last", tr, Stats{})
}

func TestWalkFilesFullPathOrdering(t *testing.T) {
	paths := []string{
		"/a!/x", "/a-file", "/a-/x", "/a./x", "/a.txt", "/a/x",
		"/a/a!/x", "/a/a-", "/a/a./x", "/a/a/x", "/a/a0", "/a/az/x",
		"/a0", "/a_/x", "/a~/x", "/é/x", "/é-/x", "/é./x", "/é.txt",
	}
	var tr Tree
	for i := len(paths) - 1; i >= 0; i-- {
		tr = mustPut(t, tr, paths[i], paths[i])
	}
	slices.Sort(paths)
	var got []string
	err := tr.WalkFiles("/", func(path string, file File) error {
		if string(file.Content) != path {
			t.Fatalf("wrong content for %q: %q", path, file.Content)
		}
		got = append(got, path)
		return nil
	})
	if err != nil || !slices.Equal(got, paths) {
		t.Fatalf("WalkFiles: got %q, want %q, error %v", got, paths, err)
	}
}

func TestWalkFilesEarlyStopDoesNotCollectSubtree(t *testing.T) {
	small := mustPut(t, Tree{}, "/a", "first")
	small = mustPut(t, small, "/z/0/file", "later")
	large := small
	for i := 1; i < 512; i++ {
		large = mustPut(t, large, fmt.Sprintf("/z/%d/file", i), "later")
	}
	stop := errors.New("stop walking")
	allocs := func(tr Tree) float64 {
		return testing.AllocsPerRun(10, func() {
			calls := 0
			err := tr.WalkFiles("/", func(path string, _ File) error {
				calls++
				if path != "/a" {
					t.Fatalf("unexpected callback for %q", path)
				}
				return stop
			})
			if err != stop || calls != 1 {
				t.Fatalf("WalkFiles: error %v, %d calls", err, calls)
			}
		})
	}
	// Both walks stop before /z. Increasing that subtree must not allocate
	// paths or traversal state for its contents before the first callback.
	smallAllocs, largeAllocs := allocs(small), allocs(large)
	if largeAllocs > smallAllocs+10 {
		t.Fatalf("early stop allocations grew with unvisited subtree: small %.0f, large %.0f", smallAllocs, largeAllocs)
	}
}

func TestTransferSnapshots(t *testing.T) {
	base := mustPut(t, Tree{}, "/source/nested/file", "source contents")
	base = mustPut(t, base, "/destination/existing", "old")
	base = mustPut(t, base, "/keep", "untouched")
	for _, tc := range []struct {
		name, source, destination string
		rename                    bool
	}{
		{"copy creates parents", "/source/nested/file", "/new/deep/copy", false},
		{"copy replaces", "/source/nested/file", "/destination/existing", false},
		{"copy self", "/source/nested/file", "/source/nested/file", false},
		{"rename creates parents", "/source/nested/file", "/new/deep/moved", true},
		{"rename replaces", "/source/nested/file", "/destination/existing", true},
		{"rename sibling", "/source/nested/file", "/source/nested/other", true},
		{"rename self", "/source/nested/file", "/source/nested/file", true},
		{"rename similar prefix", "/source/nested/file", "/source/nested/file-other/child", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, stats := base.Root(), base.Stats()
			op := base.Copy
			if tc.rename {
				op = base.Rename
			}
			got, err := op(tc.source, tc.destination)
			if err != nil {
				t.Fatal(err)
			}
			// Construct the expected snapshot through the independent public
			// Put/Delete path, checking hashes and all cached statistics.
			want := base
			if tc.rename && tc.source != tc.destination {
				want, err = want.Delete(tc.source)
				if err != nil {
					t.Fatal(err)
				}
			}
			want = mustPut(t, want, tc.destination, "source contents")
			if got.Root() != want.Root() || got.Stats() != want.Stats() {
				t.Fatalf("transfer hash/stats mismatch: got %+v, want %+v", got.Stats(), want.Stats())
			} else if base.Root() != root || base.Stats() != stats {
				t.Fatal("transfer changed original snapshot")
			}
			if tc.source == tc.destination && got.root != base.root {
				t.Fatal("self-transfer unnecessarily recreated the snapshot")
			}
		})
	}
}

func TestTransferSharesFileNode(t *testing.T) {
	content := bytes.Repeat([]byte{0xff, 0, 1, 2}, 1<<18)
	base, err := (Tree{}).Put("/source", File{Content: content})
	if err != nil {
		t.Fatal(err)
	}
	_, source, err := base.lookup([]string{"source"})
	if err != nil {
		t.Fatal(err)
	}
	copied, err := base.Copy("/source", "/nested/copy")
	if err != nil {
		t.Fatal(err)
	}
	_, copyNode, _ := copied.lookup([]string{"nested", "copy"})
	if source != copyNode {
		t.Fatal("copy did not share existing content and cached hash")
	}
	renamed, err := copied.Rename("/nested/copy", "/elsewhere/moved")
	if err != nil {
		t.Fatal(err)
	}
	_, movedNode, _ := renamed.lookup([]string{"elsewhere", "moved"})
	if source != movedNode {
		t.Fatal("rename did not share existing content and cached hash")
	}
	if _, err := renamed.Get("/nested"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("rename did not prune empty parents: %v", err)
	}
	updated := mustPut(t, renamed, "/elsewhere/moved", "replacement")
	for _, tc := range []struct {
		tree Tree
		path string
	}{
		{base, "/source"}, {copied, "/nested/copy"},
		{renamed, "/elsewhere/moved"}, {updated, "/source"},
	} {
		f, err := tc.tree.Get(tc.path)
		if err != nil || !bytes.Equal(f.Content, content) {
			t.Fatalf("replacement changed shared file at %s: %v", tc.path, err)
		}
	}
}

func TestTransferErrors(t *testing.T) {
	base := mustPut(t, Tree{}, "/dir/file", "original")
	base = mustPut(t, base, "/other", "other contents")
	for _, tc := range []struct {
		source, destination string
		want                error
	}{
		{"/missing", "/new", ErrNotFound},
		{"/missing", "/missing", ErrNotFound},
		{"/", "/new", ErrIsDir},
		{"/dir", "/new", ErrIsDir},
		{"/dir", "/dir", ErrIsDir},
		{"/dir/file/child", "/new", ErrNotDir},
		{"/dir/file", "/", ErrIsDir},
		{"/dir/file", "/dir", ErrIsDir},
		{"/dir/file", "/dir/file/child", ErrNotDir},
		{"/dir/file", "/other/child", ErrNotDir},
	} {
		for _, rename := range []bool{false, true} {
			t.Run(tc.source+" -> "+tc.destination+map[bool]string{false: " copy", true: " rename"}[rename], func(t *testing.T) {
				root, stats := base.Root(), base.Stats()
				op := base.Copy
				if rename {
					op = base.Rename
				}
				if _, err := op(tc.source, tc.destination); !errors.Is(err, tc.want) {
					t.Fatalf("got %v, want %v", err, tc.want)
				} else if base.Root() != root || base.Stats() != stats {
					t.Fatal("failed transfer changed original tree")
				}
			})
		}
	}
	for _, bad := range []string{"", "relative", "/bad/../path", "/bad//path", "/" + strings.Repeat("x", 256), "/bad\x00path"} {
		for _, op := range []func(string, string) (Tree, error){base.Copy, base.Rename} {
			for _, paths := range [][2]string{{bad, "/new"}, {"/dir/file", bad}} {
				var pe *PathError
				if _, err := op(paths[0], paths[1]); !errors.As(err, &pe) {
					t.Fatalf("invalid transfer path %q accepted: %v", bad, err)
				}
			}
		}
	}
}

func TestGraft(t *testing.T) {
	base := mustPut(t, Tree{}, "/artifact/main.go", "main")
	base = mustPut(t, base, "/source/input", "input")
	sub := mustPut(t, Tree{}, "/bin/compile.wasm", "compile")
	sub = mustPut(t, sub, "/std/fmt.a", "fmt")
	baseRoot, subRoot := base.Root(), sub.Root()
	for _, name := range []string{"a", "env", "z", "café"} {
		mounted, err := base.Graft(name, sub)
		if err != nil {
			t.Fatal(err)
		}
		want := mustPut(t, base, "/"+name+"/bin/compile.wasm", "compile")
		want = mustPut(t, want, "/"+name+"/std/fmt.a", "fmt")
		if mounted.Root() != want.Root() {
			t.Fatalf("Graft(%q) changed canonical tree hash", name)
		}
		changed := mustPut(t, mounted, "/"+name+"/std/fmt.a", "replacement")
		if changed.Root() == mounted.Root() || mounted.Root() != want.Root() {
			t.Fatal("editing mounted tree changed its source")
		}
		unmounted, err := mounted.Without(name)
		if err != nil || unmounted.Root() != baseRoot {
			t.Fatalf("Graft/Without round trip: %v", err)
		}
	}
	if base.Root() != baseRoot || sub.Root() != subRoot {
		t.Fatal("mount operations changed their source trees")
	}
	if f, err := sub.Get("/std/fmt.a"); err != nil || string(f.Content) != "fmt" {
		t.Fatalf("mounted edit changed subtree content: %q, %v", f.Content, err)
	}
}

func TestGraftCollision(t *testing.T) {
	for _, sub := range []Tree{{}, mustPut(t, Tree{}, "/tool", "tool")} {
		for _, path := range []string{"/env", "/env/existing"} {
			base := mustPut(t, Tree{}, path, "existing")
			root := base.Root()
			if _, err := base.Graft("env", sub); err == nil {
				t.Fatalf("Graft replaced %s", path)
			}
			if base.Root() != root {
				t.Fatal("failed Graft changed source tree")
			}
		}
	}
}

func TestGraftPathExpansion(t *testing.T) {
	longest := strings.Repeat("/"+strings.Repeat("a", 255), 16)
	path := longest[:len(longest)-len("/env")]
	sub := mustPut(t, Tree{}, path, "boundary")
	mounted, err := (Tree{}).Graft("env", sub)
	if err != nil {
		t.Fatal(err)
	}
	if f, err := mounted.Get("/env" + path); err != nil || string(f.Content) != "boundary" {
		t.Fatalf("maximum-length mounted path: %q, %v", f.Content, err)
	}
	sub = mustPut(t, Tree{}, path+"a", "too long after graft")
	if _, err := (Tree{}).Graft("env", sub); err == nil {
		t.Fatal("Graft accepted a path longer than the tree limit")
	}
}

func TestWithout(t *testing.T) {
	base := mustPut(t, Tree{}, "/env/bin/compile.wasm", "compile")
	base = mustPut(t, base, "/artifact/main.go", "main")
	base = mustPut(t, base, "/source", "input")
	root := base.Root()
	for _, name := range []string{"artifact", "env", "source"} {
		next, err := base.Without(name)
		if err != nil {
			t.Fatal(err)
		}
		var want Tree
		if err := base.WalkFiles("/", func(path string, f File) error {
			if path == "/"+name || strings.HasPrefix(path, "/"+name+"/") {
				return nil
			}
			var err error
			want, err = want.Put(path, f)
			return err
		}); err != nil {
			t.Fatal(err)
		}
		if next.Root() != want.Root() || base.Root() != root {
			t.Fatalf("Without(%q) returned incorrect tree or mutated source", name)
		}
		if _, err := next.Get("/" + name); !errors.Is(err, ErrNotFound) {
			t.Fatalf("Without(%q) retained removed entry: %v", name, err)
		}
	}
	for _, path := range []string{"/env", "/env/bin/tool"} {
		next, err := mustPut(t, Tree{}, path, "only").Without("env")
		if err != nil || next.Root() != (Tree{}).Root() {
			t.Fatalf("removing last root entry: %v", err)
		}
	}
}

func TestMountNoops(t *testing.T) {
	for _, base := range []Tree{{}, mustPut(t, Tree{}, "/env/tool", "tool")} {
		next, err := base.Graft("missing", Tree{})
		if err != nil || next.Root() != base.Root() {
			t.Fatalf("grafting empty subtree: %v", err)
		}
		next, err = base.Without("missing")
		if err != nil || next.Root() != base.Root() {
			t.Fatalf("removing missing entry: %v", err)
		}
	}
}

func TestMountNames(t *testing.T) {
	for _, name := range []string{"", "/", "/env", "env/", "env/bin", ".", "..", "a\x00b", "\xff", "cafe\u0301", strings.Repeat("a", 256)} {
		_, graftErr := (Tree{}).Graft(name, Tree{})
		_, withoutErr := (Tree{}).Without(name)
		for _, err := range []error{graftErr, withoutErr} {
			var pe *PathError
			if !errors.As(err, &pe) {
				t.Errorf("name %q: expected PathError, got %v", name, err)
			}
		}
	}
}
