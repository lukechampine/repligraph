package wasmtime

import (
	"reflect"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
)

func mustTree(t *testing.T, files map[string]string) tree.Tree {
	t.Helper()
	tr := tree.Tree{}
	for p, c := range files {
		var err error
		tr, err = tr.Put(p, tree.File{Content: []byte(c)})
		if err != nil {
			t.Fatal(err)
		}
	}
	return tr
}

func TestResolvePath(t *testing.T) {
	cases := []struct {
		dir, p string
		want   string
		rc     errno
	}{
		{"/", "a/b", "/a/b", errSuccess},
		{"/", "./a/../b", "/b", errSuccess},
		{"/", "../../x", "/x", errSuccess}, // clamps at root
		{"/sub", "f", "/sub/f", errSuccess},
		{"/sub", "../f", "/f", errSuccess},
		{"/sub", "/abs", "/abs", errSuccess},
		{"/", "a//b/", "/a/b", errSuccess},
		{"/", ".", "/", errSuccess},
		{"/", "bad\x01name", "", errIlseq},
		{"/", "café", "", errIlseq}, // non-NFC
	}
	for _, c := range cases {
		got, rc := resolvePath(c.dir, c.p)
		if rc != c.rc || (rc == errSuccess && got != c.want) {
			t.Errorf("resolvePath(%q, %q) = %q, %d; want %q, %d", c.dir, c.p, got, rc, c.want, c.rc)
		}
	}
}

func TestOverlayReaddir(t *testing.T) {
	base := mustTree(t, map[string]string{
		"/artifact/a.txt":     "aa",
		"/artifact/sub/b.txt": "b",
		"/artifact/sub/c.txt": "c",
		"/artifact/zzz/d.txt": "d",
	})
	o := newOverlay(base, 10000, 1<<20)
	if rc := o.setFile("/artifact/new.txt", []byte("n")); rc != errSuccess {
		t.Fatalf("setFile: %d", rc)
	}
	if rc := o.unlink("/artifact/a.txt"); rc != errSuccess {
		t.Fatalf("unlink: %d", rc)
	}
	if rc := o.mkdir("/artifact/made"); rc != errSuccess {
		t.Fatalf("mkdir: %d", rc)
	}
	if got := o.readdir("/"); !reflect.DeepEqual(got, []dirent{{"artifact", true, 0}}) {
		t.Fatalf("readdir(/) = %+v", got)
	}
	got := o.readdir("/artifact")
	want := []dirent{
		{"made", true, 0},
		{"new.txt", false, 1},
		{"sub", true, 0},
		{"zzz", true, 0},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("readdir(/artifact) = %+v, want %+v", got, want)
	}

	o.unlink("/artifact/zzz/d.txt")
	got = o.readdir("/artifact")
	want = []dirent{
		{"made", true, 0},
		{"new.txt", false, 1},
		{"sub", true, 0},
		{"zzz", true, 0},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("readdir(/artifact) after unlink = %+v, want %+v", got, want)
	}
	if rc := o.rmdir("/artifact/zzz"); rc != errSuccess {
		t.Fatalf("rmdir: %d", rc)
	}
	if o.dirExists("/artifact/zzz") {
		t.Fatal("/artifact/zzz still exists after rmdir")
	}
}

func TestOverlayMkdirRules(t *testing.T) {
	o := newOverlay(mustTree(t, map[string]string{"/artifact/f": "x"}), 10000, 1<<20)
	if rc := o.mkdir("/artifact/f"); rc != errExist {
		t.Errorf("mkdir over file: %d, want EEXIST", rc)
	}
	if rc := o.mkdir("/artifact/a/b"); rc != errNoent {
		t.Errorf("mkdir without parent: %d, want ENOENT", rc)
	}
	if rc := o.mkdir("/artifact/a"); rc != errSuccess {
		t.Errorf("mkdir /artifact/a: %d", rc)
	}
	if rc := o.mkdir("/artifact/a/b"); rc != errSuccess {
		t.Errorf("mkdir /artifact/a/b: %d", rc)
	}
	if rc := o.rmdir("/artifact/a"); rc != errNotempty {
		t.Errorf("rmdir non-empty: %d, want ENOTEMPTY", rc)
	}
	if rc := o.rmdir("/artifact/f"); rc != errNotdir {
		t.Errorf("rmdir file: %d, want ENOTDIR", rc)
	}
	if rc := o.unlink("/artifact/a"); rc != errIsdir {
		t.Errorf("unlink dir: %d, want EISDIR", rc)
	}
	if rc := o.mkdir("/elsewhere"); rc != errRofs {
		t.Errorf("mkdir outside workspace: %d, want EROFS", rc)
	}
	if rc := o.setFile("/nope", nil); rc != errRofs {
		t.Errorf("setFile outside workspace: %d, want EROFS", rc)
	}
}

func TestOverlayRename(t *testing.T) {
	base := mustTree(t, map[string]string{
		"/artifact/d/one": "1",
		"/artifact/d/two": "2",
		"/artifact/other": "o",
	})
	o := newOverlay(base, 10000, 1<<20)
	if rc := o.rename("/artifact/d/one", "/artifact/moved"); rc != errSuccess {
		t.Fatalf("file rename: %d", rc)
	}
	if c, ok := o.fileContent("/artifact/moved"); !ok || string(c) != "1" {
		t.Fatal("moved content wrong")
	}
	if o.fileExists("/artifact/d/one") {
		t.Fatal("source survived rename")
	}
	if rc := o.mkdir("/artifact/dst"); rc != errSuccess {
		t.Fatal("mkdir /dst")
	}
	if rc := o.rename("/artifact/d", "/artifact/dst/d2"); rc != errSuccess {
		t.Fatalf("dir rename: %d", rc)
	}
	if c, ok := o.fileContent("/artifact/dst/d2/two"); !ok || string(c) != "2" {
		t.Fatal("dir rename lost /d/two")
	}
	if o.dirExists("/artifact/d") {
		t.Fatal("/d survived dir rename")
	}
	if rc := o.rename("/artifact/missing", "/artifact/x"); rc != errNoent {
		t.Errorf("rename missing: %d, want ENOENT", rc)
	}
	if rc := o.rename("/artifact/other", "/artifact/dst/d2"); rc != errIsdir {
		t.Errorf("rename file onto dir: %d, want EISDIR", rc)
	}
}

func TestOverlayCommit(t *testing.T) {
	base := mustTree(t, map[string]string{"/artifact/keep": "k", "/artifact/gone": "g", "/artifact/mod": "old"})
	o := newOverlay(base, 10000, 1<<20)
	o.unlink("/artifact/gone")
	o.setFile("/artifact/mod", []byte("new"))
	o.setFile("/artifact/fresh/child", []byte("f"))
	o.mkdir("/artifact/empty") // vanishes: the tree cannot represent it

	got, err := o.commit()
	if err != nil {
		t.Fatal(err)
	}
	want := mustTree(t, map[string]string{
		"/artifact/keep":        "k",
		"/artifact/mod":         "new",
		"/artifact/fresh/child": "f",
	})
	if got.Root() != want.Root() {
		t.Fatalf("commit root mismatch:\n got %x\nwant %x", got.Root(), want.Root())
	}
	if _, err := base.Get("/artifact/gone"); err != nil {
		t.Fatal("base tree mutated by overlay")
	}
}

func TestOverlayUnlinkCreateCycle(t *testing.T) {
	o := newOverlay(mustTree(t, map[string]string{"/artifact/f": "orig"}), 10000, 1<<20)
	o.unlink("/artifact/f")
	if o.fileExists("/artifact/f") {
		t.Fatal("tombstone ignored")
	}
	o.setFile("/artifact/f", []byte("再"))
	c, ok := o.fileContent("/artifact/f")
	if !ok || string(c) != "再" {
		t.Fatal("recreate after unlink failed")
	}
	got, err := o.commit()
	if err != nil {
		t.Fatal(err)
	}
	if f, err := got.Get("/artifact/f"); err != nil || string(f.Content) != "再" {
		t.Fatal("commit lost recreated file")
	}
}

func TestOverlayReadOnlyMount(t *testing.T) {
	base := mustTree(t, map[string]string{
		"/artifact/work": "w",
		"/env/std/fmt.a": "archive",
		"/env/importcfg": "config",
	})
	o := newOverlay(base, 10000, 1<<20)
	if c, ok := o.fileContent("/env/importcfg"); !ok || string(c) != "config" {
		t.Fatalf("read-only file: %q, %v", c, ok)
	}
	for _, rc := range []errno{
		o.unlink("/env/importcfg"),
		o.mkdir("/env/new"),
		o.rmdir("/env/std"),
		o.rename("/env/importcfg", "/artifact/stolen"),
		o.rename("/artifact/work", "/env/x"),
	} {
		if rc != errRofs {
			t.Fatalf("read-only mutation: %d", rc)
		}
	}
	if got := o.readdir("/"); !reflect.DeepEqual(got, []dirent{{"artifact", true, 0}, {"env", true, 0}}) {
		t.Fatalf("readdir(/) = %+v", got)
	}
	nt, err := o.commit()
	if err != nil || nt.Root() != base.Root() {
		t.Fatalf("read-only mount changed: %v", err)
	}
}

func TestOverlayEmptyArtifact(t *testing.T) {
	o := newOverlay(tree.Tree{}, 10000, 0)
	if got := o.readdir("/"); !reflect.DeepEqual(got, []dirent{{"artifact", true, 0}}) {
		t.Fatalf("readdir(/) = %+v", got)
	}
	if got, err := o.commit(); err != nil || got.Root() != (tree.Tree{}).Root() {
		t.Fatalf("empty artifact changed committed tree: %v", err)
	}
}

func TestOverlayRmdirPreservesParent(t *testing.T) {
	o := newOverlay(mustTree(t, map[string]string{"/artifact/a/b/file": "x"}), 10000, 1<<20)
	if rc := o.unlink("/artifact/a/b/file"); rc != errSuccess {
		t.Fatal(rc)
	}
	if rc := o.rmdir("/artifact/a/b"); rc != errSuccess {
		t.Fatal(rc)
	}
	if !o.dirExists("/artifact/a") || o.dirExists("/artifact/a/b") {
		t.Fatal("rmdir removed its parent or retained the removed directory")
	}
	if got := o.readdir("/artifact"); !reflect.DeepEqual(got, []dirent{{"a", true, 0}}) {
		t.Fatalf("readdir(/artifact) = %+v", got)
	}
	if rc := o.rmdir("/artifact/a"); rc != errSuccess {
		t.Fatal(rc)
	}
}

func TestOverlayRenameValidatesDescendants(t *testing.T) {
	o := newOverlay(mustTree(t, map[string]string{
		"/artifact/src/a":         "first",
		"/artifact/src/long-name": "last",
	}), 10000, 1<<20)
	prefix := "/artifact" + strings.Repeat("/"+strings.Repeat("p", 255), 15)
	dest := prefix + "/" + strings.Repeat("q", 4090-len(prefix)-1)
	if err := tree.ValidPath(dest); err != nil {
		t.Fatal(err)
	}
	before, err := o.commit()
	if err != nil {
		t.Fatal(err)
	}
	if rc := o.rename("/artifact/src", dest); rc != errNametoolong {
		t.Fatalf("rename: %d, want ENAMETOOLONG", rc)
	}
	after, err := o.commit()
	if err != nil || after.Root() != before.Root() || len(o.files)+len(o.deleted) != 0 || len(o.dirs) != 2 || o.entries != 4 {
		t.Fatalf("failed rename mutated overlay: %v", err)
	}
}

func TestOverlayReplaceFileAndDirectory(t *testing.T) {
	t.Run("file to directory", func(t *testing.T) {
		o := newOverlay(mustTree(t, map[string]string{"/artifact/a": "old"}), 10000, 1<<20)
		for _, rc := range []errno{o.unlink("/artifact/a"), o.mkdir("/artifact/a"), o.setFile("/artifact/a/b", []byte("new"))} {
			if rc != errSuccess {
				t.Fatal(rc)
			}
		}
		if _, ok := o.fileContent("/artifact/a"); ok {
			t.Fatal("base file survived directory replacement")
		}
		got, err := o.commit()
		if want := mustTree(t, map[string]string{"/artifact/a/b": "new"}); err != nil || got.Root() != want.Root() {
			t.Fatalf("file to directory commit: %v", err)
		}
	})
	t.Run("directory to file", func(t *testing.T) {
		o := newOverlay(mustTree(t, map[string]string{"/artifact/a/b": "old"}), 10000, 1<<20)
		for _, rc := range []errno{o.unlink("/artifact/a/b"), o.rmdir("/artifact/a"), o.setFile("/artifact/a", []byte("new"))} {
			if rc != errSuccess {
				t.Fatal(rc)
			}
		}
		if o.dirExists("/artifact/a") {
			t.Fatal("base directory survived file replacement")
		}
		got, err := o.commit()
		if want := mustTree(t, map[string]string{"/artifact/a": "new"}); err != nil || got.Root() != want.Root() {
			t.Fatalf("directory to file commit: %v", err)
		}
	})
}
