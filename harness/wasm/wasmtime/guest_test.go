package wasmtime

import (
	"os"
	"reflect"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/wasm"
)

func TestRealGuest(t *testing.T) {
	bin, err := os.ReadFile("testdata/wordfreq.wasm")
	if err != nil {
		t.Fatal(err)
	}
	const input = "the workspace is a value the value is a tree\n"
	const output = "a\t2\nis\t2\nthe\t2\ntree\t1\nvalue\t2\nworkspace\t1\n"
	base := testTree(t, map[string]string{"/env/essay.txt": input, "/artifact/file": "keep"})
	baseRoot := base.Root()
	req := wasm.Request{
		Module: bin,
		Args:   []string{"wordfreq", "/env/essay.txt", "/artifact/tables/freq.tsv"},
		Limits: testLimits(1 << 28),
	}
	resp, next, err := Run(base, req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Trap != wasm.TrapNone || resp.ExitCode != 0 || resp.Usage.Fuel == 0 || len(resp.Stderr) != 0 {
		t.Fatalf("response: %+v; stderr: %s", resp, resp.Stderr)
	}
	if string(resp.Stdout) != "6 distinct words -> /artifact/tables/freq.tsv\n" {
		t.Fatalf("stdout = %q", resp.Stdout)
	}
	want := testTree(t, map[string]string{
		"/env/essay.txt":            input,
		"/artifact/file":            "keep",
		"/artifact/tables/freq.tsv": output,
	})
	if next.Root() != want.Root() {
		t.Fatal("guest produced the wrong tree")
	}
	if base.Root() != baseRoot {
		t.Fatal("run mutated the input tree")
	}
	replay, replayTree, err := Run(base, req)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resp, replay) || next.Root() != replayTree.Root() {
		t.Fatalf("nondeterministic guest: %+v vs %+v", resp, replay)
	}

	for _, tc := range []struct {
		name string
		args []string
		exit uint32
		text string
	}{
		{"usage", []string{"wordfreq"}, 2, "usage: wordfreq <input> <output>"},
		{"missing input", []string{"wordfreq", "/env/missing.txt", "/artifact/freq.tsv"}, 1, "/env/missing.txt"},
		{"read-only output", []string{"wordfreq", "/env/essay.txt", "/env/essay.txt"}, 1, "/env/essay.txt"},
		{"file as parent", []string{"wordfreq", "/env/essay.txt", "/artifact/file/freq.tsv"}, 1, "/artifact/file/freq.tsv"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req.Args = tc.args
			resp, next, err := Run(base, req)
			if err != nil {
				t.Fatal(err)
			}
			if resp.Trap != wasm.TrapNone || resp.ExitCode != tc.exit || len(resp.Stdout) != 0 || !strings.Contains(string(resp.Stderr), tc.text) {
				t.Fatalf("response: %+v; stderr: %s", resp, resp.Stderr)
			}
			if next.Root() != base.Root() {
				t.Fatal("failed guest changed the tree")
			}
		})
	}
}
