package repligraph

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"slices"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

type bundleMember struct {
	name string
	data []byte
}

func bundleZIP(t testing.TB, members ...bundleMember) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	for _, m := range members {
		f, err := w.CreateHeader(&zip.FileHeader{Name: m.name, Method: zip.Store})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write(m.data); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func zipReader(buf []byte) *zip.Reader {
	r, err := zip.NewReader(bytes.NewReader(buf), int64(len(buf)))
	if err != nil {
		panic(err)
	}
	return r
}

// A tiny wire fixture, independent of the writer: initial tokens, then alternating
// empty and nonempty model/tool records. The codec does not interpret their meaning.
func codecFixture(t testing.TB) (BundleHeader, []byte, []byte) {
	t.Helper()
	h := BundleHeader{
		Model: "m", Inference: "i", Environment: "e", TreeStart: (tree.Tree{}).Root(),
		Seed: ^uint64(0), Temperature: 0.7, InitialContext: []uint32{0, 0x12345678, 0xffffffff},
		Usage: Usage{ModelTurns: 2, HarnessTurns: 2, OutputTokens: 2, ToolOutputBytes: 3, ContextTokens: 3},
	}
	u, err := json.Marshal(h.Usage)
	if err != nil {
		t.Fatal(err)
	}
	manifest := []byte(fmt.Sprintf(`{"version":0,"model":"m","inference":"i","environment":"e","tree_start":%q,"seed":18446744073709551615,"temperature":0.7,"usage":%s}`, h.TreeStart.String(), u))
	transcript, err := hex.DecodeString("03000000000000000000000078563412ffffffff" +
		"010000000000000000020000000000000000" +
		"01020000000000000004030201feffffff02030000000000000000ff0a")
	if err != nil {
		t.Fatal(err)
	}
	return h, manifest, transcript
}

func changeManifest(t testing.TB, data []byte, field string, value any) []byte {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	v, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	m[field] = v
	data, err = json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

type trackedTranscript struct {
	io.Reader
	read, closes int
}

func (r *trackedTranscript) Read(p []byte) (int, error) {
	n, err := r.Reader.Read(p)
	r.read += n
	return n, err
}
func (r *trackedTranscript) Close() error { r.closes++; return nil }

func trackTranscript(z *zip.Reader) *trackedTranscript {
	r := new(trackedTranscript)
	for _, f := range z.File {
		if f.Name == "transcript.log" {
			f.Method = 65535 // Fixtures use Store, so the source is already plaintext.
		}
	}
	z.RegisterDecompressor(65535, func(source io.Reader) io.ReadCloser { r.Reader = source; return r })
	return r
}

func TestBundleStreamingRoundTrip(t *testing.T) {
	h, _, transcript := codecFixture(t)
	initial, err := (tree.Tree{}).Put("/input/data", tree.File{Content: []byte{0, 255, '\n'}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		tree *tree.Tree
	}{{"external", nil}, {"empty", new(tree.Tree)}, {"files", &initial}} {
		t.Run(tc.name, func(t *testing.T) {
			header, usage := h, h.Usage
			header.InitialTree = tc.tree
			header.InitialCall = &ToolCall{Tool: "run", Args: []byte{0, 255, '\n'}, Ref: "call-λ"}
			if tc.tree != nil {
				header.TreeStart = tc.tree.Root()
				usage.InitialTreeBytes, usage.TreeEntries = uint64(tc.tree.Stats().Bytes), uint64(tc.tree.Stats().Entries)
				usage.TreeBytes = usage.InitialTreeBytes
			}
			header.Usage = Usage{} // Final usage is supplied only when closing.
			var dst bytes.Buffer
			w, err := NewBundleWriter(&dst, header)
			if err != nil {
				t.Fatal(err)
			}
			for _, write := range []func() error{
				func() error { return w.WriteModelOutput(nil) },
				func() error { return w.WriteToolOutput(nil) },
				func() error { return w.WriteModelOutput([]uint32{0x01020304, 0xfffffffe}) },
				func() error { return w.WriteToolOutput([]byte{0, 255, '\n'}) },
			} {
				if err := write(); err != nil {
					t.Fatal(err)
				}
			}
			if err := w.Close(usage); err != nil {
				t.Fatal(err)
			}
			z := zipReader(dst.Bytes())
			if z.File[len(z.File)-1].Name != "manifest.json" {
				t.Fatal("manifest was not finalized last")
			}
			f, _ := z.Open("transcript.log")
			encoded, err := io.ReadAll(f)
			f.Close()
			if err != nil || !bytes.Equal(encoded, transcript) {
				t.Fatalf("wire format changed: %x, %v", encoded, err)
			}
			r, err := NewBundleReader(z)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			got := r.Header()
			if got.Version != header.Version || got.TreeStart != header.TreeStart || got.Model != h.Model || got.Inference != h.Inference || got.Environment != h.Environment || got.Seed != h.Seed || got.Temperature != h.Temperature || got.Usage != usage || !slices.Equal(got.InitialContext, h.InitialContext) {
				t.Fatalf("metadata changed: %+v", got)
			}
			if (got.InitialTree == nil) != (tc.tree == nil) || got.InitialTree != nil && got.InitialTree.Root() != header.TreeStart {
				t.Fatal("initial tree changed")
			}
			if got.InitialCall.Tool != header.InitialCall.Tool || got.InitialCall.Ref != header.InitialCall.Ref || !bytes.Equal(got.InitialCall.Args, header.InitialCall.Args) {
				t.Fatal("pending call changed")
			}
			for i := range 4 {
				model, tool, err := r.Next()
				if err != nil || (model != nil) != (i%2 == 0) || (tool != nil) != (i%2 == 1) {
					t.Fatalf("record %d: %v, %v, %v", i, model, tool, err)
				}
				if i < 2 && len(model)+len(tool) != 0 || i == 2 && !slices.Equal(model, []uint32{0x01020304, 0xfffffffe}) || i == 3 && !bytes.Equal(tool, []byte{0, 255, '\n'}) {
					t.Fatalf("record %d changed", i)
				}
			}
			if _, _, err := r.Next(); err != io.EOF {
				t.Fatal(err)
			}
		})
	}
}

func TestBundleAdmissionAndStreaming(t *testing.T) {
	h, manifest, transcript := codecFixture(t)
	z := zipReader(bundleZIP(t, bundleMember{"tree/input/file", []byte("not loaded")}, bundleMember{"transcript.log", transcript}, bundleMember{"manifest.json", manifest}))
	z.File[0].Method, z.File[1].Method = 65535, 65535 // Neither member can be decompressed.
	got, err := ReadBundleHeader(z)
	if err != nil || got.Usage != h.Usage || got.InitialTree != nil || got.InitialContext != nil {
		t.Fatalf("header admission read: %+v, %v", got, err)
	}
	z = zipReader(bundleZIP(t, bundleMember{"manifest.json", manifest}, bundleMember{"transcript.log", transcript}))
	tracked := trackTranscript(z)
	r, err := NewBundleReader(z)
	if err != nil || tracked.read != 20 {
		t.Fatalf("constructor read past initial context: %d bytes, %v", tracked.read, err)
	}
	if _, _, err := r.Next(); err != nil || tracked.read != 29 {
		t.Fatalf("Next read past one record: %d bytes, %v", tracked.read, err)
	}
	if err := r.Close(); err != nil || tracked.read != 29 || tracked.closes != 1 {
		t.Fatalf("Close drained transcript: %+v, %v", tracked, err)
	}

	for _, tc := range []struct {
		name, path string
		usage      Usage
		want       string
	}{
		{"implicit directories", strings.Repeat("/a", 100) + "/file", Usage{InitialTreeBytes: 1, Usage: wasm.Usage{TreeEntries: 100, TreeBytes: 1}}, "initial tree entries"},
		{"file bytes", "/file", Usage{Usage: wasm.Usage{TreeEntries: 1}}, "initial tree bytes"},
		{"peak below initial bytes", "/file", Usage{InitialTreeBytes: 1, Usage: wasm.Usage{TreeEntries: 1}}, "tree_bytes is less than initial_tree_bytes"},
		{"reserved mount", "/env/file", Usage{}, "/env is reserved"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			z := zipReader(bundleZIP(t, bundleMember{"tree" + tc.path, []byte("x")}, bundleMember{"transcript.log", transcript}, bundleMember{"manifest.json", changeManifest(t, manifest, "usage", tc.usage)}))
			z.File[0].Method, z.File[1].Method = 65535, 65535
			if r, err := NewBundleReader(z); r != nil || err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("contents opened before preflight: %v", err)
			}
		})
	}
	for _, tag := range []byte{0, 1, 2} {
		t.Run(fmt.Sprint("oversized payload ", tag), func(t *testing.T) {
			data := binary.LittleEndian.AppendUint64(nil, 1<<40)
			if tag != 0 {
				data = append(binary.LittleEndian.AppendUint64(nil, 0), append([]byte{tag}, data...)...)
			}
			z := zipReader(bundleZIP(t, bundleMember{"manifest.json", manifest}, bundleMember{"transcript.log", data}))
			z.File[1].UncompressedSize64 = 1 << 50 // Must bound the count even if ZIP metadata lies.
			tracked := trackTranscript(z)
			r, err := NewBundleReader(z)
			if r != nil {
				defer r.Close()
				_, _, err = r.Next()
			}
			wantRead := 8
			if tag != 0 {
				wantRead = 17
			}
			if err == nil || !strings.Contains(err.Error(), "declared usage") || tracked.read != wantRead {
				t.Fatalf("payload count not bounded before allocation: read %d, %v", tracked.read, err)
			}
		})
	}
}

// Consume to checked EOF: a valid record prefix is not a valid finalized bundle.
func checkCodec(z *zip.Reader) error {
	r, err := NewBundleReader(z)
	if err != nil {
		return err
	}
	defer r.Close()
	for {
		_, _, err := r.Next()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

func TestBundleRejectsTampering(t *testing.T) {
	h, manifest, transcript := codecFixture(t)
	m, s := bundleMember{"manifest.json", manifest}, bundleMember{"transcript.log", transcript}
	for name, members := range map[string][]bundleMember{
		"missing manifest": {s}, "missing transcript": {m}, "duplicate": {m, s, s},
		"unexpected member":   {m, s, {"extra", nil}},
		"path traversal":      {m, s, {"tree/../file", nil}},
		"duplicate tree file": {m, s, {"tree/a", nil}, {"tree/a", nil}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := checkCodec(zipReader(bundleZIP(t, members...))); err == nil {
				t.Fatal("accepted malformed archive")
			}
		})
	}
	for name, data := range map[string][]byte{
		"null": []byte("null"), "trailing JSON": append(bytes.Clone(manifest), []byte(" {}")...),
		"version":          changeManifest(t, manifest, "version", FormatVersion+1),
		"missing usage":    changeManifest(t, manifest, "usage", nil),
		"incomplete usage": changeManifest(t, manifest, "usage", map[string]int{"model_turns": 2}),
		"unknown field":    changeManifest(t, manifest, "extra", true),
	} {
		t.Run(name, func(t *testing.T) {
			if err := checkCodec(zipReader(bundleZIP(t, bundleMember{"manifest.json", data}, s))); err == nil {
				t.Fatal("accepted malformed manifest")
			}
		})
	}
	for name, change := range map[string]func(*Usage){
		"model undercount": func(u *Usage) { u.ModelTurns-- }, "model overcount": func(u *Usage) { u.ModelTurns++ },
		"harness undercount": func(u *Usage) { u.HarnessTurns-- }, "harness overcount": func(u *Usage) { u.HarnessTurns++ },
		"tokens undercount": func(u *Usage) { u.OutputTokens-- }, "tokens overcount": func(u *Usage) { u.OutputTokens++ },
		"bytes undercount": func(u *Usage) { u.ToolOutputBytes-- }, "bytes overcount": func(u *Usage) { u.ToolOutputBytes++ },
	} {
		t.Run(name, func(t *testing.T) {
			u := h.Usage
			change(&u)
			if err := checkCodec(zipReader(bundleZIP(t, bundleMember{"manifest.json", changeManifest(t, manifest, "usage", u)}, s))); err == nil {
				t.Fatal("accepted false transcript usage")
			}
		})
	}
	for n := range len(transcript) {
		if err := checkCodec(zipReader(bundleZIP(t, m, bundleMember{"transcript.log", transcript[:n]}))); err == nil {
			t.Fatalf("accepted transcript truncated to %d bytes", n)
		}
	}
	badTag := append(bytes.Clone(transcript[:20]), 3)
	if err := checkCodec(zipReader(bundleZIP(t, m, bundleMember{"transcript.log", badTag}))); err == nil {
		t.Fatal("accepted unknown record tag")
	}
	for _, member := range []string{"manifest.json", "transcript.log", "tree/a"} {
		t.Run("checksum/"+member, func(t *testing.T) {
			// A zero-byte file still changes the tree hash, so patch the expected root.
			initial, _ := (tree.Tree{}).Put("/a", tree.File{})
			u := h.Usage
			u.TreeEntries = 1
			manifest := changeManifest(t, changeManifest(t, manifest, "usage", u), "tree_start", initial.Root())
			z := zipReader(bundleZIP(t, bundleMember{"manifest.json", manifest}, s, bundleMember{"tree/a", nil}))
			for _, f := range z.File {
				if f.Name == member {
					f.CRC32 ^= 1
				}
			}
			if err := checkCodec(z); !errors.Is(err, zip.ErrChecksum) {
				t.Fatalf("checksum was not checked: %v", err)
			}
		})
	}
	for _, mode := range []fs.FileMode{fs.ModeSymlink | 0600, fs.ModeDir | 0700} {
		z := zipReader(bundleZIP(t, m, s, bundleMember{"tree/a", nil}))
		z.File[2].SetMode(mode)
		if err := checkCodec(z); err == nil {
			t.Fatal("accepted a non-file tree member")
		}
	}
	// Repacking altered tree bytes repairs the ZIP checksum but must not repair
	// the content-addressed initial state. Shared parents count only once.
	var initial tree.Tree
	files := []bundleMember{{"tree/a/b/c", []byte("x")}, {"tree/a/d", []byte("y")}, {"tree/z", []byte("z")}}
	for _, file := range files {
		initial, _ = initial.Put(strings.TrimPrefix(file.name, "tree"), tree.File{Content: file.data})
	}
	u := h.Usage
	u.TreeEntries, u.InitialTreeBytes = 5, 3
	u.TreeBytes = u.InitialTreeBytes
	manifest = changeManifest(t, changeManifest(t, manifest, "usage", u), "tree_start", initial.Root())
	withTree := append([]bundleMember{{"manifest.json", manifest}, s}, files...)
	if err := checkCodec(zipReader(bundleZIP(t, withTree...))); err != nil {
		t.Fatalf("exact implicit-entry count rejected: %v", err)
	}
	withTree[2].data = []byte("q")
	if err := checkCodec(zipReader(bundleZIP(t, withTree...))); err == nil || !strings.Contains(err.Error(), "tree_start") {
		t.Fatalf("accepted repacked tree contents: %v", err)
	}
}

type bundleSink struct {
	bytes.Buffer
	err error
}

func (s *bundleSink) Write(p []byte) (int, error) {
	if s.err != nil {
		return 0, s.err
	}
	return s.Buffer.Write(p)
}

func TestBundleWriterFailure(t *testing.T) {
	for _, phase := range []string{"sink", "abort", "false usage", "false tree bytes"} {
		t.Run(phase, func(t *testing.T) {
			var dst bundleSink
			w, err := NewBundleWriter(&dst, BundleHeader{})
			if err != nil {
				t.Fatal(err)
			}
			failure := errors.New("failed recording")
			switch phase {
			case "sink":
				// Records are buffered; the sink's failure surfaces when Close finalizes.
				dst.err = failure
				if err := w.Close(Usage{}); !errors.Is(err, failure) {
					t.Fatal(err)
				}
				dst.err = nil
			case "abort":
				if err := w.abort(failure); !errors.Is(err, failure) {
					t.Fatal(err)
				}
			case "false usage":
				failure = w.Close(Usage{ModelTurns: 1})
				if failure == nil {
					t.Fatal("accepted false final measurements")
				}
			case "false tree bytes":
				failure = w.Close(Usage{InitialTreeBytes: 1})
				if failure == nil {
					t.Fatal("accepted a tree peak below the initial state")
				}
			}
			if err := w.Close(Usage{}); !errors.Is(err, failure) {
				t.Fatalf("lost original failure: %v", err)
			}
			if z, err := zip.NewReader(bytes.NewReader(dst.Bytes()), int64(dst.Len())); err == nil {
				if _, err := ReadBundleHeader(z); err == nil {
					t.Fatal("failed writer finalized a manifest")
				}
			}
		})
	}
}

func FuzzDecodeBundle(f *testing.F) {
	_, manifest, transcript := codecFixture(f)
	f.Add(bundleZIP(f, bundleMember{"manifest.json", manifest}, bundleMember{"transcript.log", transcript}))
	f.Add([]byte("PK"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			return
		}
		z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
		if err != nil || len(z.File) > 64 {
			return
		}
		var total uint64
		for _, member := range z.File {
			if member.UncompressedSize64 > 1<<20-total {
				return
			}
			total += member.UncompressedSize64
		}
		h, err := ReadBundleHeader(z)
		if err != nil || h.Usage.TreeEntries > 1024 || h.Usage.TreeBytes > 1<<20 || h.Usage.ContextTokens > 1<<18 || h.Usage.ModelTurns > 128 || h.Usage.HarnessTurns > 128 {
			return
		}
		r, err := NewBundleReader(z)
		if err != nil {
			return
		}
		defer r.Close()
		for {
			model, tool, err := r.Next()
			if err != nil {
				if _, _, again := r.Next(); again != err {
					t.Fatal("reader did not retain terminal result")
				}
				return
			}
			if (model == nil) == (tool == nil) {
				t.Fatal("record must have exactly one payload")
			}
		}
	})
}
