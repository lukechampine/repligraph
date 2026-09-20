package repligraph

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"
	"testing"

	"lukechampine.com/repligraph/harness/tools"
	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

// Rewrite the same streamed records with a different resource claim. The codec
// checks structural counts; execution measurements are checked during replay.
func alterRecordedUsage(t *testing.T, data []byte, usage Usage) []byte {
	t.Helper()
	reader := mustBundleReader(t, data)
	var archive bytes.Buffer
	writer, err := NewBundleWriter(&archive, reader.Header())
	if err != nil {
		t.Fatal(err)
	}
	for {
		output, result, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if output != nil {
			err = writer.WriteModelOutput(output)
		} else {
			err = writer.WriteToolOutput(result)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(usage); err != nil {
		t.Fatal(err)
	}
	return archive.Bytes()
}

func assertNoManifest(t *testing.T, data []byte) {
	t.Helper()
	z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err == nil {
		if _, err := ReadBundleHeader(z); err == nil {
			t.Fatal("aborted recording emitted a final manifest")
		}
	}
}

func TestRecordingUsage(t *testing.T) {
	m, output := qwenFixture(t)
	initial, err := (tree.Tree{}).Put("/input/note", tree.File{Content: []byte("seed")})
	if err != nil {
		t.Fatal(err)
	}
	var runs int
	var budgets []wasm.Limits
	env := applyFunc(func(tr tree.Tree, limits wasm.Limits, tool string, args json.RawMessage) (tree.Tree, []byte, wasm.Usage, error) {
		if tool != "run" {
			return tools.Apply(tr, tree.Tree{}, limits, tool, args)
		}
		budgets = append(budgets, limits)
		n := runs % 2
		runs++
		usage := []wasm.Usage{
			{Fuel: 3, MemoryBytes: 65536, WriteBytes: 5, StdoutBytes: 2, TreeEntries: 6, TreeBytes: 9},
			{Fuel: 5, MemoryBytes: 131072, WriteBytes: 7, StderrBytes: 1, TreeEntries: 3, TreeBytes: 11},
		}[n]
		if limits.Fuel < usage.Fuel || limits.MemoryBytes < usage.MemoryBytes || limits.WriteBytes < usage.WriteBytes || limits.StdoutBytes < usage.StdoutBytes || limits.StderrBytes < usage.StderrBytes || limits.MaxTreeEntries < usage.TreeEntries || limits.MaxTreeBytes < usage.TreeBytes {
			return tr, nil, wasm.Usage{}, wasm.ErrResourceLimit
		}
		if n == 0 {
			return tr, []byte("<result ok>\nok\n</result>"), usage, nil
		}
		return tr, []byte("<result error>\ntrap: unreachable\n!\n</result>"), usage, nil
	})
	limits := DefaultLimits()
	limits.Fuel, limits.MemoryBytes, limits.WriteBytes = 20, 1<<20, 30
	limits.StdoutBytes, limits.StderrBytes, limits.MaxTreeEntries = 40, 50, 100
	limits.MaxTreeBytes = 100
	var archive bytes.Buffer
	recorder, err := NewRecorder(&archive, m, testHeader(initial), initial, limits)
	if err != nil {
		t.Fatal(err)
	}
	var contexts, outputs [][]uint32
	var seeds []uint64
	var inputTokens, outputTokens, toolBytes, contextPeak uint64
	for _, call := range []string{
		`{"name":"write_file","arguments":{"path":"/artifact/a","content":"hi"}}`,
		`{"name":"rename_file","arguments":{"source":"/artifact/a","destination":"/artifact/b"}}`,
		`{"name":"delete_file","arguments":{"path":"/artifact/b"}}`,
		`{"name":"run","arguments":{"module":"/env/program.wasm"}}`,
		`{"name":"run","arguments":{"module":"/env/program.wasm"}}`,
		`{"name":"finish"}`,
	} {
		input, generated := recorder.Context(), output(call)
		contexts, outputs, seeds = append(contexts, input), append(outputs, generated), append(seeds, recorder.Seed())
		inputTokens += uint64(len(input))
		outputTokens += uint64(len(generated))
		contextPeak = max(contextPeak, uint64(len(input)+len(generated)))
		if err := recorder.ApplyModelTurn(generated); err != nil {
			t.Fatal(err)
		}
		if call := recorder.Pending(); call != nil {
			next, result, resources, err := env(recorder.Tree(), recorder.HarnessLimits(), call.Tool, call.Args)
			if err != nil {
				t.Fatal(err)
			}
			if err := recorder.ApplyHarnessTurn(next, result, resources); err != nil {
				t.Fatal(err)
			}
			toolBytes += uint64(len(result))
			contextPeak = max(contextPeak, uint64(len(recorder.Context())))
			if bytes.Contains(result, []byte(" of ")) {
				t.Fatal("local resource ceiling leaked into result")
			}
		}
	}
	want := Usage{
		ModelTurns: 6, HarnessTurns: 5, InputTokens: inputTokens, OutputTokens: outputTokens,
		ContextTokens: contextPeak, ToolOutputBytes: toolBytes, InitialTreeBytes: 4,
		Usage: wasm.Usage{Fuel: 8, MemoryBytes: 131072, WriteBytes: 14, StdoutBytes: 2, StderrBytes: 1, TreeEntries: 6, TreeBytes: 11},
	}
	if recorder.Usage() != want || recorder.Tree().Root() != initial.Root() || !recorder.Finished() {
		t.Fatalf("recorded state/usage: %+v, want %+v", recorder.Usage(), want)
	}
	if len(budgets) != 2 || budgets[0].Fuel != 20 || budgets[0].WriteBytes != 28 || budgets[1].Fuel != 17 || budgets[1].WriteBytes != 23 || budgets[1].StdoutBytes != 38 {
		t.Fatalf("recording budgets did not decrease: %+v", budgets)
	}
	if err := recorder.Close(); err != nil {
		t.Fatal(err)
	}
	size := archive.Len()
	if err := recorder.Close(); err != nil || archive.Len() != size {
		t.Fatal("repeated Close changed recording")
	}
	header, err := ReadBundleHeader(zipReader(archive.Bytes()))
	if err != nil || header.Usage != want || header.InitialTree != nil || header.InitialContext != nil {
		t.Fatalf("final header: %+v, %v", header, err)
	}
	newEngine := func() bundleInferenceEngine {
		calls := 0
		return func(_ context.Context, input, stops []uint32, seed uint64, maxTokens int) ([]uint32, bool, error) {
			i := calls
			calls++
			if i >= len(outputs) || !slices.Equal(input, contexts[i]) || !slices.Equal(stops, m.StopIDs()) || seed != seeds[i] || maxTokens != len(outputs[i]) {
				return nil, false, fmt.Errorf("unexpected inference input %d", i)
			}
			return slices.Clone(outputs[i]), true, nil
		}
	}
	runs, budgets = 0, nil
	replay, err := NewReplay(m, mustBundleReader(t, archive.Bytes()), initial)
	if err != nil {
		t.Fatal(err)
	}
	eng := newEngine()
	for {
		mt, ht, err := replay.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if err := applyReplayPair(t.Context(), replay, m, eng, env, mt, ht); err != nil {
			t.Fatal(err)
		}
	}
	if replay.Usage() != want || !replay.Checkpoint().Finished || replay.Checkpoint().Tree.Root() != initial.Root() {
		t.Fatal("replay lost measured usage or final state")
	}
	first := wasm.Limits{Fuel: 8, MemoryBytes: 131072, WriteBytes: 12, StdoutBytes: 2, StderrBytes: 1, MaxTreeEntries: 6, MaxTreeBytes: 11}
	second := wasm.Limits{Fuel: 5, MemoryBytes: 131072, WriteBytes: 7, StderrBytes: 1, MaxTreeEntries: 6, MaxTreeBytes: 11}
	if !reflect.DeepEqual(budgets, []wasm.Limits{first, second}) {
		t.Fatalf("replay did not use remaining measured resources: %+v", budgets)
	}
	for _, field := range []string{"fuel", "memory", "writes", "entries", "tree bytes", "overstated fuel", "overstated tree bytes"} {
		t.Run(field, func(t *testing.T) {
			claim := want
			switch field {
			case "fuel":
				claim.Fuel--
			case "memory":
				claim.MemoryBytes--
			case "writes":
				claim.WriteBytes--
			case "entries":
				claim.TreeEntries--
			case "tree bytes":
				claim.TreeBytes--
			case "overstated fuel":
				claim.Fuel++
			case "overstated tree bytes":
				claim.TreeBytes++
			}
			tampered := alterRecordedUsage(t, archive.Bytes(), claim)
			runs = 0
			p, err := NewReplay(m, mustBundleReader(t, tampered), initial)
			if err != nil {
				t.Fatal(err)
			}
			var before Checkpoint
			var usage Usage
			eng := newEngine()
			for {
				before, usage = p.Checkpoint(), p.Usage()
				var mt *ModelTurn
				var ht *HarnessTurn
				mt, ht, err = p.Next()
				if err != nil {
					break
				}
				for _, pair := range []struct {
					m *ModelTurn
					h *HarnessTurn
				}{{mt, nil}, {nil, ht}} {
					if pair.m == nil && pair.h == nil {
						continue
					}
					before, usage = p.Checkpoint(), p.Usage()
					if err = applyReplayPair(t.Context(), p, m, eng, env, pair.m, pair.h); err != nil {
						break
					}
				}
				if err != nil {
					break
				}
			}
			if strings.HasPrefix(field, "overstated") {
				if err == nil || !strings.Contains(err.Error(), "usage does not match") {
					t.Fatalf("overstatement accepted: %v", err)
				}
			} else if !errors.Is(err, ErrResourceLimit) {
				t.Fatalf("understatement accepted: %v", err)
			}
			checkCheckpoint(t, p.Checkpoint(), before)
			_, _, again := p.Next()
			if strings.HasPrefix(field, "overstated") && again != err || again == nil || p.Usage() != usage {
				t.Fatal("failed replay advanced beyond its last successful turn")
			}
		})
	}
}

func TestRecordingInitialPrefix(t *testing.T) {
	m, _ := qwenFixture(t)
	populated, err := (tree.Tree{}).Put("/input/a", tree.File{Content: []byte("initial")})
	if err != nil {
		t.Fatal(err)
	}
	for _, initial := range []tree.Tree{{}, populated} {
		limits := Limits{MaxTreeEntries: uint64(initial.Stats().Entries), MaxTreeBytes: uint64(initial.Stats().Bytes)}
		if limits.MaxTreeEntries > 0 {
			var rejected bytes.Buffer
			insufficient := limits
			insufficient.MaxTreeEntries--
			if _, err := NewRecorder(&rejected, m, testHeader(initial), initial, insufficient); !errors.Is(err, ErrResourceLimit) || rejected.Len() != 0 {
				t.Fatalf("initial tree above ceiling accepted: %v", err)
			}
		}
		var archive bytes.Buffer
		recorder, err := NewRecorder(&archive, m, testHeader(initial), initial, limits)
		if err != nil {
			t.Fatal(err)
		}
		if err := recorder.Close(); err != nil {
			t.Fatal(err)
		}
		replay, err := NewReplay(m, mustBundleReader(t, archive.Bytes()), initial)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := replay.Next(); err != io.EOF {
			t.Fatal(err)
		}
		want := Usage{ContextTokens: 1, InitialTreeBytes: uint64(initial.Stats().Bytes), Usage: wasm.Usage{TreeEntries: uint64(initial.Stats().Entries), TreeBytes: uint64(initial.Stats().Bytes)}}
		if replay.Usage() != want || recorder.Usage() != want || replay.Checkpoint().Tree.Root() != initial.Root() {
			t.Fatalf("empty prefix usage: %+v, want %+v", replay.Usage(), want)
		}
	}
}

func TestRecordingAbort(t *testing.T) {
	m, output := qwenFixture(t)
	initial := tree.Tree{}
	h := testHeader(initial)
	generated := output(`{"name":"write_file","arguments":{"path":"/artifact/a","content":"hi"}}`)
	for _, resource := range []string{"generation", "inference", "turns", "entries", "tree bytes", "supplied tree bytes", "writes", "caller"} {
		t.Run(resource, func(t *testing.T) {
			limits := DefaultLimits()
			switch resource {
			case "generation":
				limits.MaxTokens = uint64(len(generated) - 1)
			case "inference":
				limits.MaxInferenceTokens = uint64(len(h.InitialContext) + len(generated) - 1)
			case "turns":
				limits.MaxTurns = 1
			case "entries":
				limits.MaxTreeEntries = 1
			case "tree bytes", "supplied tree bytes":
				limits.MaxTreeBytes = 1
			case "writes":
				limits.WriteBytes = 1
			}
			var archive bytes.Buffer
			recorder, err := NewRecorder(&archive, m, h, initial, limits)
			if err != nil {
				t.Fatal(err)
			}
			before, usage := recorder.Checkpoint(), recorder.Usage()
			err = recorder.ApplyModelTurn(generated)
			wantErr := ErrResourceLimit
			if err == nil {
				before, usage = recorder.Checkpoint(), recorder.Usage()
				if resource == "caller" {
					wantErr = context.DeadlineExceeded
					err = recorder.Abort(wantErr)
				} else {
					env := applyWith(tree.Tree{})
					call := recorder.Pending()
					budget := recorder.HarnessLimits()
					if resource == "supplied tree bytes" {
						budget.MaxTreeBytes++ // The recorder also checks externally supplied results.
					}
					next, result, resources, applyErr := env(recorder.Tree(), budget, call.Tool, call.Args)
					if applyErr != nil {
						if result != nil {
							t.Fatal("resource abort produced a tool result")
						}
						err = recorder.Abort(applyErr)
					} else {
						err = recorder.ApplyHarnessTurn(next, result, resources)
					}
				}
			}
			if !errors.Is(err, wantErr) {
				t.Fatalf("%s did not abort: %v", resource, err)
			}
			if !reflect.DeepEqual(recorder.Checkpoint(), before) || recorder.Usage() != usage {
				t.Fatal("aborted step changed the last successful checkpoint or usage")
			}
			if err := recorder.ApplyModelTurn(generated); !errors.Is(err, wantErr) {
				t.Fatalf("recording continued after abort: %v", err)
			}
			if err := recorder.ApplyHarnessTurn(initial, nil, wasm.Usage{}); !errors.Is(err, wantErr) {
				t.Fatalf("harness continued after abort: %v", err)
			}
			if err := recorder.Close(); !errors.Is(err, wantErr) {
				t.Fatalf("Close hid abort: %v", err)
			}
			assertNoManifest(t, archive.Bytes())
		})
	}
}

type recorderSink struct {
	bytes.Buffer
	fail   error
	closed bool
}

func (w *recorderSink) Write(p []byte) (int, error) {
	if w.fail != nil {
		return 0, w.fail
	}
	return w.Buffer.Write(p)
}
func (w *recorderSink) Close() error { w.closed = true; return nil }

func TestRecordingFailures(t *testing.T) {
	m, output := qwenFixture(t)
	initial := tree.Tree{}
	t.Run("backend retry", func(t *testing.T) {
		failure := errors.New("backend unavailable")
		failNext := true
		env := applyFunc(func(tr tree.Tree, _ wasm.Limits, _ string, _ json.RawMessage) (tree.Tree, []byte, wasm.Usage, error) {
			if failNext {
				failNext = false
				return tr, nil, wasm.Usage{}, failure
			}
			next, err := tr.Put("/artifact/a", tree.File{Content: []byte("hi")})
			if err != nil {
				t.Fatal(err)
			}
			return next, []byte("<result ok>\nok\n</result>"), wasm.Usage{Fuel: 3, WriteBytes: 2, StdoutBytes: 2, TreeEntries: 2, TreeBytes: 2}, nil
		})
		var archive bytes.Buffer
		recorder, err := NewRecorder(&archive, m, testHeader(initial), initial)
		if err != nil {
			t.Fatal(err)
		}
		if err := recorder.ApplyModelTurn(output(`{"name":"run","arguments":{"module":"/env/program.wasm"}}`)); err != nil {
			t.Fatal(err)
		}
		before, usage, size := recorder.Checkpoint(), recorder.Usage(), archive.Len()
		call := recorder.Pending()
		if _, result, _, err := env(recorder.Tree(), recorder.HarnessLimits(), call.Tool, call.Args); !errors.Is(err, failure) || result != nil {
			t.Fatalf("backend failure produced result: %s, %v", result, err)
		}
		if !reflect.DeepEqual(recorder.Checkpoint(), before) || recorder.Usage() != usage || archive.Len() != size {
			t.Fatal("backend failure committed state, bytes, or usage")
		}
		next, result, resources, err := env(recorder.Tree(), recorder.HarnessLimits(), call.Tool, call.Args)
		if err != nil {
			t.Fatal(err)
		}
		// Execution alone does not advance the recording. Committing its result
		// must not execute the call again.
		if !reflect.DeepEqual(recorder.Checkpoint(), before) || recorder.Usage() != usage || archive.Len() != size {
			t.Fatal("external execution advanced recording")
		}
		failNext = true
		if err := recorder.ApplyHarnessTurn(next, result, resources); err != nil {
			t.Fatal(err)
		}
		failNext = false
		if err := recorder.ApplyModelTurn(output(`{"name":"finish"}`)); err != nil {
			t.Fatal(err)
		}
		if err := recorder.Close(); err != nil {
			t.Fatal(err)
		}
		replay, err := NewReplay(m, mustBundleReader(t, archive.Bytes()), initial)
		if err != nil {
			t.Fatal(err)
		}
		for {
			mt, ht, err := replay.Next()
			if err == io.EOF {
				break
			} else if err != nil {
				t.Fatal(err)
			}
			if err := applyReplayPair(t.Context(), replay, m, nil, env, mt, ht); err != nil {
				t.Fatal(err)
			}
		}
		if replay.Checkpoint().Tree.Root() != recorder.Tree().Root() || replay.Usage() != recorder.Usage() {
			t.Fatal("successful retry did not replay")
		}
	})
	t.Run("sink failure", func(t *testing.T) {
		var sink recorderSink
		recorder, err := NewRecorder(&sink, m, testHeader(initial), initial)
		if err != nil {
			t.Fatal(err)
		}
		generated := output(`{"name":"write_file","arguments":{"path":"/artifact/a","content":"hi"}}`)
		if err := recorder.ApplyModelTurn(generated); err != nil {
			t.Fatal(err)
		}
		env := applyWith(tree.Tree{})
		call := recorder.Pending()
		next, result, resources, err := env(recorder.Tree(), recorder.HarnessLimits(), call.Tool, call.Args)
		if err != nil {
			t.Fatal(err)
		}
		// Records are buffered; the sink's failure surfaces when Close finalizes.
		failure := errors.New("sink unavailable")
		sink.fail = failure
		if err := recorder.ApplyHarnessTurn(next, result, resources); err != nil {
			t.Fatal(err)
		}
		if err := recorder.Close(); !errors.Is(err, failure) || sink.closed {
			t.Fatal("Close hid sink error or closed caller destination")
		}
		sink.fail = nil
		if err := recorder.Close(); !errors.Is(err, failure) {
			t.Fatal("sink error was not sticky")
		}
		assertNoManifest(t, sink.Bytes())
	})
}
