package repligraph

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"lukechampine.com/repligraph/blake3"
	"lukechampine.com/repligraph/harness/tools"
	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
	"lukechampine.com/repligraph/harness/wasm/wasmtime"
	"lukechampine.com/repligraph/inference"
	"lukechampine.com/repligraph/inference/model"
	"lukechampine.com/repligraph/inference/model/mistral"
	"lukechampine.com/repligraph/inference/model/qwen"
	mtok "lukechampine.com/repligraph/inference/token/mistral"
	qtok "lukechampine.com/repligraph/inference/token/qwen"
)

type bundleInferenceEngine func(context.Context, []uint32, []uint32, uint64, int) ([]uint32, bool, error)

func (f bundleInferenceEngine) Generate(ctx context.Context, input, stops []uint32, seed uint64, n int) ([]uint32, bool, error) {
	return f(ctx, input, stops, seed, n)
}

// An applyFunc executes one tool call, as tools.Apply does for a fixed environment.
type applyFunc func(t tree.Tree, limits wasm.Limits, tool string, args json.RawMessage) (tree.Tree, []byte, wasm.Usage, error)

func applyWith(env tree.Tree) applyFunc {
	return func(t tree.Tree, limits wasm.Limits, tool string, args json.RawMessage) (tree.Tree, []byte, wasm.Usage, error) {
		return tools.Apply(t, env, limits, tool, args)
	}
}

func qwenFixture(t *testing.T) (model.Model, func(string) []uint32) {
	t.Helper()
	m, err := qwen.New()
	if err != nil {
		t.Fatal(err)
	}
	c, err := qtok.New()
	if err != nil {
		t.Fatal(err)
	}
	return m, func(body string) []uint32 { return append(c.Encode(body), 151658) }
}

func testHeader(initial tree.Tree) BundleHeader {
	return BundleHeader{
		Version: FormatVersion, Model: "fixture-qwen", Inference: "fixture-engine", Environment: "fixture-environment",
		TreeStart: initial.Root(), InitialTree: &initial, InitialContext: []uint32{1},
	}
}

func mustBundleReader(t *testing.T, data []byte) *BundleReader {
	t.Helper()
	r, err := NewBundleReader(zipReader(data))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}

func checkCheckpoint(t *testing.T, got, want Checkpoint) {
	t.Helper()
	if !slices.Equal(got.Context, want.Context) || got.Tree.Root() != want.Tree.Root() ||
		!reflect.DeepEqual(got.Pending, want.Pending) || got.Finished != want.Finished ||
		got.ModelTurns != want.ModelTurns || got.HarnessTurns != want.HarnessTurns {
		t.Fatalf("checkpoint mismatch:\ngot  %+v\nwant %+v", got, want)
	}
}

// A small scripted conversation supplies inference inputs and outputs explicitly.
// Archives and resource measurements are produced by the real recorder.
type conversationStep struct {
	input, output []uint32
	result        []byte
}

type conversationFixture struct {
	model  model.Model
	header BundleHeader
	env    applyFunc
	steps  []conversationStep
	want   tree.Tree
}

func testConversation(t *testing.T, name string) conversationFixture {
	t.Helper()
	m, encode := qwenFixture(t)
	ref := ""
	if name == "mistral" {
		var err error
		m, err = mistral.New()
		if err != nil {
			t.Fatal(err)
		}
		c, err := mtok.New()
		if err != nil {
			t.Fatal(err)
		}
		ref = "AbC123xYz"
		encode = func(body string) []uint32 {
			return slices.Concat([]uint32{mtok.ToolCalls}, c.Encode("["+body+"]"), []uint32{mtok.EOS})
		}
	}
	initial, err := (tree.Tree{}).Put("/source/note", tree.File{Content: []byte("keep")})
	if err != nil {
		t.Fatal(err)
	}
	files, err := (tree.Tree{}).Put("/guide", tree.File{Content: []byte("environment data")})
	if err != nil {
		t.Fatal(err)
	}
	want, err := initial.Put("/artifact/message", tree.File{Content: []byte("héllo\n")})
	if err != nil {
		t.Fatal(err)
	}
	f := conversationFixture{model: m, header: testHeader(initial), env: applyWith(files), want: want}
	f.header.Model = "fixture-" + name
	f.header.Seed, f.header.Temperature = 11, 0.25
	f.header.InitialContext = m.FramePrompt([]byte("Read the guide, write héllo, and read it back."))
	context := slices.Clone(f.header.InitialContext)
	for _, step := range []struct{ tool, args, result string }{
		{"", "", "<bad_call>\nmalformed tool call\n</bad_call>"},
		{"read_file", `{"path":"/env/guide"}`, "<result ok>\nenvironment data\n</result>"},
		{"write_file", `{"path":"/env/guide","content":"tampered"}`, "<result error>\nerror[read_only]: read-only path: /env/guide\n</result>"},
		{"write_file", `{ "path" : "/artifact/message", "content" : "héllo\n" }`, "<result ok>\nwrote 7 bytes: /artifact/message\n</result>"},
		{"read_file", `{"path":"/artifact/message"}`, "<result ok>\nhéllo\n\n</result>"},
		{"finish", `{}`, ""},
	} {
		body := "not JSON"
		if step.tool != "" {
			body = fmt.Sprintf(`{"name":%q,"arguments":%s`, step.tool, step.args)
			if ref != "" {
				body += fmt.Sprintf(`,"id":%q`, ref)
			}
			body += "}"
		}
		output := encode(body)
		f.steps = append(f.steps, conversationStep{input: slices.Clone(context), output: output})
		context = append(context, output[:len(output)-1]...)
		if step.tool == "" {
			context = append(context, m.FramePrompt([]byte(step.result))...)
		} else if step.tool != "finish" {
			f.steps = append(f.steps, conversationStep{result: []byte(step.result)})
			context = append(context, m.FrameResult(ref, []byte(step.result))...)
		}
	}
	return f
}

// turnSeeds independently derives the first n per-turn sampling seeds of a header seed.
func turnSeeds(header uint64, n int) []uint64 {
	var key [32]byte
	binary.LittleEndian.PutUint64(key[:], header)
	stream := make([]byte, 8*n)
	blake3.XOF(key, blake3.DomainSeed).Read(stream) // XOF reads never fail
	seeds := make([]uint64, n)
	for i := range seeds {
		seeds[i] = binary.LittleEndian.Uint64(stream[8*i:]) &^ (1 << 63)
	}
	return seeds
}

func conversationEngine(m model.Model, header uint64, steps []conversationStep) bundleInferenceEngine {
	index, turn := 0, 0
	seeds := turnSeeds(header, len(steps))
	return func(_ context.Context, input, stops []uint32, seed uint64, n int) ([]uint32, bool, error) {
		for index < len(steps) && steps[index].output == nil {
			index++
		}
		if index == len(steps) {
			return nil, false, errors.New("unexpected extra inference")
		}
		s := steps[index]
		index++
		if !slices.Equal(input, s.input) || !slices.Equal(stops, m.StopIDs()) || seed != seeds[turn] || n != len(s.output) {
			return nil, false, fmt.Errorf("wrong inference context or parameters at step %d", index-1)
		}
		turn++
		return slices.Clone(s.output), true, nil
	}
}

func recordConversation(t *testing.T, f conversationFixture, steps []conversationStep) ([]byte, []Checkpoint, Usage) {
	t.Helper()
	var buf bytes.Buffer
	r, err := NewRecorder(&buf, f.model, f.header, *f.header.InitialTree)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	checkpoints := []Checkpoint{r.Checkpoint()}
	seeds := turnSeeds(f.header.Seed, len(steps))
	for _, step := range steps {
		if step.output != nil {
			if !slices.Equal(r.Context(), step.input) || r.Seed() != seeds[r.Usage().ModelTurns] {
				t.Fatal("recording produced the wrong inference context or seed")
			}
			err = r.ApplyModelTurn(step.output)
		} else {
			call := r.Pending()
			next, result, resources, applyErr := f.env(r.Tree(), r.HarnessLimits(), call.Tool, call.Args)
			if applyErr != nil {
				t.Fatal(applyErr)
			}
			err = r.ApplyHarnessTurn(next, result, resources)
			if err == nil && !bytes.Equal(result, step.result) {
				t.Fatalf("result %q, want %q", result, step.result)
			}
		}
		if err != nil {
			t.Fatal(err)
		}
		checkpoints = append(checkpoints, r.Checkpoint())
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes(), checkpoints, r.Usage()
}

// Execute, compare, and apply a pair supplied by Next. Execution remains outside
// Replay; cancellation and backend failures leave the staged computation available.
func applyReplayPair(ctx context.Context, p *Replay, m model.Model, eng inference.Engine, env applyFunc, mt *ModelTurn, ht *HarnessTurn) error {
	if mt != nil {
		actual, stopped := mt.Output, true
		var err error
		if eng != nil {
			actual, stopped, err = eng.Generate(ctx, p.Context(), m.StopIDs(), p.Seed(), len(mt.Output))
			if err != nil {
				return err
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := p.ValidateModelTurn(actual, stopped); err != nil {
			return err
		}
		if err := p.ApplyModelTurn(actual); err != nil {
			return err
		}
	}
	if ht != nil {
		call := p.Pending()
		if call == nil {
			return fmt.Errorf("harness turn has no pending call")
		}
		next, actual, resources, err := env(p.Tree(), p.HarnessLimits(), call.Tool, call.Args)
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := p.ValidateHarnessTurn(actual); err != nil {
			return err
		}
		if err := p.ApplyHarnessTurn(next, actual, resources); err != nil {
			return err
		}
	}
	return nil
}

func TestRecordReplayAndContinue(t *testing.T) {
	for _, name := range []string{"qwen", "mistral"} {
		t.Run(name, func(t *testing.T) {
			f := testConversation(t, name)
			data, checkpoints, usage := recordConversation(t, f, f.steps)
			r := mustBundleReader(t, data)
			if h := r.Header(); h.InitialTree.Root() != f.header.TreeStart || h.Usage != usage || h.Seed != 11 || h.Temperature != 0.25 {
				t.Fatal("bundle lost initial state, configuration, or measured usage")
			}
			p, err := NewReplay(f.model, r, *r.Header().InitialTree)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := NewReplay(f.model, r, *f.header.InitialTree); err == nil {
				t.Fatal("two replays claimed the same reader")
			}
			eng := conversationEngine(f.model, f.header.Seed, f.steps)
			index := 0
			for {
				mt, ht, err := p.Next()
				if err == io.EOF {
					break
				} else if err != nil {
					t.Fatal(err)
				}
				// Check every computation boundary, including the middle of a returned pair.
				for _, pair := range []struct {
					m *ModelTurn
					h *HarnessTurn
				}{{mt, nil}, {nil, ht}} {
					if pair.m == nil && pair.h == nil {
						continue
					}
					if err := applyReplayPair(t.Context(), p, f.model, eng, f.env, pair.m, pair.h); err != nil {
						t.Fatal(err)
					}
					index++
					cp := p.Checkpoint()
					checkCheckpoint(t, cp, checkpoints[index])
					cp.Context[0]++
					if cp.Pending != nil {
						cp.Pending.Args[0]++
					}
					checkCheckpoint(t, p.Checkpoint(), checkpoints[index])
				}
			}
			if index != len(f.steps) || p.Usage() != usage {
				t.Fatalf("checked EOF: steps %d; usage %+v, want %+v", index, p.Usage(), usage)
			}
			end := p.Checkpoint()
			if !p.Finished() || p.Tree().Root() != f.want.Root() {
				t.Fatal("wrong final artifact tree or finish state")
			}
			for _, member := range zipReader(data).File {
				if strings.HasPrefix(member.Name, "tree/env") || strings.HasPrefix(member.Name, "tree/artifact") {
					t.Fatalf("bundle contains something other than the initial task tree: %s", member.Name)
				}
			}
			// Every boundary is a valid prefix and each unfinished boundary can resume,
			// including a continuation whose first computation is a harness turn.
			for n, cp := range checkpoints {
				t.Run(fmt.Sprintf("boundary_%d", n), func(t *testing.T) {
					prefix, _, _ := recordConversation(t, f, f.steps[:n])
					prefixReplay, err := NewReplay(f.model, mustBundleReader(t, prefix), *f.header.InitialTree)
					if err != nil {
						t.Fatal(err)
					}
					eng := conversationEngine(f.model, f.header.Seed, f.steps[:n])
					for {
						mt, ht, err := prefixReplay.Next()
						if err == io.EOF {
							break
						} else if err != nil {
							t.Fatal(err)
						}
						if err := applyReplayPair(t.Context(), prefixReplay, f.model, eng, f.env, mt, ht); err != nil {
							t.Fatal(err)
						}
					}
					checkCheckpoint(t, prefixReplay.Checkpoint(), cp)
					template := f.header
					template.Seed, template.Usage = 22, usage
					h, err := cp.ResumeHeader(template)
					if cp.Finished {
						if err == nil {
							t.Fatal("resumed a finished conversation")
						}
						return
					} else if err != nil {
						t.Fatal(err)
					} else if h.Seed != 22 || h.Usage != (Usage{}) {
						t.Fatal("continuation did not reset usage and retain selected seed")
					}
					continued := f
					continued.header = h
					data, states, _ := recordConversation(t, continued, f.steps[n:])
					want := end
					want.ModelTurns -= cp.ModelTurns
					want.HarnessTurns -= cp.HarnessTurns
					checkCheckpoint(t, states[len(states)-1], want)
					r := mustBundleReader(t, data)
					if !reflect.DeepEqual(r.Header().InitialCall, cp.Pending) {
						t.Fatal("continuation changed raw arguments or native call reference")
					}
					continuation, err := NewReplay(f.model, r, cp.Tree)
					if err != nil {
						t.Fatal(err)
					}
					eng = conversationEngine(f.model, h.Seed, f.steps[n:])
					first := true
					for {
						mt, ht, err := continuation.Next()
						if err == io.EOF {
							break
						} else if err != nil {
							t.Fatal(err)
						}
						if first && cp.Pending != nil && (mt != nil || ht == nil) {
							t.Fatal("pending continuation did not start with a standalone harness turn")
						}
						first = false
						if err := applyReplayPair(t.Context(), continuation, f.model, eng, f.env, mt, ht); err != nil {
							t.Fatal(err)
						}
					}
					checkCheckpoint(t, continuation.Checkpoint(), want)
					context := slices.Clone(cp.Context)
					var args []byte
					if cp.Pending != nil {
						args = slices.Clone(cp.Pending.Args)
					}
					h.InitialContext[0]++
					if h.InitialCall != nil {
						h.InitialCall.Args[0]++
					}
					if !slices.Equal(cp.Context, context) || cp.Pending != nil && !bytes.Equal(cp.Pending.Args, args) {
						t.Fatal("resumed header aliases its checkpoint")
					}
					checkCheckpoint(t, p.Checkpoint(), end)
				})
			}
		})
	}
}

// Repackage an actual recording without compression to expose exact read
// boundaries. Optional transcript edits retain the original declared usage.
func storedConversation(t *testing.T, data []byte, edit func([]byte) []byte) *zip.Reader {
	t.Helper()
	var members []bundleMember
	for _, f := range zipReader(data).File {
		r, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			t.Fatal(err)
		}
		if f.Name == "transcript.log" {
			if edit != nil {
				body = edit(body)
			}
		}
		members = append(members, bundleMember{f.Name, body})
	}
	return zipReader(bundleZIP(t, members...))
}

func TestStreamingReplayFailures(t *testing.T) {
	f := testConversation(t, "qwen")
	data, checkpoints, _ := recordConversation(t, f, f.steps)
	ends := []int{8 + 4*len(f.header.InitialContext)}
	for _, step := range f.steps {
		ends = append(ends, ends[len(ends)-1]+9+4*len(step.output)+len(step.result))
	}
	for _, mode := range []string{"valid", "unchecked inference", "inference", "cancel", "changed model", "changed result", "wrong order", "checksum", "trailing record", "truncated", "truncated result"} {
		t.Run(mode, func(t *testing.T) {
			z := storedConversation(t, data, func(transcript []byte) []byte {
				switch mode {
				case "changed model":
					transcript[ends[1]+9] ^= 1
				case "changed result":
					transcript[ends[6]+9] ^= 1
				case "wrong order":
					return slices.Concat(transcript[:ends[1]], transcript[ends[2]:ends[3]], transcript[ends[1]:ends[2]], transcript[ends[3]:])
				case "trailing record":
					return append(transcript, 1)
				case "truncated":
					return transcript[:len(transcript)-1]
				case "truncated result":
					return transcript[:ends[3]-1]
				}
				return transcript
			})
			if mode == "checksum" {
				for _, member := range z.File {
					if member.Name == "transcript.log" {
						member.CRC32 ^= 1
					}
				}
			}
			tracked := trackTranscript(z)
			r, err := NewBundleReader(z)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			p, err := NewReplay(f.model, r, *f.header.InitialTree)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			failure := errors.New("inference unavailable")
			base := conversationEngine(f.model, f.header.Seed, f.steps)
			calls, index := 0, 0
			var eng inference.Engine = bundleInferenceEngine(func(ctx context.Context, input, stops []uint32, seed uint64, n int) ([]uint32, bool, error) {
				calls++
				if calls == 2 && mode == "inference" {
					return nil, false, failure
				}
				if calls == 2 && mode == "cancel" {
					cancel()
				}
				return base.Generate(ctx, input, stops, seed, n)
			})
			if mode == "unchecked inference" {
				eng = nil
			}
			streamError := false
			for {
				var mt *ModelTurn
				var ht *HarnessTurn
				mt, ht, err = p.Next()
				if err != nil {
					streamError = true
					break
				}
				if mt == nil && ht == nil {
					t.Fatal("empty pair without EOF")
				}
				// A normal call can stage its following result; it must never read a third record.
				if tracked.read > ends[min(index+2, len(f.steps))] {
					t.Fatal("replay read beyond its bounded pair")
				}
				for _, pair := range []struct {
					m *ModelTurn
					h *HarnessTurn
				}{{mt, nil}, {nil, ht}} {
					if pair.m == nil && pair.h == nil {
						continue
					}
					before := p.Checkpoint()
					if err = applyReplayPair(ctx, p, f.model, eng, f.env, pair.m, pair.h); err != nil {
						checkCheckpoint(t, p.Checkpoint(), before)
						break
					}
					index++
					checkCheckpoint(t, p.Checkpoint(), checkpoints[index])
				}
				if err != nil {
					break
				}
			}
			if mode == "valid" || mode == "unchecked inference" {
				if err != io.EOF || index != len(f.steps) {
					t.Fatalf("valid recording: %v (%d steps)", err, index)
				}
			} else if err == nil || err == io.EOF {
				t.Fatalf("accepted %s recording", mode)
			}
			if mode == "inference" && !errors.Is(err, failure) || mode == "cancel" && !errors.Is(err, context.Canceled) || mode == "checksum" && !errors.Is(err, zip.ErrChecksum) {
				t.Fatalf("wrong failure: %v", err)
			}
			checkCheckpoint(t, p.Checkpoint(), checkpoints[index])
			if mode == "truncated result" && index != 2 {
				t.Fatal("damaged paired result prevented applying its complete preceding model turn")
			}
			read := tracked.read
			_, _, again := p.Next()
			if streamError && again != err || !streamError && again == nil || tracked.read != read || tracked.closes != 0 {
				t.Fatal("replay consumed more input after failure or pending pair")
			}
		})
	}
}

type directVerifier func(context.Context, []uint32, []uint32, []uint32, uint64) error

func (v directVerifier) Generate(context.Context, []uint32, []uint32, uint64, int) ([]uint32, bool, error) {
	return nil, false, errors.New("direct verifier must not generate")
}
func (v directVerifier) Verify(ctx context.Context, input, expected, stops []uint32, seed uint64) error {
	return v(ctx, input, expected, stops, seed)
}

func TestReplayStaging(t *testing.T) {
	f := testConversation(t, "qwen")
	data, checkpoints, _ := recordConversation(t, f, f.steps[:3])
	p, err := NewReplay(f.model, mustBundleReader(t, data), *f.header.InitialTree)
	if err != nil {
		t.Fatal(err)
	}
	mt, ht, err := p.Next()
	if err != nil || mt == nil || ht != nil {
		t.Fatalf("malformed call must remain a standalone model turn: %v", err)
	}
	before, usage := p.Checkpoint(), p.Usage()
	expected := slices.Clone(mt.Output)
	mt.Output[0] ^= 1
	if err := p.ValidateModelTurn(mt.Output, true); err == nil {
		t.Fatal("returned model output aliases private expectation")
	}
	if err := p.ValidateModelTurn(expected, false); err == nil {
		t.Fatal("unstopped generation accepted")
	}
	if _, _, err := p.Next(); err == nil {
		t.Fatal("read past unapplied model")
	}
	if err := p.ApplyModelTurn(nil); err == nil {
		t.Fatal("applied structurally invalid output")
	}
	checkCheckpoint(t, p.Checkpoint(), before)
	if p.Usage() != usage {
		t.Fatal("staging/validation changed usage")
	}
	// Direct verification remains available without generating or validating twice.
	checks := 0
	verifier := directVerifier(func(_ context.Context, input, output, stops []uint32, seed uint64) error {
		checks++
		if !slices.Equal(input, f.steps[0].input) || !slices.Equal(output, expected) || !slices.Equal(stops, f.model.StopIDs()) || seed != turnSeeds(f.header.Seed, 1)[0] {
			return errors.New("wrong verification inputs")
		}
		return nil
	})
	if err := inference.Verify(t.Context(), verifier, p.Context(), expected, f.model.StopIDs(), p.Seed()); err != nil {
		t.Fatal(err)
	}
	if err := p.ApplyModelTurn(expected); err != nil {
		t.Fatal(err)
	}
	if checks != 1 {
		t.Fatal("direct verifier was not used")
	}
	mt, ht, err = p.Next()
	if err != nil || mt == nil || ht == nil {
		t.Fatalf("normal call and result must be paired: %v", err)
	}
	before = p.Checkpoint()
	if err := p.ApplyHarnessTurn(p.Tree(), ht.Output, wasm.Usage{}); err == nil {
		t.Fatal("applied harness before paired model")
	}
	if _, _, err := p.Next(); err == nil {
		t.Fatal("read past unapplied pair")
	}
	checkCheckpoint(t, p.Checkpoint(), before)
	if err := p.ValidateModelTurn(mt.Output, true); err != nil {
		t.Fatal(err)
	}
	if err := p.ApplyModelTurn(mt.Output); err != nil {
		t.Fatal(err)
	}
	checkCheckpoint(t, p.Checkpoint(), checkpoints[2])
	before, usage = p.Checkpoint(), p.Usage()
	contextCopy := p.Context()
	contextCopy[0]++
	call := p.Pending()
	call.Args[0]++
	checkCheckpoint(t, p.Checkpoint(), before)
	call = p.Pending()
	next, actual, resources, err := f.env(p.Tree(), p.HarnessLimits(), call.Tool, call.Args)
	if err != nil {
		t.Fatal(err)
	}
	ht.Output[0] ^= 1
	if err := p.ValidateHarnessTurn(ht.Output); err == nil {
		t.Fatal("returned harness output aliases private expectation")
	}
	for range 2 {
		if err := p.ValidateHarnessTurn(actual); err != nil {
			t.Fatal(err)
		}
	}
	checkCheckpoint(t, p.Checkpoint(), before)
	if p.Usage() != usage {
		t.Fatal("external execution or validation advanced state")
	}
	if err := p.ApplyHarnessTurn(next, actual, resources); err != nil {
		t.Fatal(err)
	}
	checkCheckpoint(t, p.Checkpoint(), checkpoints[3])
	if _, _, err := p.Next(); err != io.EOF {
		t.Fatal(err)
	}

	// Comparison is optional: application consumes the actual supplied output,
	// even after the validator has rejected it against the recording.
	prefix, _, _ := recordConversation(t, f, f.steps[:1])
	p, err = NewReplay(f.model, mustBundleReader(t, prefix), *f.header.InitialTree)
	if err != nil {
		t.Fatal(err)
	}
	mt, _, err = p.Next()
	if err != nil {
		t.Fatal(err)
	}
	mt.Output[0] ^= 1 // Still malformed JSON with the same valid final stop token.
	if err := p.ValidateModelTurn(mt.Output, true); err == nil {
		t.Fatal("altered output matched the recording")
	}
	if err := p.ApplyModelTurn(mt.Output); err != nil {
		t.Fatalf("application unexpectedly required recorded output agreement: %v", err)
	}
	if slices.Equal(p.Context(), checkpoints[1].Context) {
		t.Fatal("application used the staged output instead of the supplied output")
	}
}

// Per-turn sampling seeds are part of the bundle format: turn n takes bytes
// [8n, 8n+8) of the header seed's keyed XOF, little-endian, top bit cleared.
// These vectors were computed directly with the underlying BLAKE3 implementation
// and an explicit domain literal. Only an applied model turn advances the seed;
// a rejected one leaves it for the retry.
func TestTurnSeeds(t *testing.T) {
	m, encode := qwenFixture(t)
	write, finish := encode(`{"name":"write_file","arguments":{"path":"/artifact/a","content":"hi"}}`), encode(`{"name":"finish"}`)
	env := applyWith(tree.Tree{})
	for _, tc := range []struct {
		header uint64
		want   []uint64
	}{
		// Turn 8 starts the second 64-byte XOF block.
		{0, []uint64{0x016b1ebe5bfb42e8, 0x752d496492cfda6d, 0x76996c77234986c4, 0x1b4a193474a429ec, 0x3da00b667f992829, 0x0035d465c24d3f90, 0x081f6a8972f0bd8e, 0x32ad1882cc31c3e5, 0x4a071cdd38fc0480}},
		{1, []uint64{0x519797f8b89920a3, 0x6d082d1e267e71eb, 0x52909ab405dd1331, 0x4dde964042502704}},
		{^uint64(0), []uint64{0x0024d65812ac0d51, 0x0540f1939cfb906e, 0x2d090fd81d0e523d, 0x6dd131533a1de7f3}},
	} {
		t.Run(fmt.Sprint(tc.header), func(t *testing.T) {
			if got := turnSeeds(tc.header, len(tc.want)); !slices.Equal(got, tc.want) {
				t.Fatalf("seed stream changed: %#x", got)
			}
			h := testHeader(tree.Tree{})
			h.Seed = tc.header
			var buf bytes.Buffer
			r, err := NewRecorder(&buf, m, h, tree.Tree{})
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			for i, want := range tc.want {
				if err := r.ApplyModelTurn(nil); err == nil || r.Seed() != want || want > math.MaxInt64 {
					t.Fatalf("recorder turn %d: seed %#x, want %#x (%v)", i, r.Seed(), want, err)
				}
				output := write
				if i == len(tc.want)-1 {
					output = finish
				}
				if err := r.ApplyModelTurn(output); err != nil {
					t.Fatal(err)
				}
				if call := r.Pending(); call != nil {
					next, result, resources, err := env(r.Tree(), r.HarnessLimits(), call.Tool, call.Args)
					if err != nil {
						t.Fatal(err)
					}
					seed := r.Seed()
					if err := r.ApplyHarnessTurn(next, result, resources); err != nil || r.Seed() != seed {
						t.Fatalf("recorder harness turn %d advanced seed: %v", i, err)
					}
				}
			}
			if err := r.Close(); err != nil {
				t.Fatal(err)
			}
			p, err := NewReplay(m, mustBundleReader(t, buf.Bytes()), tree.Tree{})
			if err != nil {
				t.Fatal(err)
			}
			for i, want := range tc.want {
				mt, ht, err := p.Next()
				if err != nil || mt == nil {
					t.Fatalf("replay turn %d: %v", i, err)
				}
				if p.ValidateModelTurn(mt.Output, false) == nil || p.ApplyModelTurn(nil) == nil || p.Seed() != want {
					t.Fatalf("replay turn %d: seed %#x, want %#x", i, p.Seed(), want)
				}
				if err := applyReplayPair(t.Context(), p, m, nil, env, mt, nil); err != nil {
					t.Fatal(err)
				}
				seed := p.Seed()
				if err := applyReplayPair(t.Context(), p, m, nil, env, nil, ht); err != nil || p.Seed() != seed {
					t.Fatalf("replay harness turn %d advanced seed: %v", i, err)
				}
			}
			if _, _, err := p.Next(); err != io.EOF {
				t.Fatal(err)
			}
		})
	}
}

func TestWASMRecordingReplay(t *testing.T) {
	m, encode := qwenFixture(t)
	bin, err := wasmtime.Wat2Wasm(`(module (memory (export "memory") 1) (func (export "_start") nop nop nop))`)
	if err != nil {
		t.Fatal(err)
	}
	files, err := (tree.Tree{}).Put("/noop.wasm", tree.File{Content: bin})
	if err != nil {
		t.Fatal(err)
	}
	env := applyWith(files)
	initial, err := (tree.Tree{}).Put("/artifact/a", tree.File{Content: []byte("original")})
	if err != nil {
		t.Fatal(err)
	}
	h := testHeader(initial)
	var buf bytes.Buffer
	r, err := NewRecorder(&buf, m, h, initial)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	call := encode(`{"name":"run","arguments":{"module":"/env/noop.wasm"}}`)
	if err := r.ApplyModelTurn(call); err != nil {
		t.Fatal(err)
	}
	pending := r.Pending()
	next, result, resources, err := env(r.Tree(), r.HarnessLimits(), pending.Tool, pending.Args)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.ApplyHarnessTurn(next, result, resources); err != nil {
		t.Fatal(err)
	}
	finish := encode(`{"name":"finish"}`)
	steps := []conversationStep{{input: h.InitialContext, output: call}, {input: r.Context(), output: finish}}
	if err := r.ApplyModelTurn(finish); err != nil {
		t.Fatal(err)
	} else if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if u := r.Usage(); u.Fuel == 0 || u.MemoryBytes != 65536 {
		t.Fatalf("WASM resource usage was not measured: %+v", u)
	}
	p, err := NewReplay(m, mustBundleReader(t, buf.Bytes()), initial)
	if err != nil {
		t.Fatal(err)
	}
	eng := conversationEngine(m, h.Seed, steps)
	for {
		mt, ht, err := p.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if err := applyReplayPair(t.Context(), p, m, eng, env, mt, ht); err != nil {
			t.Fatalf("WASM replay under exact measured ceilings: %v", err)
		}
	}
	checkCheckpoint(t, p.Checkpoint(), r.Checkpoint())

	// Cancellation arriving while a tool executes must discard its returned tree
	// and measurements, even when the executor itself completed successfully.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	runs := 0
	canceling := applyFunc(func(tr tree.Tree, limits wasm.Limits, tool string, args json.RawMessage) (tree.Tree, []byte, wasm.Usage, error) {
		runs++
		next, output, resources, err := env(tr, limits, tool, args)
		if err != nil {
			t.Fatal(err)
		}
		next, err = next.Put("/artifact/a", tree.File{Content: []byte("replaced")})
		if err != nil {
			t.Fatal(err)
		}
		cancel()
		return next, output, resources, nil
	})
	p, err = NewReplay(m, mustBundleReader(t, buf.Bytes()), initial)
	if err != nil {
		t.Fatal(err)
	}
	eng = conversationEngine(m, h.Seed, steps)
	mt, ht, err := p.Next()
	if err != nil || mt == nil || ht == nil {
		t.Fatalf("missing WASM pair: %v", err)
	}
	if err := applyReplayPair(ctx, p, m, eng, env, mt, nil); err != nil {
		t.Fatal(err)
	}
	before, usage := p.Checkpoint(), p.Usage()
	if err := applyReplayPair(ctx, p, m, eng, canceling, nil, ht); !errors.Is(err, context.Canceled) {
		t.Fatalf("caller failed to discard canceled execution: %v", err)
	}
	checkCheckpoint(t, p.Checkpoint(), before)
	if _, _, err := p.Next(); err == nil || runs != 1 || p.Usage() != usage {
		t.Fatal("unapplied harness was consumed after cancellation")
	}
	// Cancellation belongs to the caller, so the same staged result can be retried.
	if err := applyReplayPair(t.Context(), p, m, eng, env, nil, ht); err != nil {
		t.Fatal(err)
	}
	for {
		mt, ht, err := p.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if err := applyReplayPair(t.Context(), p, m, eng, env, mt, ht); err != nil {
			t.Fatal(err)
		}
	}
	checkCheckpoint(t, p.Checkpoint(), r.Checkpoint())
}
