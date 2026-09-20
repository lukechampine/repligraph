package repligraph

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"unicode/utf8"

	"lukechampine.com/repligraph/blake3"
	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
	"lukechampine.com/repligraph/inference"
	"lukechampine.com/repligraph/inference/model"
)

// ToolCall is an issued call awaiting execution. Args contains the exact raw
// argument bytes; JSON encodes them as base64 to preserve their spelling. Tool
// and Ref must be valid UTF-8.
type ToolCall struct {
	Tool string `json:"tool"`
	Args []byte `json:"args"`
	Ref  string `json:"ref"`
}

func (c *ToolCall) validate() error {
	if c != nil && (c.Tool == "" || c.Tool == model.ToolFinish || !utf8.ValidString(c.Tool) || !utf8.ValidString(c.Ref)) {
		return fmt.Errorf("invalid initial call")
	}
	return nil
}

func cloneCall(c *ToolCall) *ToolCall {
	if c == nil {
		return nil
	}
	return &ToolCall{Tool: c.Tool, Args: slices.Clone(c.Args), Ref: c.Ref}
}

// Checkpoint captures the state after a prefix of computations. It is not a
// proof of inference, checked archive EOF, or ancestry. Context and Pending are
// owned by the caller; Tree follows tree.Tree's immutable-value contract. Turn
// counts refer to this bundle and reset when resumed into a new bundle.
type Checkpoint struct {
	Context      []uint32
	Tree         tree.Tree
	Pending      *ToolCall
	Finished     bool
	ModelTurns   uint64
	HarnessTurns uint64
}

// ResumeHeader constructs a new bundle's starting premises from c, retaining the
// supplied configuration. A pending call resumes as the first harness operation.
// The caller must select a model matching the context and call framing, and check
// ancestry separately. Compacted or translated inputs should instead be supplied
// explicitly in a new header. A finished transcript cannot resume unchanged.
// Per-turn seeds restart with the new bundle.
func (c Checkpoint) ResumeHeader(b BundleHeader) (BundleHeader, error) {
	if c.Finished {
		return BundleHeader{}, fmt.Errorf("cannot resume a finished checkpoint")
	}
	b.InitialContext = slices.Clone(c.Context)
	initial := c.Tree
	b.InitialTree = &initial
	b.TreeStart = initial.Root()
	b.InitialCall = cloneCall(c.Pending)
	b.Usage = Usage{}
	return b, nil
}

// runState is the canonical transition logic shared by recording and replay.
// Prepare methods leave it unchanged; recording commits after writing succeeds,
// while replay commits the computation supplied by its caller.
type runState struct {
	model                    model.Model
	usage, ceiling           Usage
	context                  []uint32
	tree                     tree.Tree
	pending                  *ToolCall
	finished                 bool
	modelTurns, harnessTurns uint64
	seeds                    io.Reader // XOF keyed by the header seed
	seed                     uint64    // sampling seed for the next model turn
}

func validateInitial(b BundleHeader, initial tree.Tree) error {
	if b.Version != FormatVersion {
		return fmt.Errorf("unsupported format version %d", b.Version)
	} else if err := b.InitialCall.validate(); err != nil {
		return err
	} else if b.Model == "" || b.Inference == "" || b.Environment == "" {
		return fmt.Errorf("bundle requires model, inference, and environment pins")
	} else if math.IsNaN(b.Temperature) || math.IsInf(b.Temperature, 0) || b.Temperature < 0 {
		return fmt.Errorf("temperature must be finite and non-negative")
	} else if initial.Root() != b.TreeStart {
		return fmt.Errorf("initial tree does not match tree_start")
	} else if b.InitialTree != nil && b.InitialTree.Root() != b.TreeStart {
		return fmt.Errorf("bundled initial tree does not match tree_start")
	} else if len(b.InitialContext) == 0 {
		return fmt.Errorf("bundle requires an initial context")
	}
	return validateTaskTree(initial)
}

func newRunState(m model.Model, b BundleHeader, initial tree.Tree, ceiling Usage) (runState, error) {
	if err := validateInitial(b, initial); err != nil {
		return runState{}, err
	} else if m == nil {
		return runState{}, fmt.Errorf("model must be non-nil")
	}
	stats := initial.Stats()
	usage := Usage{InitialTreeBytes: uint64(stats.Bytes), ContextTokens: uint64(len(b.InitialContext)), Usage: wasm.Usage{TreeEntries: uint64(stats.Entries), TreeBytes: uint64(stats.Bytes)}}
	if err := usage.within(ceiling); err != nil {
		return runState{}, err
	}
	var key [32]byte
	binary.LittleEndian.PutUint64(key[:], b.Seed)
	s := runState{
		model: m, usage: usage, ceiling: ceiling, context: slices.Clone(b.InitialContext),
		tree: initial, pending: cloneCall(b.InitialCall), seeds: blake3.XOF(key, blake3.DomainSeed),
	}
	s.nextSeed()
	return s, nil
}

// nextSeed reads eight little-endian bytes of the seed stream per model turn,
// clearing the top bit for engines that take signed seeds.
func (s *runState) nextSeed() {
	var buf [8]byte
	s.seeds.Read(buf[:]) // XOF reads never fail
	s.seed = binary.LittleEndian.Uint64(buf[:]) &^ (1 << 63)
}

func (s *runState) checkpoint() Checkpoint {
	return Checkpoint{
		Context: slices.Clone(s.context), Tree: s.tree, Pending: cloneCall(s.pending),
		Finished: s.finished, ModelTurns: s.modelTurns, HarnessTurns: s.harnessTurns,
	}
}

type modelStep struct {
	generated, frame []uint32
	pending          *ToolCall
	finished         bool
	usage            Usage
}

func (s *runState) prepareModel(output []uint32) (modelStep, error) {
	if s.finished {
		return modelStep{}, fmt.Errorf("record after finish")
	} else if s.pending != nil {
		return modelStep{}, fmt.Errorf("a tool call is pending")
	} else if err := inference.ValidateOutput(output, s.model.StopIDs()); err != nil {
		return modelStep{}, err
	}
	usage, err := s.usage.add(Usage{ModelTurns: 1, InputTokens: uint64(len(s.context)), OutputTokens: uint64(len(output)), ContextTokens: uint64(len(s.context)) + uint64(len(output))})
	if err != nil {
		return modelStep{}, err
	}
	if err := usage.within(s.ceiling); err != nil {
		return modelStep{}, err
	}
	step := modelStep{generated: output[:len(output)-1], usage: usage}
	tool, args, ref, reject := s.model.Parse(step.generated)
	if reject != "" {
		step.frame = s.model.FramePrompt(model.BadCall(reject))
	} else if tool == model.ToolFinish {
		step.finished = true
	} else {
		step.pending = &ToolCall{Tool: tool, Args: slices.Clone(args), Ref: ref}
	}
	nextContext := uint64(len(s.context)) + uint64(len(step.generated)) + uint64(len(step.frame))
	if nextContext > uint64(math.MaxInt) {
		return modelStep{}, resourceError("context allocation")
	}
	step.usage.ContextTokens = max(step.usage.ContextTokens, nextContext)
	if err := step.usage.within(s.ceiling); err != nil {
		return modelStep{}, err
	}
	return step, nil
}

func (s *runState) commitModel(step modelStep) {
	s.context = append(s.context, step.generated...)
	s.context = append(s.context, step.frame...)
	s.pending, s.finished = step.pending, step.finished
	s.modelTurns++
	s.usage = step.usage
	s.nextSeed()
}

type harnessStep struct {
	tree   tree.Tree
	output []byte
	frame  []uint32
	usage  Usage
}

func (s *runState) prepareHarness(next tree.Tree, output []byte, resources wasm.Usage) (harnessStep, error) {
	if s.finished {
		return harnessStep{}, fmt.Errorf("record after finish")
	} else if s.pending == nil {
		return harnessStep{}, fmt.Errorf("no tool call is pending")
	}
	if s.usage.HarnessTurns == s.ceiling.HarnessTurns {
		return harnessStep{}, resourceError("harness turns")
	}
	if err := validateTaskTree(next); err != nil {
		return harnessStep{}, err
	}
	before, after := s.tree.Stats(), next.Stats()
	if uint64(after.Entries) > s.ceiling.TreeEntries {
		return harnessStep{}, resourceError("tree entries")
	} else if uint64(after.Bytes) > s.ceiling.TreeBytes {
		return harnessStep{}, resourceError("tree bytes")
	}
	if resources.TreeEntries < uint64(max(before.Entries, after.Entries)) || resources.TreeBytes < uint64(max(before.Bytes, after.Bytes)) {
		return harnessStep{}, fmt.Errorf("harness usage understates task tree size")
	}
	// Reject known resource excesses before allocating the framed result.
	usage, err := s.usage.add(Usage{HarnessTurns: 1, ToolOutputBytes: uint64(len(output)), Usage: resources})
	if err != nil {
		return harnessStep{}, err
	} else if err := usage.within(s.ceiling); err != nil {
		return harnessStep{}, err
	}
	frame := s.model.FrameResult(s.pending.Ref, output)
	nextContext := uint64(len(s.context)) + uint64(len(frame))
	if nextContext > uint64(math.MaxInt) {
		return harnessStep{}, resourceError("context allocation")
	}
	usage.ContextTokens = max(usage.ContextTokens, nextContext)
	if err := usage.within(s.ceiling); err != nil {
		return harnessStep{}, err
	}
	return harnessStep{tree: next, output: output, frame: frame, usage: usage}, nil
}

func (s *runState) commitHarness(step harnessStep) {
	s.tree = step.tree
	s.context = append(s.context, step.frame...)
	s.pending = nil
	s.harnessTurns++
	s.usage = step.usage
}

// ModelTurn contains the recorded output of an inference computation, including
// its stop token. Output belongs to the caller; changing it does not change what
// Replay's validator expects. The input context and sampling seed are available
// from Replay.Context and Replay.Seed.
type ModelTurn struct {
	Output []uint32
}

// HarnessTurn contains the recorded output of a tool execution. Output belongs
// to the caller. After applying the preceding model turn, use Replay.Pending,
// Tree, and HarnessLimits to execute the call. The resulting tree and resource
// measurements are computed by the caller; they are not stored in this record.
type HarnessTurn struct {
	Output []byte
}

// Replay streams recorded computations for the caller to execute, validate, and
// apply. It retains current state and at most one model/harness pair. Applying a
// turn does not compare outputs: callers choose whether to validate them first.
type Replay struct {
	state       runState
	reader      *BundleReader
	modelOutput []uint32
	toolOutput  []byte
	err         error
}

// NewReplay takes exclusive use of a fresh reader's record stream. The caller
// admits the declared Usage, supplies implementations matching the header pins,
// and runs inference with the recorded temperature and each turn's Seed. Neither
// inference nor tool execution is performed by Replay.
//
// Drive Next through io.EOF to check the archive checksum and exact resource
// usage, and close the reader on every path. Valid prefixes, including pending
// calls at EOF, are accepted; require Finished separately when needed. EOF does
// not prove output agreement when the caller has skipped validation.
func NewReplay(m model.Model, r *BundleReader, initial tree.Tree) (*Replay, error) {
	if r == nil {
		return nil, fmt.Errorf("bundle reader is nil")
	} else if r.closed {
		return nil, io.ErrClosedPipe
	} else if r.started {
		return nil, fmt.Errorf("bundle reader has already been consumed")
	}
	b := r.Header()
	state, err := newRunState(m, b, initial, b.Usage)
	if err != nil {
		return nil, err
	}
	r.started = true
	return &Replay{state: state, reader: r}, nil
}

func (p *Replay) Context() []uint32 { return slices.Clone(p.state.context) }
func (p *Replay) Tree() tree.Tree   { return p.state.tree }
func (p *Replay) Finished() bool    { return p.state.finished }

// Seed returns the sampling seed for the next model turn, derived from the header
// seed. It never exceeds math.MaxInt64.
func (p *Replay) Seed() uint64 { return p.state.seed }

// Pending returns a copy of the issued call, or nil if none awaits execution.
func (p *Replay) Pending() *ToolCall { return cloneCall(p.state.pending) }

// HarnessLimits returns the remaining declared execution budget.
func (p *Replay) HarnessLimits() wasm.Limits { return p.state.usage.remaining(p.state.ceiling) }

// Checkpoint copies the current applied prefix. It remains available after EOF
// or an error; failed applications never advance the state. It does not attest
// that the caller validated each computation.
func (p *Replay) Checkpoint() Checkpoint { return p.state.checkpoint() }

// Usage returns measured totals and peaks for the applied prefix.
func (p *Replay) Usage() Usage { return p.state.usage }

// Next returns a model turn and its immediately following harness turn, when
// present. Either may be nil, but not both on success. A leading harness turn is
// returned alone; consecutive model turns are returned by separate calls.
// Apply the model before the harness, and apply both before calling Next again.
//
// Reading does not advance execution state. A read error encountered after a
// complete model turn is deferred until the next Next, allowing that prefix to
// be applied. Archive/protocol errors and EOF are sticky; requesting another
// pair before applying the current one is a retryable error. io.EOF checks exact
// aggregate usage as well as the transcript checksum, even after Finished.
func (p *Replay) Next() (mt *ModelTurn, ht *HarnessTurn, err error) {
	if p.err != nil {
		return nil, nil, p.err
	} else if p.modelOutput != nil || p.toolOutput != nil {
		return nil, nil, fmt.Errorf("apply the pending turns before calling Next")
	}
	defer func() {
		if err != nil {
			p.err = err
		}
	}()
	output, result, err := p.reader.Next()
	if err == io.EOF {
		if p.state.usage != p.state.ceiling {
			return nil, nil, fmt.Errorf("recorded usage does not match replay: declared %+v, measured %+v", p.state.ceiling, p.state.usage)
		}
		return nil, nil, io.EOF
	} else if err != nil {
		return nil, nil, fmt.Errorf("transcript: %w", err)
	} else if p.state.finished {
		return nil, nil, fmt.Errorf("record after finish")
	}
	if output != nil {
		// Preflight the recorded protocol before handing work to the caller.
		// ApplyModelTurn separately checks the output actually supplied.
		step, err := p.state.prepareModel(output)
		if err != nil {
			return nil, nil, fmt.Errorf("model output %d: %w", p.state.modelTurns, err)
		}
		p.modelOutput = output
		mt = &ModelTurn{Output: slices.Clone(output)}
		if step.pending != nil {
			// Peek only the tag, never another model payload. A malformed
			// harness record leaves its error on the reader for the next Next.
			if tag, err := p.reader.peekTag(); err == nil && tag == 2 {
				if _, result, err := p.reader.Next(); err == nil {
					p.toolOutput = result
					ht = &HarnessTurn{Output: slices.Clone(result)}
				}
			}
		}
		return mt, ht, nil
	}
	if p.state.pending == nil {
		return nil, nil, fmt.Errorf("unexpected tool output %d", p.state.harnessTurns)
	}
	p.toolOutput = result
	return nil, &HarnessTurn{Output: slices.Clone(result)}, nil
}

func (p *Replay) expectModel() error {
	if p.err != nil {
		return p.err
	} else if p.modelOutput == nil {
		return fmt.Errorf("no model turn awaits application")
	}
	return nil
}

func (p *Replay) expectHarness() error {
	if p.err != nil {
		return p.err
	} else if p.modelOutput != nil {
		return fmt.Errorf("apply the model turn before the harness turn")
	} else if p.toolOutput == nil {
		return fmt.Errorf("no harness turn awaits application")
	} else if p.state.pending == nil {
		return fmt.Errorf("no tool call is pending")
	}
	return nil
}

// ValidateModelTurn compares a regenerated output with the staged model turn.
// It neither executes inference nor applies the turn. A mismatch is retryable.
// Pass the same output to ApplyModelTurn after successful validation.
// Callers using inference.Verify may instead verify ModelTurn.Output directly
// and apply it without calling this method.
func (p *Replay) ValidateModelTurn(output []uint32, stopped bool) error {
	if err := p.expectModel(); err != nil {
		return err
	}
	for i := 0; i < min(len(output), len(p.modelOutput)); i++ {
		if output[i] != p.modelOutput[i] {
			return fmt.Errorf("model output %d: token mismatch at position %d: recorded %d, regenerated %d", p.state.modelTurns, i, p.modelOutput[i], output[i])
		}
	}
	if len(output) != len(p.modelOutput) {
		return fmt.Errorf("model output %d: output length mismatch: recorded %d tokens, regenerated %d", p.state.modelTurns, len(p.modelOutput), len(output))
	} else if !stopped {
		return fmt.Errorf("model output %d: regenerated turn did not end on a stop token", p.state.modelTurns)
	}
	return nil
}

// ValidateHarnessTurn compares a tool execution's output with the staged harness
// turn, without executing or applying it. A mismatch is retryable. Pass the same
// output, with its resulting tree and usage, to ApplyHarnessTurn after validation.
// Individual trees and resource measurements are not stored in transcript records;
// Replay checks structural bounds during application and aggregate usage at EOF.
func (p *Replay) ValidateHarnessTurn(output []byte) error {
	if err := p.expectHarness(); err != nil {
		return err
	} else if !bytes.Equal(output, p.toolOutput) {
		return fmt.Errorf("tool output %d does not match", p.state.harnessTurns)
	}
	return nil
}

// ApplyModelTurn applies a complete generation, including its stop token, using
// the same transition rules as Recorder.ApplyModelTurn. It checks protocol and
// resources but does not compare the output with the recorded one.
func (p *Replay) ApplyModelTurn(output []uint32) error {
	if err := p.expectModel(); err != nil {
		return err
	}
	step, err := p.state.prepareModel(output)
	if err != nil {
		return p.abortLimit(err)
	}
	p.state.commitModel(step)
	p.modelOutput = nil
	return nil
}

// ApplyHarnessTurn applies the externally computed tree, output, and usage using
// the same transition rules as Recorder.ApplyHarnessTurn. It checks protocol and
// resources but does not compare the output with the recorded one.
func (p *Replay) ApplyHarnessTurn(next tree.Tree, output []byte, resources wasm.Usage) error {
	if err := p.expectHarness(); err != nil {
		return err
	}
	step, err := p.state.prepareHarness(next, output, resources)
	if err != nil {
		return p.abortLimit(err)
	}
	p.state.commitHarness(step)
	p.toolOutput = nil
	return nil
}

func (p *Replay) abortLimit(err error) error {
	if errors.Is(err, ErrResourceLimit) {
		p.err = err
	}
	return err
}
