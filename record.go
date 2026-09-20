package repligraph

import (
	"errors"
	"fmt"
	"io"
	"slices"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
	"lukechampine.com/repligraph/inference/model"
)

// A Recorder writes one model or harness computation at a time.
type Recorder struct {
	writer *BundleWriter
	state  runState
	limits Limits
	closed bool
	err    error
}

// NewRecorder starts a bundle with no recorded outputs. The caller supplies
// implementations matching its pins and drives inference using Context and Seed.
// If InitialCall is present, ApplyHarnessTurn must precede the first ApplyModelTurn.
// Omitted ceilings use DefaultLimits. Header.Usage is replaced with measurements
// when Close finalizes the manifest; it is never a recording ceiling.
func NewRecorder(w io.Writer, m model.Model, b BundleHeader, initial tree.Tree, policy ...Limits) (*Recorder, error) {
	limits := DefaultLimits()
	if len(policy) > 1 {
		return nil, fmt.Errorf("expected at most one recording ceiling")
	}
	if len(policy) == 1 {
		limits = policy[0]
	}
	state, err := newRunState(m, b, initial, limits.ceiling())
	if err != nil {
		return nil, err
	}
	writer, err := NewBundleWriter(w, b)
	if err != nil {
		return nil, err
	}
	return &Recorder{writer: writer, state: state, limits: limits}, nil
}

func (r *Recorder) Context() []uint32 { return slices.Clone(r.state.context) }
func (r *Recorder) Tree() tree.Tree   { return r.state.tree }
func (r *Recorder) Finished() bool    { return r.state.finished }

// Seed returns the sampling seed for the next model turn, derived from the header
// seed. It never exceeds math.MaxInt64.
func (r *Recorder) Seed() uint64 { return r.state.seed }

// Pending returns a copy of the issued call, or nil if none awaits execution.
func (r *Recorder) Pending() *ToolCall { return cloneCall(r.state.pending) }

// HarnessLimits returns the remaining execution budget to pass to Environment.Apply.
// Fuel, write, and output allowances decrease; memory and tree ceilings stay fixed.
func (r *Recorder) HarnessLimits() wasm.Limits { return r.state.usage.remaining(r.state.ceiling) }

// Usage returns measured totals and peaks for the recorded prefix.
func (r *Recorder) Usage() Usage { return r.state.usage }

// Checkpoint copies the current recorded prefix. It does not verify inference or ancestry.
func (r *Recorder) Checkpoint() Checkpoint { return r.state.checkpoint() }

func (r *Recorder) active() error {
	if r.err != nil {
		return r.err
	} else if r.closed {
		return io.ErrClosedPipe
	} else if r.state.finished {
		return fmt.Errorf("recorder is finished")
	}
	return nil
}

// ApplyModelTurn records a complete generation, including its stop token.
// It adds malformed-call feedback or leaves a call pending for ApplyHarnessTurn.
func (r *Recorder) ApplyModelTurn(output []uint32) error {
	if err := r.active(); err != nil {
		return err
	}
	if uint64(len(output)) > r.limits.MaxTokens {
		return r.abortLimit(resourceError("generation tokens"))
	}
	if err := r.checkTurn(); err != nil {
		return err
	}
	step, err := r.state.prepareModel(output)
	if err != nil {
		return r.abortLimit(err)
	}
	if step.usage.OutputTokens > r.limits.MaxInferenceTokens-step.usage.InputTokens {
		return r.abortLimit(resourceError("inference tokens"))
	}
	if err := r.writer.WriteModelOutput(output); err != nil {
		r.err = err
		return err
	}
	r.state.commitModel(step)
	return nil
}

// ApplyHarnessTurn records the result of executing the pending call, including its
// next task tree and measured resources. It performs no execution. The caller
// obtains these values from Environment.Apply using Tree, HarnessLimits, and
// Pending, and must Abort if that execution exhausts resources or is canceled.
// Supplied measurements are checked against the recording ceilings. To verify,
// the replay caller independently executes the call and validates its result.
func (r *Recorder) ApplyHarnessTurn(next tree.Tree, output []byte, resources wasm.Usage) error {
	if err := r.active(); err != nil {
		return err
	}
	if err := r.checkTurn(); err != nil {
		return err
	}
	step, err := r.state.prepareHarness(next, output, resources)
	if err != nil {
		return r.abortLimit(err)
	}
	if err := r.writer.WriteToolOutput(step.output); err != nil {
		r.err = err
		return err
	}
	r.state.commitHarness(step)
	return nil
}

func (r *Recorder) abortLimit(err error) error {
	if errors.Is(err, ErrResourceLimit) {
		r.err = err
	}
	return err
}

func (r *Recorder) checkTurn() error {
	u := r.state.usage
	if u.ModelTurns >= r.limits.MaxTurns || u.HarnessTurns >= r.limits.MaxTurns-u.ModelTurns {
		return r.abortLimit(resourceError("turns"))
	}
	return nil
}

// Abort stops recording without a final manifest. Call this when inference or
// another caller-managed operation exhausts its budget or is canceled. The last
// successful checkpoint remains available; later operations return this error.
func (r *Recorder) Abort(err error) error {
	if r.closed {
		if r.err != nil {
			return r.err
		}
		return io.ErrClosedPipe
	}
	if r.err == nil {
		if err == nil {
			err = errors.New("recording aborted")
		}
		r.err = err
	}
	r.closed = true
	return r.writer.abort(r.err)
}

// Close finalizes the ZIP at the current prefix without closing the destination.
func (r *Recorder) Close() error {
	if !r.closed {
		r.closed = true
		if r.err != nil {
			return r.writer.abort(r.err)
		}
		r.err = r.writer.Close(r.state.usage)
	}
	return r.err
}
