package repligraph

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"

	"lukechampine.com/repligraph/harness/wasm"
)

// ErrResourceLimit aborts recording or replay; it is never a tool result.
var ErrResourceLimit = wasm.ErrResourceLimit

// Usage describes the work recorded in a bundle. Counts and bytes are totals;
// ContextTokens, MemoryBytes, TreeEntries, and TreeBytes are high-water marks.
// MemoryBytes measures guest linear memory, not the host process or inference backend.
// InitialTreeBytes excludes the separately supplied, read-only environment.
type Usage struct {
	ModelTurns       uint64 `json:"model_turns"`
	HarnessTurns     uint64 `json:"harness_turns"`
	InputTokens      uint64 `json:"input_tokens"`
	OutputTokens     uint64 `json:"output_tokens"`
	ContextTokens    uint64 `json:"context_tokens"` // Includes the stop token during inference, and framed context between turns.
	ToolOutputBytes  uint64 `json:"tool_output_bytes"`
	InitialTreeBytes uint64 `json:"initial_tree_bytes"`
	wasm.Usage
}

func (u *Usage) UnmarshalJSON(data []byte) error {
	type plain Usage
	var v plain
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&v); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for _, name := range []string{"model_turns", "harness_turns", "input_tokens", "output_tokens", "context_tokens", "tool_output_bytes", "initial_tree_bytes", "fuel", "memory_bytes", "write_bytes", "stdout_bytes", "stderr_bytes", "tree_entries", "tree_bytes"} {
		if value, ok := fields[name]; !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("usage is missing required field %q", name)
		}
	}
	*u = Usage(v)
	return nil
}

// Limits are local recording ceilings. They are never written to the bundle.
// Fuel, writes, and output bytes are cumulative across all calls; memory, tree
// entries, and tree bytes bound peaks. MaxTokens bounds one generation;
// MaxInferenceTokens counts the sum of all inference inputs and outputs.
// MaxTurns counts all recorded outputs.
// Zero permits none. Callers must also bound their inference requests and I/O.
type Limits struct {
	MaxTokens          uint64
	MaxInferenceTokens uint64
	MaxTurns           uint64
	Fuel               uint64
	MemoryBytes        uint64
	WriteBytes         uint64
	StdoutBytes        uint64
	StderrBytes        uint64
	MaxTreeEntries     uint64
	MaxTreeBytes       uint64
}

func DefaultLimits() Limits {
	return Limits{
		MaxTokens: 65536, MaxInferenceTokens: 1 << 24, MaxTurns: 2048,
		Fuel: 1 << 40, MemoryBytes: 1 << 30, WriteBytes: 256 << 20,
		StdoutBytes: 1 << 20, StderrBytes: 1 << 20, MaxTreeEntries: 65536, MaxTreeBytes: 256 << 20,
	}
}

func (l Limits) ceiling() Usage {
	return Usage{
		ModelTurns: l.MaxTurns, HarnessTurns: l.MaxTurns,
		InputTokens: l.MaxInferenceTokens, OutputTokens: l.MaxInferenceTokens,
		ContextTokens: math.MaxUint64, ToolOutputBytes: math.MaxUint64, InitialTreeBytes: math.MaxUint64,
		Usage: wasm.Usage{Fuel: l.Fuel, MemoryBytes: l.MemoryBytes, WriteBytes: l.WriteBytes,
			StdoutBytes: l.StdoutBytes, StderrBytes: l.StderrBytes, TreeEntries: l.MaxTreeEntries, TreeBytes: l.MaxTreeBytes},
	}
}

func resourceError(name string) error { return fmt.Errorf("%w: %s", ErrResourceLimit, name) }

func (u Usage) within(cap Usage) error {
	for _, f := range []struct {
		name        string
		used, limit uint64
	}{
		{"model turns", u.ModelTurns, cap.ModelTurns}, {"harness turns", u.HarnessTurns, cap.HarnessTurns},
		{"input tokens", u.InputTokens, cap.InputTokens}, {"output tokens", u.OutputTokens, cap.OutputTokens},
		{"context tokens", u.ContextTokens, cap.ContextTokens}, {"tool output bytes", u.ToolOutputBytes, cap.ToolOutputBytes},
		{"initial tree bytes", u.InitialTreeBytes, cap.InitialTreeBytes}, {"fuel", u.Fuel, cap.Fuel},
		{"memory bytes", u.MemoryBytes, cap.MemoryBytes}, {"tree entries", u.TreeEntries, cap.TreeEntries},
		{"tree bytes", u.TreeBytes, cap.TreeBytes},
		{"write bytes", u.WriteBytes, cap.WriteBytes}, {"stdout bytes", u.StdoutBytes, cap.StdoutBytes},
		{"stderr bytes", u.StderrBytes, cap.StderrBytes},
	} {
		if f.used > f.limit {
			return resourceError(f.name)
		}
	}
	return nil
}

func (u Usage) add(v Usage) (Usage, error) {
	for _, f := range []struct {
		dst *uint64
		n   uint64
	}{
		{&u.ModelTurns, v.ModelTurns}, {&u.HarnessTurns, v.HarnessTurns},
		{&u.InputTokens, v.InputTokens}, {&u.OutputTokens, v.OutputTokens}, {&u.ToolOutputBytes, v.ToolOutputBytes},
		{&u.Fuel, v.Fuel}, {&u.WriteBytes, v.WriteBytes}, {&u.StdoutBytes, v.StdoutBytes}, {&u.StderrBytes, v.StderrBytes},
	} {
		if f.n > math.MaxUint64-*f.dst {
			return Usage{}, resourceError("usage overflow")
		}
		*f.dst += f.n
	}
	u.ContextTokens = max(u.ContextTokens, v.ContextTokens)
	u.MemoryBytes = max(u.MemoryBytes, v.MemoryBytes)
	u.TreeEntries = max(u.TreeEntries, v.TreeEntries)
	u.TreeBytes = max(u.TreeBytes, v.TreeBytes)
	return u, nil
}

func (u Usage) remaining(cap Usage) wasm.Limits {
	return wasm.Limits{Fuel: cap.Fuel - u.Fuel, MemoryBytes: cap.MemoryBytes,
		WriteBytes: cap.WriteBytes - u.WriteBytes, StdoutBytes: cap.StdoutBytes - u.StdoutBytes,
		StderrBytes: cap.StderrBytes - u.StderrBytes, MaxTreeEntries: cap.TreeEntries, MaxTreeBytes: cap.TreeBytes}
}
