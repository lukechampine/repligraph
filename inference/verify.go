package inference

import (
	"context"
	"fmt"
	"slices"
)

// An Engine deterministically generates until a stop token (included in output) or maxTokens.
// Each call uses the configured temperature and restarts sampling from seed.
type Engine interface {
	Generate(ctx context.Context, contextIDs, stopIDs []uint32, seed uint64, maxTokens int) (output []uint32, stopped bool, err error)
}

// A Verifier optionally checks an expected output without regenerating it.
type Verifier interface {
	Verify(ctx context.Context, contextIDs, expected, stopIDs []uint32, seed uint64) error
}

// ValidateOutput checks that output contains exactly one stop token, at its end.
func ValidateOutput(output, stopIDs []uint32) error {
	if len(output) == 0 || !slices.Contains(stopIDs, output[len(output)-1]) {
		return fmt.Errorf("generated turn does not end on a stop token")
	}
	for i, id := range output[:len(output)-1] {
		if slices.Contains(stopIDs, id) {
			return fmt.Errorf("generated turn contains a stop token before its end at position %d", i)
		}
	}
	return nil
}

// Verify checks that eng generates expected from contextIDs using stopIDs and seed.
func Verify(ctx context.Context, eng Engine, contextIDs, expected, stopIDs []uint32, seed uint64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(contextIDs) == 0 {
		return fmt.Errorf("inference context must be non-empty")
	}
	if err := ValidateOutput(expected, stopIDs); err != nil {
		return err
	}
	if eng == nil {
		return fmt.Errorf("inference engine is nil")
	}
	if v, ok := eng.(Verifier); ok {
		if err := v.Verify(ctx, slices.Clone(contextIDs), slices.Clone(expected), slices.Clone(stopIDs), seed); err != nil {
			return fmt.Errorf("inference: %w", err)
		}
		return ctx.Err()
	}
	got, stopped, err := eng.Generate(ctx, slices.Clone(contextIDs), slices.Clone(stopIDs), seed, len(expected))
	if err != nil {
		return fmt.Errorf("inference: %w", err)
	}
	for i := 0; i < len(got) && i < len(expected); i++ {
		if got[i] != expected[i] {
			return fmt.Errorf("token mismatch at position %d: recorded %d, regenerated %d", i, expected[i], got[i])
		}
	}
	if len(got) != len(expected) {
		return fmt.Errorf("output length mismatch: recorded %d tokens, regenerated %d", len(expected), len(got))
	}
	if !stopped {
		return fmt.Errorf("regenerated turn matches but did not end on a stop token")
	}
	return nil
}
