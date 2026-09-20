package vllm

import (
	"context"
	"fmt"
	"math"
	"slices"

	"lukechampine.com/repligraph/inference"
)

// ParallelVerifier scores greedy turns; sampled or ambiguous turns regenerate.
// Prompt scoring is validated with m-qwen2.5-coder-7b under the registry's
// i-vllm-0.29.0-h100 configuration, using MaxScoringTokens 8191 to leave room
// for the scoring API's extra generated token.
type ParallelVerifier struct {
	*Client
	MaxScoringTokens int // context plus output; larger turns regenerate; zero disables scoring
}

type promptLogprob struct {
	Logprob *float64 `json:"logprob"`
	Rank    *uint64  `json:"rank"`
}

func (v *ParallelVerifier) Verify(ctx context.Context, contextIDs, expected, stopIDs []uint32, seed uint64) error {
	if err := ctx.Err(); err != nil {
		return err
	} else if v == nil || v.Client == nil {
		return fmt.Errorf("vLLM client is nil")
	} else if len(contextIDs) == 0 {
		return fmt.Errorf("inference context must be non-empty")
	} else if err := inference.ValidateOutput(expected, stopIDs); err != nil {
		return err
	} else if v.MaxScoringTokens < 0 {
		return fmt.Errorf("scoring token limit must be non-negative")
	}
	if v.Temperature != 0 || len(contextIDs) >= v.MaxScoringTokens || len(expected) > v.MaxScoringTokens-len(contextIDs) {
		return inference.Verify(ctx, v.Client, contextIDs, expected, stopIDs, seed)
	}
	prompt := slices.Concat(contextIDs, expected)
	ch, err := v.complete(ctx, prompt, stopIDs, seed, 0, true, false)
	if err != nil {
		return err
	} else if len(ch.PromptLogprobs) != len(prompt) || ch.PromptLogprobs[0] != nil {
		return fmt.Errorf("vLLM returned invalid prompt logprobs")
	}
	ambiguous := false
	for i, token := range expected {
		row := ch.PromptLogprobs[len(contextIDs)+i]
		if (len(row) != 2 && len(row) != 3) || row[token] == nil {
			return fmt.Errorf("vLLM omitted candidate or top scores at position %d", i)
		}
		var topID uint32
		var first, second *float64
		for id, score := range row {
			if score == nil || score.Logprob == nil || score.Rank == nil || *score.Rank == 0 || math.IsNaN(*score.Logprob) || math.IsInf(*score.Logprob, 0) || *score.Logprob > 0 {
				return fmt.Errorf("vLLM returned an invalid score at position %d", i)
			}
			if len(row) == 3 && id == token {
				continue
			}
			switch *score.Rank {
			case 1:
				if first != nil {
					return fmt.Errorf("vLLM returned duplicate top ranks at position %d", i)
				}
				topID, first = id, score.Logprob
			case 2:
				if second != nil {
					return fmt.Errorf("vLLM returned duplicate top ranks at position %d", i)
				}
				second = score.Logprob
			default:
				return fmt.Errorf("vLLM returned an unexpected score at position %d", i)
			}
		}
		if first == nil || second == nil || *first < *second || (len(row) == 3 && *row[token].Logprob > *second) {
			return fmt.Errorf("vLLM returned inconsistent top scores at position %d", i)
		}
		if *row[token].Logprob < *first {
			return fmt.Errorf("token mismatch at position %d: recorded %d, scored %d", i, token, topID)
		} else if *first == *second {
			ambiguous = true
		}
	}
	if ambiguous {
		return inference.Verify(ctx, v.Client, contextIDs, expected, stopIDs, seed)
	}
	return ctx.Err()
}
