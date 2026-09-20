// Package vllm implements token-level inference with vLLM 0.29.
package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"
)

// A Client implements inference.Engine using a vLLM completions endpoint.
type Client struct {
	Endpoint    string // Full URL, including /v1/completions.
	Model       string
	Temperature float64
	HTTP        *http.Client // nil uses http.DefaultClient.
}

type completionChoice struct {
	Index          *uint64                     `json:"index"`
	PromptTokenIDs []*uint32                   `json:"prompt_token_ids"`
	TokenIDs       []*uint32                   `json:"token_ids"`
	FinishReason   string                      `json:"finish_reason"`
	StopReason     *uint32                     `json:"stop_reason"`
	PromptLogprobs []map[uint32]*promptLogprob `json:"prompt_logprobs"`
	Logprobs       *completionLogprobs         `json:"logprobs"`
}

type completionLogprobs struct {
	Tokens        []*string             `json:"tokens"`
	TokenLogprobs []*float64            `json:"token_logprobs"`
	TopLogprobs   []map[string]*float64 `json:"top_logprobs"`
}

// TokenLogprobs contains one generated token's log probability and the two
// highest-scoring tokens. Top includes the generated token, adding a third entry
// when it is outside the top two. Scores include the terminating stop token.
type TokenLogprobs struct {
	Logprob float64            `json:"logprob"`
	Top     map[uint32]float64 `json:"top"`
}

func (c *Client) complete(ctx context.Context, contextIDs, stopIDs []uint32, seed uint64, maxTokens int, scoring, logprobs bool) (completionChoice, error) {
	if err := ctx.Err(); err != nil {
		return completionChoice{}, err
	} else if c.Model == "" {
		return completionChoice{}, fmt.Errorf("model must be non-empty")
	} else if seed > math.MaxInt64 {
		return completionChoice{}, fmt.Errorf("seed exceeds vLLM's signed 64-bit range")
	} else if math.IsNaN(c.Temperature) || math.IsInf(c.Temperature, 0) || (c.Temperature != 0 && (c.Temperature < 0.01 || c.Temperature > 2)) {
		return completionChoice{}, fmt.Errorf("temperature must be zero or between 0.01 and 2")
	}
	if stopIDs == nil {
		stopIDs = []uint32{}
	}
	params := map[string]any{
		"model":              c.Model,
		"prompt":             contextIDs,
		"seed":               seed,
		"temperature":        c.Temperature,
		"max_tokens":         maxTokens,
		"stop_token_ids":     stopIDs,
		"stop":               []string{},
		"ignore_eos":         true,
		"min_tokens":         0,
		"return_token_ids":   true,
		"add_special_tokens": false,
		"top_p":              1,
		"top_k":              0,
		"min_p":              0,
		"repetition_penalty": 1,
		"presence_penalty":   0,
		"frequency_penalty":  0,
		"n":                  1,
		"stream":             false,
		"echo":               scoring,
	}
	if scoring {
		params["prompt_logprobs"] = 2
	}
	if logprobs {
		params["logprobs"] = 2
		params["return_tokens_as_token_ids"] = true
	}
	body, err := json.Marshal(params)
	if err != nil {
		return completionChoice{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint, bytes.NewReader(body))
	if err != nil {
		return completionChoice{}, err
	} else if (req.URL.Scheme != "http" && req.URL.Scheme != "https") || req.URL.Hostname() == "" {
		return completionChoice{}, fmt.Errorf("endpoint must be an absolute HTTP or HTTPS URL")
	}
	req.Header.Set("Content-Type", "application/json")
	cl := c.HTTP
	if cl == nil {
		cl = http.DefaultClient
	}
	resp, err := cl.Do(req)
	if err != nil {
		return completionChoice{}, fmt.Errorf("vLLM request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return completionChoice{}, fmt.Errorf("vLLM HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var out struct {
		Choices []completionChoice `json:"choices"`
	}
	var bodyReader io.Reader = resp.Body
	var limited *io.LimitedReader
	if scoring {
		limited = &io.LimitedReader{R: resp.Body, N: int64(len(contextIDs))*4096 + 16385}
		bodyReader = limited
	} else if logprobs {
		// Include echoed prompt IDs and bounded top-two rows. Saturate the
		// arithmetic for unusually large caller-supplied token budgets.
		limit := 16385 + min(uint64(len(contextIDs)), (math.MaxInt64-16385)/16)*16
		limit += min(uint64(maxTokens), (math.MaxInt64-limit)/4096) * 4096
		limited = &io.LimitedReader{R: resp.Body, N: int64(limit)}
		bodyReader = limited
	}
	d := json.NewDecoder(bodyReader)
	if err := d.Decode(&out); err != nil {
		return completionChoice{}, fmt.Errorf("vLLM response: %w", err)
	} else if err := d.Decode(new(any)); err != io.EOF {
		if err != nil {
			return completionChoice{}, fmt.Errorf("vLLM response: %w", err)
		}
		return completionChoice{}, fmt.Errorf("vLLM response contains trailing data")
	} else if limited != nil && limited.N == 0 {
		if scoring {
			return completionChoice{}, fmt.Errorf("vLLM scoring response too large")
		}
		return completionChoice{}, fmt.Errorf("vLLM logprobs response too large")
	} else if len(out.Choices) != 1 || out.Choices[0].Index == nil || *out.Choices[0].Index != 0 {
		return completionChoice{}, fmt.Errorf("vLLM must return exactly one choice with index zero")
	}
	ch := out.Choices[0]
	if len(ch.PromptTokenIDs) != len(contextIDs) {
		return completionChoice{}, fmt.Errorf("vLLM changed the prompt tokens")
	}
	for i, id := range ch.PromptTokenIDs {
		if id == nil || *id != contextIDs[i] {
			return completionChoice{}, fmt.Errorf("vLLM changed the prompt token at position %d", i)
		}
	}
	return ch, nil
}

func (c *Client) Generate(ctx context.Context, contextIDs, stopIDs []uint32, seed uint64, maxTokens int) ([]uint32, bool, error) {
	ch, err := c.generateChoice(ctx, contextIDs, stopIDs, seed, maxTokens, false)
	if err != nil {
		return nil, false, err
	}
	return generationOutput(ch, stopIDs, maxTokens)
}

// GenerateWithLogprobs generates as Generate does, additionally requesting the
// chosen token's score and top two scores at every output position. The server
// must use raw_logprobs mode (the vLLM V1 default) for scores before temperature
// scaling and penalties. This mode cannot be established from the response.
// vLLM clips log probabilities below -9999 to -9999 in its completion response.
func (c *Client) GenerateWithLogprobs(ctx context.Context, contextIDs, stopIDs []uint32, seed uint64, maxTokens int) (tokens []uint32, stopped bool, logprobs []TokenLogprobs, err error) {
	ch, err := c.generateChoice(ctx, contextIDs, stopIDs, seed, maxTokens, true)
	if err != nil {
		return nil, false, nil, err
	}
	tokens, stopped, err = generationOutput(ch, stopIDs, maxTokens)
	if err != nil {
		return nil, false, nil, err
	}
	lp := ch.Logprobs
	if lp == nil || len(lp.Tokens) != len(tokens) || len(lp.TokenLogprobs) != len(tokens) || len(lp.TopLogprobs) != len(tokens) {
		return nil, false, nil, fmt.Errorf("vLLM returned misaligned generated-token logprobs")
	}
	logprobs = make([]TokenLogprobs, len(tokens))
	for i, token := range tokens {
		if lp.Tokens[i] == nil || *lp.Tokens[i] != "token_id:"+strconv.FormatUint(uint64(token), 10) {
			return nil, false, nil, fmt.Errorf("vLLM logprobs changed the token at position %d", i)
		}
		validScore := func(score *float64) bool {
			return score != nil && !math.IsNaN(*score) && !math.IsInf(*score, 0) && *score <= 0
		}
		row := lp.TopLogprobs[i]
		if !validScore(lp.TokenLogprobs[i]) || (len(row) != 2 && len(row) != 3) {
			return nil, false, nil, fmt.Errorf("vLLM returned invalid generated-token logprobs at position %d", i)
		}
		top := make(map[uint32]float64, len(row))
		for name, score := range row {
			id, err := strconv.ParseUint(strings.TrimPrefix(name, "token_id:"), 10, 32)
			if err != nil || name != "token_id:"+strconv.FormatUint(id, 10) || !validScore(score) {
				return nil, false, nil, fmt.Errorf("vLLM returned an invalid top logprob at position %d", i)
			}
			top[uint32(id)] = *score
		}
		chosen, ok := top[token]
		if !ok || chosen != *lp.TokenLogprobs[i] {
			return nil, false, nil, fmt.Errorf("vLLM returned an inconsistent chosen-token logprob at position %d", i)
		}
		if len(top) == 3 {
			for id, score := range top {
				if id != token && score < chosen {
					return nil, false, nil, fmt.Errorf("vLLM returned inconsistent top logprobs at position %d", i)
				}
			}
		}
		logprobs[i] = TokenLogprobs{Logprob: chosen, Top: top}
	}
	return tokens, stopped, logprobs, nil
}

func (c *Client) generateChoice(ctx context.Context, contextIDs, stopIDs []uint32, seed uint64, maxTokens int, logprobs bool) (completionChoice, error) {
	if err := ctx.Err(); err != nil {
		return completionChoice{}, err
	} else if len(contextIDs) == 0 || maxTokens <= 0 {
		return completionChoice{}, fmt.Errorf("context and token budget must be non-empty")
	}
	return c.complete(ctx, contextIDs, stopIDs, seed, maxTokens, false, logprobs)
}

func generationOutput(ch completionChoice, stopIDs []uint32, maxTokens int) ([]uint32, bool, error) {
	if len(ch.TokenIDs) == 0 || len(ch.TokenIDs) > maxTokens {
		return nil, false, fmt.Errorf("vLLM returned an invalid output token count")
	}
	output := make([]uint32, len(ch.TokenIDs))
	for i, id := range ch.TokenIDs {
		if id == nil {
			return nil, false, fmt.Errorf("vLLM returned a null token at position %d", i)
		} else if i < len(output)-1 && slices.Contains(stopIDs, *id) {
			return nil, false, fmt.Errorf("vLLM returned tokens after a stop token")
		}
		output[i] = *id
	}
	last := output[len(output)-1]
	switch ch.FinishReason {
	case "stop":
		if ch.StopReason == nil || *ch.StopReason != last || !slices.Contains(stopIDs, last) {
			return nil, false, fmt.Errorf("vLLM returned an unexpected stop")
		}
		return output, true, nil
	case "length":
		if ch.StopReason != nil || slices.Contains(stopIDs, last) || len(output) != maxTokens {
			return nil, false, fmt.Errorf("vLLM returned an inconsistent length termination")
		}
		return output, false, nil
	default:
		return nil, false, fmt.Errorf("vLLM returned unexpected finish reason %q", ch.FinishReason)
	}
}
