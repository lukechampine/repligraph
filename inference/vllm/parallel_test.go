package vllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"lukechampine.com/repligraph/inference"
)

func parallelRow(id uint32) string {
	return fmt.Sprintf(`{"%d":{"logprob":-1,"rank":1},"42":{"logprob":-2,"rank":2}}`, id)
}

func parallelResponse(prompt []uint32, rows ...string) string {
	ids, _ := json.Marshal(prompt)
	return fmt.Sprintf(`{"choices":[{"index":0,"prompt_token_ids":%s,"prompt_logprobs":[%s],"token_ids":[999],"finish_reason":"length"}]}`, ids, strings.Join(rows, ","))
}

func TestParallelVerifierRequest(t *testing.T) {
	var calls atomic.Int32
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var req map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
		}
		for key, want := range map[string]string{
			"model": `"model"`, "prompt": "[1,2,3,8]", "max_tokens": "0", "min_tokens": "0",
			"seed": "9007199254740993", "temperature": "0", "stop_token_ids": "[8,9]", "stop": "[]",
			"ignore_eos": "true", "return_token_ids": "true", "add_special_tokens": "false",
			"top_p": "1", "top_k": "0", "min_p": "0", "repetition_penalty": "1",
			"frequency_penalty": "0", "presence_penalty": "0", "n": "1", "stream": "false",
			"echo": "true", "prompt_logprobs": "2",
		} {
			if string(req[key]) != want {
				t.Errorf("%s = %s; want %s", key, req[key], want)
			}
		}
		if _, ok := req["logprobs"]; ok {
			t.Error("scoring request includes generated-token logprobs")
		}
		io.WriteString(w, parallelResponse([]uint32{1, 2, 3, 8}, "null", parallelRow(2), parallelRow(3), parallelRow(8)))
	})
	const seed = 1<<53 + 1
	v := &ParallelVerifier{Client: c, MaxScoringTokens: 4}
	input := make([]uint32, 2, 4)
	copy(input, []uint32{1, 2})
	copy(input[2:cap(input)], []uint32{77, 78})
	expected, stops := []uint32{3, 8}, []uint32{8, 9}
	if err := v.Verify(context.Background(), input, expected, stops, seed); err != nil {
		t.Fatal(err)
	}
	if err := inference.Verify(context.Background(), v, input, expected, stops, seed); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 || !slices.Equal(input[:cap(input)], []uint32{1, 2, 77, 78}) || !slices.Equal(expected, []uint32{3, 8}) || !slices.Equal(stops, []uint32{8, 9}) {
		t.Fatal("wrong call count or mutated caller inputs")
	}
}

func TestParallelVerifierMismatch(t *testing.T) {
	for i := range 3 {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			rows := []string{"null", parallelRow(2), parallelRow(3), parallelRow(4), parallelRow(8)}
			candidate := []uint32{3, 4, 8}[i]
			rows[2+i] = fmt.Sprintf(`{"%d":{"logprob":-2,"rank":2},"42":{"logprob":-1,"rank":1}}`, candidate)
			var calls atomic.Int32
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				io.WriteString(w, parallelResponse([]uint32{1, 2, 3, 4, 8}, rows...))
			})
			v := &ParallelVerifier{Client: c, MaxScoringTokens: 16}
			if err := v.Verify(context.Background(), []uint32{1, 2}, []uint32{3, 4, 8}, []uint32{8, 9}, 0); err == nil {
				t.Fatal("accepted non-greedy candidate")
			}
			if calls.Load() != 1 {
				t.Fatal("mismatch fell back to generation")
			}
		})
	}
}

func TestParallelVerifierMalformed(t *testing.T) {
	prompt := []uint32{1, 2, 3, 8}
	valid := parallelResponse(prompt, "null", parallelRow(2), parallelRow(3), parallelRow(8))
	responses := map[string]string{
		"missing response": "", "null response": "null", "no choices": `{"choices":[]}`,
		"two choices":       strings.Replace(valid, `]}`, `,{}]}`, 1),
		"missing index":     strings.Replace(valid, `"index":0,`, "", 1),
		"wrong index":       strings.Replace(valid, `"index":0`, `"index":1`, 1),
		"null prompt ID":    strings.Replace(valid, `[1,2,3,8]`, `[1,2,null,8]`, 1),
		"changed prompt":    strings.Replace(valid, `[1,2,3,8]`, `[1,2,4,8]`, 1),
		"truncated prompt":  strings.Replace(valid, `[1,2,3,8]`, `[1,2,3]`, 1),
		"missing scores":    strings.Replace(valid, `"prompt_logprobs"`, `"unused"`, 1),
		"short scores":      parallelResponse(prompt, "null", parallelRow(2), parallelRow(3)),
		"long scores":       parallelResponse(prompt, "null", parallelRow(2), parallelRow(3), parallelRow(8), parallelRow(9)),
		"nonnull first row": parallelResponse(prompt, parallelRow(1), parallelRow(2), parallelRow(3), parallelRow(8)),
		"trailing JSON":     valid + `{}`, "trailing junk": valid + ` broken`,
		"oversized body": valid + strings.Repeat(" ", 1<<20),
	}
	for name, row := range map[string]string{
		"null row": `null`, "empty row": `{}`, "one score": `{"3":{"logprob":-1,"rank":1}}`,
		"missing candidate":     `{"4":{"logprob":-1,"rank":1},"42":{"logprob":-2,"rank":2}}`,
		"null score":            `{"3":null,"42":{"logprob":-2,"rank":2}}`,
		"missing logprob":       `{"3":{"rank":1},"42":{"logprob":-2,"rank":2}}`,
		"null logprob":          `{"3":{"logprob":null,"rank":1},"42":{"logprob":-2,"rank":2}}`,
		"overflow logprob":      `{"3":{"logprob":1e999,"rank":1},"42":{"logprob":-2,"rank":2}}`,
		"positive logprob":      `{"3":{"logprob":0.1,"rank":1},"42":{"logprob":-2,"rank":2}}`,
		"missing rank":          `{"3":{"logprob":-1},"42":{"logprob":-2,"rank":2}}`,
		"null rank":             `{"3":{"logprob":-1,"rank":null},"42":{"logprob":-2,"rank":2}}`,
		"zero rank":             `{"3":{"logprob":-1,"rank":0},"42":{"logprob":-2,"rank":2}}`,
		"negative rank":         `{"3":{"logprob":-1,"rank":-1},"42":{"logprob":-2,"rank":2}}`,
		"duplicate rank":        `{"3":{"logprob":-1,"rank":1},"42":{"logprob":-2,"rank":1}}`,
		"missing rank two":      `{"3":{"logprob":-1,"rank":1},"42":{"logprob":-2,"rank":3}}`,
		"rank score inversion":  `{"3":{"logprob":-2,"rank":1},"42":{"logprob":-1,"rank":2}}`,
		"third score inversion": `{"3":{"logprob":-1,"rank":1},"42":{"logprob":-3,"rank":2},"43":{"logprob":-2,"rank":3}}`,
		"negative token":        `{"-1":{"logprob":-1,"rank":1},"42":{"logprob":-2,"rank":2}}`,
		"too many scores":       `{"3":{"logprob":-1,"rank":1},"42":{"logprob":-2,"rank":2},"43":{"logprob":-3,"rank":3},"44":{"logprob":-4,"rank":4}}`,
	} {
		responses[name] = parallelResponse(prompt, "null", parallelRow(2), row, parallelRow(8))
	}
	responses["malformed after tie"] = parallelResponse(prompt, "null", parallelRow(2), `{"3":{"logprob":-1,"rank":1},"42":{"logprob":-1,"rank":2}}`, `null`)
	for name, response := range responses {
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				io.WriteString(w, response)
			})
			v := &ParallelVerifier{Client: c, MaxScoringTokens: 16}
			if err := v.Verify(context.Background(), []uint32{1, 2}, []uint32{3, 8}, []uint32{8}, 0); err == nil {
				t.Fatal("accepted malformed score response")
			}
			if calls.Load() != 1 {
				t.Fatal("malformed response fell back to generation")
			}
		})
	}
}

func TestParallelVerifierFallback(t *testing.T) {
	for _, tc := range []struct {
		name        string
		temperature float64
		limit       int
		tieRank     int
		mismatch    bool
	}{
		{"sampling", 0.7, 16, 0, false}, {"oversize", 0, 3, 0, false}, {"disabled", 0, 0, 0, false},
		{"tied rank one", 0, 16, 1, false}, {"tied rank two", 0, 16, 2, false},
		{"tied rank two mismatch", 0, 16, 2, true},
		{"three-way tie", 0, 16, 3, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				n := calls.Add(1)
				var req map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				if tc.tieRank != 0 && n == 1 {
					if string(req["echo"]) != "true" {
						t.Error("tie case did not score first")
					}
					row := fmt.Sprintf(`{"3":{"logprob":-1,"rank":%d},"42":{"logprob":-1,"rank":%d}}`, tc.tieRank, 3-tc.tieRank)
					if tc.tieRank == 3 {
						row = `{"3":{"logprob":-1,"rank":1},"42":{"logprob":-1,"rank":1},"43":{"logprob":-1,"rank":2}}`
					}
					io.WriteString(w, parallelResponse([]uint32{1, 2, 3, 8}, "null", parallelRow(2), row, parallelRow(8)))
					return
				}
				if string(req["prompt"]) != "[1,2]" || string(req["echo"]) != "false" || string(req["max_tokens"]) != "2" || string(req["seed"]) != "42" || req["prompt_logprobs"] != nil {
					t.Errorf("fallback did not generate original whole turn: %s", req)
				}
				first := 3
				if tc.mismatch {
					first = 4
				}
				fmt.Fprintf(w, `{"choices":[{"index":0,"prompt_token_ids":[1,2],"token_ids":[%d,8],"finish_reason":"stop","stop_reason":8}]}`, first)
			})
			c.Temperature = tc.temperature
			v := &ParallelVerifier{Client: c, MaxScoringTokens: tc.limit}
			err := inference.Verify(context.Background(), v, []uint32{1, 2}, []uint32{3, 8}, []uint32{8}, 42)
			if (err != nil) != tc.mismatch {
				t.Fatalf("got %v; mismatch=%t", err, tc.mismatch)
			}
			wantCalls := int32(1)
			if tc.tieRank != 0 {
				wantCalls++
			}
			if calls.Load() != wantCalls {
				t.Fatalf("got %d calls; want %d", calls.Load(), wantCalls)
			}
		})
	}
}

func TestParallelVerifierValidation(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, tc := range []struct {
		name                   string
		ctx                    context.Context
		input, expected, stops []uint32
		limit                  int
		temperature            float64
	}{
		{"canceled", canceled, []uint32{1}, []uint32{8}, []uint32{8}, 16, 0},
		{"empty context", context.Background(), nil, []uint32{8}, []uint32{8}, 16, 0},
		{"empty expected", context.Background(), []uint32{1}, nil, []uint32{8}, 16, 0},
		{"no terminal stop", context.Background(), []uint32{1}, []uint32{3}, []uint32{8}, 16, 0},
		{"interior stop", context.Background(), []uint32{1}, []uint32{8, 8}, []uint32{8}, 16, 0},
		{"negative limit", context.Background(), []uint32{1}, []uint32{8}, []uint32{8}, -1, 0},
		{"NaN temperature", context.Background(), []uint32{1}, []uint32{8}, []uint32{8}, 16, math.NaN()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t, func(http.ResponseWriter, *http.Request) { t.Error("invalid input reached server") })
			c.Temperature = tc.temperature
			v := &ParallelVerifier{Client: c, MaxScoringTokens: tc.limit}
			if err := v.Verify(tc.ctx, tc.input, tc.expected, tc.stops, 0); err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
	if err := (&ParallelVerifier{MaxScoringTokens: 16}).Verify(context.Background(), []uint32{1}, []uint32{8}, []uint32{8}, 0); err == nil {
		t.Fatal("accepted nil client")
	}
}

func TestParallelVerifierRequestErrors(t *testing.T) {
	t.Run("HTTP", func(t *testing.T) {
		var calls atomic.Int32
		c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			http.Error(w, "scoring unavailable", http.StatusServiceUnavailable)
		})
		v := &ParallelVerifier{Client: c, MaxScoringTokens: 16}
		if err := v.Verify(context.Background(), []uint32{1}, []uint32{8}, []uint32{8}, 0); err == nil || !strings.Contains(err.Error(), "503") {
			t.Fatalf("missing HTTP failure: %v", err)
		}
		if calls.Load() != 1 {
			t.Fatal("HTTP error fell back to generation")
		}
	})
	t.Run("transport", func(t *testing.T) {
		want := errors.New("scoring transport failure")
		var calls atomic.Int32
		c := &Client{Endpoint: "http://unused", Model: "model", HTTP: &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, want
		})}}
		v := &ParallelVerifier{Client: c, MaxScoringTokens: 16}
		if err := v.Verify(context.Background(), []uint32{1}, []uint32{8}, []uint32{8}, 0); !errors.Is(err, want) {
			t.Fatalf("got %v; want transport failure", err)
		}
		if calls.Load() != 1 {
			t.Fatal("transport error fell back to generation")
		}
	})
	t.Run("cancellation", func(t *testing.T) {
		started := make(chan struct{})
		c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
			io.Copy(io.Discard, r.Body)
			close(started)
			<-r.Context().Done()
		})
		v := &ParallelVerifier{Client: c, MaxScoringTokens: 16}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		done := make(chan error, 1)
		go func() { done <- v.Verify(ctx, []uint32{1}, []uint32{8}, []uint32{8}, 0) }()
		select {
		case <-started:
		case err := <-done:
			t.Fatalf("request ended before reaching server: %v", err)
		}
		cancel()
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("got %v; want context.Canceled", err)
		}
	})
}
