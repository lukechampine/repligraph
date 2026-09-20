package vllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"lukechampine.com/repligraph/inference"
)

func testClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	s := httptest.NewServer(handler)
	t.Cleanup(s.Close)
	return &Client{Endpoint: s.URL + "/custom/completions?recipe=test", Model: "model", HTTP: s.Client()}
}

func TestGenerateRequest(t *testing.T) {
	for _, tc := range []struct {
		name        string
		seed        uint64
		temperature float64
		stops       []uint32
	}{
		{"greedy", 0, 0, []uint32{8, 9}},
		{"large seed", 1<<53 + 1, 0.7, []uint32{8, 9}},
		{"largest seed", math.MaxInt64, 2, []uint32{8, 9}},
		{"no stops", 42, 0.01, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.Method != http.MethodPost || r.URL.RequestURI() != "/custom/completions?recipe=test" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.RequestURI())
				}
				if r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("unexpected content type: %q", r.Header.Get("Content-Type"))
				}
				var req map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				stops := "[8,9]"
				if tc.stops == nil {
					stops = "[]"
				}
				for key, want := range map[string]string{
					"model": `"model"`, "prompt": "[1,4294967295]", "max_tokens": "2", "min_tokens": "0",
					"seed": strconv.FormatUint(tc.seed, 10), "temperature": strconv.FormatFloat(tc.temperature, 'f', -1, 64),
					"stop_token_ids": stops, "stop": "[]", "ignore_eos": "true", "return_token_ids": "true",
					"add_special_tokens": "false", "top_p": "1", "top_k": "0", "min_p": "0",
					"repetition_penalty": "1", "frequency_penalty": "0", "presence_penalty": "0",
					"n": "1", "stream": "false", "echo": "false",
				} {
					if got := string(req[key]); got != want {
						t.Errorf("%s = %s; want %s", key, got, want)
					}
				}
				if req["logprobs"] != nil || req["return_tokens_as_token_ids"] != nil {
					t.Error("ordinary request includes logprobs options")
				}
				if tc.stops == nil {
					fmt.Fprint(w, `{"choices":[{"index":0,"prompt_token_ids":[1,4294967295],"token_ids":[3,4],"finish_reason":"length","stop_reason":null}]}`)
				} else {
					fmt.Fprint(w, `{"choices":[{"index":0,"prompt_token_ids":[1,4294967295],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}]}`)
				}
			})
			c.Temperature = tc.temperature
			contextIDs := []uint32{1, math.MaxUint32}
			for range 2 {
				got, stopped, err := c.Generate(context.Background(), contextIDs, tc.stops, tc.seed, 2)
				want := []uint32{3, 8}
				if tc.stops == nil {
					want = []uint32{3, 4}
				}
				if err != nil || !slices.Equal(got, want) || stopped != (tc.stops != nil) {
					t.Fatalf("got %v, %t, %v", got, stopped, err)
				}
			}
			if calls.Load() != 2 || !slices.Equal(contextIDs, []uint32{1, math.MaxUint32}) {
				t.Fatal("incorrect call count or modified context")
			}
		})
	}
}

func TestGenerateInvalidInputs(t *testing.T) {
	for _, tc := range []struct {
		name        string
		contextIDs  []uint32
		maxTokens   int
		seed        uint64
		temperature float64
	}{
		{"empty context", nil, 2, 0, 0},
		{"zero budget", []uint32{1}, 0, 0, 0},
		{"negative budget", []uint32{1}, -1, 0, 0},
		{"seed overflow", []uint32{1}, 2, math.MaxInt64 + 1, 0},
		{"negative temperature", []uint32{1}, 2, 0, -1},
		{"high temperature", []uint32{1}, 2, 0, 2.01},
		{"NaN temperature", []uint32{1}, 2, 0, math.NaN()},
		{"infinite temperature", []uint32{1}, 2, 0, math.Inf(1)},
		{"clamped temperature", []uint32{1}, 2, 0, 0.005},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t, func(http.ResponseWriter, *http.Request) {
				t.Error("invalid inputs reached server")
			})
			c.Temperature = tc.temperature
			if _, _, err := c.Generate(context.Background(), tc.contextIDs, []uint32{8}, tc.seed, tc.maxTokens); err == nil {
				t.Fatal("accepted invalid inputs")
			}
		})
	}
}

func TestGenerateResponses(t *testing.T) {
	for _, tc := range []struct {
		name     string
		response string
		want     []uint32
		stopped  bool
	}{
		{"stop", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, []uint32{3, 8}, true},
		{"immediate stop", `{"index":0,"prompt_token_ids":[1],"token_ids":[8],"finish_reason":"stop","stop_reason":8}`, []uint32{8}, true},
		{"length", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,4294967295],"finish_reason":"length","stop_reason":null}`, []uint32{3, math.MaxUint32}, false},
		{"missing prompt", `{"index":0,"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"null prompt", `{"index":0,"prompt_token_ids":null,"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"changed prompt", `{"index":0,"prompt_token_ids":[2],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"truncated prompt", `{"index":0,"prompt_token_ids":[],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"extended prompt", `{"index":0,"prompt_token_ids":[1,2],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"null prompt token", `{"index":0,"prompt_token_ids":[null],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"overflowing prompt token", `{"index":0,"prompt_token_ids":[4294967297],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"fractional prompt token", `{"index":0,"prompt_token_ids":[1.5],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"missing tokens", `{"index":0,"prompt_token_ids":[1],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"null tokens", `{"index":0,"prompt_token_ids":[1],"token_ids":null,"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"empty tokens", `{"index":0,"prompt_token_ids":[1],"token_ids":[],"finish_reason":"length"}`, nil, false},
		{"null token", `{"index":0,"prompt_token_ids":[1],"token_ids":[null,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"negative token", `{"index":0,"prompt_token_ids":[1],"token_ids":[-1,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"overflowing token", `{"index":0,"prompt_token_ids":[1],"token_ids":[4294967296,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"fractional token", `{"index":0,"prompt_token_ids":[1],"token_ids":[3.5,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"string token", `{"index":0,"prompt_token_ids":[1],"token_ids":["3",8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"over budget", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,4,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"interior stop", `{"index":0,"prompt_token_ids":[1],"token_ids":[8,9],"finish_reason":"stop","stop_reason":9}`, nil, false},
		{"unexpected stop", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,7],"finish_reason":"stop","stop_reason":7}`, nil, false},
		{"mismatched stop", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":9}`, nil, false},
		{"missing stop reason", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop"}`, nil, false},
		{"null stop reason", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":null}`, nil, false},
		{"string stop reason", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":"8"}`, nil, false},
		{"fractional stop reason", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8.5}`, nil, false},
		{"overflowing stop reason", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":4294967304}`, nil, false},
		{"early length", `{"index":0,"prompt_token_ids":[1],"token_ids":[3],"finish_reason":"length","stop_reason":null}`, nil, false},
		{"length with stop", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"length","stop_reason":null}`, nil, false},
		{"length with reason", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,4],"finish_reason":"length","stop_reason":8}`, nil, false},
		{"unknown finish", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"content_filter","stop_reason":8}`, nil, false},
		{"missing finish", `{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"stop_reason":8}`, nil, false},
		{"wrong index", `{"index":1,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"missing index", `{"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
		{"null index", `{"index":null,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}`, nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprintf(w, `{"choices":[%s]}`, tc.response)
			})
			got, stopped, err := c.Generate(context.Background(), []uint32{1}, []uint32{8, 9}, 0, 2)
			if tc.want == nil {
				if err == nil {
					t.Fatalf("accepted invalid response: %v, %t", got, stopped)
				}
			} else if err != nil || !slices.Equal(got, tc.want) || stopped != tc.stopped {
				t.Fatalf("got %v, %t, %v; want %v, %t", got, stopped, err, tc.want, tc.stopped)
			}
		})
	}
}

func TestGenerateMalformedResponse(t *testing.T) {
	choice := `{"index":0,"prompt_token_ids":[1],"token_ids":[8],"finish_reason":"stop","stop_reason":8}`
	for _, response := range []string{
		``, `null`, `{}`, `{"choices":null}`, `{"choices":[]}`, `{"choices":[null]}`,
		`{"choices":[` + choice + `,` + choice + `]}`,
		`{"choices":[` + choice + `]}{}`, `{"choices":[` + choice + `]} garbage`,
	} {
		t.Run(response, func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, response) })
			if _, _, err := c.Generate(context.Background(), []uint32{1}, []uint32{8}, 0, 2); err == nil {
				t.Fatal("accepted malformed response")
			}
		})
	}
}

func TestGenerateHTTPError(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, "invalid model: "+strings.Repeat("x", 1<<16))
	})
	_, _, err := c.Generate(context.Background(), []uint32{1}, []uint32{8}, 0, 2)
	if err == nil || !strings.Contains(err.Error(), "400") || !strings.Contains(err.Error(), "invalid model") {
		t.Fatalf("missing HTTP error details: %v", err)
	} else if len(err.Error()) > 16<<10 {
		t.Fatalf("unbounded HTTP error: %d bytes", len(err.Error()))
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestGenerateTransportError(t *testing.T) {
	want := errors.New("transport failure")
	c := &Client{Endpoint: "http://unused", Model: "model", HTTP: &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return nil, want
	})}}
	if _, _, err := c.Generate(context.Background(), []uint32{1}, []uint32{8}, 0, 2); !errors.Is(err, want) {
		t.Fatalf("got %v; want wrapped transport error", err)
	}
}

func TestGenerateInvalidConfiguration(t *testing.T) {
	for _, tc := range []struct {
		endpoint string
		model    string
	}{
		{"http://unused", ""},
		{"", "model"},
		{"/v1/completions", "model"},
		{"ftp://unused", "model"},
		{"http://", "model"},
	} {
		c := &Client{Endpoint: tc.endpoint, Model: tc.model, HTTP: &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			t.Error("invalid configuration reached transport")
			return nil, errors.New("unexpected request")
		})}}
		if _, _, err := c.Generate(context.Background(), []uint32{1}, []uint32{8}, 0, 2); err == nil {
			t.Fatalf("accepted configuration: %+v", tc)
		}
	}
}

func TestGenerateCancellation(t *testing.T) {
	started := make(chan struct{})
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		close(started)
		<-r.Context().Done()
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, _, err := c.Generate(ctx, []uint32{1}, []uint32{8}, 0, 2)
		done <- err
	}()
	select {
	case <-started:
	case err := <-done:
		t.Fatalf("request ended before reaching server: %v", err)
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v; want context.Canceled", err)
	}
	if _, _, err := c.Generate(ctx, []uint32{1}, []uint32{8}, 0, 2); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v; want context.Canceled before sending request", err)
	}
}

func TestVerify(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8}]}`)
	})
	c.HTTP = nil
	if err := inference.Verify(context.Background(), c, []uint32{1}, []uint32{3, 8}, []uint32{8}, 0); err != nil {
		t.Fatal(err)
	}
	if err := inference.Verify(context.Background(), c, []uint32{1}, []uint32{4, 8}, []uint32{8}, 0); err == nil {
		t.Fatal("accepted tampered output")
	}
}

func TestGenerateWithLogprobs(t *testing.T) {
	for _, tc := range []struct {
		name        string
		temperature float64
		tokens      []uint32
		stopped     bool
		rows        []TokenLogprobs
	}{
		{"greedy", 0, []uint32{3, 8}, true, []TokenLogprobs{{-0.25, map[uint32]float64{3: -0.25, 4: -2}}, {-0.5, map[uint32]float64{8: -0.5, 4: -1}}}},
		{"sample outside top two", 0.7, []uint32{3, 8}, true, []TokenLogprobs{{-4, map[uint32]float64{3: -4, 4: -0.5, 5: -1.5}}, {-1, map[uint32]float64{8: -1, 4: -0.5}}}},
		{"length and tie", 0.7, []uint32{3, math.MaxUint32}, false, []TokenLogprobs{{-1, map[uint32]float64{3: -1, 4: -1}}, {-9999, map[uint32]float64{math.MaxUint32: -9999, 4: 0, 5: -20}}}},
		{"immediate stop", 0, []uint32{8}, true, []TokenLogprobs{{0, map[uint32]float64{8: 0, 4: -9999}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				var req map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Fatal(err)
				}
				for key, want := range map[string]string{"logprobs": "2", "return_tokens_as_token_ids": "true", "return_token_ids": "true", "echo": "false", "prompt": "[1]", "max_tokens": "2", "seed": "42", "stop_token_ids": "[8]"} {
					if string(req[key]) != want {
						t.Errorf("%s = %s; want %s", key, req[key], want)
					}
				}
				if req["prompt_logprobs"] != nil {
					t.Error("generation requested prompt scoring")
				}
				var lp completionLogprobs
				for i, token := range tc.tokens {
					lp.Tokens = append(lp.Tokens, new("token_id:"+strconv.FormatUint(uint64(token), 10)))
					lp.TokenLogprobs = append(lp.TokenLogprobs, new(tc.rows[i].Logprob))
					row := make(map[string]*float64)
					for id, score := range tc.rows[i].Top {
						row["token_id:"+strconv.FormatUint(uint64(id), 10)] = new(score)
					}
					lp.TopLogprobs = append(lp.TopLogprobs, row)
				}
				choice := map[string]any{"index": 0, "prompt_token_ids": []uint32{1}, "token_ids": tc.tokens, "finish_reason": "length", "logprobs": lp}
				if tc.stopped {
					choice["finish_reason"], choice["stop_reason"] = "stop", tc.tokens[len(tc.tokens)-1]
				}
				json.NewEncoder(w).Encode(map[string]any{"choices": []any{choice}})
			})
			c.Temperature = tc.temperature
			tokens, stopped, rows, err := c.GenerateWithLogprobs(t.Context(), []uint32{1}, []uint32{8}, 42, 2)
			if err != nil || !slices.Equal(tokens, tc.tokens) || stopped != tc.stopped || !reflect.DeepEqual(rows, tc.rows) {
				t.Fatalf("got %v, %t, %+v, %v; want %v, %t, %+v", tokens, stopped, rows, err, tc.tokens, tc.stopped, tc.rows)
			}
		})
	}
}

func TestGenerateWithLogprobsRejectsMalformedScores(t *testing.T) {
	const valid = `{"choices":[{"index":0,"prompt_token_ids":[1],"token_ids":[3,8],"finish_reason":"stop","stop_reason":8,"logprobs":{"tokens":["token_id:3","token_id:8"],"token_logprobs":[-2,-0.5],"top_logprobs":[{"token_id:3":-2,"token_id:4":-1},{"token_id:8":-0.5,"token_id:4":-1}]}}]}`
	for _, tc := range []struct{ name, from, to string }{
		{"missing scores", `"logprobs":`, `"unused":`},
		{"null score object", `"logprobs":{`, `"logprobs":null,"unused":{`},
		{"null tokens", `"tokens":["token_id:3","token_id:8"]`, `"tokens":null`},
		{"truncated tokens", `"tokens":["token_id:3","token_id:8"]`, `"tokens":["token_id:3"]`},
		{"mismatched token", `"tokens":["token_id:3"`, `"tokens":["token_id:4"`},
		{"null token", `"tokens":["token_id:3"`, `"tokens":[null`},
		{"decoded text token", `"tokens":["token_id:3"`, `"tokens":["word"`},
		{"missing stop score", `"token_logprobs":[-2,-0.5]`, `"token_logprobs":[-2]`},
		{"extra score", `"token_logprobs":[-2,-0.5]`, `"token_logprobs":[-2,-0.5,-1]`},
		{"null chosen score", `"token_logprobs":[-2`, `"token_logprobs":[null`},
		{"positive chosen score", `"token_logprobs":[-2`, `"token_logprobs":[0.1`},
		{"overflow chosen score", `"token_logprobs":[-2`, `"token_logprobs":[1e999`},
		{"chosen score disagreement", `"token_logprobs":[-2`, `"token_logprobs":[-1`},
		{"null top row", `{"token_id:3":-2,"token_id:4":-1}`, `null`},
		{"missing top two", `{"token_id:3":-2,"token_id:4":-1}`, `{"token_id:3":-2}`},
		{"missing chosen token", `{"token_id:3":-2,"token_id:4":-1}`, `{"token_id:5":-2,"token_id:4":-1}`},
		{"too many top tokens", `{"token_id:3":-2,"token_id:4":-1}`, `{"token_id:3":-2,"token_id:4":-1,"token_id:5":-1,"token_id:6":-1}`},
		{"third token outranks second", `{"token_id:3":-2,"token_id:4":-1}`, `{"token_id:3":-2,"token_id:4":-1,"token_id:5":-3}`},
		{"null top score", `"token_id:4":-1`, `"token_id:4":null`},
		{"positive top score", `"token_id:4":-1`, `"token_id:4":0.1`},
		{"invalid top ID", `"token_id:4":-1`, `"word":-1`},
		{"overflow top ID", `"token_id:4":-1`, `"token_id:4294967296":-1`},
		{"noncanonical top ID", `"token_id:4":-1`, `"token_id:04":-1`},
		{"wrong stop reason", `"stop_reason":8`, `"stop_reason":3`},
		{"oversized response", valid, valid + strings.Repeat(" ", 1<<20)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				io.WriteString(w, strings.Replace(valid, tc.from, tc.to, 1))
			})
			tokens, stopped, rows, err := c.GenerateWithLogprobs(t.Context(), []uint32{1}, []uint32{8}, 0, 2)
			if err == nil || tokens != nil || stopped || rows != nil {
				t.Fatalf("invalid scores returned partial success: %v, %t, %v, %v", tokens, stopped, rows, err)
			}
		})
	}
}
