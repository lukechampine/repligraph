package inference

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

type engineFunc func(context.Context, []uint32, []uint32, uint64, int) ([]uint32, bool, error)

func (f engineFunc) Generate(ctx context.Context, input, stops []uint32, seed uint64, maxTokens int) ([]uint32, bool, error) {
	return f(ctx, input, stops, seed, maxTokens)
}

type verifierFunc func(context.Context, []uint32, []uint32, []uint32, uint64) error

func (f verifierFunc) Generate(context.Context, []uint32, []uint32, uint64, int) ([]uint32, bool, error) {
	panic("Generate called on verifier")
}

func (f verifierFunc) Verify(ctx context.Context, input, expected, stops []uint32, seed uint64) error {
	return f(ctx, input, expected, stops, seed)
}

func TestVerify(t *testing.T) {
	ctx := context.Background()
	contextIDs := []uint32{1, 2}
	stops := []uint32{8, 9}
	for _, output := range [][]uint32{{3, 4, 8}, {3, 4, 9}, {8}} {
		called := false
		eng := engineFunc(func(gotCtx context.Context, input, gotStops []uint32, _ uint64, maxTokens int) ([]uint32, bool, error) {
			called = true
			if gotCtx != ctx || !slices.Equal(input, contextIDs) || !slices.Equal(gotStops, stops) || maxTokens != len(output) {
				t.Fatalf("wrong generation inputs: context=%v stops=%v budget=%d", input, gotStops, maxTokens)
			}
			return slices.Clone(output), true, nil
		})
		if err := Verify(ctx, eng, contextIDs, output, stops, 0); err != nil {
			t.Fatal(err)
		} else if !called {
			t.Fatal("engine was not called")
		}
	}
}

func TestVerifyInvalidInput(t *testing.T) {
	for _, tc := range []struct {
		name       string
		contextIDs []uint32
		expected   []uint32
	}{
		{"empty context", nil, []uint32{3, 4, 8}},
		{"empty output", []uint32{1, 2}, nil},
		{"missing final stop", []uint32{1, 2}, []uint32{3, 4}},
		{"interior stop", []uint32{1, 2}, []uint32{3, 9, 8}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eng := engineFunc(func(context.Context, []uint32, []uint32, uint64, int) ([]uint32, bool, error) {
				t.Fatal("invalid turn reached engine")
				return nil, false, nil
			})
			if err := Verify(context.Background(), eng, tc.contextIDs, tc.expected, []uint32{8, 9}, 0); err == nil {
				t.Fatal("accepted invalid turn")
			}
		})
	}
	if err := Verify(context.Background(), nil, []uint32{1, 2}, []uint32{3, 4, 8}, []uint32{8, 9}, 0); err == nil {
		t.Fatal("accepted nil engine")
	}
}

func TestVerifyDivergentOutput(t *testing.T) {
	contextIDs, expected := []uint32{1, 2}, []uint32{3, 4, 8}
	for _, tc := range []struct {
		name    string
		output  []uint32
		stopped bool
	}{
		{"different token", []uint32{3, 5, 8}, true},
		{"different stop", []uint32{3, 4, 9}, true},
		{"short output", []uint32{3, 4}, false},
		{"long output", []uint32{3, 4, 8, 5}, true},
		{"not stopped", expected, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eng := engineFunc(func(context.Context, []uint32, []uint32, uint64, int) ([]uint32, bool, error) {
				return tc.output, tc.stopped, nil
			})
			if err := Verify(context.Background(), eng, contextIDs, expected, []uint32{8, 9}, 0); err == nil {
				t.Fatal("accepted divergent output")
			}
		})
	}
}

func TestVerifyEngineError(t *testing.T) {
	want := errors.New("engine failure")
	eng := engineFunc(func(context.Context, []uint32, []uint32, uint64, int) ([]uint32, bool, error) {
		return nil, false, want
	})
	if err := Verify(context.Background(), eng, []uint32{1}, []uint32{8}, []uint32{8}, 0); !errors.Is(err, want) {
		t.Fatalf("got %v; want wrapped engine error", err)
	}
}

func TestVerifyInputIsolation(t *testing.T) {
	contextIDs, expected := []uint32{1, 2}, []uint32{3, 8}
	stops := []uint32{8, 9}
	eng := engineFunc(func(_ context.Context, input, gotStops []uint32, _ uint64, _ int) ([]uint32, bool, error) {
		input[0] = 42
		gotStops[0] = 42
		return slices.Clone(expected), true, nil
	})
	if err := Verify(context.Background(), eng, contextIDs, expected, stops, 0); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(contextIDs, []uint32{1, 2}) || !slices.Equal(stops, []uint32{8, 9}) {
		t.Fatal("engine mutated caller inputs")
	}
}

func TestVerifyCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	eng := engineFunc(func(context.Context, []uint32, []uint32, uint64, int) ([]uint32, bool, error) {
		t.Fatal("canceled verification reached engine")
		return nil, false, nil
	})
	if err := Verify(ctx, eng, []uint32{1}, []uint32{8}, []uint32{8}, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v; want context.Canceled", err)
	}
}

func TestVerifyVerifier(t *testing.T) {
	ctx := context.Background()
	contextIDs, expected, stops := []uint32{1, 2}, []uint32{3, 8}, []uint32{8, 9}
	called := false
	eng := verifierFunc(func(gotCtx context.Context, input, output, gotStops []uint32, _ uint64) error {
		called = true
		if gotCtx != ctx || !slices.Equal(input, contextIDs) || !slices.Equal(output, expected) || !slices.Equal(gotStops, stops) {
			t.Fatalf("wrong verification inputs: context=%v output=%v stops=%v", input, output, gotStops)
		}
		input[0], output[0], gotStops[0] = 42, 42, 42
		return nil
	})
	if err := Verify(ctx, eng, contextIDs, expected, stops, 0); err != nil {
		t.Fatal(err)
	} else if !called {
		t.Fatal("verifier was not called")
	}
	if !slices.Equal(contextIDs, []uint32{1, 2}) || !slices.Equal(expected, []uint32{3, 8}) || !slices.Equal(stops, []uint32{8, 9}) {
		t.Fatal("verifier mutated caller inputs")
	}
}

func TestVerifyVerifierError(t *testing.T) {
	want := errors.New("verification failure")
	eng := verifierFunc(func(context.Context, []uint32, []uint32, []uint32, uint64) error {
		return want
	})
	if err := Verify(context.Background(), eng, []uint32{1}, []uint32{8}, []uint32{8}, 0); !errors.Is(err, want) {
		t.Fatalf("got %v; want wrapped verifier error", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eng = verifierFunc(func(context.Context, []uint32, []uint32, []uint32, uint64) error {
		cancel()
		return nil
	})
	if err := Verify(ctx, eng, []uint32{1}, []uint32{8}, []uint32{8}, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v; want context.Canceled after verifier returned", err)
	}
}

func TestVerifyVerifierValidation(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, tc := range []struct {
		name                          string
		ctx                           context.Context
		contextIDs, expected, stopIDs []uint32
	}{
		{"canceled", canceled, []uint32{1}, []uint32{8}, []uint32{8}},
		{"empty context", context.Background(), nil, []uint32{8}, []uint32{8}},
		{"empty output", context.Background(), []uint32{1}, nil, []uint32{8}},
		{"empty stops", context.Background(), []uint32{1}, []uint32{8}, nil},
		{"missing final stop", context.Background(), []uint32{1}, []uint32{3}, []uint32{8}},
		{"interior stop", context.Background(), []uint32{1}, []uint32{8, 8}, []uint32{8}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eng := verifierFunc(func(context.Context, []uint32, []uint32, []uint32, uint64) error {
				t.Fatal("invalid verification reached verifier")
				return nil
			})
			if err := Verify(tc.ctx, eng, tc.contextIDs, tc.expected, tc.stopIDs, 0); err == nil {
				t.Fatal("accepted invalid verification")
			}
		})
	}
}

func TestVerifySeed(t *testing.T) {
	contextIDs, expected, stops := []uint32{1, 2}, []uint32{3, 8}, []uint32{8, 9}
	for _, seed := range []uint64{0, 1, math.MaxInt64, 1 << 63, math.MaxUint64} {
		var got []uint64
		for _, eng := range []Engine{
			engineFunc(func(_ context.Context, _, _ []uint32, gotSeed uint64, _ int) ([]uint32, bool, error) {
				got = append(got, gotSeed)
				return slices.Clone(expected), true, nil
			}),
			verifierFunc(func(_ context.Context, _, _, _ []uint32, gotSeed uint64) error {
				got = append(got, gotSeed)
				return nil
			}),
		} {
			if err := Verify(context.Background(), eng, contextIDs, expected, stops, seed); err != nil {
				t.Fatal(err)
			}
		}
		if !slices.Equal(got, []uint64{seed, seed}) {
			t.Fatalf("engine and verifier got seeds %d; want %d twice", got, seed)
		}
	}
}
