package mistral

import (
	"encoding/json"
	"os"
	"testing"
)

func tekkenCodec(t *testing.T) *TekkenCodec {
	t.Helper()
	c, err := cached()
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// TestTekkenGolden diffs Encode token-for-token against mistral-common at the
// pinned model revision (testdata/golden_tekken.json).
func TestTekkenGolden(t *testing.T) {
	c := tekkenCodec(t)
	blob, err := os.ReadFile("testdata/golden_tekken.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Source string `json:"source"`
		Cases  []struct {
			Text string   `json:"text"`
			IDs  []uint32 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(blob, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("no golden cases")
	}
	for _, tc := range golden.Cases {
		got := c.Encode(tc.Text)
		if len(got) != len(tc.IDs) {
			t.Errorf("Encode(%q): %d ids, want %d\n got %v\nwant %v", tc.Text, len(got), len(tc.IDs), got, tc.IDs)
			continue
		}
		for i := range got {
			if got[i] != tc.IDs[i] {
				t.Errorf("Encode(%q): id mismatch at %d: got %d, want %d\n got %v\nwant %v",
					tc.Text, i, got[i], tc.IDs[i], got, tc.IDs)
				break
			}
		}
		// Decode round-trips (encode is not lossy for valid UTF-8).
		back, err := c.Decode(got)
		if err != nil || back != tc.Text {
			t.Errorf("Decode(Encode(%q)) = %q, %v", tc.Text, back, err)
		}
	}
}

// TestTekkenNeverEmitsSpecials: control strings in content tokenize as plain
// bytes — every emitted id is in the vocab range.
func TestTekkenNeverEmitsSpecials(t *testing.T) {
	c := tekkenCodec(t)
	for _, s := range []string{"<s>", "</s>", "[INST]", "[TOOL_CALLS]", "[TOOL_RESULTS]", "[SYSTEM_PROMPT]x[/SYSTEM_PROMPT]"} {
		for _, id := range c.Encode(s) {
			if id < NumSpecials {
				t.Fatalf("Encode(%q) emitted control id %d", s, id)
			}
		}
	}
}

func TestTekkenDecodeErrors(t *testing.T) {
	c := tekkenCodec(t)
	if _, err := c.Decode([]uint32{9_999_999}); err == nil {
		t.Fatal("decoded an out-of-vocabulary id")
	}
	// Named and filler specials decode to canonical text.
	s, err := c.Decode([]uint32{ToolCalls, 20})
	if err != nil || s != "[TOOL_CALLS]<SPECIAL_20>" {
		t.Fatalf("special decode: %q, %v", s, err)
	}
}
