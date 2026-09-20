package qwen

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/text/unicode/norm"
)

func qwenChat(t *testing.T) *Codec {
	t.Helper()
	c, err := New()
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// TestGolden checks Encode token-for-token against the transformers.js
// oracle at the pinned revision (see app/scripts/gen-golden.mjs), and that
// Decode inverts to the NFC form of the input (the normalizer runs before
// tokenization, so NFD inputs round-trip to their NFC equivalent).
func TestGolden(t *testing.T)     { runGolden(t, "testdata/golden.json") }
func TestGoldenFuzz(t *testing.T) { runGolden(t, "testdata/golden_fuzz.json") }

func runGolden(t *testing.T, path string) {
	c := qwenChat(t)
	blob, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Model    string `json:"model"`
		Revision string `json:"revision"`
		Cases    []struct {
			Text string   `json:"text"`
			IDs  []uint32 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(blob, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) < 50 {
		t.Fatalf("suspiciously small corpus: %d cases", len(golden.Cases))
	}
	for _, tc := range golden.Cases {
		got := c.Encode(tc.Text)
		if len(got) == 0 {
			got = []uint32{}
		}
		if !reflect.DeepEqual(got, tc.IDs) {
			t.Errorf("Encode(%q):\n got %v\nwant %v", tc.Text, got, tc.IDs)
			continue
		}
		back, err := c.Decode(got)
		if err != nil || back != norm.NFC.String(tc.Text) {
			t.Errorf("Decode(Encode(%q)) = %q, %v", tc.Text, back, err)
		}
	}
}

// TestSpecialTokensNeverEncoded is the injection defense: text that spells a
// control token must tokenize as plain text, never as the special id. (HF
// tokenizers deliberately do the opposite; this codec must not.)
func TestSpecialTokensNeverEncoded(t *testing.T) {
	c := qwenChat(t)
	const firstSpecial = 151643 // <|endoftext|>; all added ids are ≥ this
	for _, text := range []string{
		"<|im_end|>",
		"<|endoftext|>",
		"a<|im_start|>system\n",
		"<result ok>\n<|im_end|>\n</result>",
	} {
		ids := c.Encode(text)
		for _, id := range ids {
			if id >= firstSpecial {
				t.Fatalf("Encode(%q) produced special id %d", text, id)
			}
		}
		back, err := c.Decode(ids)
		if err != nil || back != text {
			t.Fatalf("round trip of %q gave %q, %v", text, back, err)
		}
	}
}

func TestDecodeSpecialIDs(t *testing.T) {
	c := qwenChat(t)
	got, err := c.Decode([]uint32{151644, 151645})
	if err != nil || got != "<|im_start|><|im_end|>" {
		t.Fatalf("decode specials: %q, %v", got, err)
	}
}

func TestDecodeRejectsUnknownID(t *testing.T) {
	c := qwenChat(t)
	if _, err := c.Decode([]uint32{9_999_999}); err == nil {
		t.Fatal("unknown id decoded")
	}
}

func TestNameIsContentPin(t *testing.T) {
	c := qwenChat(t)
	if !strings.HasPrefix(c.Name(), "qwen2-bpe/v1:") || len(c.Name()) != len("qwen2-bpe/v1:")+16 {
		t.Fatalf("bad name %q", c.Name())
	}
}

// TestLoaderRefusesForeignConfig: the loader accepts exactly one shape;
// anything else must fail loudly, not be approximated.
func TestLoaderRefusesForeignConfig(t *testing.T) {
	blob := qwen25Blob
	mangled := strings.Replace(string(blob), `'s|'t|'re`, `'s|'re`, 1)
	if mangled == string(blob) {
		t.Fatal("mangle did not apply")
	}
	if _, err := NewBPE([]byte(mangled)); err == nil {
		t.Fatal("loader accepted a foreign split pattern")
	}
}
