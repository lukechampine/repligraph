package harmony

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

func TestGolden(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Fatal(err)
	}
	blob, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Text   string   `json:"text"`
			IDs    []uint32 `json:"ids"`
			Pieces []string `json:"pieces"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(blob, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("empty fixture")
	}
	for _, tc := range fixture.Cases {
		var pieces []string
		for rs := []rune(tc.Text); len(rs) > 0; {
			n := match(rs)
			pieces = append(pieces, string(rs[:n]))
			rs = rs[n:]
		}
		if !slices.Equal(pieces, tc.Pieces) {
			t.Errorf("split(%q):\n got %q\nwant %q", tc.Text, pieces, tc.Pieces)
		}
		ids := c.Encode(tc.Text)
		if !slices.Equal(ids, tc.IDs) {
			t.Errorf("Encode(%q):\n got %v\nwant %v", tc.Text, ids, tc.IDs)
		}
		if back, err := c.Decode(ids); err != nil || back != tc.Text {
			t.Errorf("Decode(Encode(%q)) = %q, %v", tc.Text, back, err)
		}
	}
}

func TestSpecials(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range specials {
		id, ok := c.AddedID(s)
		if !ok || id != uint32(numOrdinary+i) {
			t.Fatalf("AddedID(%q) = %d, %v", s, id, ok)
		}
		if text, err := c.Decode([]uint32{id}); err != nil || text != s {
			t.Errorf("Decode(%d) = %q, %v", id, text, err)
		}
		for _, id := range c.Encode(s) {
			if id >= numOrdinary {
				t.Fatalf("Encode(%q) emitted special token %d", s, id)
			}
		}
	}
	if _, ok := c.AddedID("<|reserved_200019|>"); ok {
		t.Fatal("accepted an undeclared special token")
	}
	for _, id := range []uint32{200019, 201087, ^uint32(0)} {
		if _, err := c.Decode([]uint32{id}); err == nil {
			t.Fatalf("decoded undeclared token %d", id)
		}
	}
}
