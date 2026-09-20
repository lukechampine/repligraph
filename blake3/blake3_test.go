package blake3

import (
	"encoding/hex"
	"encoding/json"
	"testing"
)

// TestPinnedValues locks the v0 domains and NUL separator framing. These
// vectors were computed directly with the underlying BLAKE3 implementation,
// using explicit domain literals rather than this package's helpers.
func TestPinnedValues(t *testing.T) {
	vectors := []struct {
		name string
		got  Hash
		want string
	}{
		{"file", Sum(DomainTreeFile, []byte("hello")), "778cfbcbaf3b1ab9fc25aa73b94b07794fc937df0ed1211b0f5057555a03b146"},
		{"dir", Sum(DomainTreeDir, nil), "241efc080020330aabf90f66bab575c10a4c1dc777269c3b8d1c6824839dfefc"},
		{"module", Sum(DomainModule, []byte("\x00asm")), "a945e9875d51fa96ebd02a82ad1f673f876d9bed89bf6c60dd471b361307f035"},
		{"ctx", Sum(DomainContext, []byte{1, 0, 0, 0}), "625ef7c63b218237b6bbc5e8c72c3fbe48faa23232046bb9a840446ac6876715"},
		{"content", Sum256([]byte("hello")), "ea8f163db38682925e4491c5e58d4bb3506ef8c14eb78a86e908c5624a67200f"},
	}
	for _, v := range vectors {
		if v.got.String() != v.want {
			t.Errorf("%s: %s, want %s", v.name, v.got, v.want)
		}
	}

	for _, v := range []struct{ domain, want string }{
		{DomainRandom, "034aa4dacea65f23"},
		{DomainSeed, "e842fb5bbe1e6b01"},
	} {
		var xof [8]byte
		XOF([32]byte{}, v.domain).Read(xof[:])
		if got := hex.EncodeToString(xof[:]); got != v.want {
			t.Errorf("%s xof stream: %s, want %s", v.domain, got, v.want)
		}
	}
}

// TestIncrementalMatchesOneShot: Hasher over split writes equals Sum.
func TestIncrementalMatchesOneShot(t *testing.T) {
	h := New(DomainTreeDir)
	h.Write([]byte("ab"))
	h.Write([]byte("c"))
	if h.Sum() != Sum(DomainTreeDir, []byte("abc")) {
		t.Fatal("incremental hash diverges from one-shot")
	}
}

func TestRawIncrementalMatchesSum256(t *testing.T) {
	payload := []byte("a streaming transcript\x00with a few chunks")
	h := NewRaw()
	h.Write(payload[:1])
	h.Write(payload[1:10])
	h.Write(payload[10:])
	if h.Sum() != Sum256(payload) {
		t.Fatal("incremental raw hash diverges from Sum256")
	}
}

// TestDomainsSeparate: the same payload under two domains never collides,
// and a domain'd hash never equals the plain content hash.
func TestDomainsSeparate(t *testing.T) {
	payload := []byte("payload")
	if Sum(DomainTreeFile, payload) == Sum(DomainTreeDir, payload) {
		t.Fatal("domains do not separate")
	}
	if Sum(DomainTreeFile, payload) == Sum256(payload) {
		t.Fatal("domain'd hash equals plain content hash")
	}
}

func TestHashText(t *testing.T) {
	h := Sum256([]byte("roundtrip"))
	blob, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	var back Hash
	if err := json.Unmarshal(blob, &back); err != nil {
		t.Fatal(err)
	}
	if back != h {
		t.Fatal("hex round-trip diverges")
	}
	if err := back.UnmarshalText([]byte("abc")); err == nil {
		t.Fatal("short text accepted")
	}
}
