// Package blake3 defines repligraph's BLAKE3-256 hashes. Structured hashes
// are prefixed with a domain string and a NUL byte; Sum256 hashes raw content.
package blake3

import (
	"encoding/hex"
	"errors"
	"io"

	"lukechampine.com/blake3"
)

// Hash domains are part of the bundle format.
const (
	DomainTreeFile = "repligraph.tree.file.v0"   // tree: file node, over content
	DomainTreeDir  = "repligraph.tree.dir.v0"    // tree: directory node, over (kind, len, name, child hash) records
	DomainModule   = "repligraph.wasm.module.v0" // wasm: .wasm module bytes
	DomainRandom   = "repligraph.wasm.random.v0" // wasm: guest random_get stream (keyed XOF, key = request seed)
	DomainContext  = "repligraph.ctx.v0"         // transcript: running context, over little-endian token ids
	DomainSeed     = "repligraph.seed.v0"        // inference: per-turn sampling seeds (keyed XOF, key = little-endian header seed, zero-padded)
)

// Hash is a BLAKE3-256 digest, rendered as 64 hex characters in text form.
type Hash [32]byte

func (h Hash) String() string { return hex.EncodeToString(h[:]) }

func (h Hash) MarshalText() ([]byte, error) { return []byte(h.String()), nil }

func (h *Hash) UnmarshalText(text []byte) error {
	if len(text) != 64 {
		return errors.New("invalid hash length")
	}
	_, err := hex.Decode(h[:], text)
	return err
}

// A Hasher accumulates a BLAKE3 hash.
type Hasher struct {
	h *blake3.Hasher
}

// NewRaw returns a Hasher without a domain separator, equivalent to Sum256.
func NewRaw() *Hasher { return &Hasher{blake3.New(32, nil)} }

// New returns a Hasher with the domain separator already written.
func New(domain string) *Hasher {
	h := blake3.New(32, nil)
	h.Write([]byte(domain))
	h.Write([]byte{0})
	return &Hasher{h}
}

func (h *Hasher) Write(p []byte) (int, error) { return h.h.Write(p) }

func (h *Hasher) Sum() Hash {
	var out Hash
	h.h.Sum(out[:0])
	return out
}

// Sum is the one-shot domain-separated hash.
func Sum(domain string, data []byte) Hash {
	h := New(domain)
	h.Write(data)
	return h.Sum()
}

// Sum256 hashes raw content without a domain separator.
func Sum256(data []byte) Hash { return blake3.Sum256(data) }

// XOF returns the deterministic output stream for a domain: BLAKE3 keyed
// with key, domain-separated, extended output. Reads never fail.
func XOF(key [32]byte, domain string) io.Reader {
	h := blake3.New(32, key[:])
	h.Write([]byte(domain))
	h.Write([]byte{0})
	return h.XOF()
}
