package harmony

import (
	_ "embed"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"

	"lukechampine.com/repligraph/blake3"
)

// This is the ordinary vocabulary of openai/gpt-oss-120b at
// b5c939de8f754692c1647ca79fbf85e8c1e70f8a, in OpenAI's compact rank format.
//
//go:embed o200k_base.tiktoken
var vocabulary string

// Unicode classes used by the pinned HF tokenizer's split expression.
//
//go:embed unicode.bin
var classes []byte

var specials = [...]string{
	"<|startoftext|>", "<|endoftext|>", "<|reserved_200000|>", "<|reserved_200001|>",
	"<|return|>", "<|constrain|>", "<|reserved_200004|>", "<|channel|>",
	"<|start|>", "<|end|>", "<|message|>", "<|reserved_200009|>",
	"<|reserved_200010|>", "<|reserved_200011|>", "<|call|>", "<|reserved_200013|>",
	"<|reserved_200014|>", "<|reserved_200015|>", "<|reserved_200016|>", "<|reserved_200017|>",
	"<|endofprompt|>",
}

const numOrdinary = 199998

type Codec struct {
	name  string
	ranks map[string]uint32
	words []string
}

var cached = sync.OnceValues(func() (*Codec, error) {
	sum := blake3.Sum256(append([]byte(vocabulary), classes...))
	c := &Codec{
		name:  "o200k-harmony/v1:" + hex.EncodeToString(sum[:8]),
		ranks: make(map[string]uint32, numOrdinary),
		words: make([]string, 0, numOrdinary),
	}
	for line := range strings.Lines(vocabulary) {
		encoded, rank, ok := strings.Cut(strings.TrimSuffix(line, "\n"), " ")
		id, err := strconv.Atoi(rank)
		if !ok || err != nil || id != len(c.words) {
			return nil, fmt.Errorf("invalid harmony vocabulary rank %q", rank)
		}
		word, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || len(word) == 0 {
			return nil, fmt.Errorf("invalid harmony vocabulary word at %d", id)
		}
		c.words = append(c.words, string(word))
		c.ranks[string(word)] = uint32(id)
	}
	if len(c.words) != numOrdinary {
		return nil, fmt.Errorf("invalid harmony vocabulary size %d", len(c.words))
	}
	for b := range 256 {
		if _, ok := c.ranks[string([]byte{byte(b)})]; !ok {
			return nil, fmt.Errorf("harmony vocabulary lacks byte %d", b)
		}
	}
	return c, nil
})

func New() (*Codec, error) { return cached() }

func (c *Codec) Name() string { return c.name }

func (c *Codec) AddedID(content string) (uint32, bool) {
	for i, s := range specials {
		if s == content {
			return uint32(numOrdinary + i), true
		}
	}
	return 0, false
}

// Encode treats special-token spellings as ordinary text.
func (c *Codec) Encode(text string) []uint32 {
	var ids []uint32
	rs := []rune(text)
	for len(rs) > 0 {
		n := match(rs)
		ids = append(ids, c.bpe(string(rs[:n]))...)
		rs = rs[n:]
	}
	return ids
}

func (c *Codec) bpe(piece string) []uint32 {
	if id, ok := c.ranks[piece]; ok {
		return []uint32{id}
	}
	// Each boundary initially separates bytes; merge the lowest-ranked pair.
	bounds := make([]int, len(piece)+1)
	for i := range bounds {
		bounds[i] = i
	}
	for len(bounds) > 2 {
		best, bestID := -1, uint32(0)
		for i := 0; i+2 < len(bounds); i++ {
			if id, ok := c.ranks[piece[bounds[i]:bounds[i+2]]]; ok && (best < 0 || id < bestID) {
				best, bestID = i, id
			}
		}
		if best < 0 {
			break
		}
		bounds = append(bounds[:best+1], bounds[best+2:]...)
	}
	ids := make([]uint32, 0, len(bounds)-1)
	for i := 0; i+1 < len(bounds); i++ {
		ids = append(ids, c.ranks[piece[bounds[i]:bounds[i+1]]])
	}
	return ids
}

func (c *Codec) Decode(ids []uint32) (string, error) {
	var out strings.Builder
	for _, id := range ids {
		switch {
		case id < numOrdinary:
			out.WriteString(c.words[id])
		case id < numOrdinary+uint32(len(specials)):
			out.WriteString(specials[id-numOrdinary])
		default:
			return "", fmt.Errorf("token id %d is outside the %s vocabulary", id, c.name)
		}
	}
	return out.String(), nil
}

const (
	letterClass = 1 << iota
	numberClass
	spaceClass
	upperClass
	lowerClass
)

func class(r rune) byte {
	if r < 128 {
		return classes[r]
	}
	ranges := classes[128:]
	i := sort.Search(len(ranges)/9, func(i int) bool {
		return uint32(r) <= binary.LittleEndian.Uint32(ranges[i*9+4:])
	})
	if i == len(ranges)/9 || uint32(r) < binary.LittleEndian.Uint32(ranges[i*9:]) {
		return 0
	}
	return ranges[i*9+8]
}

func upper(r rune) bool  { return class(r)&upperClass != 0 }
func lower(r rune) bool  { return class(r)&lowerClass != 0 }
func letter(r rune) bool { return class(r)&letterClass != 0 }
func number(r rune) bool { return class(r)&numberClass != 0 }
func space(r rune) bool  { return class(r)&spaceClass != 0 }

// word matches either [upper]*[lower]+ or [upper]+[lower]*.
func word(rs []rune, start int, lowerRequired bool) int {
	u := start
	for u < len(rs) && upper(rs[u]) {
		u++
	}
	if !lowerRequired && u == start {
		return -1
	}
	if lowerRequired {
		if u == len(rs) || !lower(rs[u]) {
			u--
			for u >= start && !lower(rs[u]) {
				u--
			}
			if u < start {
				return -1
			}
		}
	}
	end := u
	for end < len(rs) && lower(rs[end]) {
		end++
	}
	return end + contraction(rs[end:])
}

func contraction(rs []rune) int {
	if len(rs) < 2 || rs[0] != '\'' {
		return 0
	}
	for _, s := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
		if len(rs) <= len(s) {
			continue
		}
		match := true
		for i, want := range s {
			r := rs[i+1]
			if r >= 'A' && r <= 'Z' {
				r += 'a' - 'A'
			} else if r == 'ſ' {
				r = 's'
			}
			match = match && r == want
		}
		if match {
			return 1 + len(s)
		}
	}
	return 0
}

// match implements o200k's ordered regex alternatives; Go's regexp lacks
// the lookahead used to leave one space attached to the following word.
func match(rs []rune) int {
	r := rs[0]
	prefix := r != '\r' && r != '\n' && !letter(r) && !number(r)
	for _, lowerRequired := range []bool{true, false} {
		if prefix && len(rs) > 1 {
			if end := word(rs, 1, lowerRequired); end >= 0 {
				return end
			}
		}
		if end := word(rs, 0, lowerRequired); end >= 0 {
			return end
		}
	}
	if number(r) {
		n := 1
		for n < len(rs) && n < 3 && number(rs[n]) {
			n++
		}
		return n
	}
	other := func(r rune) bool {
		return !space(r) && !letter(r) && !number(r)
	}
	start := 0
	if r == ' ' && len(rs) > 1 && other(rs[1]) {
		start = 1
	}
	if other(rs[start]) {
		n := start + 1
		for n < len(rs) && other(rs[n]) {
			n++
		}
		for n < len(rs) && (rs[n] == '\r' || rs[n] == '\n' || rs[n] == '/') {
			n++
		}
		return n
	}
	n, newline := 1, 0
	if r == '\r' || r == '\n' {
		newline = 1
	}
	for n < len(rs) && space(rs[n]) {
		if rs[n] == '\r' || rs[n] == '\n' {
			newline = n + 1
		}
		n++
	}
	if newline > 0 {
		return newline
	} else if n < len(rs) && n > 1 {
		return n - 1
	}
	return n
}
