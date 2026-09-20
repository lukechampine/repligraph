package qwen

import (
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/text/unicode/norm"
	"lukechampine.com/repligraph/blake3"
)

// The Qwen2-family tokenizer as a pure Go Codec: NFC normalizer, the split
// pattern below, byte-level BPE. Deliberately not a general HF tokenizers
// port — the loader accepts exactly one configuration shape, because every
// accepted variation is behavior pinned forever. Differential-tested
// token-for-token against transformers.js at the pinned revision.
//
// Encode never emits added/special ids: "<|im_end|>" in content tokenizes as
// plain text. This diverges from HF tokenizers on purpose.

// qwen2Pattern is the only split pattern the loader accepts. The scanner
// below hand-implements it: RE2 has no lookahead, and backtracking engines
// have pathological cases.
const qwen2Pattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// The Qwen2.5-Coder tokenizer.json, pinned at model revision c03e6d35….
//
//go:embed qwen25_tokenizer.json
var qwen25Blob []byte

var cached = sync.OnceValues(func() (*Codec, error) {
	return NewBPE(qwen25Blob)
})

type Codec struct {
	name       string
	vocab      map[string]uint32 // byte-mapped token string → id
	vocabRev   []string          // id → byte-mapped token string ("" = gap)
	ranks      map[string]int    // "left right" → merge priority
	addedRev   map[uint32]string // added-token id → literal content (decode only)
	addedFwd   map[string]uint32 // literal content → added-token id (framing only)
	byteToRune [256]rune
	runeToByte map[rune]byte
}

type tokenizerJSON struct {
	Normalizer *struct {
		Type string `json:"type"`
	} `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Type                    string            `json:"type"`
		Vocab                   map[string]uint32 `json:"vocab"`
		Merges                  []json.RawMessage `json:"merges"`
		ContinuingSubwordPrefix *string           `json:"continuing_subword_prefix"`
		EndOfWordSuffix         *string           `json:"end_of_word_suffix"`
		ByteFallback            bool              `json:"byte_fallback"`
	} `json:"model"`
	AddedTokens []struct {
		ID      uint32 `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

// New returns the embedded Qwen2.5-Coder codec, constructed once.
func New() (*Codec, error) { return cached() }

// NewBPE builds a codec from tokenizer.json bytes; the Name embeds their
// BLAKE3, so the name is the pin.
func NewBPE(blob []byte) (*Codec, error) {
	var tj tokenizerJSON
	if err := json.Unmarshal(blob, &tj); err != nil {
		return nil, fmt.Errorf("bad tokenizer.json: %w", err)
	}
	if tj.Normalizer == nil || tj.Normalizer.Type != "NFC" {
		return nil, fmt.Errorf("unsupported normalizer (want NFC)")
	}
	if err := checkPreTokenizer(tj.PreTokenizer); err != nil {
		return nil, err
	}
	m := tj.Model
	if m.Type != "BPE" || m.ByteFallback ||
		(m.ContinuingSubwordPrefix != nil && *m.ContinuingSubwordPrefix != "") ||
		(m.EndOfWordSuffix != nil && *m.EndOfWordSuffix != "") {
		return nil, fmt.Errorf("unsupported model configuration (want plain byte-level BPE)")
	}

	sum := blake3.Sum256(blob)
	c := &Codec{
		name:       "qwen2-bpe/v1:" + hex.EncodeToString(sum[:8]),
		vocab:      m.Vocab,
		ranks:      make(map[string]int, len(m.Merges)),
		addedRev:   make(map[uint32]string, len(tj.AddedTokens)),
		addedFwd:   make(map[string]uint32, len(tj.AddedTokens)),
		runeToByte: make(map[rune]byte, 256),
	}

	maxID := uint32(0)
	for _, id := range m.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	c.vocabRev = make([]string, maxID+1)
	for tok, id := range m.Vocab {
		c.vocabRev[id] = tok
	}

	for i, raw := range m.Merges {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			// Newer serializations use ["left", "right"].
			var pair [2]string
			if err := json.Unmarshal(raw, &pair); err != nil {
				return nil, fmt.Errorf("bad merge at index %d", i)
			}
			s = pair[0] + " " + pair[1]
		}
		if strings.Count(s, " ") != 1 {
			return nil, fmt.Errorf("bad merge %q at index %d", s, i)
		}
		c.ranks[s] = i
	}

	for _, at := range tj.AddedTokens {
		// Shadowed base-vocab ids would make decode ambiguous.
		if at.ID < uint32(len(c.vocabRev)) && c.vocabRev[at.ID] != "" {
			return nil, fmt.Errorf("added token %d shadows the base vocabulary", at.ID)
		}
		c.addedRev[at.ID] = at.Content
		c.addedFwd[at.Content] = at.ID
	}
	// GPT-2 byte↔unicode table: printable latin-1 bytes map to themselves,
	// the rest to U+0100+n.
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		printable := (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
		if !printable {
			r = rune(256 + n)
			n++
		}
		c.byteToRune[b] = r
		c.runeToByte[r] = byte(b)
	}
	return c, nil
}

// checkPreTokenizer pins the pre_tokenizer shape: Split on qwen2Pattern
// (Isolated, not inverted) followed by ByteLevel without its own regex.
func checkPreTokenizer(raw json.RawMessage) error {
	var pt struct {
		Type          string `json:"type"`
		Pretokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
			Behavior       string `json:"behavior"`
			Invert         bool   `json:"invert"`
			AddPrefixSpace bool   `json:"add_prefix_space"`
			UseRegex       bool   `json:"use_regex"`
		} `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &pt); err != nil {
		return fmt.Errorf("bad pre_tokenizer: %w", err)
	}
	ok := pt.Type == "Sequence" && len(pt.Pretokenizers) == 2 &&
		pt.Pretokenizers[0].Type == "Split" &&
		pt.Pretokenizers[0].Pattern.Regex == qwen2Pattern &&
		pt.Pretokenizers[0].Behavior == "Isolated" &&
		!pt.Pretokenizers[0].Invert &&
		pt.Pretokenizers[1].Type == "ByteLevel" &&
		!pt.Pretokenizers[1].AddPrefixSpace &&
		!pt.Pretokenizers[1].UseRegex
	if !ok {
		return fmt.Errorf("unsupported pre_tokenizer (this codec implements exactly the Qwen2 split pattern)")
	}
	return nil
}

func (c *Codec) Name() string { return c.name }

// AddedID looks up an added (control) token by its literal content. Framing
// injects control ids through this; Encode never emits them.
func (c *Codec) AddedID(content string) (uint32, bool) {
	id, ok := c.addedFwd[content]
	return id, ok
}

// ── encode ──────────────────────────────────────────────────────────────

func (c *Codec) Encode(text string) []uint32 {
	text = norm.NFC.String(text)
	var ids []uint32
	for _, piece := range pretokenize(text) {
		// Byte-map the piece: one rune per UTF-8 byte.
		var sb strings.Builder
		for i := 0; i < len(piece); i++ {
			sb.WriteRune(c.byteToRune[piece[i]])
		}
		ids = append(ids, c.bpe(sb.String())...)
	}
	return ids
}

// bpe repeatedly applies the lowest-ranked adjacent pair, then looks the
// symbols up. Every mapped byte is in the base vocabulary, so lookup is total.
func (c *Codec) bpe(mapped string) []uint32 {
	if id, ok := c.vocab[mapped]; ok {
		return []uint32{id}
	}
	var symbols []string
	for _, r := range mapped {
		symbols = append(symbols, string(r))
	}
	for len(symbols) > 1 {
		best, bestRank := -1, -1
		for i := 0; i < len(symbols)-1; i++ {
			if rank, ok := c.ranks[symbols[i]+" "+symbols[i+1]]; ok && (best < 0 || rank < bestRank) {
				best, bestRank = i, rank
			}
		}
		if best < 0 {
			break
		}
		merged := symbols[best] + symbols[best+1]
		symbols = append(symbols[:best], append([]string{merged}, symbols[best+2:]...)...)
	}
	ids := make([]uint32, len(symbols))
	for i, s := range symbols {
		id, ok := c.vocab[s]
		if !ok {
			panic(fmt.Sprintf("bpe symbol %q not in vocabulary", s))
		}
		ids[i] = id
	}
	return ids
}

// ── decode ──────────────────────────────────────────────────────────────

func (c *Codec) Decode(ids []uint32) (string, error) {
	var out []byte
	for _, id := range ids {
		if content, ok := c.addedRev[id]; ok {
			// Added tokens carry literal text, not byte-mapped symbols.
			out = append(out, content...)
			continue
		}
		if int(id) >= len(c.vocabRev) || c.vocabRev[id] == "" {
			return "", fmt.Errorf("token id %d is outside the %s vocabulary", id, c.name)
		}
		for _, r := range c.vocabRev[id] {
			b, ok := c.runeToByte[r]
			if !ok {
				return "", fmt.Errorf("token id %d contains a non-byte symbol", id)
			}
			out = append(out, b)
		}
	}
	return string(out), nil
}

// ── pretokenizer ────────────────────────────────────────────────────────

// pretokenize hand-implements qwen2Pattern with backtracking-alternation
// semantics (first alternative at the position wins, each greedy) in linear
// time. One divergence: the contraction alternative folds case in ASCII only,
// where (?i) also folds exotic pairs (U+017F ſ → s); Encode is the pinned
// truth, so the simpler rule wins.
func pretokenize(text string) []string {
	rs := []rune(text)
	n := len(rs)
	var pieces []string
	for i := 0; i < n; {
		j := i + matchAt(rs[i:])
		pieces = append(pieces, string(rs[i:j]))
		i = j
	}
	return pieces
}

// matchAt returns the length (in runes) of the piece starting at rs[0].
func matchAt(rs []rune) int {
	n := len(rs)
	r0 := rs[0]

	// 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if r0 == '\'' && n >= 2 {
		a := asciiLower(rs[1])
		var b rune
		if n >= 3 {
			b = asciiLower(rs[2])
		}
		switch {
		case a == 's', a == 't', a == 'm', a == 'd':
			return 2
		case a == 'r' && b == 'e', a == 'v' && b == 'e', a == 'l' && b == 'l':
			return 3
		}
	}

	isL := unicode.IsLetter
	isN := unicode.IsNumber
	isS := unicode.IsSpace
	isOther := func(r rune) bool { return !isS(r) && !isL(r) && !isN(r) }

	// 2. [^\r\n\p{L}\p{N}]?\p{L}+
	if isL(r0) {
		j := 1
		for j < n && isL(rs[j]) {
			j++
		}
		return j
	}
	if r0 != '\r' && r0 != '\n' && !isN(r0) && n >= 2 && isL(rs[1]) {
		j := 2
		for j < n && isL(rs[j]) {
			j++
		}
		return j
	}

	// 3. \p{N} — one number rune at a time.
	if isN(r0) {
		return 1
	}

	// 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
	punctFrom := func(j int) int {
		k := j
		for k < n && isOther(rs[k]) {
			k++
		}
		if k == j {
			return -1
		}
		for k < n && (rs[k] == '\r' || rs[k] == '\n') {
			k++
		}
		return k
	}
	if r0 == ' ' && n >= 2 {
		if k := punctFrom(1); k >= 0 {
			return k
		}
	}
	if k := punctFrom(0); k >= 0 {
		return k
	}

	if !isS(r0) {
		// Unreachable: every rune is a letter, number, whitespace, or other.
		return 1
	}
	w := 1
	for w < n && isS(rs[w]) {
		w++
	}

	// 5. \s*[\r\n]+ — the whitespace run up to and including its last newline.
	last := -1
	for k := 0; k < w; k++ {
		if rs[k] == '\r' || rs[k] == '\n' {
			last = k
		}
	}
	if last >= 0 {
		return last + 1
	}

	// 6. \s+(?!\S) — all of a trailing run; all but the last char otherwise,
	// leaving one whitespace rune to prefix the next word via alternative 2.
	if w == n {
		return w
	}
	if w >= 2 {
		return w - 1
	}

	// 7. \s+
	return w
}

func asciiLower(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r + ('a' - 'A')
	}
	return r
}
