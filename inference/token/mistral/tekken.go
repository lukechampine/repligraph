package mistral

import (
	_ "embed"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"unicode"

	"lukechampine.com/repligraph/blake3"
)

// The Mistral Tekken v7 tokenizer as a pure Go Codec. Like the Qwen loader,
// deliberately not a general port: it accepts exactly one configuration shape.
// Differential-tested against mistral-common at the pinned revision.
//
// The vocabulary is raw bytes (tiktoken-style); merges go by rank of the
// merged token. Special ids occupy the reserved range [0, 1000) below the
// vocabulary, so Encode can never produce a control id.
//
// IMPORTANT: the HF-converted tokenizer.json shipped alongside tekken.json
// declares a different split pattern and is unfaithful to mistral-common;
// tekken.json is the defining artifact.

// tekkenPattern is the only split pattern the loader accepts. The scanner
// below hand-implements it (RE2 has no lookahead).
const tekkenPattern = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// NumSpecials is the reserved control-id range [0, NumSpecials); vocab token
// with rank r has id r + NumSpecials.
const NumSpecials = 1000

// tekkenSpecials is mistral-common's built-in v7 control-id set (tekken.json
// v7 carries no special_tokens list). Ids in [len(tekkenSpecials), 1000) are
// unnamed fillers, decoded as "<SPECIAL_n>".
var tekkenSpecials = []string{
	"<unk>", "<s>", "</s>", "[INST]", "[/INST]",
	"[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]", "[TOOL_CALLS]",
	"[IMG]", "<pad>", "[IMG_BREAK]", "[IMG_END]", "[PREFIX]",
	"[MIDDLE]", "[SUFFIX]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[TOOL_CONTENT]",
}

// The v7 control ids the chat framing injects.
const (
	BOS          = 1
	EOS          = 2
	Inst         = 3
	InstEnd      = 4
	Tools        = 5
	ToolsEnd     = 6
	Results      = 7
	ResultsEnd   = 8
	ToolCalls    = 9
	SysPrompt    = 17
	SysPromptEnd = 18
	ToolContent  = 19
)

// The Mistral Small 3.1 tekken.json, pinned at model revision 68faf511….
//
//go:embed tekken_v7.json
var tekkenBlob []byte

var cached = sync.OnceValues(func() (*TekkenCodec, error) {
	return NewTekken(tekkenBlob)
})

// New returns the embedded Mistral Small 3.1 codec, constructed once.
func New() (*TekkenCodec, error) { return cached() }

type TekkenCodec struct {
	name     string
	vocab    map[string]uint32 // token bytes → id (rank + NumSpecials)
	vocabRev [][]byte          // id - NumSpecials → token bytes
}

type tekkenJSON struct {
	Config struct {
		Pattern           string `json:"pattern"`
		NumVocabTokens    int    `json:"num_vocab_tokens"`
		DefaultVocabSize  int    `json:"default_vocab_size"`
		DefaultNumSpecial int    `json:"default_num_special_tokens"`
		Version           string `json:"version"`
	} `json:"config"`
	Vocab []struct {
		Rank       uint32 `json:"rank"`
		TokenBytes string `json:"token_bytes"`
	} `json:"vocab"`
	SpecialTokens json.RawMessage `json:"special_tokens"`
}

// NewTekken builds a codec from tekken.json bytes; the Name embeds their
// BLAKE3, so the name is the pin.
func NewTekken(blob []byte) (*TekkenCodec, error) {
	var tj tekkenJSON
	if err := json.Unmarshal(blob, &tj); err != nil {
		return nil, fmt.Errorf("bad tekken.json: %w", err)
	}
	cfg := tj.Config
	if cfg.Version != "v7" || cfg.Pattern != tekkenPattern || cfg.DefaultNumSpecial != NumSpecials {
		return nil, fmt.Errorf("unsupported tekken configuration (this codec implements exactly Tekken v7)")
	}
	if len(tj.SpecialTokens) != 0 && string(tj.SpecialTokens) != "null" {
		return nil, fmt.Errorf("unsupported tekken configuration: explicit special_tokens list")
	}
	// Only the first default_vocab_size - 1000 vocab entries are addressable:
	// vocab ids share the [0, default_vocab_size) space with the specials.
	usable := cfg.DefaultVocabSize - NumSpecials
	if usable <= 0 || usable > len(tj.Vocab) {
		return nil, fmt.Errorf("bad tekken vocab sizing: %d usable of %d", usable, len(tj.Vocab))
	}

	sum := blake3.Sum256(blob)
	c := &TekkenCodec{
		name:     "tekken-v7/v1:" + hex.EncodeToString(sum[:8]),
		vocab:    make(map[string]uint32, usable),
		vocabRev: make([][]byte, usable),
	}
	var singles [256]bool
	for i, v := range tj.Vocab[:usable] {
		if int(v.Rank) != i {
			return nil, fmt.Errorf("tekken vocab ranks are not contiguous at %d", i)
		}
		b, err := base64.StdEncoding.DecodeString(v.TokenBytes)
		if err != nil || len(b) == 0 {
			return nil, fmt.Errorf("bad token_bytes at rank %d", v.Rank)
		}
		c.vocab[string(b)] = v.Rank + NumSpecials
		c.vocabRev[v.Rank] = b
		if len(b) == 1 {
			singles[b[0]] = true
		}
	}
	for b := 0; b < 256; b++ {
		if !singles[b] {
			return nil, fmt.Errorf("tekken vocab lacks single byte %#x", b)
		}
	}
	return c, nil
}

func (c *TekkenCodec) Name() string { return c.name }

// ── encode ──────────────────────────────────────────────────────────────

// Encode maps text to vocab ids; it can never emit a control id.
func (c *TekkenCodec) Encode(text string) []uint32 {
	var ids []uint32
	for _, piece := range tekkenPretokenize(text) {
		ids = append(ids, c.bpe([]byte(piece))...)
	}
	return ids
}

// bpe merges tiktoken-style: repeatedly merge the adjacent pair whose
// concatenation has the lowest rank. Ids are rank + 1000, so comparing ids
// compares ranks.
func (c *TekkenCodec) bpe(piece []byte) []uint32 {
	if id, ok := c.vocab[string(piece)]; ok {
		return []uint32{id}
	}
	// bounds[i] is the start of part i; the sentinel len(piece) closes the
	// last part.
	bounds := make([]int, len(piece)+1)
	for i := range bounds {
		bounds[i] = i
	}
	for len(bounds) > 2 {
		best, bestID := -1, uint32(0)
		for i := 0; i+2 < len(bounds); i++ {
			if id, ok := c.vocab[string(piece[bounds[i]:bounds[i+2]])]; ok && (best < 0 || id < bestID) {
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
		id, ok := c.vocab[string(piece[bounds[i]:bounds[i+1]])]
		if !ok {
			panic(fmt.Sprintf("tekken part %q not in vocabulary", piece[bounds[i]:bounds[i+1]]))
		}
		ids = append(ids, id)
	}
	return ids
}

// ── decode ──────────────────────────────────────────────────────────────

func (c *TekkenCodec) Decode(ids []uint32) (string, error) {
	var out []byte
	for _, id := range ids {
		switch {
		case id < uint32(len(tekkenSpecials)):
			out = append(out, tekkenSpecials[id]...)
		case id < NumSpecials:
			out = append(out, fmt.Sprintf("<SPECIAL_%d>", id)...)
		case int(id-NumSpecials) < len(c.vocabRev):
			out = append(out, c.vocabRev[id-NumSpecials]...)
		default:
			return "", fmt.Errorf("token id %d is outside the %s vocabulary", id, c.name)
		}
	}
	return string(out), nil
}

// ── pretokenizer ────────────────────────────────────────────────────────

// tekkenPretokenize hand-implements tekkenPattern with backtracking-
// alternation semantics (first alternative at the position wins, quantifiers
// greedy with minimal give-back) in linear time per piece.
func tekkenPretokenize(text string) []string {
	rs := []rune(text)
	n := len(rs)
	var pieces []string
	for i := 0; i < n; {
		j := i + tekkenMatchAt(rs[i:])
		pieces = append(pieces, string(rs[i:j]))
		i = j
	}
	return pieces
}

// "Upperish" and "lowerish" overlap on Lm, Lo, and M — that overlap is what
// the backtracking below navigates.
func tekkenUpperish(r rune) bool {
	return unicode.In(r, unicode.Lu, unicode.Lt, unicode.Lm, unicode.Lo, unicode.M)
}
func tekkenLowerish(r rune) bool {
	return unicode.In(r, unicode.Ll, unicode.Lm, unicode.Lo, unicode.M)
}

// caseWord1 matches `[upperish]*[lowerish]+` at rs[i:], returning the end or
// -1. The upperish run yields characters from its tail until a lowerish match
// can start.
func caseWord1(rs []rune, i int) int {
	n := len(rs)
	u := i
	for u < n && tekkenUpperish(rs[u]) {
		u++
	}
	if u < n && tekkenLowerish(rs[u]) {
		l := u + 1
		for l < n && tekkenLowerish(rs[l]) {
			l++
		}
		return l
	}
	for j := u - 1; j >= i; j-- {
		if tekkenLowerish(rs[j]) {
			l := j + 1
			for l < n && tekkenLowerish(rs[l]) {
				l++
			}
			return l
		}
	}
	return -1
}

// caseWord2 matches `[upperish]+[lowerish]*` at rs[i:], returning the end or
// -1.
func caseWord2(rs []rune, i int) int {
	n := len(rs)
	u := i
	for u < n && tekkenUpperish(rs[u]) {
		u++
	}
	if u == i {
		return -1
	}
	l := u
	for l < n && tekkenLowerish(rs[l]) {
		l++
	}
	return l
}

// tekkenMatchAt returns the length (in runes) of the piece starting at rs[0].
func tekkenMatchAt(rs []rune) int {
	n := len(rs)
	r0 := rs[0]

	isL := unicode.IsLetter
	isN := unicode.IsNumber
	isS := unicode.IsSpace
	// The optional word prefix: anything but CR, LF, letters, and numbers.
	isPrefix := r0 != '\r' && r0 != '\n' && !isL(r0) && !isN(r0)

	// 1. [^\r\n\p{L}\p{N}]?[upperish]*[lowerish]+ — the greedy `?` tries the
	// prefix first.
	if isPrefix && n >= 2 {
		if end := caseWord1(rs, 1); end > 0 {
			return end
		}
	}
	if end := caseWord1(rs, 0); end > 0 {
		return end
	}

	// 2. [^\r\n\p{L}\p{N}]?[upperish]+[lowerish]*
	if isPrefix && n >= 2 {
		if end := caseWord2(rs, 1); end > 0 {
			return end
		}
	}
	if end := caseWord2(rs, 0); end > 0 {
		return end
	}

	// 3. \p{N} — one number rune at a time.
	if isN(r0) {
		return 1
	}

	// 4. ` ?[^\s\p{L}\p{N}]+[\r\n/]*` — the trailing class admits '/'.
	isOther := func(r rune) bool { return !isS(r) && !isL(r) && !isN(r) }
	punctFrom := func(j int) int {
		k := j
		for k < n && isOther(rs[k]) {
			k++
		}
		if k == j {
			return -1
		}
		for k < n && (rs[k] == '\r' || rs[k] == '\n' || rs[k] == '/') {
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
	// leaving one whitespace rune to prefix the next word via alternative 1.
	if w == n {
		return w
	}
	if w >= 2 {
		return w - 1
	}

	// 7. \s+
	return w
}
