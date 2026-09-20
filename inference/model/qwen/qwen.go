package qwen

import (
	"encoding/json"
	"fmt"
	"strings"

	"lukechampine.com/repligraph/inference/model"
	tok "lukechampine.com/repligraph/inference/token/qwen"
)

type chat struct {
	codec                                    *tok.Codec
	imStart, imEnd, eot, callOpen, callClose uint32
	preopenCall                              bool
}

// New returns the Qwen2.5-Coder-7B framing, with a pre-opened tool call.
func New() (model.Model, error) { return newModel(true) }

// NewSmall returns the Qwen2.5-Coder-1.5B framing, with bare JSON calls.
func NewSmall() (model.Model, error) { return newModel(false) }

func newModel(preopenCall bool) (model.Model, error) {
	c, err := tok.New()
	if err != nil {
		return nil, err
	}
	ch := &chat{codec: c, preopenCall: preopenCall}
	for _, t := range []struct {
		content string
		dst     *uint32
	}{
		{"<|im_start|>", &ch.imStart},
		{"<|im_end|>", &ch.imEnd},
		{"<|endoftext|>", &ch.eot},
		{"<tool_call>", &ch.callOpen},
		{"</tool_call>", &ch.callClose},
	} {
		id, ok := c.AddedID(t.content)
		if !ok {
			return nil, fmt.Errorf("codec lacks control token %s", t.content)
		}
		*t.dst = id
	}
	return ch, nil
}

func (c *chat) StopIDs() []uint32 {
	return []uint32{c.imEnd, c.imStart, c.eot, c.callOpen, c.callClose}
}

func (c *chat) Parse(gen []uint32) (tool string, args json.RawMessage, ref string, reject string) {
	body, err := c.codec.Decode(gen)
	if err != nil {
		return "", nil, "", "malformed tool call"
	}
	var call struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	dec := json.NewDecoder(strings.NewReader(body))
	dec.DisallowUnknownFields()
	// Only the first object executes; trailing text remains in context.
	if err := dec.Decode(&call); err != nil || call.Name == "" {
		return "", nil, "", "malformed tool call"
	}
	return call.Name, call.Arguments, "", ""
}

func (c *chat) FrameResult(ref string, body []byte) []uint32 {
	return c.FramePrompt([]byte("<tool_response>\n" + string(body) + "\n</tool_response>"))
}

func (c *chat) FramePrompt(body []byte) []uint32 {
	out := append([]uint32{c.imEnd}, c.codec.Encode("\n")...)
	out = append(out, c.imStart)
	out = append(out, c.codec.Encode("user\n"+string(body))...)
	out = append(out, c.imEnd)
	out = append(out, c.codec.Encode("\n")...)
	out = append(out, c.imStart)
	out = append(out, c.codec.Encode("assistant\n")...)
	if c.preopenCall {
		out = append(out, c.callOpen)
		out = append(out, c.codec.Encode("\n")...)
	}
	return out
}
