package mistral

import (
	"encoding/json"
	"io"
	"strings"

	"lukechampine.com/repligraph/inference/model"
	tok "lukechampine.com/repligraph/inference/token/mistral"
)

type chat struct {
	codec *tok.TekkenCodec
}

func New() (model.Model, error) {
	c, err := tok.New()
	if err != nil {
		return nil, err
	}
	return &chat{c}, nil
}

func (m *chat) StopIDs() []uint32 { return []uint32{tok.EOS} }

func badCall(reject string) (string, json.RawMessage, string, string) {
	return "", nil, "", reject
}

func (m *chat) Parse(gen []uint32) (tool string, args json.RawMessage, ref string, reject string) {
	i := len(gen) - 1
	for i >= 0 && gen[i] != tok.ToolCalls {
		i--
	}
	if i < 0 {
		return badCall("no tool call found")
	}
	body, err := m.codec.Decode(gen[i+1:])
	if err != nil {
		return badCall("malformed tool call")
	}
	var calls []struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		ID        string          `json:"id"`
	}
	dec := json.NewDecoder(strings.NewReader(body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&calls); err != nil {
		return badCall("malformed tool call")
	}
	if dec.Decode(new(json.RawMessage)) != io.EOF {
		return badCall("trailing content after tool call")
	}
	switch {
	case len(calls) == 0:
		return badCall("malformed tool call")
	case len(calls) > 1:
		return badCall("multiple tool calls in one turn")
	case calls[0].Name == "":
		return badCall("malformed tool call")
	}
	return calls[0].Name, calls[0].Arguments, calls[0].ID, ""
}

func (m *chat) FrameResult(ref string, body []byte) []uint32 {
	out := []uint32{tok.EOS, tok.Results}
	out = append(out, m.codec.Encode(ref)...)
	out = append(out, tok.ToolContent)
	out = append(out, m.codec.Encode(string(body))...)
	return append(out, tok.ResultsEnd)
}

func (m *chat) FramePrompt(body []byte) []uint32 {
	out := []uint32{tok.EOS, tok.Inst}
	out = append(out, m.codec.Encode(string(body))...)
	return append(out, tok.InstEnd)
}
