package model

import "encoding/json"

// A Model defines stop tokens, call parsing, and continuation framing.
type Model interface {
	StopIDs() []uint32
	// Parse takes output without its final stop; rejection returns a model-visible
	// reason instead of a call.
	Parse(gen []uint32) (tool string, args json.RawMessage, ref string, reject string)
	// Frames follow the generated tokens with their final stop removed.
	FrameResult(ref string, body []byte) []uint32
	FramePrompt(body []byte) []uint32
}

// BadCall renders the feedback for a rejected model output. No call was issued,
// so this is not a tool result.
func BadCall(reject string) []byte {
	return []byte("<bad_call>\n" + reject + "\n</bad_call>")
}

// ToolFinish ends the transcript without executing a harness call; its arguments are ignored.
const ToolFinish = "finish"
