package tools

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
	"lukechampine.com/repligraph/harness/wasm/wasmtime"
)

const (
	maxOutputBytes = 32768
	truncHeadBytes = 24576
	truncTailBytes = 4096
)

func truncate(s string) string {
	if len(s) <= maxOutputBytes {
		return s
	}
	head := truncHeadBytes
	for k := 0; k < 3 && head > 0 && s[head]&0xC0 == 0x80; k++ {
		head--
	}
	tail := len(s) - truncTailBytes
	for k := 0; k < 3 && tail < len(s) && s[tail]&0xC0 == 0x80; k++ {
		tail++
	}
	return s[:head] + fmt.Sprintf("\n[truncated: %d bytes omitted]\n", tail-head) + s[tail:]
}

type result struct {
	status string
	output string
}

// Apply mounts env read-only at /env for one tool call, running modules with
// wasmtime.Run. It returns the next task tree, complete model-visible result
// bytes, and usage. Reads span the whole tree; writes are limited to /artifact/
// descendants. Host errors, including resource exhaustion, leave the tree
// unchanged and return no result bytes or usage.
func Apply(t, env tree.Tree, limits wasm.Limits, tool string, args json.RawMessage) (tree.Tree, []byte, wasm.Usage, error) {
	mounted, err := t.Graft("env", env)
	if err != nil {
		return t, nil, wasm.Usage{}, err
	}
	next, output, usage, err := apply(mounted, wasmtime.Run, limits, tool, args)
	if err != nil {
		return t, nil, wasm.Usage{}, err
	}
	next, err = next.Without("env")
	if err != nil {
		return t, nil, wasm.Usage{}, err
	}
	return next, output, usage, nil
}

// apply is Apply over a tree with the environment already mounted.
func apply(t tree.Tree, ex wasm.Executor, limits wasm.Limits, tool string, args json.RawMessage) (tree.Tree, []byte, wasm.Usage, error) {
	initial := t
	task, err := t.Without("env")
	if err != nil {
		return initial, nil, wasm.Usage{}, err
	} else if uint64(task.Stats().Entries) > limits.MaxTreeEntries {
		return initial, nil, wasm.Usage{}, fmt.Errorf("%w: initial task tree entries", wasm.ErrResourceLimit)
	} else if task.Stats().Bytes < 0 || uint64(task.Stats().Bytes) > limits.MaxTreeBytes {
		return initial, nil, wasm.Usage{}, fmt.Errorf("%w: initial task tree bytes", wasm.ErrResourceLimit)
	}
	initialEntries := uint64(task.Stats().Entries)
	initialBytes := uint64(task.Stats().Bytes)
	var usage wasm.Usage
	var r result
	switch tool {
	case "read_file":
		r = applyRead(t, args)
	case "write_file":
		r, t, usage.WriteBytes, err = applyWrite(t, limits, initialBytes, args)
	case "list_files":
		r = applyList(t, args)
	case "search_files":
		r = applySearch(t, args)
	case "edit_file":
		r, t, usage.WriteBytes, err = applyEdit(t, limits, initialBytes, args)
	case "copy_file":
		r, t, err = applyCopy(t, limits, initialBytes, args, false)
	case "rename_file":
		r, t, err = applyCopy(t, limits, initialBytes, args, true)
	case "delete_file":
		r, t = applyDelete(t, args)
	case "run":
		r, t, usage, err = applyRun(t, ex, limits, args)
	default:
		r = errResult("unknown_tool", "unknown tool: %s", tool)
	}
	if err != nil {
		return initial, nil, wasm.Usage{}, err
	}
	task, err = t.Without("env")
	if err != nil {
		return initial, nil, wasm.Usage{}, err
	}
	usage.TreeEntries = max(usage.TreeEntries, initialEntries, uint64(task.Stats().Entries))
	usage.TreeBytes = max(usage.TreeBytes, initialBytes, uint64(task.Stats().Bytes))
	if usage.TreeEntries > limits.MaxTreeEntries {
		return initial, nil, wasm.Usage{}, fmt.Errorf("%w: task tree entries", wasm.ErrResourceLimit)
	} else if task.Stats().Bytes < 0 || usage.TreeBytes > limits.MaxTreeBytes {
		return initial, nil, wasm.Usage{}, fmt.Errorf("%w: task tree bytes", wasm.ErrResourceLimit)
	}
	return t, render(r), usage, nil
}

func render(r result) []byte {
	return []byte("<result " + r.status + ">\n" + truncate(r.output) + "\n</result>")
}

func errResult(code, format string, args ...any) result {
	return result{"error", fmt.Sprintf("error[%s]: %s", code, fmt.Sprintf(format, args...))}
}

func treeErr(err error, path string) result {
	var pe *tree.PathError
	switch {
	case errors.As(err, &pe):
		return errResult("bad_path", "%s: %s", pe.Reason, path)
	case errors.Is(err, tree.ErrNotFound):
		return errResult("not_found", "file not found: %s", path)
	case errors.Is(err, tree.ErrIsDir):
		return errResult("is_dir", "path is a directory: %s", path)
	case errors.Is(err, tree.ErrNotDir):
		return errResult("not_dir", "path component is a file: %s", path)
	}
	panic(fmt.Sprintf("unmapped tree error: %v", err))
}

func writable(path string) *result {
	if err := tree.ValidPath(path); err != nil {
		r := treeErr(err, path)
		return &r
	}
	if !strings.HasPrefix(path, "/artifact/") {
		r := errResult("read_only", "read-only path: %s", path)
		return &r
	}
	return nil
}

// Argument errors use fixed wording because these bytes are replayed.
func decodeArgs(tool string, raw json.RawMessage, v any) *result {
	failf := func(format string, args ...any) *result {
		r := errResult("bad_args", "%s: %s", tool, fmt.Sprintf(format, args...))
		return &r
	}
	if len(raw) == 0 {
		raw = json.RawMessage("{}")
	}
	var obj map[string]json.RawMessage
	dec := json.NewDecoder(bytes.NewReader(raw))
	if err := dec.Decode(&obj); err != nil || obj == nil {
		return failf("arguments are not a JSON object")
	}
	if dec.Decode(new(json.RawMessage)) != io.EOF {
		return failf("trailing data after arguments")
	}
	t := reflect.TypeOf(v).Elem()
	allowed := make(map[string]bool, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		name, _, _ := strings.Cut(t.Field(i).Tag.Get("json"), ",")
		allowed[name] = true
	}
	names := make([]string, 0, len(obj))
	for name := range obj {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if !allowed[name] {
			return failf("unknown argument %q", name)
		}
	}
	for _, name := range names {
		if bytes.Equal(bytes.TrimSpace(obj[name]), []byte("null")) {
			return failf("argument %q has the wrong type", name)
		}
	}
	if err := json.Unmarshal(raw, v); err != nil {
		var te *json.UnmarshalTypeError
		if errors.As(err, &te) && te.Field != "" {
			return failf("argument %q has the wrong type", te.Field)
		}
		return failf("malformed arguments")
	}
	return nil
}
