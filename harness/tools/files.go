package tools

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"unicode/utf8"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

// Even JSON's worst-case escaping fits in one untruncated tool result.
const maxReadBytes = 4096

type readArgs struct {
	Path     string  `json:"path"`
	Offset   *uint64 `json:"offset,omitempty"`
	Length   *uint64 `json:"length,omitempty"`
	Encoding string  `json:"encoding,omitempty"`
}

func applyRead(t tree.Tree, args json.RawMessage) result {
	var a readArgs
	if r := decodeArgs("read_file", args, &a); r != nil {
		return *r
	}
	f, err := t.Get(a.Path)
	if err != nil {
		return treeErr(err, a.Path)
	}
	// Preserve the original short-form result. Explicit range/encoding arguments
	// select paged output, which can reach every byte without truncation.
	if a.Offset == nil && a.Length == nil && a.Encoding == "" {
		if !utf8.Valid(f.Content) {
			return errResult("binary", "file is not valid UTF-8: %s", a.Path)
		}
		return result{"ok", string(f.Content)}
	}
	if a.Encoding == "" {
		a.Encoding = "utf8"
	}
	if a.Encoding != "utf8" && a.Encoding != "base64" {
		return errResult("bad_args", "read_file: encoding must be utf8 or base64")
	}
	var offset uint64
	length := uint64(maxReadBytes)
	if a.Offset != nil {
		offset = *a.Offset
	}
	if a.Length != nil {
		length = *a.Length
	}
	if offset > uint64(len(f.Content)) {
		return errResult("bad_args", "read_file: offset exceeds file size")
	} else if length == 0 || length > maxReadBytes {
		return errResult("bad_args", "read_file: length must be between 1 and %d", maxReadBytes)
	}
	end := offset + min(length, uint64(len(f.Content))-offset)
	if a.Encoding == "utf8" {
		for pos := offset; pos < end; {
			_, size := utf8.DecodeRune(f.Content[pos:])
			if size == 1 && f.Content[pos] >= utf8.RuneSelf {
				return errResult("binary", "range is not valid UTF-8; use base64: %s", a.Path)
			} else if pos+uint64(size) > end {
				end = pos
				break
			}
			pos += uint64(size)
		}
		if end == offset && offset < uint64(len(f.Content)) {
			return errResult("bad_args", "read_file: range cuts through a UTF-8 character; increase length or use base64")
		}
	}
	data := f.Content[offset:end]
	var content string
	if a.Encoding == "base64" {
		content = base64.StdEncoding.EncodeToString(data)
	} else {
		content = string(data)
	}
	body, _ := json.Marshal(struct {
		Offset     uint64 `json:"offset"`
		NextOffset uint64 `json:"next_offset"`
		Size       uint64 `json:"size"`
		Encoding   string `json:"encoding"`
		Content    string `json:"content"`
	}{offset, end, uint64(len(f.Content)), a.Encoding, content})
	return result{"ok", string(body)}
}

type writeArgs struct {
	Path     string `json:"path"`
	Content  string `json:"content"`
	Encoding string `json:"encoding,omitempty"`
	Append   bool   `json:"append,omitempty"`
}

// Check the logical size before allocating replacement content or placing a
// shared file at another path. Tree's cached byte count must also fit in an int.
func checkTreeBytes(total, removed, added, limit uint64) error {
	limit = min(limit, uint64(^uint(0)>>1))
	if removed > total || total-removed > limit || added > limit-(total-removed) {
		return fmt.Errorf("%w: task tree bytes", wasm.ErrResourceLimit)
	}
	return nil
}

func applyWrite(t tree.Tree, limits wasm.Limits, taskBytes uint64, args json.RawMessage) (result, tree.Tree, uint64, error) {
	var a writeArgs
	if r := decodeArgs("write_file", args, &a); r != nil {
		return *r, t, 0, nil
	} else if r := writable(a.Path); r != nil {
		return *r, t, 0, nil
	}
	// Resolve path errors before checking a local write ceiling. Only a valid
	// operation can be aborted for exceeding its resource allowance.
	f, pathErr := t.Get(a.Path)
	if pathErr != nil && !errors.Is(pathErr, tree.ErrNotFound) {
		return treeErr(pathErr, a.Path), t, 0, nil
	}
	checkSize := func(n uint64) error {
		if n > limits.WriteBytes {
			return fmt.Errorf("%w: builtin write bytes", wasm.ErrResourceLimit)
		}
		oldSize := uint64(len(f.Content))
		if a.Append {
			n += oldSize // Both lengths fit in an int, so their sum fits uint64.
		}
		return checkTreeBytes(taskBytes, oldSize, n, limits.MaxTreeBytes)
	}
	var data []byte
	switch a.Encoding {
	case "", "utf8":
		if err := checkSize(uint64(len(a.Content))); err != nil {
			return result{}, t, 0, err
		}
		data = []byte(a.Content)
	case "base64":
		var err error
		data, err = base64.StdEncoding.Strict().DecodeString(a.Content)
		if err != nil {
			return errResult("bad_args", "write_file: content is not valid base64"), t, 0, nil
		}
		if err := checkSize(uint64(len(data))); err != nil {
			return result{}, t, 0, err
		}
	default:
		return errResult("bad_args", "write_file: encoding must be utf8 or base64"), t, 0, nil
	}
	n := len(data)
	if a.Append {
		data = append(bytes.Clone(f.Content), data...)
	}
	nt, err := t.Put(a.Path, tree.File{Content: data})
	if err != nil {
		return treeErr(err, a.Path), t, 0, nil
	}
	verb := "wrote"
	if a.Append {
		verb = "appended"
	}
	return result{"ok", fmt.Sprintf("%s %d bytes: %s", verb, n, a.Path)}, nt, uint64(n), nil
}

func applyEdit(t tree.Tree, limits wasm.Limits, taskBytes uint64, args json.RawMessage) (result, tree.Tree, uint64, error) {
	var a struct {
		Path string `json:"path"`
		Old  string `json:"old"`
		New  string `json:"new"`
	}
	if r := decodeArgs("edit_file", args, &a); r != nil {
		return *r, t, 0, nil
	} else if r := writable(a.Path); r != nil {
		return *r, t, 0, nil
	} else if a.Old == "" {
		return errResult("bad_args", "edit_file: old must not be empty"), t, 0, nil
	}
	f, err := t.Get(a.Path)
	if err != nil {
		return treeErr(err, a.Path), t, 0, nil
	} else if !utf8.Valid(f.Content) {
		return errResult("binary", "file is not valid UTF-8: %s", a.Path), t, 0, nil
	}
	i := bytes.Index(f.Content, []byte(a.Old))
	if i < 0 {
		return errResult("no_match", "edit_file: old text was not found"), t, 0, nil
	} else if bytes.Contains(f.Content[i+1:], []byte(a.Old)) {
		return errResult("multiple_matches", "edit_file: old text must match exactly once"), t, 0, nil
	}
	size := uint64(len(f.Content)-len(a.Old)) + uint64(len(a.New))
	if size > limits.WriteBytes {
		return result{}, t, 0, fmt.Errorf("%w: builtin write bytes", wasm.ErrResourceLimit)
	}
	if err := checkTreeBytes(taskBytes, uint64(len(f.Content)), size, limits.MaxTreeBytes); err != nil {
		return result{}, t, 0, err
	}
	data := make([]byte, 0, int(size))
	data = append(data, f.Content[:i]...)
	data = append(data, a.New...)
	data = append(data, f.Content[i+len(a.Old):]...)
	nt, err := t.Put(a.Path, tree.File{Content: data})
	if err != nil {
		return treeErr(err, a.Path), t, 0, nil
	}
	return result{"ok", fmt.Sprintf("edited %s: replaced %d bytes with %d bytes", a.Path, len(a.Old), len(a.New))}, nt, uint64(len(data)), nil
}

func applyCopy(t tree.Tree, limits wasm.Limits, taskBytes uint64, args json.RawMessage, rename bool) (result, tree.Tree, error) {
	var a struct {
		Source      string `json:"source"`
		Destination string `json:"destination"`
	}
	tool, verb := "copy_file", "copied"
	if rename {
		tool, verb = "rename_file", "renamed"
	}
	if r := decodeArgs(tool, args, &a); r != nil {
		return *r, t, nil
	} else if r := writable(a.Destination); r != nil {
		return *r, t, nil
	}
	if rename {
		if r := writable(a.Source); r != nil {
			return *r, t, nil
		}
	}
	f, err := t.Get(a.Source)
	if err != nil {
		return treeErr(err, a.Source), t, nil
	}
	if !rename {
		old, err := t.Get(a.Destination)
		if err != nil && !errors.Is(err, tree.ErrNotFound) {
			return treeErr(err, a.Destination), t, nil
		}
		if err := checkTreeBytes(taskBytes, uint64(len(old.Content)), uint64(len(f.Content)), limits.MaxTreeBytes); err != nil {
			return result{}, t, err
		}
	}
	var nt tree.Tree
	if rename {
		nt, err = t.Rename(a.Source, a.Destination)
	} else {
		nt, err = t.Copy(a.Source, a.Destination)
	}
	if err != nil {
		return treeErr(err, a.Destination), t, nil
	}
	return result{"ok", fmt.Sprintf("%s %d bytes: %s -> %s", verb, len(f.Content), a.Source, a.Destination)}, nt, nil
}

func applyDelete(t tree.Tree, args json.RawMessage) (result, tree.Tree) {
	var a struct {
		Path string `json:"path"`
	}
	if r := decodeArgs("delete_file", args, &a); r != nil {
		return *r, t
	} else if r := writable(a.Path); r != nil {
		return *r, t
	}
	nt, err := t.Delete(a.Path)
	if err != nil {
		return treeErr(err, a.Path), t
	}
	return result{"ok", "deleted " + a.Path}, nt
}
