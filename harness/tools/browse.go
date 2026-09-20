package tools

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"unicode/utf8"

	"lukechampine.com/repligraph/harness/tree"
)

// These operations construct bounded JSON results themselves, so render must
// never truncate their continuation cursors.
var errBrowseDone = errors.New("browse page complete")

func browseJSON(v any) []byte {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		panic(err) // Only strings, integers, and concrete structs are encoded.
	}
	return bytes.TrimSuffix(b.Bytes(), []byte("\n"))
}

func beneath(path, scope string) bool {
	return scope == "/" || path == scope || strings.HasPrefix(path, scope+"/")
}

type listArgs struct {
	Path  string `json:"path"`
	After string `json:"after"`
	Limit uint64 `json:"limit"`
}

type listedFile struct {
	Path string `json:"path"`
	Size uint64 `json:"size"`
}

type listPage struct {
	Files     []listedFile `json:"files"`
	NextAfter string       `json:"next_after,omitempty"`
}

func applyList(t tree.Tree, raw json.RawMessage) result {
	a := listArgs{Path: "/", Limit: 100}
	if r := decodeArgs("list_files", raw, &a); r != nil {
		return *r
	} else if a.Limit == 0 || a.Limit > 1000 {
		return errResult("bad_args", "list_files: limit must be between 1 and 1000")
	} else if a.After != "" {
		if err := tree.ValidPath(a.After); err != nil {
			return treeErr(err, a.After)
		} else if !beneath(a.After, a.Path) {
			return errResult("bad_args", "list_files: after must be at or below path")
		}
	}
	page := listPage{Files: make([]listedFile, 0)}
	encodedEntries := 0
	err := t.WalkFiles(a.Path, func(path string, f tree.File) error {
		if path <= a.After {
			return nil
		}
		entry := listedFile{path, uint64(len(f.Content))}
		n := len(browseJSON(entry))
		if len(page.Files) != 0 {
			n++ // Comma before this entry.
		}
		// Reserve enough space for a continuation even when this entry is the
		// last one that fits. Valid tree paths guarantee one entry always fits.
		withCursor := len(`{"files":[],"next_after":}`) + encodedEntries + n + len(browseJSON(path))
		if uint64(len(page.Files)) == a.Limit || withCursor > maxOutputBytes {
			page.NextAfter = page.Files[len(page.Files)-1].Path
			return errBrowseDone
		}
		page.Files = append(page.Files, entry)
		encodedEntries += n
		return nil
	})
	if err != nil && !errors.Is(err, errBrowseDone) {
		return treeErr(err, a.Path)
	}
	return result{"ok", string(browseJSON(page))}
}

const (
	maxSearchBytes = 1 << 20
	maxSearchFiles = 1000
	maxQueryBytes  = 4096
	// A valid tree path is at most 4096 bytes, and JSON escaping can at most
	// double it (control bytes are forbidden). Reserve room for a cursor in a
	// different file from the last match, including its offset and field names.
	maxSearchCursorBytes = 2*4096 + 100
)

type searchCursor struct {
	Path   string `json:"path"`
	Offset uint64 `json:"offset"`
}

type searchArgs struct {
	Path   string          `json:"path"`
	Query  string          `json:"query"`
	Cursor json.RawMessage `json:"cursor"`
	Limit  uint64          `json:"limit"`
}

type searchPage struct {
	Matches []searchCursor `json:"matches"`
	Next    *searchCursor  `json:"next,omitempty"`
}

// applySearch searches literal UTF-8 query bytes, including in binary files.
// Matches do not overlap. Cursors are byte offsets of the next candidate match;
// the per-page scan budget can produce an empty page with a continuation.
func applySearch(t tree.Tree, raw json.RawMessage) result {
	a := searchArgs{Path: "/", Limit: 50}
	if r := decodeArgs("search_files", raw, &a); r != nil {
		return *r
	} else if len(a.Query) == 0 || len(a.Query) > maxQueryBytes || !utf8.ValidString(a.Query) {
		return errResult("bad_args", "search_files: query must be 1 to 4096 UTF-8 bytes")
	} else if a.Limit == 0 || a.Limit > 200 {
		return errResult("bad_args", "search_files: limit must be between 1 and 200")
	}
	var cursor *searchCursor
	if len(a.Cursor) != 0 {
		cursor = new(searchCursor)
		if r := decodeArgs("search_files cursor", a.Cursor, cursor); r != nil {
			return *r
		} else if !beneath(cursor.Path, a.Path) {
			return errResult("bad_args", "search_files: cursor must be at or below path")
		}
		f, err := t.Get(cursor.Path)
		if err != nil {
			return treeErr(err, cursor.Path)
		} else if cursor.Offset > uint64(len(f.Content)) {
			return errResult("bad_args", "search_files: cursor offset exceeds file size")
		}
	}
	page := searchPage{Matches: make([]searchCursor, 0)}
	remaining, visited, encodedMatches := maxSearchBytes, 0, 0
	query := []byte(a.Query)
	stop := func(path string, offset int) error {
		page.Next = &searchCursor{path, uint64(offset)}
		return errBrowseDone
	}
	err := t.WalkFiles(a.Path, func(path string, f tree.File) error {
		pos := 0
		if cursor != nil {
			if path < cursor.Path {
				return nil
			} else if path == cursor.Path {
				pos = int(cursor.Offset)
			}
		}
		if remaining == 0 || visited == maxSearchFiles || uint64(len(page.Matches)) == a.Limit {
			return stop(path, pos)
		}
		visited++
		end := pos + min(remaining, len(f.Content)-pos)
		// Look ahead enough to find matches beginning before the scan boundary.
		// The cursor advances past the whole match, including this lookahead.
		lookahead := end + min(len(query)-1, len(f.Content)-end)
		for pos < end {
			start := pos
			i := bytes.Index(f.Content[pos:lookahead], query)
			if i < 0 || pos+i >= end {
				remaining -= end - pos
				pos = end
				break
			}
			match := searchCursor{path, uint64(pos + i)}
			next := pos + i + len(query)
			n := len(browseJSON(match))
			if len(page.Matches) != 0 {
				n++
			}
			withCursor := len(`{"matches":[],"next":}`) + encodedMatches + n + maxSearchCursorBytes
			if withCursor > maxOutputBytes {
				return stop(path, pos+i)
			}
			page.Matches = append(page.Matches, match)
			encodedMatches += n
			remaining -= min(remaining, next-start)
			pos = next
			if uint64(len(page.Matches)) == a.Limit {
				break
			}
		}
		if pos < len(f.Content) {
			return stop(path, pos)
		}
		return nil
	})
	if err != nil && !errors.Is(err, errBrowseDone) {
		return treeErr(err, a.Path)
	}
	return result{"ok", string(browseJSON(page))}
}
