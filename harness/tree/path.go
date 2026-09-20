package tree

import (
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// PathError's reason is model-visible.
type PathError struct{ Reason string }

func (e *PathError) Error() string { return e.Reason }

const (
	maxPathBytes      = 4096
	maxComponentBytes = 255
)

func ValidPath(p string) error {
	if _, perr := splitPath(p); perr != nil {
		return perr
	}
	return nil
}

func splitPath(p string) ([]string, *PathError) {
	if len(p) > maxPathBytes {
		return nil, &PathError{"path is too long"}
	}
	if !strings.HasPrefix(p, "/") {
		return nil, &PathError{"path must be absolute"}
	}
	if !utf8.ValidString(p) {
		return nil, &PathError{"path is not valid UTF-8"}
	}
	if !norm.NFC.IsNormalString(p) {
		return nil, &PathError{"path is not NFC-normalized"}
	}
	for i := 0; i < len(p); i++ {
		if p[i] < 0x20 || p[i] == 0x7f {
			return nil, &PathError{"path contains control characters"}
		}
	}
	if p == "/" {
		return nil, nil
	}
	parts := strings.Split(p[1:], "/")
	for _, part := range parts {
		switch {
		case part == "":
			return nil, &PathError{"path contains an empty component"}
		case part == "." || part == "..":
			return nil, &PathError{"path contains '.' or '..'"}
		case len(part) > maxComponentBytes:
			return nil, &PathError{"path component is too long"}
		}
	}
	return parts, nil
}
