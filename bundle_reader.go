package repligraph

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"strings"

	"lukechampine.com/repligraph/harness/tree"
)

// A BundleReader decodes one output at a time. Read through io.EOF to check the
// transcript's ZIP checksum, and call Close even when stopping early.
type BundleReader struct {
	header     BundleHeader
	transcript io.ReadCloser
	remaining  uint64
	readUsage  Usage
	nextTag    byte
	hasTag     bool
	err        error
	started    bool
	closed     bool
}

// NewBundleReader loads the initial state and opens the transcript for streaming.
// ReadBundleHeader permits caller admission before this allocation. Callers also
// inspect and bound ZIP member sizes; the decoder imposes no local policy.
func NewBundleReader(z *zip.Reader) (*BundleReader, error) {
	for _, f := range z.File {
		if f.UncompressedSize64 >= uint64(^uint(0)>>1) {
			return nil, fmt.Errorf("bundle member %q is too large", f.Name)
		}
	}
	seen := make(map[string]bool)
	var transcript *zip.File
	for _, f := range z.File {
		if seen[f.Name] {
			return nil, fmt.Errorf("duplicate bundle member %q", f.Name)
		}
		seen[f.Name] = true
		isInitial := strings.HasPrefix(f.Name, "tree/")
		switch {
		case f.Name == "tree/":
			if f.Mode().Type() != fs.ModeDir || f.Method != zip.Store || f.CompressedSize64 != 0 || f.UncompressedSize64 != 0 || f.CRC32 != 0 {
				return nil, fmt.Errorf("invalid initial tree directory entry")
			}
		case isInitial:
			path := strings.TrimPrefix(f.Name, "tree")
			if path == "/env" || strings.HasPrefix(path, "/env/") {
				return nil, fmt.Errorf("/env is reserved for the environment")
			} else if !f.Mode().IsRegular() {
				return nil, fmt.Errorf("initial tree member %q is not a regular file", f.Name)
			} else if err := tree.ValidPath(path); err != nil {
				return nil, fmt.Errorf("initial tree member %q: %w", f.Name, err)
			}
		default:
			if (f.Name != "manifest.json" && f.Name != "transcript.log") || !f.Mode().IsRegular() {
				return nil, fmt.Errorf("unexpected bundle member %q", f.Name)
			}
		}
		if f.Name == "transcript.log" {
			transcript = f
		}
	}
	for _, name := range []string{"manifest.json", "transcript.log"} {
		if !seen[name] {
			return nil, fmt.Errorf("missing bundle member %q", name)
		}
	}

	b, err := ReadBundleHeader(z)
	if err != nil {
		return nil, err
	}
	// Count all implicit directories before allocating nodes or file contents.
	entryLimit := b.Usage.TreeEntries
	hasInitial, initialBytes := seen["tree/"], uint64(0)
	entries := make(map[string]struct{})
	for _, f := range z.File {
		if !strings.HasPrefix(f.Name, "tree/") || f.Name == "tree/" {
			continue
		}
		hasInitial = true
		if f.UncompressedSize64 > b.Usage.InitialTreeBytes-initialBytes {
			return nil, fmt.Errorf("initial tree bytes exceed declared usage")
		}
		initialBytes += f.UncompressedSize64
		for path := strings.TrimPrefix(f.Name, "tree"); path != ""; path = path[:strings.LastIndexByte(path, '/')] {
			if _, seen := entries[path]; seen {
				break // This prefix's parents were already counted.
			} else if uint64(len(entries)) == entryLimit {
				return nil, fmt.Errorf("initial tree entries exceed declared usage")
			}
			entries[path] = struct{}{}
		}
	}
	if hasInitial && initialBytes != b.Usage.InitialTreeBytes {
		return nil, fmt.Errorf("initial tree bytes do not match declared usage")
	}
	var initial *tree.Tree
	for _, f := range z.File {
		if !strings.HasPrefix(f.Name, "tree/") {
			continue
		}
		data, err := readBundleMember(f)
		if err != nil {
			return nil, err
		}
		if initial == nil {
			initial = new(tree.Tree)
		}
		if f.Name == "tree/" {
			if len(data) != 0 {
				return nil, fmt.Errorf("initial tree directory entry contains data")
			}
		} else if next, err := initial.Put(strings.TrimPrefix(f.Name, "tree"), tree.File{Content: data}); err != nil {
			return nil, fmt.Errorf("initial tree member %q: %w", f.Name, err)
		} else {
			*initial = next
		}
	}
	b.InitialTree = initial
	if initial != nil && initial.Root() != b.TreeStart {
		return nil, fmt.Errorf("bundled initial tree does not match tree_start")
	}
	r, err := transcript.Open()
	if err != nil {
		return nil, fmt.Errorf("transcript: %w", err)
	}
	br := &BundleReader{header: b, transcript: r, remaining: transcript.UncompressedSize64}
	if br.header.InitialContext, err = br.readTokens(b.Usage.ContextTokens); err != nil {
		r.Close()
		return nil, fmt.Errorf("initial context: %w", err)
	}
	return br, nil
}

// ReadBundleHeader reads only manifest.json. InitialTree and InitialContext are
// absent; NewBundleReader materializes those after the caller admits this usage.
// ZIP metadata remains available to bound archive and member bytes separately.
func ReadBundleHeader(z *zip.Reader) (BundleHeader, error) {
	var f *zip.File
	for _, member := range z.File {
		if member.Name != "manifest.json" {
			continue
		} else if f != nil {
			return BundleHeader{}, fmt.Errorf("duplicate bundle member %q", member.Name)
		}
		f = member
	}
	if f == nil {
		return BundleHeader{}, fmt.Errorf("missing bundle member %q", "manifest.json")
	} else if !f.Mode().IsRegular() || f.UncompressedSize64 >= uint64(^uint(0)>>1) {
		return BundleHeader{}, fmt.Errorf("invalid manifest member")
	}
	data, err := readBundleMember(f)
	if err != nil {
		return BundleHeader{}, err
	}
	var m bundleManifest
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&m); err != nil {
		return BundleHeader{}, fmt.Errorf("manifest: %w", err)
	} else if m.Version == nil || m.Model == nil || m.Inference == nil || m.Environment == nil || m.TreeStart == nil || m.Usage == nil || m.Seed == nil || m.Temperature == nil {
		return BundleHeader{}, fmt.Errorf("manifest is missing required fields")
	} else if *m.Version != FormatVersion {
		return BundleHeader{}, fmt.Errorf("unsupported bundle format version %d", *m.Version)
	} else if *m.Temperature < 0 {
		return BundleHeader{}, fmt.Errorf("temperature must be non-negative")
	} else if m.Usage.TreeBytes < m.Usage.InitialTreeBytes {
		return BundleHeader{}, fmt.Errorf("tree_bytes is less than initial_tree_bytes")
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return BundleHeader{}, fmt.Errorf("manifest contains trailing data")
	}
	return BundleHeader{
		Version: *m.Version, Model: *m.Model, Inference: *m.Inference,
		Environment: *m.Environment, TreeStart: *m.TreeStart, Usage: *m.Usage,
		Seed: *m.Seed, Temperature: *m.Temperature, InitialCall: m.InitialCall,
	}, nil
}

// readBundleMember reads one member within its already-checked declared size and
// verifies its checksum. The transcript is opened separately and read lazily.
func readBundleMember(f *zip.File) ([]byte, error) {
	r, err := f.Open()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", f.Name, err)
	}
	data, err := io.ReadAll(io.LimitReader(r, int64(f.UncompressedSize64)+1))
	r.Close()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", f.Name, err)
	} else if uint64(len(data)) != f.UncompressedSize64 {
		return nil, fmt.Errorf("%s: %w", f.Name, zip.ErrFormat)
	}
	return data, nil
}

// Header contains the metadata, optional tree, and initial context, without outputs.
// Its context, tree, and pending call share storage with the reader and must be
// treated as immutable.
func (r *BundleReader) Header() BundleHeader { return r.header }

func (r *BundleReader) readData(width, limit uint64) ([]byte, error) {
	if r.remaining < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	var count [8]byte
	if _, err := io.ReadFull(r.transcript, count[:]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	r.remaining -= 8
	n := binary.LittleEndian.Uint64(count[:])
	if n > limit {
		return nil, fmt.Errorf("record exceeds declared usage")
	}
	if n > r.remaining/width || n > uint64(^uint(0)>>1)/width {
		return nil, io.ErrUnexpectedEOF
	}
	data, err := io.ReadAll(io.LimitReader(r.transcript, int64(n*width)))
	r.remaining -= uint64(len(data))
	if err != nil {
		return nil, err
	} else if uint64(len(data)) != n*width {
		return nil, io.ErrUnexpectedEOF
	}
	return data, nil
}

func (r *BundleReader) readTokens(limit uint64) ([]uint32, error) {
	data, err := r.readData(4, limit)
	if err != nil {
		return nil, err
	}
	ids := make([]uint32, len(data)/4)
	for i := range ids {
		ids[i] = binary.LittleEndian.Uint32(data[i*4:])
	}
	return ids, nil
}

// peekTag buffers only the next record's tag. It also checks EOF and retains any
// read error, so replay can defer an error found while looking ahead until the
// caller has applied the preceding complete record.
func (r *BundleReader) peekTag() (tag byte, err error) {
	if r.err != nil {
		return 0, r.err
	} else if r.closed {
		return 0, io.ErrClosedPipe
	} else if r.hasTag {
		return r.nextTag, nil
	}
	defer func() {
		if err != nil {
			r.err = err
		}
	}()
	var b [1]byte
	if _, err := io.ReadFull(r.transcript, b[:]); err == io.EOF {
		if r.remaining != 0 {
			return 0, io.ErrUnexpectedEOF
		}
		u, want := r.readUsage, r.header.Usage
		if u.ModelTurns != want.ModelTurns || u.HarnessTurns != want.HarnessTurns || u.OutputTokens != want.OutputTokens || u.ToolOutputBytes != want.ToolOutputBytes {
			return 0, fmt.Errorf("transcript usage does not match manifest")
		}
		return 0, io.EOF
	} else if err != nil {
		return 0, err
	} else if r.remaining == 0 {
		return 0, zip.ErrFormat
	}
	r.nextTag, r.hasTag = b[0], true
	return r.nextTag, nil
}

// Next returns exactly one non-nil slice, including for an empty output.
// Outputs follow their order in transcript.log; returned slices belong to the caller.
// Parsing does not check execution order; Replay checks the model's protocol.
func (r *BundleReader) Next() (modelOutput []uint32, toolOutput []byte, err error) {
	r.started = true
	defer func() {
		if err != nil {
			r.err = err
		}
	}()
	tag, err := r.peekTag()
	if err != nil {
		return nil, nil, err
	}
	r.hasTag = false
	r.remaining--
	switch tag {
	case 1:
		if r.readUsage.ModelTurns == r.header.Usage.ModelTurns {
			return nil, nil, fmt.Errorf("model turns exceed declared usage")
		}
		output, err := r.readTokens(min(r.header.Usage.ContextTokens, r.header.Usage.OutputTokens-r.readUsage.OutputTokens))
		if err != nil {
			return nil, nil, fmt.Errorf("model output: %w", err)
		}
		r.readUsage.ModelTurns++
		r.readUsage.OutputTokens += uint64(len(output))
		return output, nil, nil
	case 2:
		if r.readUsage.HarnessTurns == r.header.Usage.HarnessTurns {
			return nil, nil, fmt.Errorf("harness turns exceed declared usage")
		}
		output, err := r.readData(1, r.header.Usage.ToolOutputBytes-r.readUsage.ToolOutputBytes)
		if err != nil {
			return nil, nil, fmt.Errorf("tool output: %w", err)
		}
		r.readUsage.HarnessTurns++
		r.readUsage.ToolOutputBytes += uint64(len(output))
		return nil, output, nil
	default:
		return nil, nil, fmt.Errorf("unknown record tag %d", tag)
	}
}

// Close releases the transcript reader without draining it or closing the archive.
func (r *BundleReader) Close() error {
	if !r.closed {
		r.closed = true
		if err := r.transcript.Close(); err != nil && (r.err == nil || r.err == io.EOF) {
			r.err = err
		}
		r.transcript = nil
	}
	if r.err == io.EOF {
		return nil
	}
	return r.err
}
