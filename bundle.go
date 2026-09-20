package repligraph

import (
	"archive/zip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"

	"lukechampine.com/repligraph/blake3"
	"lukechampine.com/repligraph/harness/tree"
)

// FormatVersion is the supported bundle format version. Version 0 is unstable.
const FormatVersion uint64 = 0

// A BundleHeader contains a bundle's metadata and initial state. Model and tool
// outputs follow separately, in execution order, through BundleReader or BundleWriter.
type BundleHeader struct {
	Version     uint64 // must equal FormatVersion
	Model       string
	Inference   string
	Environment string
	TreeStart   blake3.Hash // initial task tree hash
	Usage       Usage
	Seed        uint64 // keys the stream of per-turn sampling seeds
	Temperature float64

	InitialTree    *tree.Tree // task files; nil when supplied separately
	InitialContext []uint32
	InitialCall    *ToolCall // pending call; nil when the next turn is inference
}

type bundleManifest struct {
	Version     *uint64      `json:"version"`
	Model       *string      `json:"model"`
	Inference   *string      `json:"inference"`
	Environment *string      `json:"environment"`
	TreeStart   *blake3.Hash `json:"tree_start"`
	Usage       *Usage       `json:"usage"`
	Seed        *uint64      `json:"seed"`
	Temperature *float64     `json:"temperature"`
	InitialCall *ToolCall    `json:"initial_call,omitempty"`
}

// A BundleWriter appends model and tool outputs to a .zip bundle in execution
// order. It encodes records without parsing model outputs; use Recorder to check
// the ordering and derive tool calls while recording.
type BundleWriter struct {
	z           *zip.Writer
	transcript  io.Writer
	err         error
	closed      bool
	manifest    bundleManifest
	measured    Usage
	bundledTree bool
}

// NewBundleWriter writes optional tree/ files and the initial context. Close
// writes the manifest with final measured usage; b.Usage is not used.
func NewBundleWriter(w io.Writer, b BundleHeader) (_ *BundleWriter, err error) {
	if b.Version != FormatVersion {
		return nil, fmt.Errorf("unsupported bundle format version %d", b.Version)
	} else if err := b.InitialCall.validate(); err != nil {
		return nil, err
	} else if b.Temperature < 0 {
		return nil, fmt.Errorf("temperature must be non-negative")
	}
	if b.InitialTree != nil {
		if err := validateTaskTree(*b.InitialTree); err != nil {
			return nil, err
		} else if b.InitialTree.Root() != b.TreeStart {
			return nil, fmt.Errorf("bundled initial tree does not match tree_start")
		}
	}
	bw := &BundleWriter{z: zip.NewWriter(w), bundledTree: b.InitialTree != nil}
	bw.measured.ContextTokens = uint64(len(b.InitialContext))
	if b.InitialTree != nil {
		stats := b.InitialTree.Stats()
		bw.measured.TreeEntries, bw.measured.InitialTreeBytes = uint64(stats.Entries), uint64(stats.Bytes)
	}
	defer func() {
		if err != nil {
			bw.err = err
			bw.Close(Usage{})
		}
	}()
	b.InitialCall = cloneCall(b.InitialCall)
	bw.manifest = bundleManifest{
		Version: &b.Version, Model: &b.Model, Inference: &b.Inference,
		Environment: &b.Environment, TreeStart: &b.TreeStart,
		Seed: &b.Seed, Temperature: &b.Temperature, InitialCall: b.InitialCall,
	}
	if b.InitialTree != nil {
		h := &zip.FileHeader{Name: "tree/"}
		h.SetMode(fs.ModeDir | 0755)
		if _, err := bw.z.CreateHeader(h); err != nil {
			return nil, err
		}
		if err := b.InitialTree.WalkFiles("/", func(path string, file tree.File) error {
			f, err := bw.z.Create("tree" + path)
			if err != nil {
				return err
			}
			_, err = f.Write(file.Content)
			return err
		}); err != nil {
			return nil, fmt.Errorf("initial tree: %w", err)
		}
	}
	bw.transcript, err = bw.z.Create("transcript.log")
	if err != nil {
		return nil, err
	}
	if err := bw.write(uint64(len(b.InitialContext)), b.InitialContext); err != nil {
		return nil, err
	}
	b.InitialContext, b.InitialTree = nil, nil
	return bw, nil
}

func (w *BundleWriter) write(values ...any) error {
	if w.err != nil {
		return w.err
	} else if w.closed {
		return io.ErrClosedPipe
	}
	for _, v := range values {
		if w.err = binary.Write(w.transcript, binary.LittleEndian, v); w.err != nil {
			return w.err
		}
	}
	return nil
}

func (w *BundleWriter) WriteModelOutput(output []uint32) error {
	if w.err != nil {
		return w.err
	} else if w.closed {
		return io.ErrClosedPipe
	}
	if uint64(len(output)) > ^uint64(0)-w.measured.OutputTokens || w.measured.ModelTurns == ^uint64(0) {
		w.err = fmt.Errorf("transcript usage overflow")
		return w.err
	}
	if err := w.write(uint8(1), uint64(len(output)), output); err != nil {
		return err
	}
	w.measured.ModelTurns++
	w.measured.OutputTokens += uint64(len(output))
	w.measured.ContextTokens = max(w.measured.ContextTokens, uint64(len(output)))
	return nil
}

func (w *BundleWriter) WriteToolOutput(output []byte) error {
	if w.err != nil {
		return w.err
	} else if w.closed {
		return io.ErrClosedPipe
	}
	if uint64(len(output)) > ^uint64(0)-w.measured.ToolOutputBytes || w.measured.HarnessTurns == ^uint64(0) {
		w.err = fmt.Errorf("transcript usage overflow")
		return w.err
	}
	if err := w.write(uint8(2), uint64(len(output)), output); err != nil {
		return err
	}
	w.measured.HarnessTurns++
	w.measured.ToolOutputBytes += uint64(len(output))
	return nil
}

func (w *BundleWriter) checkUsage(u Usage) error {
	m := w.measured
	if u.ModelTurns != m.ModelTurns || u.HarnessTurns != m.HarnessTurns || u.OutputTokens != m.OutputTokens || u.ToolOutputBytes != m.ToolOutputBytes {
		return fmt.Errorf("transcript usage does not match recorded outputs")
	} else if u.ContextTokens < m.ContextTokens {
		return fmt.Errorf("context_tokens is less than recorded token length")
	} else if u.TreeBytes < u.InitialTreeBytes {
		return fmt.Errorf("tree_bytes is less than initial_tree_bytes")
	} else if w.bundledTree && (u.InitialTreeBytes != m.InitialTreeBytes || u.TreeEntries < m.TreeEntries) {
		return fmt.Errorf("initial tree usage does not match bundled tree")
	}
	return nil
}

func (w *BundleWriter) abort(err error) error {
	if !w.closed && w.err == nil {
		w.err = err
	}
	return w.Close(Usage{})
}

// Close writes the final manifest and finishes the archive without closing its
// destination. Usage must describe the completed computations. The writer checks
// transcript counts and initial state; execution-specific measurements belong to
// the caller. Repeated Close calls retain the first outcome.
func (w *BundleWriter) Close(usage Usage) error {
	if !w.closed {
		w.closed = true
		if w.err == nil {
			w.err = w.checkUsage(usage)
		}
		if w.err == nil {
			w.manifest.Usage = &usage
			var f io.Writer
			if f, w.err = w.z.Create("manifest.json"); w.err == nil {
				if err := json.NewEncoder(f).Encode(w.manifest); err != nil {
					w.err = fmt.Errorf("manifest: %w", err)
				}
			}
		}
		if err := w.z.Close(); w.err == nil {
			w.err = err
		}
		w.z, w.transcript = nil, nil
	}
	return w.err
}
