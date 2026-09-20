package wasmtime

import (
	"errors"
	"sort"
	"strings"

	"lukechampine.com/repligraph/harness/tree"
)

// overlay is a copy-on-write view of the session tree. Only /artifact is
// writable. Empty directories survive within a run but disappear at commit.
type overlay struct {
	base        tree.Tree
	files       map[string][]byte // written / created files
	deleted     map[string]bool   // tombstoned paths
	dirs        map[string]bool   // all live task directories, including empty ones
	entries     uint64            // task files and directories, excluding / and /env
	peakEntries uint64
	maxEntries  uint64
	bytes       uint64 // logical task file sizes; content shared by paths counts per path
	peakBytes   uint64
	maxBytes    uint64
}

func newOverlay(base tree.Tree, maxEntries, maxBytes uint64) *overlay {
	o := &overlay{
		base:       base,
		files:      make(map[string][]byte),
		deleted:    make(map[string]bool),
		dirs:       make(map[string]bool),
		maxEntries: maxEntries,
		maxBytes:   maxBytes,
	}
	task, _ := base.Without("env")
	task.WalkFiles("/", func(p string, _ tree.File) error {
		o.entries++
		for _, d := range o.missingDirs(p) {
			o.dirs[d] = true
			o.entries++
		}
		return nil
	})
	o.peakEntries = o.entries
	o.bytes = uint64(task.Stats().Bytes)
	o.peakBytes = o.bytes
	return o
}

// resolvePath resolves a guest path against the canonical directory dir,
// removes empty and dot segments, and clamps ".." at the root.
func resolvePath(dir, p string) (string, errno) {
	var parts []string
	if !strings.HasPrefix(p, "/") && dir != "/" {
		parts = strings.Split(dir[1:], "/")
	}
	for _, seg := range strings.Split(p, "/") {
		switch seg {
		case "", ".":
		case "..":
			if len(parts) > 0 {
				parts = parts[:len(parts)-1]
			}
		default:
			parts = append(parts, seg)
		}
	}
	out := "/" + strings.Join(parts, "/")
	if err := tree.ValidPath(out); err != nil {
		return "", pathErrno(err)
	}
	return out, errSuccess
}

const artifactRoot = "/artifact"

func writable(p string) bool { return strings.HasPrefix(p, artifactRoot+"/") }

func (o *overlay) fileContent(p string) ([]byte, bool) {
	if c, ok := o.files[p]; ok {
		return c, true
	}
	if o.deleted[p] {
		return nil, false
	}
	f, err := o.base.Get(p)
	if err != nil {
		return nil, false
	}
	return f.Content, true
}

func (o *overlay) fileExists(p string) bool {
	_, ok := o.fileContent(p)
	return ok
}

func (o *overlay) dirExists(p string) bool {
	if p == "/" || p == artifactRoot {
		return true
	}
	if p == "/env" || strings.HasPrefix(p, "/env/") {
		_, err := o.base.Get(p)
		return errors.Is(err, tree.ErrIsDir)
	}
	return o.dirs[p]
}

func (o *overlay) parentDirExists(p string) bool {
	i := strings.LastIndexByte(p, '/')
	if i == 0 {
		return true
	}
	return o.dirExists(p[:i])
}

func (o *overlay) fileOnPath(p string) bool {
	for i := strings.LastIndexByte(p, '/'); i > 0; i = strings.LastIndexByte(p[:i], '/') {
		if o.fileExists(p[:i]) {
			return true
		}
	}
	return false
}

func (o *overlay) missingDirs(p string) []string {
	var dirs []string
	for i := strings.LastIndexByte(p, '/'); i > 0; i = strings.LastIndexByte(p[:i], '/') {
		if !o.dirs[p[:i]] {
			dirs = append(dirs, p[:i])
		}
	}
	return dirs
}

func (o *overlay) checkEntries(add, remove uint64) {
	if o.entries-remove > o.maxEntries || add > o.maxEntries-(o.entries-remove) {
		abortResource("tree entries")
	}
}

// checkBytes checks the logical size after replacing remove bytes with add
// bytes. WASI writes call this before allocating their replacement buffer.
func (o *overlay) checkBytes(add, remove uint64) {
	// The committed tree stores its cached byte count in an int.
	limit := min(o.maxBytes, uint64(^uint(0)>>1))
	remaining := o.bytes - remove
	if remove > o.bytes || remaining > limit || add > limit-remaining {
		abortResource("tree bytes")
	}
}

// putFile and removeFile mutate content only; their callers reserve entries
// and update the live directory set before committing the operation.
func (o *overlay) putFile(p string, content []byte) {
	o.files[p] = content
	delete(o.deleted, p)
}

func (o *overlay) removeFile(p string) {
	delete(o.files, p)
	if _, err := o.base.Get(p); err == nil {
		o.deleted[p] = true
	} else {
		delete(o.deleted, p)
	}
}

func (o *overlay) setFile(p string, content []byte) errno {
	if !writable(p) {
		return errRofs
	}
	if o.dirExists(p) {
		return errIsdir
	} else if o.fileOnPath(p) {
		return errNotdir
	}
	old, exists := o.fileContent(p)
	o.checkBytes(uint64(len(content)), uint64(len(old)))
	if !exists {
		dirs := o.missingDirs(p)
		o.checkEntries(uint64(len(dirs))+1, 0)
		for _, d := range dirs {
			o.dirs[d] = true
		}
		o.entries += uint64(len(dirs)) + 1
		o.peakEntries = max(o.peakEntries, o.entries)
	}
	o.putFile(p, content)
	o.bytes = o.bytes - uint64(len(old)) + uint64(len(content))
	o.peakBytes = max(o.peakBytes, o.bytes)
	return errSuccess
}

func (o *overlay) unlink(p string) errno {
	if !writable(p) {
		if o.fileExists(p) || o.dirExists(p) {
			return errRofs
		}
		return errNoent
	}
	content, exists := o.fileContent(p)
	if !exists {
		if o.dirExists(p) {
			return errIsdir
		}
		return errNoent
	}
	o.removeFile(p)
	o.entries--
	o.bytes -= uint64(len(content))
	return errSuccess
}

func (o *overlay) mkdir(p string) errno {
	if p == "/" || p == artifactRoot {
		return errExist
	}
	if !writable(p) {
		return errRofs
	}
	if o.fileExists(p) || o.dirExists(p) {
		return errExist
	}
	if !o.parentDirExists(p) {
		return errNoent
	}
	dirs := o.missingDirs(p)
	o.checkEntries(uint64(len(dirs))+1, 0)
	for _, d := range dirs {
		o.dirs[d] = true
	}
	o.dirs[p] = true
	o.entries += uint64(len(dirs)) + 1
	o.peakEntries = max(o.peakEntries, o.entries)
	return errSuccess
}

func (o *overlay) rmdir(p string) errno {
	if !writable(p) {
		if p == "/" || p == artifactRoot || o.dirExists(p) {
			return errRofs
		}
		return errNoent
	}
	if o.fileExists(p) {
		return errNotdir
	}
	if !o.dirExists(p) {
		return errNoent
	}
	if len(o.readdir(p)) > 0 {
		return errNotempty
	}
	delete(o.dirs, p)
	o.entries--
	return errSuccess
}

func (o *overlay) rename(old, new string) errno {
	if !writable(old) {
		if o.fileExists(old) || o.dirExists(old) || old == "/" || old == artifactRoot {
			return errRofs
		}
		return errNoent
	}
	if !writable(new) {
		return errRofs
	}
	if old == new {
		if !o.fileExists(old) && !o.dirExists(old) {
			return errNoent
		}
		return errSuccess
	}
	if o.fileOnPath(new) {
		return errNotdir
	}
	_, sourceFile := o.fileContent(old)
	if sourceFile {
		if o.dirExists(new) {
			return errIsdir
		}
	} else {
		if !o.dirExists(old) {
			return errNoent
		}
		if o.fileExists(new) {
			return errNotdir
		}
		if strings.HasPrefix(new+"/", old+"/") || strings.HasPrefix(old+"/", new+"/") {
			return errInval
		}
		if o.dirExists(new) && len(o.readdir(new)) > 0 {
			return errNotempty
		}
	}

	// Collect and validate the entire move before changing content or quota.
	// Renames consume entries only for newly created parents and release an
	// entry when replacing an existing target; moving a full subtree is neutral.
	type move struct {
		from, to string
		content  []byte
		dir      bool
	}
	var moves []move
	if sourceFile {
		c, _ := o.fileContent(old)
		moves = append(moves, move{old, new, c, false})
	} else {
		for d := range o.dirs {
			if d == old || strings.HasPrefix(d, old+"/") {
				moves = append(moves, move{d, new + d[len(old):], nil, true})
			}
		}
		o.base.WalkFiles(old, func(p string, f tree.File) error {
			if _, shadowed := o.files[p]; !shadowed && !o.deleted[p] {
				moves = append(moves, move{p, new + p[len(old):], f.Content, false})
			}
			return nil
		})
		for p, c := range o.files {
			if strings.HasPrefix(p, old+"/") {
				moves = append(moves, move{p, new + p[len(old):], c, false})
			}
		}
	}
	sort.Slice(moves, func(i, j int) bool { return moves[i].from < moves[j].from })
	for _, m := range moves {
		if err := tree.ValidPath(m.to); err != nil {
			return pathErrno(err)
		}
	}
	dirs := o.missingDirs(new)
	var replaced uint64
	if o.fileExists(new) || o.dirExists(new) {
		replaced = 1
	}
	o.checkEntries(uint64(len(dirs)), replaced)
	replacedContent, _ := o.fileContent(new)
	for _, m := range moves {
		if m.dir {
			delete(o.dirs, m.from)
		} else {
			o.removeFile(m.from)
		}
	}
	for _, d := range dirs {
		o.dirs[d] = true
	}
	for _, m := range moves {
		if m.dir {
			o.dirs[m.to] = true
		} else {
			o.putFile(m.to, m.content)
		}
	}
	o.entries = o.entries - replaced + uint64(len(dirs))
	o.peakEntries = max(o.peakEntries, o.entries)
	o.bytes -= uint64(len(replacedContent))
	return errSuccess
}

type dirent struct {
	name string
	dir  bool
	size int
}

func (o *overlay) readdir(p string) []dirent {
	prefix := p + "/"
	if p == "/" {
		prefix = "/"
	}
	type info struct {
		dir  bool
		size int
	}
	children := make(map[string]info)
	if p == "/" {
		children["artifact"] = info{dir: true}
	}
	child := func(full string) (string, bool, bool) { // name, isLeaf, under
		if !strings.HasPrefix(full, prefix) || full == p {
			return "", false, false
		}
		rest := full[len(prefix):]
		if i := strings.IndexByte(rest, '/'); i >= 0 {
			return rest[:i], false, true
		}
		return rest, true, true
	}
	o.base.WalkFiles(p, func(fp string, f tree.File) error {
		name, leaf, ok := child(fp)
		if !ok {
			return nil
		}
		if leaf {
			if !o.deleted[fp] {
				if _, shadowed := o.files[fp]; !shadowed {
					children[name] = info{dir: false, size: len(f.Content)}
				}
			}
		} else if _, seen := children[name]; !seen {
			if o.dirExists(prefix + name) {
				children[name] = info{dir: true}
			}
		}
		return nil
	})
	for fp, c := range o.files {
		if name, leaf, ok := child(fp); ok {
			if leaf {
				children[name] = info{dir: false, size: len(c)}
			} else {
				children[name] = info{dir: true}
			}
		}
	}
	for d := range o.dirs {
		if name, leaf, ok := child(d); ok {
			if leaf || o.dirExists(prefix+name) {
				children[name] = info{dir: true}
			}
		}
	}
	names := make([]string, 0, len(children))
	for name := range children {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]dirent, len(names))
	for i, name := range names {
		out[i] = dirent{name: name, dir: children[name].dir, size: children[name].size}
	}
	return out
}

// commit applies the overlay to the session tree, dropping empty directories.
func (o *overlay) commit() (tree.Tree, error) {
	t := o.base
	deleted := make([]string, 0, len(o.deleted))
	for p := range o.deleted {
		deleted = append(deleted, p)
	}
	sort.Strings(deleted)
	for _, p := range deleted {
		if nt, err := t.Delete(p); err == nil {
			t = nt
		}
	}
	written := make([]string, 0, len(o.files))
	for p := range o.files {
		written = append(written, p)
	}
	sort.Strings(written)
	for _, p := range written {
		nt, err := t.Put(p, tree.File{Content: o.files[p]})
		if err != nil {
			return tree.Tree{}, err
		}
		t = nt
	}
	return t, nil
}
