package tree

import (
	"bytes"
	"errors"
	"sort"
	"strings"

	"lukechampine.com/repligraph/blake3"
)

var (
	ErrNotFound = errors.New("not found")
	ErrIsDir    = errors.New("is a directory")
	ErrNotDir   = errors.New("not a directory")
)

type File struct {
	Content []byte
}

// Tree is an immutable workspace snapshot. The zero value is the empty tree.
type Tree struct{ root *dirNode }

func (t Tree) Root() blake3.Hash { return dirHash(t.root) }

// Stats describes the files and directories in a tree. Entries includes files
// and directories, excluding the root; Bytes counts file content only.
type Stats struct {
	Entries      int
	Files        int
	Bytes        int
	MaxFileBytes int
}

// Stats returns cached statistics for this snapshot in constant time.
func (t Tree) Stats() Stats {
	if t.root == nil {
		return Stats{}
	}
	return Stats{
		Entries:      t.root.entryCount,
		Files:        t.root.files,
		Bytes:        t.root.bytes,
		MaxFileBytes: t.root.maxFileBytes,
	}
}

type fileNode struct {
	content []byte
	h       blake3.Hash
}

type dirEntry struct {
	name string
	file *fileNode // exactly one of file/dir is non-nil
	dir  *dirNode
}

type dirNode struct {
	entries      []dirEntry // bytewise-sorted by name
	entryCount   int        // files and directories below this node
	files        int
	bytes        int // total content bytes at or below this node
	maxFileBytes int
	h            blake3.Hash
}

func newFile(f File) *fileNode {
	return &fileNode{content: bytes.Clone(f.Content), h: fileHash(f.Content)}
}

func newDirNode(entries []dirEntry) *dirNode {
	c, n, b, m := len(entries), 0, 0, 0
	for _, e := range entries {
		if e.dir != nil {
			c += e.dir.entryCount
			n += e.dir.files
			b += e.dir.bytes
			m = max(m, e.dir.maxFileBytes)
		} else {
			n++
			b += len(e.file.content)
			m = max(m, len(e.file.content))
		}
	}
	return &dirNode{entries: entries, entryCount: c, files: n, bytes: b, maxFileBytes: m, h: dirHashEntries(entries)}
}

func findEntry(d *dirNode, name string) (dirEntry, int, bool) {
	i := sort.Search(len(d.entries), func(k int) bool { return d.entries[k].name >= name })
	if i < len(d.entries) && d.entries[i].name == name {
		return d.entries[i], i, true
	}
	return dirEntry{}, i, false
}

func (t Tree) lookup(parts []string) (*dirNode, *fileNode, error) {
	d := t.root
	for i, name := range parts {
		if d == nil {
			return nil, nil, ErrNotFound
		}
		e, _, ok := findEntry(d, name)
		if !ok {
			return nil, nil, ErrNotFound
		}
		if e.file != nil {
			if i == len(parts)-1 {
				return nil, e.file, nil
			}
			return nil, nil, ErrNotDir
		}
		d = e.dir
	}
	return d, nil, nil
}

// Get returns a file whose Content must not be modified.
func (t Tree) Get(path string) (File, error) {
	parts, perr := splitPath(path)
	if perr != nil {
		return File{}, perr
	}
	if len(parts) == 0 {
		return File{}, ErrIsDir
	}
	_, f, err := t.lookup(parts)
	if err != nil {
		return File{}, err
	}
	if f == nil {
		return File{}, ErrIsDir
	}
	return File{Content: f.content}, nil
}

// Put copies f into a new snapshot, creating parent directories as needed.
func (t Tree) Put(path string, f File) (Tree, error) {
	parts, perr := splitPath(path)
	if perr != nil {
		return Tree{}, perr
	}
	if len(parts) == 0 {
		return Tree{}, ErrIsDir
	}
	root, err := putNode(t.root, parts, newFile(f))
	if err != nil {
		return Tree{}, err
	}
	return Tree{root: root}, nil
}

func putNode(d *dirNode, parts []string, f *fileNode) (*dirNode, error) {
	name := parts[0]
	var entries []dirEntry
	if d != nil {
		entries = d.entries
	}
	i := sort.Search(len(entries), func(k int) bool { return entries[k].name >= name })
	found := i < len(entries) && entries[i].name == name

	var ne dirEntry
	if len(parts) == 1 {
		if found && entries[i].dir != nil {
			return nil, ErrIsDir
		}
		ne = dirEntry{name: name, file: f}
	} else {
		var child *dirNode
		if found {
			if entries[i].file != nil {
				return nil, ErrNotDir
			}
			child = entries[i].dir
		}
		nc, err := putNode(child, parts[1:], f)
		if err != nil {
			return nil, err
		}
		ne = dirEntry{name: name, dir: nc}
	}

	out := make([]dirEntry, 0, len(entries)+1)
	out = append(out, entries[:i]...)
	out = append(out, ne)
	if found {
		out = append(out, entries[i+1:]...)
	} else {
		out = append(out, entries[i:]...)
	}
	return newDirNode(out), nil
}

// Copy places the source file at destination, replacing an existing file and
// creating parent directories as needed. The immutable content and its hash are
// shared; copying does not read or allocate the file's bytes.
func (t Tree) Copy(source, destination string) (Tree, error) {
	return t.transfer(source, destination, false)
}

// Rename moves the source file to destination, replacing an existing file,
// creating parent directories, and pruning empty source directories. Content and
// its hash are shared. A destination directory or descendant of source is an
// error, even if removing source would make that destination available.
func (t Tree) Rename(source, destination string) (Tree, error) {
	return t.transfer(source, destination, true)
}

func (t Tree) transfer(source, destination string, rename bool) (Tree, error) {
	src, perr := splitPath(source)
	if perr != nil {
		return Tree{}, perr
	} else if len(src) == 0 {
		return Tree{}, ErrIsDir
	}
	_, file, err := t.lookup(src)
	if err != nil {
		return Tree{}, err
	} else if file == nil {
		return Tree{}, ErrIsDir
	}
	dst, perr := splitPath(destination)
	if perr != nil {
		return Tree{}, perr
	} else if len(dst) == 0 {
		return Tree{}, ErrIsDir
	} else if source == destination {
		return t, nil
	}
	// Validate against the original snapshot. Deleting the source first could
	// turn a forbidden destination (its parent or descendant) into a valid one.
	_, existing, err := t.lookup(dst)
	if err != nil && !errors.Is(err, ErrNotFound) {
		return Tree{}, err
	} else if err == nil && existing == nil {
		return Tree{}, ErrIsDir
	}
	root := t.root
	if rename {
		root, err = deleteNode(root, src)
		if err != nil {
			return Tree{}, err
		}
	}
	root, err = putNode(root, dst, file)
	if err != nil {
		return Tree{}, err
	}
	return Tree{root: root}, nil
}

// Delete removes a file and prunes empty parent directories.
func (t Tree) Delete(path string) (Tree, error) {
	parts, perr := splitPath(path)
	if perr != nil {
		return Tree{}, perr
	}
	if len(parts) == 0 {
		return Tree{}, ErrIsDir
	}
	root, err := deleteNode(t.root, parts)
	if err != nil {
		return Tree{}, err
	}
	return Tree{root: root}, nil
}

func deleteNode(d *dirNode, parts []string) (*dirNode, error) {
	if d == nil {
		return nil, ErrNotFound
	}
	e, i, found := findEntry(d, parts[0])
	if !found {
		return nil, ErrNotFound
	}
	var replacement *dirEntry
	if len(parts) == 1 {
		if e.dir != nil {
			return nil, ErrIsDir
		}
	} else {
		if e.file != nil {
			return nil, ErrNotDir
		}
		nc, err := deleteNode(e.dir, parts[1:])
		if err != nil {
			return nil, err
		}
		if nc != nil {
			replacement = &dirEntry{name: e.name, dir: nc}
		}
	}
	out := make([]dirEntry, 0, len(d.entries))
	out = append(out, d.entries[:i]...)
	if replacement != nil {
		out = append(out, *replacement)
	}
	out = append(out, d.entries[i+1:]...)
	if len(out) == 0 {
		return nil, nil
	}
	return newDirNode(out), nil
}

func validRootName(name string) error {
	parts, err := splitPath("/" + name)
	if err != nil {
		return err
	} else if len(parts) != 1 {
		return &PathError{"name must be a single component"}
	}
	return nil
}

// Graft attaches sub at a new root directory; an empty sub leaves t unchanged.
func (t Tree) Graft(name string, sub Tree) (Tree, error) {
	if err := validRootName(name); err != nil {
		return Tree{}, err
	}
	var entries []dirEntry
	if t.root != nil {
		entries = t.root.entries
	}
	i := sort.Search(len(entries), func(k int) bool { return entries[k].name >= name })
	if i < len(entries) && entries[i].name == name {
		return Tree{}, &PathError{"graft target already exists"}
	} else if sub.root == nil {
		return t, nil
	}
	if err := sub.WalkFiles("/", func(path string, _ File) error {
		return ValidPath("/" + name + path)
	}); err != nil {
		return Tree{}, err
	}
	out := make([]dirEntry, 0, len(entries)+1)
	out = append(out, entries[:i]...)
	out = append(out, dirEntry{name: name, dir: sub.root})
	out = append(out, entries[i:]...)
	return Tree{root: newDirNode(out)}, nil
}

// Without removes a root file or directory; a missing name leaves t unchanged.
func (t Tree) Without(name string) (Tree, error) {
	if err := validRootName(name); err != nil {
		return Tree{}, err
	} else if t.root == nil {
		return t, nil
	}
	_, i, found := findEntry(t.root, name)
	if !found {
		return t, nil
	} else if len(t.root.entries) == 1 {
		return Tree{}, nil
	}
	out := make([]dirEntry, 0, len(t.root.entries)-1)
	out = append(out, t.root.entries[:i]...)
	out = append(out, t.root.entries[i+1:]...)
	return Tree{root: newDirNode(out)}, nil
}

// WalkFiles visits files at or below path in bytewise full-path order, stopping
// immediately if fn returns an error. It sorts only the entries of directories
// it reaches, without collecting file paths for the entire subtree. Content must
// not be modified.
func (t Tree) WalkFiles(path string, fn func(path string, f File) error) error {
	parts, perr := splitPath(path)
	if perr != nil {
		return perr
	}
	d, f, err := t.lookup(parts)
	if err != nil {
		return err
	}
	if f != nil {
		return fn(path, File{Content: f.content})
	}
	return walkFiles(strings.TrimSuffix(path, "/"), d, fn)
}

func walkFiles(prefix string, d *dirNode, fn func(string, File) error) error {
	if d == nil {
		return nil
	}
	// Directory names sort as prefixes ending in '/': /a-file must be visited
	// before /a/file even though the directory entry "a" sorts before "a-file".
	type item struct {
		name string
		dir  *dirNode
		file *fileNode
	}
	items := make([]item, len(d.entries))
	for i, e := range d.entries {
		name := e.name
		if e.dir != nil {
			name += "/"
		}
		items[i] = item{name, e.dir, e.file}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].name < items[j].name })
	for _, e := range items {
		path := prefix + "/" + e.name
		var err error
		if e.dir != nil {
			err = walkFiles(path[:len(path)-1], e.dir, fn)
		} else {
			err = fn(path, File{Content: e.file.content})
		}
		if err != nil {
			return err
		}
	}
	return nil
}
