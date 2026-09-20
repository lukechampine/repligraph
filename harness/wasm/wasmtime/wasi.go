package wasmtime

import (
	"encoding/binary"
	"io"

	wt "github.com/bytecodealliance/wasmtime-go/v38"

	"lukechampine.com/repligraph/blake3"
	"lukechampine.com/repligraph/harness/wasm"
)

// WASI preview 1 file types and rights bits (the ones we distinguish).
const (
	filetypeCharDevice = 2
	filetypeDirectory  = 3
	filetypeRegular    = 4

	rightFDRead    = 1 << 1
	rightFDWrite   = 1 << 6
	rightFDReaddir = 1 << 14

	oflagCreat     = 1 << 0
	oflagDirectory = 1 << 1
	oflagExcl      = 1 << 2
	oflagTrunc     = 1 << 3

	fdflagAppend = 1 << 0
)

type fdKind uint8

const (
	fdStdin fdKind = iota
	fdStdout
	fdStderr
	fdDir
	fdFile
)

type fdEntry struct {
	kind                fdKind
	path                string // descriptors look up the current file at this path
	off                 uint64
	read, write, append bool
}

type run struct {
	ov             *overlay
	limits         wasm.Limits
	clock, written uint64
	random         io.Reader
	fds            map[int32]*fdEntry
	nextFD         int32
	args, env      []string
	stdin          []byte
	stdinOff       int
	stdout, stderr []byte
	exited         bool
	exitCode       uint32
}

func newRun(ov *overlay, req wasm.Request) *run {
	return &run{
		ov:     ov,
		limits: req.Limits,
		random: blake3.XOF(req.RandomSeed, blake3.DomainRandom),
		fds: map[int32]*fdEntry{
			0: {kind: fdStdin, read: true},
			1: {kind: fdStdout, write: true},
			2: {kind: fdStderr, write: true},
			3: {kind: fdDir, path: "/", read: true, write: true},
		},
		nextFD: 4,
		args:   req.Args,
		env:    req.Env,
		stdin:  req.Stdin,
	}
}

func (r *run) mem(c *wt.Caller) []byte {
	ext := c.GetExport("memory")
	if ext == nil {
		return nil
	}
	defer ext.Close()
	m := ext.Memory()
	if m == nil {
		return nil
	}
	return m.UnsafeData(c)
}

func memRange(mem []byte, ptr, n int32) ([]byte, bool) {
	return memBytes(mem, ptr, uint64(uint32(n)))
}

func putU32(mem []byte, ptr int32, v uint32) bool {
	b, ok := memRange(mem, ptr, 4)
	if ok {
		binary.LittleEndian.PutUint32(b, v)
	}
	return ok
}

func putU64(mem []byte, ptr int32, v uint64) bool {
	b, ok := memRange(mem, ptr, 8)
	if ok {
		binary.LittleEndian.PutUint64(b, v)
	}
	return ok
}

func memBytes(mem []byte, ptr int32, n uint64) ([]byte, bool) {
	// Wasm i32 parameters carry unsigned wasm32 addresses and lengths. Widen
	// before adding so ranges can cross 2 GiB without signed overflow, and reject
	// ranges past the end of memory instead of wrapping at 4 GiB.
	start := uint64(uint32(ptr))
	if start > uint64(len(mem)) || n > uint64(len(mem))-start {
		return nil, false
	}
	end := start + n
	return mem[start:end:end], true
}

func (r *run) guestPath(mem []byte, fd, ptr, n int32) (string, errno) {
	e, ok := r.fds[fd]
	if !ok {
		return "", errBadf
	}
	if e.kind != fdDir {
		return "", errNotdir
	}
	b, ok := memRange(mem, ptr, n)
	if !ok {
		return "", errFault
	}
	return resolvePath(e.path, string(b))
}

func (r *run) define(linker *wt.Linker) error {
	const m = "wasi_snapshot_preview1"
	type def struct {
		name string
		fn   any
	}
	defs := []def{
		{"args_sizes_get", func(c *wt.Caller, argc, bufSize int32) int32 {
			return r.sizesGet(c, r.args, argc, bufSize)
		}},
		{"args_get", func(c *wt.Caller, argv, buf int32) int32 {
			return r.listGet(c, r.args, argv, buf)
		}},
		{"environ_sizes_get", func(c *wt.Caller, argc, bufSize int32) int32 {
			return r.sizesGet(c, r.env, argc, bufSize)
		}},
		{"environ_get", func(c *wt.Caller, argv, buf int32) int32 {
			return r.listGet(c, r.env, argv, buf)
		}},
		{"clock_res_get", func(c *wt.Caller, id, out int32) int32 {
			if id < 0 || id > 3 {
				return errInval
			}
			if !putU64(r.mem(c), out, clockQuantumNanos) {
				return errFault
			}
			return errSuccess
		}},
		{"clock_time_get", func(c *wt.Caller, id int32, precision int64, out int32) int32 {
			if id < 0 || id > 3 {
				return errInval
			}
			mem := r.mem(c)
			if _, ok := memRange(mem, out, 8); !ok {
				return errFault
			}
			putU64(mem, out, r.now(id))
			return errSuccess
		}},
		{"random_get", func(c *wt.Caller, buf, n int32) int32 {
			b, ok := memRange(r.mem(c), buf, n)
			if !ok {
				return errFault
			}
			r.random.Read(b) // XOF reads never fail
			return errSuccess
		}},
		{"sched_yield", func(c *wt.Caller) int32 { return errSuccess }},
		{"fd_close", r.fdClose},
		{"fd_fdstat_get", r.fdFdstatGet},
		{"fd_fdstat_set_flags", r.fdFdstatSetFlags},
		{"fd_filestat_get", r.fdFilestatGet},
		{"fd_filestat_set_size", r.fdFilestatSetSize},
		{"fd_filestat_set_times", func(c *wt.Caller, fd int32, atim, mtim int64, flags int32) int32 {
			if _, ok := r.fds[fd]; !ok {
				return errBadf
			}
			return errSuccess // the workspace has no timestamps; setting them is a no-op
		}},
		{"fd_prestat_get", r.fdPrestatGet},
		{"fd_prestat_dir_name", r.fdPrestatDirName},
		{"fd_read", r.fdRead},
		{"fd_pread", r.fdPread},
		{"fd_write", r.fdWrite},
		{"fd_pwrite", r.fdPwrite},
		{"fd_seek", r.fdSeek},
		{"fd_tell", r.fdTell},
		{"fd_readdir", r.fdReaddir},
		{"fd_sync", r.fdNoopSync},
		{"fd_datasync", r.fdNoopSync},
		{"fd_advise", func(c *wt.Caller, fd int32, off, n int64, advice int32) int32 {
			if _, ok := r.fds[fd]; !ok {
				return errBadf
			}
			return errSuccess
		}},
		{"path_open", r.pathOpen},
		{"path_filestat_get", r.pathFilestatGet},
		{"path_filestat_set_times", func(c *wt.Caller, fd, flags, p, plen int32, atim, mtim int64, fstflags int32) int32 {
			_, rc := r.guestPath(r.mem(c), fd, p, plen)
			return rc
		}},
		{"path_create_directory", func(c *wt.Caller, fd, p, plen int32) int32 {
			path, rc := r.guestPath(r.mem(c), fd, p, plen)
			if rc != errSuccess {
				return rc
			}
			return r.ov.mkdir(path)
		}},
		{"path_remove_directory", func(c *wt.Caller, fd, p, plen int32) int32 {
			path, rc := r.guestPath(r.mem(c), fd, p, plen)
			if rc != errSuccess {
				return rc
			}
			return r.ov.rmdir(path)
		}},
		{"path_unlink_file", func(c *wt.Caller, fd, p, plen int32) int32 {
			path, rc := r.guestPath(r.mem(c), fd, p, plen)
			if rc != errSuccess {
				return rc
			}
			return r.ov.unlink(path)
		}},
		{"path_rename", func(c *wt.Caller, fd, p, plen, newFd, newP, newPlen int32) int32 {
			mem := r.mem(c)
			old, rc := r.guestPath(mem, fd, p, plen)
			if rc != errSuccess {
				return rc
			}
			new, rc := r.guestPath(mem, newFd, newP, newPlen)
			if rc != errSuccess {
				return rc
			}
			return r.ov.rename(old, new)
		}},
		{"path_readlink", func(c *wt.Caller, fd, p, plen, buf, buflen, used int32) int32 {
			path, rc := r.guestPath(r.mem(c), fd, p, plen)
			if rc != errSuccess {
				return rc
			}
			if !r.ov.fileExists(path) && !r.ov.dirExists(path) {
				return errNoent
			}
			return errInval // nothing in the workspace is a symlink
		}},
		{"path_link", func(c *wt.Caller, fd, flags, p, plen, newFd, newP, newPlen int32) int32 {
			return errNosys
		}},
		{"path_symlink", func(c *wt.Caller, p, plen, fd, newP, newPlen int32) int32 {
			return errNosys
		}},
		{"poll_oneoff", r.pollOneoff},
		{"proc_raise", func(c *wt.Caller, sig int32) int32 { return errNosys }},
		{"fd_allocate", func(c *wt.Caller, fd int32, off, n int64) int32 { return errNosys }},
		{"fd_renumber", func(c *wt.Caller, from, to int32) int32 { return errNosys }},
		{"fd_fdstat_set_rights", func(c *wt.Caller, fd int32, base, inheriting int64) int32 { return errNosys }},
		{"sock_accept", func(c *wt.Caller, fd, flags, out int32) int32 { return errNosys }},
		{"sock_recv", func(c *wt.Caller, fd, iovs, iovsLen, flags, n, oflags int32) int32 { return errNosys }},
		{"sock_send", func(c *wt.Caller, fd, iovs, iovsLen, flags, n int32) int32 { return errNosys }},
		{"sock_shutdown", func(c *wt.Caller, fd, how int32) int32 { return errNosys }},
	}
	for _, d := range defs {
		if err := linker.FuncWrap(m, d.name, d.fn); err != nil {
			return err
		}
	}

	exitArg := wt.NewValType(wt.KindI32)
	defer exitArg.Close()
	exitTy := wt.NewFuncType([]*wt.ValType{exitArg}, nil)
	defer exitTy.Close()
	return linker.FuncNew(m, "proc_exit", exitTy, func(c *wt.Caller, vals []wt.Val) ([]wt.Val, *wt.Trap) {
		r.exited = true
		r.exitCode = uint32(vals[0].I32())
		return nil, wt.NewTrap("proc_exit")
	})
}

func (r *run) now(id int32) uint64 {
	r.clock = addTime(r.clock, clockQuantumNanos)
	if id == 0 {
		return addTime(clockEpochNanos, r.clock)
	}
	return r.clock
}

func addTime(a, b uint64) uint64 {
	if b > ^uint64(0)-a {
		return ^uint64(0)
	}
	return a + b
}

func (r *run) sizesGet(c *wt.Caller, list []string, countPtr, bufSizePtr int32) int32 {
	mem := r.mem(c)
	if _, ok := memRange(mem, countPtr, 4); !ok {
		return errFault
	}
	if _, ok := memRange(mem, bufSizePtr, 4); !ok {
		return errFault
	}
	var total uint64
	for _, s := range list {
		total += uint64(len(s)) + 1
	}
	if total > uint64(^uint32(0)) || uint64(len(list)) > uint64(^uint32(0)) {
		return errNospc
	}
	putU32(mem, countPtr, uint32(len(list)))
	putU32(mem, bufSizePtr, uint32(total))
	return errSuccess
}

func (r *run) listGet(c *wt.Caller, list []string, arrPtr, bufPtr int32) int32 {
	mem := r.mem(c)
	var total uint64
	for _, s := range list {
		total += uint64(len(s)) + 1
	}
	arr, ok := memBytes(mem, arrPtr, uint64(len(list))*4)
	if !ok {
		return errFault
	}
	buf, ok := memBytes(mem, bufPtr, total)
	if !ok {
		return errFault
	}
	off := 0
	for i, s := range list {
		binary.LittleEndian.PutUint32(arr[i*4:], uint32(bufPtr)+uint32(off))
		copy(buf[off:], s)
		off += len(s)
		buf[off] = 0
		off++
	}
	return errSuccess
}

func (r *run) fdClose(c *wt.Caller, fd int32) int32 {
	if _, ok := r.fds[fd]; !ok {
		return errBadf
	}
	delete(r.fds, fd)
	return errSuccess
}

func (r *run) fdNoopSync(c *wt.Caller, fd int32) int32 {
	if _, ok := r.fds[fd]; !ok {
		return errBadf
	}
	return errSuccess
}

// rights returns the rights bitmask reflecting an entry's capabilities.
func (e *fdEntry) rights() uint64 {
	var rights uint64 = ^uint64(0) &^ (rightFDRead | rightFDWrite | rightFDReaddir)
	if e.read {
		rights |= rightFDRead
	}
	if e.write {
		rights |= rightFDWrite
	}
	if e.kind == fdDir {
		rights |= rightFDReaddir
	}
	return rights
}

func (e *fdEntry) filetype() byte {
	switch e.kind {
	case fdDir:
		return filetypeDirectory
	case fdFile:
		return filetypeRegular
	default:
		return filetypeCharDevice
	}
}

func (r *run) fdFdstatGet(c *wt.Caller, fd, out int32) int32 {
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	mem := r.mem(c)
	b, ok := memRange(mem, out, 24)
	if !ok {
		return errFault
	}
	clear(b)
	b[0] = e.filetype()
	if e.append {
		binary.LittleEndian.PutUint16(b[2:], fdflagAppend)
	}
	binary.LittleEndian.PutUint64(b[8:], e.rights())
	binary.LittleEndian.PutUint64(b[16:], e.rights())
	return errSuccess
}

func (r *run) fdFdstatSetFlags(c *wt.Caller, fd, flags int32) int32 {
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	e.append = flags&fdflagAppend != 0
	return errSuccess
}

func putFilestat(mem []byte, out int32, filetype byte, size uint64) errno {
	b, ok := memRange(mem, out, 64)
	if !ok {
		return errFault
	}
	clear(b)
	b[16] = filetype
	binary.LittleEndian.PutUint64(b[24:], 1) // nlink
	binary.LittleEndian.PutUint64(b[32:], size)
	return errSuccess
}

func (r *run) fdFilestatGet(c *wt.Caller, fd, out int32) int32 {
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	var size uint64
	if e.kind == fdFile {
		content, ok := r.ov.fileContent(e.path)
		if !ok {
			return errNoent
		}
		size = uint64(len(content))
	}
	return putFilestat(r.mem(c), out, e.filetype(), size)
}

func (r *run) fdFilestatSetSize(c *wt.Caller, fd int32, size int64) int32 {
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	if e.kind != fdFile || !e.write {
		return errInval
	}
	content, ok := r.ov.fileContent(e.path)
	if !ok {
		return errNoent
	}
	if size < 0 {
		return errInval
	}
	if size == int64(len(content)) {
		return errSuccess
	}
	var grow uint64
	if uint64(size) > uint64(len(content)) {
		grow = uint64(size) - uint64(len(content))
	}
	if grow > r.limits.WriteBytes-r.written || uint64(size) > uint64(^uint(0)>>1) {
		abortResource("file writes")
	}
	r.ov.checkBytes(uint64(size), uint64(len(content)))
	next := make([]byte, int(size))
	copy(next, content)
	if rc := r.ov.setFile(e.path, next); rc != errSuccess {
		return rc
	}
	r.written += grow
	return errSuccess
}

func (r *run) fdPrestatGet(c *wt.Caller, fd, out int32) int32 {
	e, ok := r.fds[fd]
	if !ok || e.kind != fdDir || e.path != "/" {
		return errBadf
	}
	mem := r.mem(c)
	b, ok := memRange(mem, out, 8)
	if !ok {
		return errFault
	}
	clear(b)
	binary.LittleEndian.PutUint32(b[4:], 1) // strlen("/")
	return errSuccess
}

func (r *run) fdPrestatDirName(c *wt.Caller, fd, buf, buflen int32) int32 {
	e, ok := r.fds[fd]
	if !ok || e.kind != fdDir || e.path != "/" {
		return errBadf
	}
	b, ok := memRange(r.mem(c), buf, buflen)
	if !ok {
		return errFault
	}
	if len(b) < 1 {
		return errInval
	}
	b[0] = '/'
	return errSuccess
}

// Snapshot descriptors so reads into the original table cannot alter later buffers.
func readIOVecs(mem []byte, iovs, iovsLen int32) (ioVecs, errno) {
	table, ok := memBytes(mem, iovs, uint64(uint32(iovsLen))*8)
	if !ok {
		return ioVecs{}, errFault
	}
	var total uint64
	for i := 0; i < len(table); i += 8 {
		ptr := binary.LittleEndian.Uint32(table[i:])
		n := binary.LittleEndian.Uint32(table[i+4:])
		if _, ok := memBytes(mem, int32(ptr), uint64(n)); !ok {
			return ioVecs{}, errFault
		}
		total += uint64(n)
		if total > uint64(^uint32(0)) {
			return ioVecs{}, errInval
		}
	}
	return ioVecs{mem, append([]byte(nil), table...)}, errSuccess
}

type ioVecs struct{ mem, table []byte }

func (v ioVecs) buffers(yield func([]byte) bool) {
	for i := 0; i < len(v.table); i += 8 {
		ptr := int(binary.LittleEndian.Uint32(v.table[i:]))
		n := int(binary.LittleEndian.Uint32(v.table[i+4:]))
		if !yield(v.mem[ptr : ptr+n : ptr+n]) {
			return
		}
	}
}

func readAt(iovs ioVecs, src []byte, off uint64) uint32 {
	var total uint32
	for b := range iovs.buffers {
		if off >= uint64(len(src)) {
			break
		}
		n := copy(b, src[off:])
		off += uint64(n)
		total += uint32(n)
	}
	return total
}

func (r *run) fdRead(c *wt.Caller, fd, iovs, iovsLen, out int32) int32 {
	mem := r.mem(c)
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	if !e.read {
		return errNotcapable
	}
	if _, ok := memRange(mem, out, 4); !ok {
		return errFault
	}
	bufs, rc := readIOVecs(mem, iovs, iovsLen)
	if rc != errSuccess {
		return rc
	}
	var n uint32
	switch e.kind {
	case fdStdin:
		n = readAt(bufs, r.stdin, uint64(r.stdinOff))
		r.stdinOff += int(n)
	case fdFile:
		content, ok := r.ov.fileContent(e.path)
		if !ok {
			return errNoent
		}
		n = readAt(bufs, content, e.off)
		e.off += uint64(n)
	default:
		return errIsdir
	}
	if !putU32(mem, out, n) {
		return errFault
	}
	return errSuccess
}

func (r *run) fdPread(c *wt.Caller, fd, iovs, iovsLen int32, off int64, out int32) int32 {
	mem := r.mem(c)
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	if e.kind != fdFile {
		return errSpipe
	}
	if !e.read {
		return errNotcapable
	}
	if _, ok := memRange(mem, out, 4); !ok {
		return errFault
	}
	if off < 0 {
		return errInval
	}
	bufs, rc := readIOVecs(mem, iovs, iovsLen)
	if rc != errSuccess {
		return rc
	}
	content, ok := r.ov.fileContent(e.path)
	if !ok {
		return errNoent
	}
	if !putU32(mem, out, readAt(bufs, content, uint64(off))) {
		return errFault
	}
	return errSuccess
}

func writeStream(dst *[]byte, limit uint64, bufs ioVecs) (uint32, errno) {
	limit = min(limit, uint64(^uint(0)>>1))
	var total uint64
	for b := range bufs.buffers {
		if uint64(len(b)) > limit-uint64(len(*dst))-total {
			abortResource("output stream")
		}
		total += uint64(len(b))
	}
	for b := range bufs.buffers {
		*dst = append(*dst, b...)
	}
	return uint32(total), errSuccess
}

func (r *run) writeFile(e *fdEntry, bufs ioVecs, off uint64, advance bool) (uint32, errno) {
	content, ok := r.ov.fileContent(e.path)
	if !ok {
		return 0, errNoent
	}
	var n uint64
	for b := range bufs.buffers {
		if uint64(len(b)) > r.limits.WriteBytes-r.written-n {
			abortResource("file writes")
		}
		n += uint64(len(b))
	}
	if n == 0 {
		return 0, errSuccess
	}
	var fill uint64
	if off > uint64(len(content)) {
		fill = off - uint64(len(content))
	}
	if fill > r.limits.WriteBytes-r.written-n || off > uint64(^uint(0)>>1) || n > uint64(^uint(0)>>1)-off {
		abortResource("file writes")
	}
	end := off + n
	size := max(uint64(len(content)), end)
	r.ov.checkBytes(size, uint64(len(content)))
	next := make([]byte, size)
	copy(next, content)
	p := off
	for b := range bufs.buffers {
		copy(next[p:], b)
		p += uint64(len(b))
	}
	if rc := r.ov.setFile(e.path, next); rc != errSuccess {
		return 0, rc
	}
	r.written += n + fill
	if advance {
		e.off = p
	}
	return uint32(n), errSuccess
}

func (r *run) fdWrite(c *wt.Caller, fd, iovs, iovsLen, out int32) int32 {
	mem := r.mem(c)
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	if !e.write {
		return errNotcapable
	}
	if _, ok := memRange(mem, out, 4); !ok {
		return errFault
	}
	bufs, rc := readIOVecs(mem, iovs, iovsLen)
	if rc != errSuccess {
		return rc
	}
	var n uint32
	switch e.kind {
	case fdStdout:
		n, rc = writeStream(&r.stdout, r.limits.StdoutBytes, bufs)
	case fdStderr:
		n, rc = writeStream(&r.stderr, r.limits.StderrBytes, bufs)
	case fdFile:
		off := e.off
		if e.append {
			content, _ := r.ov.fileContent(e.path)
			off = uint64(len(content))
		}
		n, rc = r.writeFile(e, bufs, off, true)
	default:
		return errIsdir
	}
	if rc != errSuccess {
		return rc
	}
	if !putU32(mem, out, n) {
		return errFault
	}
	return errSuccess
}

func (r *run) fdPwrite(c *wt.Caller, fd, iovs, iovsLen int32, off int64, out int32) int32 {
	mem := r.mem(c)
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	if e.kind != fdFile {
		return errSpipe
	}
	if !e.write {
		return errNotcapable
	}
	if _, ok := memRange(mem, out, 4); !ok {
		return errFault
	}
	if off < 0 {
		return errInval
	}
	bufs, rc := readIOVecs(mem, iovs, iovsLen)
	if rc != errSuccess {
		return rc
	}
	n, rc := r.writeFile(e, bufs, uint64(off), false)
	if rc != errSuccess {
		return rc
	}
	if !putU32(mem, out, n) {
		return errFault
	}
	return errSuccess
}

func (r *run) fdSeek(c *wt.Caller, fd int32, offset int64, whence, out int32) int32 {
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	if e.kind != fdFile {
		return errSpipe
	}
	var base uint64
	switch whence {
	case 0:
	case 1:
		base = e.off
	case 2:
		content, ok := r.ov.fileContent(e.path)
		if !ok {
			return errNoent
		}
		base = uint64(len(content))
	default:
		return errInval
	}
	var next uint64
	if offset < 0 {
		delta := uint64(-(offset + 1)) + 1
		if delta > base {
			return errInval
		}
		next = base - delta
	} else {
		if uint64(offset) > ^uint64(0)-base {
			return errInval
		}
		next = base + uint64(offset)
	}
	if !putU64(r.mem(c), out, next) {
		return errFault
	}
	e.off = next
	return errSuccess
}

func (r *run) fdTell(c *wt.Caller, fd, out int32) int32 {
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	if e.kind != fdFile {
		return errSpipe
	}
	if !putU64(r.mem(c), out, e.off) {
		return errFault
	}
	return errSuccess
}

func (r *run) fdReaddir(c *wt.Caller, fd, buf, buflen int32, cookie int64, out int32) int32 {
	mem := r.mem(c)
	e, ok := r.fds[fd]
	if !ok {
		return errBadf
	}
	if e.kind != fdDir {
		return errNotdir
	}
	if cookie < 0 {
		return errInval
	}
	if _, ok := memRange(mem, out, 4); !ok {
		return errFault
	}
	entries := r.ov.readdir(e.path)
	b, ok := memRange(mem, buf, buflen)
	if !ok {
		return errFault
	}
	used := 0
	for i := uint64(cookie); i < uint64(len(entries)); i++ {
		ent := entries[i]
		var head [24]byte
		binary.LittleEndian.PutUint64(head[0:], uint64(i)+1) // d_next
		binary.LittleEndian.PutUint64(head[8:], uint64(i)+1) // d_ino: stable nonzero ordinal
		binary.LittleEndian.PutUint32(head[16:], uint32(len(ent.name)))
		if ent.dir {
			head[20] = filetypeDirectory
		} else {
			head[20] = filetypeRegular
		}
		used += copy(b[used:], head[:])
		used += copy(b[used:], ent.name)
		if used == len(b) {
			break
		}
	}
	if !putU32(mem, out, uint32(used)) {
		return errFault
	}
	return errSuccess
}

func (r *run) pathOpen(c *wt.Caller, fd, dirflags, p, plen int32, oflags int32, rightsBase, rightsInheriting int64, fdflags, out int32) int32 {
	mem := r.mem(c)
	path, rc := r.guestPath(mem, fd, p, plen)
	if rc != errSuccess {
		return rc
	}
	if _, ok := memRange(mem, out, 4); !ok {
		return errFault
	}
	if r.nextFD < 0 {
		return errNospc
	}
	read := rightsBase&(rightFDRead|rightFDReaddir) != 0
	write := rightsBase&rightFDWrite != 0
	appendMode := fdflags&fdflagAppend != 0

	if !writable(path) && (oflags&(oflagCreat|oflagTrunc) != 0 || write || appendMode) {
		return errRofs
	}
	if oflags&oflagTrunc != 0 && !write {
		return errNotcapable
	}

	if r.ov.dirExists(path) {
		if oflags&(oflagCreat|oflagExcl) == (oflagCreat | oflagExcl) {
			return errExist
		}
		if oflags&oflagTrunc != 0 {
			return errIsdir
		}
		r.fds[r.nextFD] = &fdEntry{kind: fdDir, path: path, read: read, write: write}
	} else {
		if oflags&oflagDirectory != 0 {
			if r.ov.fileExists(path) {
				return errNotdir
			}
			return errNoent
		}
		exists := r.ov.fileExists(path)
		switch {
		case exists && oflags&(oflagCreat|oflagExcl) == (oflagCreat|oflagExcl):
			return errExist
		case !exists && oflags&oflagCreat == 0:
			return errNoent
		case !exists:
			if r.ov.fileOnPath(path) {
				return errNotdir
			}
			if rc := r.ov.setFile(path, nil); rc != errSuccess {
				return rc
			}
		}
		if oflags&oflagTrunc != 0 {
			if rc := r.ov.setFile(path, nil); rc != errSuccess {
				return rc
			}
		}
		r.fds[r.nextFD] = &fdEntry{kind: fdFile, path: path, read: read, write: write, append: appendMode}
	}
	if !putU32(mem, out, uint32(r.nextFD)) {
		delete(r.fds, r.nextFD)
		return errFault
	}
	r.nextFD++
	return errSuccess
}

func (r *run) pathFilestatGet(c *wt.Caller, fd, flags, p, plen, out int32) int32 {
	mem := r.mem(c)
	path, rc := r.guestPath(mem, fd, p, plen)
	if rc != errSuccess {
		return rc
	}
	if content, ok := r.ov.fileContent(path); ok {
		return putFilestat(mem, out, filetypeRegular, uint64(len(content)))
	}
	if r.ov.dirExists(path) {
		return putFilestat(mem, out, filetypeDirectory, 0)
	}
	return errNoent
}

// Poll advances to the earliest clock deadline only when no fd or error is ready.
func (r *run) pollOneoff(c *wt.Caller, in, out, nsubs, nevents int32) int32 {
	mem := r.mem(c)
	if nsubs == 0 {
		return errInval
	}
	subs, ok := memBytes(mem, in, uint64(uint32(nsubs))*48)
	if !ok {
		return errFault
	}
	evs, ok := memBytes(mem, out, uint64(uint32(nsubs))*32)
	if !ok {
		return errFault
	}
	if _, ok := memRange(mem, nevents, 4); !ok {
		return errFault
	}
	subs = append([]byte(nil), subs...)
	start := r.clock
	until := ^uint64(0)
	for i := 0; i < len(subs); i += 48 {
		deadline, _, _ := r.subscription(subs[i:i+48], start)
		until = min(until, deadline)
	}
	r.clock = until
	clear(evs)
	var count uint32
	for i := 0; i < len(subs); i += 48 {
		sub := subs[i : i+48]
		deadline, rc, nbytes := r.subscription(sub, start)
		if deadline > until {
			continue
		}
		ev := evs[int(count)*32:][:32]
		copy(ev[:8], sub[:8])
		binary.LittleEndian.PutUint16(ev[8:], uint16(rc))
		ev[10] = sub[8]
		binary.LittleEndian.PutUint64(ev[16:], nbytes)
		count++
	}
	putU32(mem, nevents, count)
	return errSuccess
}

func (r *run) subscription(sub []byte, start uint64) (deadline uint64, rc errno, nbytes uint64) {
	switch sub[8] {
	case 0:
		id := binary.LittleEndian.Uint32(sub[16:])
		timeout := binary.LittleEndian.Uint64(sub[24:])
		flags := binary.LittleEndian.Uint16(sub[40:])
		if id > 3 || flags&^1 != 0 {
			return start, errInval, 0
		}
		if flags&1 == 0 {
			return addTime(start, timeout), errSuccess, 0
		}
		if id == 0 {
			if timeout < clockEpochNanos {
				timeout = 0
			} else {
				timeout -= clockEpochNanos
			}
		}
		return max(start, timeout), errSuccess, 0
	case 1, 2:
		fd := int32(binary.LittleEndian.Uint32(sub[16:]))
		e, ok := r.fds[fd]
		if !ok {
			return start, errBadf, 0
		}
		if (sub[8] == 1 && !e.read) || (sub[8] == 2 && !e.write) {
			return start, errNotcapable, 0
		}
		if e.kind == fdDir {
			return start, errIsdir, 0
		}
		nbytes = 1 << 16
		if sub[8] == 1 {
			if e.kind == fdStdin {
				nbytes = uint64(len(r.stdin) - r.stdinOff)
			}
			if e.kind == fdFile {
				content, ok := r.ov.fileContent(e.path)
				if !ok {
					return start, errNoent, 0
				}
				nbytes = uint64(len(content)) - min(e.off, uint64(len(content)))
			}
		}
		return start, errSuccess, nbytes
	default:
		return start, errInval, 0
	}
}
