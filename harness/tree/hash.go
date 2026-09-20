package tree

import (
	"encoding/binary"

	"lukechampine.com/repligraph/blake3"
)

func fileHash(content []byte) blake3.Hash {
	return blake3.Sum(blake3.DomainTreeFile, content)
}

// Directory hashes cover sorted (kind, name length, name, child hash) records.
func dirHashEntries(entries []dirEntry) blake3.Hash {
	h := blake3.New(blake3.DomainTreeDir)
	var lenBuf [4]byte
	for _, e := range entries {
		kind := byte(0)
		child := blake3.Hash{}
		if e.dir != nil {
			kind = 1
			child = e.dir.h
		} else {
			child = e.file.h
		}
		h.Write([]byte{kind})
		binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(e.name)))
		h.Write(lenBuf[:])
		h.Write([]byte(e.name))
		h.Write(child[:])
	}
	return h.Sum()
}

func dirHash(d *dirNode) blake3.Hash {
	if d == nil {
		return dirHashEntries(nil)
	}
	return d.h
}
