package wasmtime

import (
	"errors"

	"lukechampine.com/repligraph/harness/tree"
)

type errno = int32

const (
	errSuccess     errno = 0
	errBadf        errno = 8
	errExist       errno = 20
	errFault       errno = 21
	errIlseq       errno = 25
	errInval       errno = 28
	errIsdir       errno = 31
	errNametoolong errno = 37
	errNoent       errno = 44
	errNospc       errno = 51
	errNosys       errno = 52
	errNotdir      errno = 54
	errNotempty    errno = 55
	errNotcapable  errno = 76
	errRofs        errno = 69
	errSpipe       errno = 70
)

func pathErrno(err error) errno {
	var pe *tree.PathError
	if errors.As(err, &pe) {
		switch pe.Reason {
		case "path is too long", "path component is too long":
			return errNametoolong
		}
		return errIlseq
	}
	switch {
	case errors.Is(err, tree.ErrNotFound):
		return errNoent
	case errors.Is(err, tree.ErrIsDir):
		return errIsdir
	case errors.Is(err, tree.ErrNotDir):
		return errNotdir
	}
	return errInval
}
