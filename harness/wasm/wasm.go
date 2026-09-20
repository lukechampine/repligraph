package wasm

import (
	"errors"

	"lukechampine.com/repligraph/blake3"
	"lukechampine.com/repligraph/harness/tree"
)

func ModuleHash(bin []byte) blake3.Hash {
	return blake3.Sum(blake3.DomainModule, bin)
}

var (
	ErrBadModule     = errors.New("not a runnable WASM module")
	ErrResourceLimit = errors.New("execution resource limit exceeded")
)

// Limits apply to one run; zero permits none of the resource.
type Limits struct {
	Fuel        uint64
	MemoryBytes uint64
	WriteBytes  uint64
	StdoutBytes uint64
	StderrBytes uint64
	// MaxTreeEntries counts task files and directories, excluding the root and
	// /env mount. Empty directories count while a run is in progress.
	MaxTreeEntries uint64
	// MaxTreeBytes bounds the sum of task file sizes, excluding /env.
	MaxTreeBytes uint64
}

type Request struct {
	Module     []byte
	Args       []string // includes argv[0]
	Env        []string
	Stdin      []byte
	Limits     Limits
	RandomSeed [32]byte
}

// Usage records consumed resources. MemoryBytes, TreeEntries, and TreeBytes are peaks;
// the remaining fields are totals, including work before a semantic trap.
type Usage struct {
	Fuel        uint64 `json:"fuel"`
	MemoryBytes uint64 `json:"memory_bytes"`
	WriteBytes  uint64 `json:"write_bytes"`
	StdoutBytes uint64 `json:"stdout_bytes"`
	StderrBytes uint64 `json:"stderr_bytes"`
	TreeEntries uint64 `json:"tree_entries"`
	TreeBytes   uint64 `json:"tree_bytes"`
}

// Trap strings are part of the model-visible output format.
type Trap string

const (
	TrapNone            Trap = ""
	TrapUnreachable     Trap = "unreachable"
	TrapMemoryOOB       Trap = "memory_out_of_bounds"
	TrapTableOOB        Trap = "table_out_of_bounds"
	TrapIndirectNull    Trap = "indirect_call_to_null"
	TrapBadSignature    Trap = "indirect_call_type_mismatch"
	TrapDivByZero       Trap = "integer_divide_by_zero"
	TrapIntegerOverflow Trap = "integer_overflow"
	TrapBadConversion   Trap = "bad_conversion_to_integer"
)

type Response struct {
	ExitCode uint32
	Trap     Trap
	Stdout   []byte
	Stderr   []byte
	Usage    Usage
}

// An Executor deterministically runs WASM over a tree. Changes commit on any exit
// code and roll back on traps; stdout and stderr are retained. Errors leave the
// tree unchanged and return no response. Exceeding a local ceiling returns
// ErrResourceLimit; resource exhaustion is never a guest-visible result.
type Executor func(t tree.Tree, req Request) (Response, tree.Tree, error)
