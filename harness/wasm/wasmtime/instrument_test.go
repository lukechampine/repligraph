package wasmtime

import (
	"errors"
	"fmt"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func TestInstrumentationSkipsInstructionImmediates(t *testing.T) {
	// 0x40 is memory.grow only at an instruction boundary. It also appears
	// in signed constants, float/vector literals, block types and offsets.
	bin := watModule(t, `(module
      (memory 1)
      (data "@")
      (func (export "_start")
        (block nop)
        (drop (i32.const -64))
        (drop (i64.const 64))
        (drop (f32.const 2))
        (drop (f64.const 2))
        (drop (v128.const i8x16 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64))
        (v128.store offset=64 (i32.const 0) (v128.const i32x4 0 0 0 0))
        (drop (v128.load32_lane offset=64 0 (i32.const 0) (v128.const i32x4 0 0 0 0)))
        (drop (v128.load32_zero offset=64 (i32.const 0)))
        (memory.init 0 (i32.const 64) (i32.const 0) (i32.const 1))
        (data.drop 0)
        (memory.copy (i32.const 0) (i32.const 64) (i32.const 1))
        (memory.fill (i32.const 64) (i32.const 64) (i32.const 1))
        (drop (i32.load offset=64 (i32.const 0)))
        (drop (memory.grow (i32.const 1)))))`)
	resp, _ := runModule(t, bin, tree.Tree{}, nil)
	if resp.Trap != wasm.TrapNone || resp.Usage.MemoryBytes != 131072 {
		t.Fatalf("instrumented immediates: %+v", resp)
	}
}

func TestInstrumentationRejectsPrivateExports(t *testing.T) {
	for _, name := range []string{privateMemory, privateLimit, privateAbort, privateStart} {
		t.Run(name, func(t *testing.T) {
			bin := watModule(t, fmt.Sprintf(`(module (memory 1) (func (export "%s")) (func (export "_start")))`, name))
			resp, nt, err := Run(tree.Tree{}, wasm.Request{Module: bin, Limits: testLimits(10000)})
			if !errors.Is(err, wasm.ErrBadModule) || resp.Usage.Fuel != 0 || nt.Root() != (tree.Tree{}).Root() {
				t.Fatalf("reserved export: %+v, %v", resp, err)
			}
		})
	}
}

func TestInstrumentationValidatesOriginalIndices(t *testing.T) {
	// Index zero would refer to a host-added private global after rewriting.
	bin := watModule(t, `(module (memory 1) (func (export "_start") (drop (global.get 0))))`)
	_, _, err := Run(tree.Tree{}, wasm.Request{Module: bin, Limits: testLimits(10000)})
	if !errors.Is(err, wasm.ErrBadModule) {
		t.Fatalf("invalid original module gained access to helper global: %v", err)
	}
}
