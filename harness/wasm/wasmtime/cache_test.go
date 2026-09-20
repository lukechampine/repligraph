package wasmtime

import (
	"bytes"
	"errors"
	"sync"
	"testing"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

func testModuleCache(t *testing.T, entries, sourceBytes int) *moduleCache {
	t.Helper()
	c := &moduleCache{maxEntries: entries, maxSourceBytes: sourceBytes}
	t.Cleanup(func() {
		for _, m := range c.modules {
			m.module.Close()
			m.engine.Close()
		}
	})
	return c
}

func cachedModule(t *testing.T, c *moduleCache, bin []byte) *compiledModule {
	t.Helper()
	m, err := c.acquire(bin)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestModuleCacheReuseAndEviction(t *testing.T) {
	a := watModule(t, `(module (func (export "_start")))`)
	b := watModule(t, `(module (func (export "_start") nop))`)
	d := watModule(t, `(module (func (export "_start") nop nop))`)
	c := testModuleCache(t, 1, 1<<20)

	active := cachedModule(t, c, a)
	c.release(active)
	if got := cachedModule(t, c, bytes.Clone(a)); got != active {
		t.Fatal("identical bytes did not reuse compiled module")
	}
	// Keep a leased while b and d compete for the single idle slot.
	b1 := cachedModule(t, c, b)
	c.release(b1)
	c.release(cachedModule(t, c, d))
	if got := cachedModule(t, c, a); got != active {
		t.Fatal("eviction discarded an active module")
	} else if len(got.module.Exports()) == 0 {
		t.Fatal("active module was closed")
	}
	c.release(active)
	b2 := cachedModule(t, c, b)
	if b1 == b2 {
		t.Fatal("least recently used idle module was not evicted")
	}
	c.release(b2)
	c.release(active)
	if len(c.modules) != 1 || c.idle.Len() != 1 {
		t.Fatal("released modules exceeded idle entry limit")
	}
}

func TestModuleCacheSourceBudget(t *testing.T) {
	a := watModule(t, `(module (func (export "_start")))`)
	b := watModule(t, `(module (func (export "_start") nop))`)
	t.Run("combined size", func(t *testing.T) {
		c := testModuleCache(t, 16, len(a)+len(b)-1)
		a1 := cachedModule(t, c, a)
		c.release(a1)
		c.release(cachedModule(t, c, b))
		if c.sourceBytes != len(b) || len(c.modules) != 1 {
			t.Fatal("idle cache exceeded source-byte budget")
		}
		a2 := cachedModule(t, c, a)
		if a1 == a2 {
			t.Fatal("byte budget did not evict the oldest module")
		}
		c.release(a2)
	})
	t.Run("oversized module", func(t *testing.T) {
		c := testModuleCache(t, 16, len(a)-1)
		m := cachedModule(t, c, a)
		if len(m.module.Exports()) == 0 {
			t.Fatal("oversized module could not be used")
		}
		c.release(m)
		if len(c.modules) != 0 || c.sourceBytes != 0 || c.idle.Len() != 0 {
			t.Fatal("oversized module was retained")
		}
	})
}

func TestModuleCacheConcurrentAcquire(t *testing.T) {
	c := testModuleCache(t, 1, 1<<20)
	bin := watModule(t, helloWAT)
	const n = 16
	type result struct {
		m   *compiledModule
		err error
	}
	results := make(chan result, n)
	start, release := make(chan struct{}), make(chan struct{})
	var wg sync.WaitGroup
	for range n {
		wg.Go(func() {
			<-start
			m, err := c.acquire(bin)
			results <- result{m, err}
			<-release
			if err == nil {
				c.release(m)
			}
		})
	}
	close(start)
	var first *compiledModule
	for range n {
		r := <-results
		if r.err != nil {
			t.Error(r.err)
		} else if first == nil {
			first = r.m
		} else if r.m != first {
			t.Error("concurrent callers compiled the same module independently")
		}
	}
	close(release)
	wg.Wait()
	if len(c.modules) != 1 || c.idle.Len() != 1 {
		t.Fatal("concurrent leases did not return to one idle entry")
	}
}

func TestModuleCacheBadModule(t *testing.T) {
	c := testModuleCache(t, 2, 1<<20)
	for _, bin := range [][]byte{[]byte("bad wasm"), watModule(t, `(module)`)} {
		for range 2 {
			if _, err := c.acquire(bin); !errors.Is(err, wasm.ErrBadModule) {
				t.Fatalf("error = %v, want ErrBadModule", err)
			}
			if len(c.modules) != 0 {
				t.Fatal("failed compilation was retained")
			}
		}
	}
	c.release(cachedModule(t, c, watModule(t, helloWAT)))
}

func TestCachedRunsHaveFreshState(t *testing.T) {
	bin := watModule(t, `(module
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (global $count (mut i32) (i32.const 0))
  (func (export "_start")
    (global.set $count (i32.add (global.get $count) (i32.const 1)))
    (i32.store (i32.const 0) (i32.add (i32.load (i32.const 0)) (i32.const 1)))
    (call $exit (i32.add (global.get $count) (i32.load (i32.const 0))))))`)
	req := wasm.Request{Module: bin, Limits: testLimits(100)}
	baseline, _, err := Run(tree.Tree{}, req)
	if err != nil || baseline.ExitCode != 2 || baseline.Trap != wasm.TrapNone {
		t.Fatalf("initial run: %+v, %v", baseline, err)
	}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			got, tr, err := Run(tree.Tree{}, req)
			if err != nil || got.ExitCode != 2 || got.Trap != wasm.TrapNone || got.Usage.Fuel != baseline.Usage.Fuel || tr.Root() != (tree.Tree{}).Root() {
				t.Errorf("cached run leaked state or fuel: %+v, %v", got, err)
			}
		})
	}
	wg.Wait()
	req.Limits.Fuel = 0
	got, nt, err := Run(tree.Tree{}, req)
	checkResourceAbort(t, got, nt, err, tree.Tree{})
}
