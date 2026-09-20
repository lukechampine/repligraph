package tools

import (
	"encoding/json"
	"errors"
	"fmt"

	"lukechampine.com/repligraph/harness/tree"
	"lukechampine.com/repligraph/harness/wasm"
)

type runArgs struct {
	Module string    `json:"module"`
	Args   []*string `json:"args,omitempty"`
	Stdin  string    `json:"stdin,omitempty"`
}

func applyRun(t tree.Tree, ex wasm.Executor, limits wasm.Limits, args json.RawMessage) (result, tree.Tree, wasm.Usage, error) {
	var a runArgs
	if r := decodeArgs("run", args, &a); r != nil {
		return *r, t, wasm.Usage{}, nil
	}
	argv := []string{a.Module}
	for _, arg := range a.Args {
		if arg == nil {
			return errResult("bad_args", `run: argument "args" has the wrong type`), t, wasm.Usage{}, nil
		}
		argv = append(argv, *arg)
	}
	f, err := t.Get(a.Module)
	if err != nil {
		return treeErr(err, a.Module), t, wasm.Usage{}, nil
	}
	if ex == nil {
		return result{}, t, wasm.Usage{}, errors.New("run executor is nil")
	}
	// The guest environment is empty and the random seed is zero.
	req := wasm.Request{
		Module: f.Content,
		Args:   argv,
		Stdin:  []byte(a.Stdin),
		Limits: limits,
	}
	resp, nt, err := ex(t, req)
	if errors.Is(err, wasm.ErrBadModule) {
		return errResult("bad_module", "not a runnable WASM module: %s", a.Module), t, wasm.Usage{}, nil
	} else if err != nil {
		return result{}, t, wasm.Usage{}, err
	}
	return renderRun(resp), nt, resp.Usage, nil
}

func renderRun(resp wasm.Response) result {
	status := "ok"
	head := fmt.Sprintf("exit %d", resp.ExitCode)
	if resp.Trap != wasm.TrapNone {
		status = "error"
		head = fmt.Sprintf("trap: %s", resp.Trap)
	} else if resp.ExitCode != 0 {
		status = "error"
	}
	return result{
		status: status,
		output: fmt.Sprintf("%s (fuel used: %d)\n--- stdout ---\n%s\n--- stderr ---\n%s",
			head, resp.Usage.Fuel, resp.Stdout, resp.Stderr),
	}
}
