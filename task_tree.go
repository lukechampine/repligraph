package repligraph

import (
	"fmt"

	"lukechampine.com/repligraph/harness/tree"
)

func validateTaskTree(t tree.Tree) error {
	if _, err := t.Get("/env"); err == nil || err == tree.ErrIsDir {
		return fmt.Errorf("/env is reserved for the environment")
	}
	return nil
}
