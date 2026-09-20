//go:build ignore

// Root prints the tree root of a directory of regular files.
package main

import (
	"fmt"
	"io/fs"
	"os"

	"lukechampine.com/repligraph/harness/tree"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: go run root.go <dir>")
		os.Exit(2)
	}
	disk := os.DirFS(os.Args[1])
	var files tree.Tree
	if err := fs.WalkDir(disk, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		} else if !d.Type().IsRegular() {
			return fmt.Errorf("non-regular file: %s", path)
		}
		content, err := fs.ReadFile(disk, path)
		if err == nil {
			files, err = files.Put("/"+path, tree.File{Content: content})
		}
		return err
	}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(files.Root())
}
