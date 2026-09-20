For a program at `/artifact/main.go`, call `run` with these arguments in order:

```json
{"module":"/env/bin/go-compile.wasm","args":["-p","main","-complete","-importcfg","/env/importcfg","-o","/artifact/main.o","/artifact/main.go"]}
{"module":"/env/bin/go-link.wasm","args":["-importcfg","/env/link.cfg","-o","/artifact/main.wasm","/artifact/main.o"]}
{"module":"/artifact/main.wasm"}
```

These build a single `package main`; list any additional source files after
`main.go`. The toolkit has no `go` command or `go test` driver. Tests can be
ordinary programs that exit nonzero on failure.
