set -eu
toolkit_version=go1.26.4
toolkit_tree=040702735852323ea4288651ec9083816206674116f8adf05fd6c6fdab676183
toolkit_dir=$(cd "$(dirname "$0")" && pwd)
toolkit_out="$PWD/go-toolkit"
toolkit_cache="$PWD/go-toolkit-cache"
# Any go command can fetch the exact toolchain, verified by the checksum database.
toolkit_root=$(GOTOOLCHAIN="$toolkit_version" go env GOROOT)
toolkit_go="$toolkit_root/bin/go"

wasi_go() {
    env -i PATH="$PATH" GOROOT="$toolkit_root" \
        GOENV=off GOTOOLCHAIN=local GO111MODULE=off GOWORK=off \
        GOOS=wasip1 GOARCH=wasm CGO_ENABLED=0 \
        GOFLAGS= GOEXPERIMENT= GOWASM= GODEBUG= GOFIPS140=off \
        GOCACHEPROG= GOCACHE="$toolkit_cache" GOPATH="$toolkit_cache/gopath" \
        GOPROXY=off GOSUMDB=off "$toolkit_go" "$@"
}

[ "$(wasi_go env GOVERSION)" = "$toolkit_version" ]
mkdir -p "$toolkit_out/bin" "$toolkit_out/std"
wasi_go build -trimpath -o "$toolkit_out/bin/go-compile.wasm" cmd/compile
wasi_go build -trimpath -o "$toolkit_out/bin/go-link.wasm" cmd/link
wasi_go list -export -deps -trimpath -f '{{if .Export}}{{.ImportPath}}={{.Export}}{{end}}' std > "$toolkit_cache/exports"
: > "$toolkit_out/importcfg"
while IFS='=' read -r toolkit_pkg toolkit_archive; do
    [ -n "$toolkit_archive" ] || continue
    mkdir -p "$toolkit_out/std/$(dirname "$toolkit_pkg")"
    cp "$toolkit_archive" "$toolkit_out/std/$toolkit_pkg.a"
    printf 'packagefile %s=/env/std/%s.a\n' "$toolkit_pkg" "$toolkit_pkg" >> "$toolkit_out/importcfg"
done < "$toolkit_cache/exports"
LC_ALL=C sort -o "$toolkit_out/importcfg" "$toolkit_out/importcfg"
cp "$toolkit_out/importcfg" "$toolkit_out/link.cfg"
printf 'packagefile main=/artifact/main.o\n' >> "$toolkit_out/link.cfg"
cp "$toolkit_root/LICENSE" "$toolkit_out/LICENSE"

# The environment pin is this tree root; any other result is a different environment.
toolkit_got=$(cd "$toolkit_dir" && GOTOOLCHAIN=local "$toolkit_go" run root.go "$toolkit_out")
if [ "$toolkit_got" != "$toolkit_tree" ]; then
    echo "toolkit tree root is $toolkit_got, want $toolkit_tree" >&2
    exit 1
fi