`wordfreq.wasm` is the prototype's Rust `wasm32-wasip1` fixture, copied unchanged.
It exercises runtime initialization, preopen discovery, arguments, file I/O,
and buffered stdout/stderr. `wordfreq.rs` preserves its source.

The prototype records rustc 1.96.0 and this build:

```sh
cargo new wordfreq
cp wordfreq.rs wordfreq/src/main.rs
cd wordfreq
cat >> Cargo.toml <<'EOF'

[profile.release]
opt-level = "s"
lto = true
strip = true
panic = "abort"
EOF
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1
```

The committed WASM bytes are the test fixture; tests do not require Rust.
