package wasmtime

// resourceAbort unwinds a host function immediately, without returning an
// errno to the guest. wasmtime-go propagates host panics out of the Wasm call;
// Run converts only this private value to wasm.ErrResourceLimit.
type resourceAbort string

func abortResource(resource string) { panic(resourceAbort(resource)) }
