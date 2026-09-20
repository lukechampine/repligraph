package wasmtime

import (
	"encoding/binary"
	"fmt"
	"strings"
)

const (
	privatePrefix = "repligraph.internal."
	privateMemory = privatePrefix + "memory"
	privateLimit  = privatePrefix + "memory_limit"
	privateAbort  = privatePrefix + "memory_abort"
	privateStart  = privatePrefix + "start"
)

// Wasmtime's C limiter reports a refused memory.grow as -1 to the guest.
// Guard growth in the module instead, so local ceilings cannot affect a
// recorded result. Original modules are validated before this transformation.
// Appending globals and exports leaves every existing index unchanged. Only
// memory.grow receives additional instructions (and deterministic fuel cost).
// Moving the start section lets the host install the ceiling before any code
// runs, and retain access to memory after a start-section trap.
func instrumentMemory(bin []byte) (out []byte, err error) {
	defer func() {
		if p := recover(); p != nil {
			if e, ok := p.(instrumentError); ok {
				out, err = nil, fmt.Errorf("memory instrumentation: %s", e)
			} else {
				panic(p)
			}
		}
	}()
	type section struct {
		id   byte
		data []byte
	}
	r := wasmCursor(bin)
	magic := r.take(8)
	var sections []section
	var globalCount uint64
	var globals, exports []byte
	var memory, start bool
	var startIndex uint64
	maxPages := uint64(65536)
	for len(r) > 0 {
		id := r.byte()
		data := r.take(r.u())
		sections = append(sections, section{id, data})
		s := wasmCursor(data)
		switch id {
		case 2:
			for n := s.u(); n > 0; n-- {
				s.take(s.u())
				s.take(s.u())
				if s.byte() != 0 {
					failInstrument("only function imports are supported")
				}
				s.u()
			}
		case 5:
			n := s.u()
			if n > 1 {
				failInstrument("multiple memories are unsupported")
			}
			if n == 1 {
				memory = true
				flags := s.u()
				if flags > 1 {
					failInstrument("unsupported memory limits")
				}
				s.u()
				if flags == 1 {
					maxPages = s.u()
				}
			}
		case 6:
			globalCount, globals = s.u(), data
		case 7:
			exports = data
			for n := s.u(); n > 0; n-- {
				name := string(s.take(s.u()))
				if strings.HasPrefix(name, privatePrefix) {
					failInstrument("reserved export name")
				}
				s.byte()
				s.u()
			}
		case 8:
			start, startIndex = true, s.u()
		}
	}
	appendVector := func(original []byte, count uint64, items []byte) []byte {
		s := wasmCursor(original)
		if len(s) > 0 {
			count += s.u()
		}
		return append(append(binary.AppendUvarint(nil, count), s...), items...)
	}
	addExport := func(dst []byte, name string, kind byte, index uint64) []byte {
		dst = append(binary.AppendUvarint(dst, uint64(len(name))), name...)
		return binary.AppendUvarint(append(dst, kind), index)
	}
	var addedExports []byte
	var exportCount uint64
	if memory {
		// Ceiling pages, desired pages, delta/result, and abort flag.
		globals = appendVector(globals, 4, []byte{
			0x7e, 1, 0x42, 0, 0x0b, 0x7e, 1, 0x42, 0, 0x0b,
			0x7f, 1, 0x41, 0, 0x0b, 0x7f, 1, 0x41, 0, 0x0b,
		})
		addedExports = addExport(addedExports, privateMemory, 2, 0)
		addedExports = addExport(addedExports, privateLimit, 3, globalCount)
		addedExports = addExport(addedExports, privateAbort, 3, globalCount+3)
		exportCount += 3
	}
	if start {
		addedExports = addExport(addedExports, privateStart, 0, startIndex)
		exportCount++
	}
	if exportCount > 0 {
		exports = appendVector(exports, exportCount, addedExports)
	}
	out = append(out, magic...)
	writeSection := func(id byte, data []byte) {
		out = append(out, id)
		out = append(binary.AppendUvarint(out, uint64(len(data))), data...)
	}
	writtenGlobals, writtenExports := !memory, exportCount == 0
	for _, section := range sections {
		if section.id != 0 {
			if !writtenGlobals && section.id >= 6 {
				writeSection(6, globals)
				writtenGlobals = true
			}
			if !writtenExports && section.id >= 7 {
				writeSection(7, exports)
				writtenExports = true
			}
		}
		if section.id == 8 || section.id == 6 && memory || section.id == 7 && exportCount > 0 {
			continue
		}
		data := section.data
		if section.id == 10 && memory {
			s := wasmCursor(data)
			n := s.u()
			data = binary.AppendUvarint(nil, n)
			for ; n > 0; n-- {
				body := instrumentBody(s.take(s.u()), globalCount, maxPages)
				data = append(binary.AppendUvarint(data, uint64(len(body))), body...)
			}
		}
		writeSection(section.id, data)
	}
	if !writtenGlobals {
		writeSection(6, globals)
	}
	if !writtenExports {
		writeSection(7, exports)
	}
	return out, nil
}

type instrumentError string

func failInstrument(s string) { panic(instrumentError(s)) }

type wasmCursor []byte

func (r *wasmCursor) take(n uint64) []byte {
	if n > uint64(len(*r)) {
		failInstrument("truncated binary")
	}
	b := (*r)[:n]
	*r = (*r)[n:]
	return b
}
func (r *wasmCursor) byte() byte { return r.take(1)[0] }
func (r *wasmCursor) u() uint64 {
	x, n := binary.Uvarint(*r)
	if n <= 0 {
		failInstrument("invalid unsigned LEB128")
	}
	*r = (*r)[n:]
	return x
}
func (r *wasmCursor) leb() {
	for i := 0; i < 10; i++ {
		if r.byte()&0x80 == 0 {
			return
		}
	}
	failInstrument("invalid signed LEB128")
}
func signedLEB(dst []byte, n int64) []byte {
	for {
		b := byte(n) & 0x7f
		n >>= 7
		if n == 0 && b&0x40 == 0 || n == -1 && b&0x40 != 0 {
			return append(dst, b)
		}
		dst = append(dst, b|0x80)
	}
}

func instrumentBody(body []byte, global, maxPages uint64) []byte {
	r := wasmCursor(body)
	for n := r.u(); n > 0; n-- {
		r.u()
		skipValueType(&r)
	}
	out := append([]byte(nil), body[:len(body)-len(r)]...)
	for len(r) > 0 {
		before := r
		op := r.byte()
		switch {
		case op == 0x40:
			if r.u() != 0 {
				failInstrument("multiple memories are unsupported")
			}
			out = append(out, guardedGrow(global, maxPages)...)
			continue
		case op == 0x02 || op == 0x03 || op == 0x04 || op == 0x41 || op == 0x42 || op == 0xd0:
			r.leb()
		case op == 0x0c || op == 0x0d || op == 0x10 || op == 0x12 || op == 0x14 || op == 0x15 || op >= 0x20 && op <= 0x26 || op == 0x3f || op == 0xd2:
			r.u()
		case op == 0x0e:
			for n := r.u() + 1; n > 0; n-- {
				r.u()
			}
		case op == 0x11 || op == 0x13 || op >= 0x28 && op <= 0x3e:
			r.u()
			r.u()
		case op == 0x1c:
			for n := r.u(); n > 0; n-- {
				skipValueType(&r)
			}
		case op == 0x43:
			r.take(4)
		case op == 0x44:
			r.take(8)
		case op == 0xfc:
			sub := r.u()
			switch {
			case sub <= 7:
			case sub == 8 || sub == 10 || sub == 12 || sub == 14:
				r.u()
				r.u()
			case sub == 9 || sub == 11 || sub == 13 || sub >= 15 && sub <= 17:
				r.u()
			default:
				failInstrument("unsupported miscellaneous instruction")
			}
		case op == 0xfd:
			sub := r.u()
			switch {
			case sub <= 11 || sub == 92 || sub == 93:
				r.u()
				r.u()
			case sub == 12 || sub == 13:
				r.take(16)
			case sub >= 21 && sub <= 34:
				r.take(1)
			case sub >= 84 && sub <= 91:
				r.u()
				r.u()
				r.take(1)
			case sub <= 275:
			default:
				failInstrument("unsupported SIMD instruction")
			}
		case op == 0x00 || op == 0x01 || op == 0x05 || op == 0x0b || op == 0x0f || op == 0x1a || op == 0x1b || op >= 0x45 && op <= 0xc4 || op == 0xd1:
		default:
			failInstrument(fmt.Sprintf("unsupported opcode 0x%x", op))
		}
		out = append(out, before[:len(before)-len(r)]...)
	}
	return out
}

func skipValueType(r *wasmCursor) {
	t := r.byte()
	if t == 0x63 || t == 0x64 {
		r.leb()
	}
}

func guardedGrow(g, maxPages uint64) []byte {
	var b []byte
	instruction := func(op byte, index uint64) { b = binary.AppendUvarint(append(b, op), index) }
	abort := func() { b = append(b, 0x41, 1); instruction(0x24, g+3); b = append(b, 0x00) }
	instruction(0x24, g+2)       // delta
	b = append(b, 0x3f, 0, 0xad) // memory.size; i64.extend_i32_u
	instruction(0x23, g+2)
	b = append(b, 0xad, 0x7c) // extend; add
	instruction(0x24, g+1)    // desired pages
	instruction(0x23, g+1)
	b = signedLEB(append(b, 0x42), int64(maxPages))
	b = append(b, 0x56, 0x04, 0x7f, 0x41, 0x7f, 0x05) // > max: -1, else
	instruction(0x23, g+1)
	instruction(0x23, g)
	b = append(b, 0x56, 0x04, 0x40)
	abort()
	b = append(b, 0x0b)
	instruction(0x23, g+2)
	b = append(b, 0x40, 0) // memory.grow
	instruction(0x24, g+2)
	instruction(0x23, g+2)
	b = append(b, 0x41, 0x7f, 0x46, 0x04, 0x40) // allocation failure: abort
	abort()
	b = append(b, 0x0b)
	instruction(0x23, g+2)
	b = append(b, 0x0b)
	return b
}
