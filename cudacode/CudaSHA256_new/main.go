// main.go
package main

/*
#cgo LDFLAGS: -L. -lcuda_hash -L/usr/local/cuda/lib64 -lcudart
#include "sha256.cuh"
#include "main.cu"
*/
import "C"
import (
	"unsafe"
)

//export HashString
func HashString(input string) string {
	cstr := C.CString(input)
	defer C.free(unsafe.Pointer(cstr))
	output := make([]byte, 64) // SHA-256 hash size
	C.run_hash(C.uchar(cstr), C.ulong(len(input)), (*C.uchar)(&output[0]))
	return string(output)
}

func main() {
	hashedString := HashString("your string here")
	println(hashedString)
}
