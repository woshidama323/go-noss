package utils

/*
#cgo LDFLAGS: -L../ -lcuda_hash -L/usr/local/cuda/lib64 -lcudart
#include <stdlib.h>
#include "../cgoinclude/cuda_sha256.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func formatCDigests(cDigests unsafe.Pointer, numStrs int) []string {
	// 将 cDigests 转换为指向指针数组的指针
	digestPtrs := (*[1 << 30]*C.uchar)(cDigests)

	// 创建一个字符串切片来存储格式化的哈希值
	formattedHashes := make([]string, numStrs)

	for i := 0; i < numStrs; i++ {
		// 获取第 i 个哈希值的指针
		digestPtr := digestPtrs[i]

		// 创建一个指向相应哈希值的字节切片
		hashSlice := (*[64]byte)(unsafe.Pointer(digestPtr))[:64:64]

		// 使用 fmt.Sprintf 构建每个哈希值的十六进制字符串表示
		hashStr := ""
		for _, b := range hashSlice {
			hashStr += fmt.Sprintf("%02x", b)
		}

		// 将格式化的哈希字符串添加到切片中
		formattedHashes[i] = hashStr
	}
	// fmt.Printf("Hello formattedHashes %+v: ", formattedHashes)
	return formattedHashes
}

//export HashStringsWithGPU
func HashStringsWithGPU(inputs []string) []string {
	numStrs := C.int(len(inputs))
	cstrs := C.malloc(C.size_t(numStrs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(cstrs)

	cDigests := C.malloc(C.size_t(numStrs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(cDigests)

	for i, s := range inputs {
		cs := C.CString(s)
		*(*uintptr)(unsafe.Pointer(uintptr(cstrs) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = uintptr(unsafe.Pointer(cs))
		defer C.free(unsafe.Pointer(cs))
	}

	digests := make([][]byte, numStrs)
	for i := range digests {
		digests[i] = make([]byte, 32) // SHA-256 hash size
		*(*uintptr)(unsafe.Pointer(uintptr(cDigests) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = uintptr(unsafe.Pointer(&digests[i][0]))
	}

	C.hashStrings((**C.char)(cstrs), numStrs, (**C.uchar)(cDigests))

	//output := make([]string, numStrs)

	return formatCDigests(cDigests, len(inputs))
}

//export HashStringsNew
func HashStringsNew(inputs []string) []string {
	numStrs := C.int(len(inputs))
	cstrs := C.malloc(C.size_t(numStrs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	// defer C.free(cstrs)

	cDigests := C.malloc(C.size_t(numStrs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	// defer C.free(cDigests)

	for i, s := range inputs {
		cs := C.CString(s)
		*(*uintptr)(unsafe.Pointer(uintptr(cstrs) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = uintptr(unsafe.Pointer(cs))
		// defer C.free(unsafe.Pointer(cs))
	}

	digests := make([][]byte, numStrs)
	for i := range digests {
		digests[i] = make([]byte, 64) // SHA-256 hash size
		*(*uintptr)(unsafe.Pointer(uintptr(cDigests) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = uintptr(unsafe.Pointer(&digests[i][0]))
	}

	C.hashTest((**C.char)(cstrs), numStrs, (**C.uchar)(cDigests))

	C.free(cstrs)
	C.free(cDigests)

	return formatCDigests(cDigests, len(inputs))
}
