// main.go
package main

/*
#cgo LDFLAGS: -L. -lcuda_hash -L/usr/local/cuda/lib64 -lcudart
#include <stdlib.h>
#include "cuda_sha256.h"
*/
import "C"
import (
	"unsafe"
	//"time"
	//"encoding/hex"
	"fmt"
)

////export HashString
//func HashStringOld(input string) string {
//	cstr := C.CString(input)
//	defer C.free(unsafe.Pointer(cstr))
//	output := make([]byte, 64) // SHA-256 hash size
//	C.hashString(cstr, C.size_t(len(input)), (*C.uchar)(&output[0]))
//	return string(output)
//}

//export HashStrings
func HashStrings(inputs []string) []string {
    numStrs := C.int(len(inputs))
    cstrs := C.malloc(C.size_t(numStrs) * C.size_t(unsafe.Sizeof(uintptr(0))))
    defer C.free(cstrs)

    cDigests := C.malloc(C.size_t(numStrs) * C.size_t(unsafe.Sizeof(uintptr(0))))
    defer C.free(cDigests)

    for i, s := range inputs {
        cs := C.CString(s)
        defer C.free(unsafe.Pointer(cs))
        *(*uintptr)(unsafe.Pointer(uintptr(cstrs) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = uintptr(unsafe.Pointer(cs))
    }

     digests := make([][]byte, numStrs)
     for i := range digests {
        digests[i] = make([]byte, 64) // SHA-256 hash size
        *(*uintptr)(unsafe.Pointer(uintptr(cDigests) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = uintptr(unsafe.Pointer(&digests[i][0]))
     }

     C.hashStrings((**C.char)(cstrs), numStrs, (**C.uchar)(cDigests))

     //output := make([]string, numStrs)
     printCDigests(cDigests,len(inputs))
     
     //for i, digest := range digests {
     //    println(i,digest)
     //    output[i] = string(digest)
     //	 hashStr := hex.EncodeToString(digest)
     //    println(hashStr)
   // }
    //return output
    return formatCDigests(cDigests,len(inputs))
}

func printCDigests(cDigests unsafe.Pointer, numStrs int) {
    // 将 cDigests 转换为指向指针数组的指针
    digestPtrs := (*[1 << 30]*C.uchar)(cDigests)

    for i := 0; i < numStrs; i++ {
        // 获取第 i 个哈希值的指针
        digestPtr := digestPtrs[i]

        // 创建一个指向相应哈希值的字节切片
        hashSlice := (*[32]byte)(unsafe.Pointer(digestPtr))[:32:32]

        // 打印哈希值的十六进制表示
        fmt.Printf("Harry Hash %d: ", i)
        for _, b := range hashSlice {
            fmt.Printf("%02x", b)
        }
        fmt.Println()
    }
}

func formatCDigests(cDigests unsafe.Pointer, numStrs int) []string {
    // 将 cDigests 转换为指向指针数组的指针
    digestPtrs := (*[1 << 30]*C.uchar)(cDigests)

    // 创建一个字符串切片来存储格式化的哈希值
    formattedHashes := make([]string, numStrs)

    for i := 0; i < numStrs; i++ {
        // 获取第 i 个哈希值的指针
        digestPtr := digestPtrs[i]

        // 创建一个指向相应哈希值的字节切片
        hashSlice := (*[32]byte)(unsafe.Pointer(digestPtr))[:32:32]

        // 使用 fmt.Sprintf 构建每个哈希值的十六进制字符串表示
        hashStr := fmt.Sprintf("Hash %d: ", i)
        for _, b := range hashSlice {
            hashStr += fmt.Sprintf("%02x", b)
        }

        // 将格式化的哈希字符串添加到切片中
        fmt.Printf("Hello Hash %s: ", hashStr)
	formattedHashes[i] = hashStr
    }
   fmt.Printf("Hello formattedHashes %+v: ", formattedHashes)
    return formattedHashes
}


func main() {
	hashedString := HashStrings([]string{"your string here"})
	fmt.Println(hashedString)
}
