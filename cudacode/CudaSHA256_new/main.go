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

	"github.com/nbd-wtf/go-nostr"
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
	printCDigests(cDigests, len(inputs))

	//for i, digest := range digests {
	//    println(i,digest)
	//    output[i] = string(digest)
	//	 hashStr := hex.EncodeToString(digest)
	//    println(hashStr)
	// }
	//return output
	return formatCDigests(cDigests, len(inputs))
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

	foundEvent := nostr.Event{
		Kind:      1,
		CreatedAt: 1704575164,
		Tags:      nostr.Tags{{"p", "9be107b0d7218c67b4954ee3e6bd9e4dba06ef937a93f684e42f730a0c3d053c"}, {"e", "51ed7939a984edee863bfbb2e66fdc80436b000a8ddca442d83e6a2bf1636a95", "wss://relay.noscription.org/", "root"}, {"e", "00000571cb4fabce45e915cce67ec468051d550847f166137fd8aea8615bcd8c", "wss://relay.noscription.org/", "reply"}, {"seq_witness", "167798002", "0xec4bb82180016b3e050cc9c5deceb672e360bfb251e7543c6323348d1505d99e"}, {"nonce", "ctlqrejf99", "21"}},
		Content:   "{\"p\":\"nrc-20\",\"op\":\"mint\",\"tick\":\"noss\",\"amt\":\"10\"}",
		PubKey:    "66313c9225464c64e8cbab0d48b16a9b5a25f206e00bb79371b684743aa9d288",
	}

	foundEvent.GetID()

	// foundEvent.Sign("710155b5a9e39097669893d132b0a34b7302e78f2a9d75fcd304bf7951eeb878")

	GPUID := HashStrings([]string{string(foundEvent.Serialize())})

	fmt.Println("GPU ID:", GPUID)
	fmt.Println("CPU ID:", foundEvent.GetID())
}
