package main

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcrypto
#include "hash256.cu"
*/
import "C"
import (
	"fmt"
)

func main() {
	numElements := 10
	input := make([]byte, numElements*10)
	output := make([]byte, numElements*C.SHA256_DIGEST_LENGTH)

	// 填充输入数据
	for i := range input {
		input[i] = byte(i % 256)
	}

	C.runHash256((*C.uchar)(&input[0]), (*C.uchar)(&output[0]), C.int(numElements))

	for i := 0; i < numElements; i++ {
		fmt.Printf("Hash %d: %x\n", i, output[i*C.SHA256_DIGEST_LENGTH:(i+1)*C.SHA256_DIGEST_LENGTH])
	}
}
