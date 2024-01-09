package utils

/*
cgo LDFLAGS: -L../ -lcuda_hash -L/usr/local/cuda/lib64 -lcudart
#include <stdlib.h>
#include "../cgoinclude/cuda_sha256.h"
*/
import "C"

import (
	"fmt"
	"nostr/logger"
	"sync"
	"unsafe"

	"github.com/nbd-wtf/go-nostr"
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
		hashSlice := (*[32]byte)(unsafe.Pointer(digestPtr))[:32:32]

		// 使用 fmt.Sprintf 构建每个哈希值的十六进制字符串表示
		hashStr := ""
		for _, b := range hashSlice {
			hashStr += fmt.Sprintf("%02x", b)
		}

		// 将格式化的哈希字符串添加到切片中
		formattedHashes[i] = hashStr
	}
	fmt.Printf("Hello formattedHashes %+v: ", formattedHashes)
	return formattedHashes
}

type EventMan struct {
	NonceMan   *NonceMan
	ConnectMan *ConnectManager
	EventIDMan *EventIDMan
	Num        int
	NonceChan  chan string

	UnsignedEvent nostr.Event
}

// func NewEventMan(nonceMan NonceMan, connectMan ConnectManager, preEventMan EventIDMan, num int) *EventMan {
func NewEventMan(num int) *EventMan {

	nonChan := make(chan string, 1000)
	nonceMan := NewNonceMan(nonChan)
	connectMan := NewConnectManager()
	preEventMan := NewEventIDMan()

	return &EventMan{
		NonceMan:   nonceMan,
		ConnectMan: connectMan,
		EventIDMan: preEventMan,
		Num:        num,
		NonceChan:  nonChan,
	}
}

func (e *EventMan) Run() {

	var wg sync.WaitGroup
	//start a gorountine for generate random string continuously

	wg.Add(1)
	go e.NonceMan.GenNonceRun(&wg)

	e.ConnectMan.Connect()

	e.ConnectMan.GetBlockInfo()

	wg.Add(1)
	go e.EventIDMan.GetPreviousID(&wg, "wss://arb-mainnet.g.alchemy.com/v2/demo")

	wg.Wait()
}

func (e *EventMan) AssembleBaseEvent(newNonce string) nostr.Event {
	event := nostr.Event{
		Content:   "{\"p\":\"nrc-20\",\"op\":\"mint\",\"tick\":\"noss\",\"amt\":\"10\"}",
		CreatedAt: nostr.Now(),
		ID:        "",
		Kind:      nostr.KindTextNote,
		PubKey:    "66313c9225464c64e8cbab0d48b16a9b5a25f206e00bb79371b684743aa9d288",
		Sig:       "",
		Tags:      nil,
	}
	event.Tags = event.Tags.AppendUnique(nostr.Tag{"p", "9be107b0d7218c67b4954ee3e6bd9e4dba06ef937a93f684e42f730a0c3d053c"})
	event.Tags = event.Tags.AppendUnique(nostr.Tag{"e", "51ed7939a984edee863bfbb2e66fdc80436b000a8ddca442d83e6a2bf1636a95", WSURL, "root"})
	event.Tags = event.Tags.AppendUnique(nostr.Tag{"e", e.EventIDMan.PreID.PreviesId, WSURL, "reply"})
	event.Tags = event.Tags.AppendUnique(nostr.Tag{"seq_witness", string(e.ConnectMan.GetLatestBlockInfo().BlockNumber), e.ConnectMan.GetLatestBlockInfo().BlockHash})
	event.Tags = event.Tags.AppendUnique(nostr.Tag{"nonce", newNonce, "21"})

	return event
}

func (e *EventMan) HashCalculate() {

	forHash := []nostr.Event{}
	forHashString := []string{}
	for nonce := range e.NonceChan {
		// fmt.Println("nonce:", nonce)

		logger.GLogger.Info("nonce:", nonce)

		//assemble event

		ev := e.AssembleBaseEvent(nonce)
		forHash = append(forHash, e.AssembleBaseEvent(nonce))
		forHashString = append(forHashString, string(ev.Serialize()))
		if len(forHash) > e.Num {

			logger.GLogger.Info("len(forHash) > e.Num", len(forHash))
			hashGPU := HashStrings(forHashString)
			logger.GLogger.Info("hashGPU:", hashGPU[0])

		}

	}
}

//export HashStrings
func (e *EventMan) HashStrings(inputs []string) []string {
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
	return formatCDigests(cDigests, len(inputs))
}
