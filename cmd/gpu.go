// main.go
package cmd

/*
#cgo LDFLAGS: -L. -lcuda_hash -L/usr/local/cuda/lib64 -lcudart
#include <stdlib.h>
#include "cuda_sha256.h"
*/
import "C"
import (
	"fmt"
	"math/rand"
	"nostr/utils"
	"time"
	"unsafe"

	"github.com/nbd-wtf/go-nostr"
	"github.com/nbd-wtf/go-nostr/nip13"
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli/v2"
)

func init() {

	logrus.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true, // Enable timestamp
		ForceColors:   true, // Force colored output even when stdout is not a terminal

	})
}

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
	return formatCDigests(cDigests, len(inputs))
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

var GPUCmd = &cli.Command{
	Name:  "gpu",
	Usage: "gpu for create sign",
	Flags: []cli.Flag{
		&cli.StringFlag{
			Name:  "rpcurl",
			Usage: "trb rpc url",
			Value: "https://arb1.arbitrum.io/rpc",
		},
		&cli.StringFlag{
			Name:  "noscriptionWss",
			Usage: "noscription wss url",
			Value: "wss://report-worker-2.noscription.org",
		},
		&cli.StringFlag{
			Name:  "pk",
			Usage: "private key",
			Value: "x",
		},
		&cli.IntFlag{
			Name:  "num",
			Usage: "num times",
			Value: 1000,
		},
	},
	Action: func(cctx *cli.Context) error {
		//随机1000次
		var letterRunes = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

		// ranNonce := []string{}
		ranSerial := []string{}

		for {
			rand.Seed(time.Now().UnixNano())

			b := make([]rune, 10)
			for i := range b {
				b[i] = letterRunes[rand.Intn(len(letterRunes))]
			}
			fmt.Println("random nonce:", string(b))
			// ranNonce = append(ranNonce, string(b))

			ranTest := nostr.Event{
				Kind:      1,
				CreatedAt: 1704575164,
				Tags:      nostr.Tags{{"p", "9be107b0d7218c67b4954ee3e6bd9e4dba06ef937a93f684e42f730a0c3d053c"}, {"e", "51ed7939a984edee863bfbb2e66fdc80436b000a8ddca442d83e6a2bf1636a95", "wss://relay.noscription.org/", "root"}, {"e", "00000571cb4fabce45e915cce67ec468051d550847f166137fd8aea8615bcd8c", "wss://relay.noscription.org/", "reply"}, {"seq_witness", "167798002", "0xec4bb82180016b3e050cc9c5deceb672e360bfb251e7543c6323348d1505d99e"}, {"nonce", fmt.Sprintf("%s", string(b)), "21"}},
				Content:   "{\"p\":\"nrc-20\",\"op\":\"mint\",\"tick\":\"noss\",\"amt\":\"10\"}",
				PubKey:    "66313c9225464c64e8cbab0d48b16a9b5a25f206e00bb79371b684743aa9d288",
			}

			ranSerial = append(ranSerial, string(ranTest.Serialize()))

			if len(ranSerial) > cctx.Int("num") {
				// go func(t []string) {
				logrus.Info("start GPU hash =================")
				GPUID := HashStrings(ranSerial)
				logrus.Warn("end GPU hash =================")
				//verify hash
				for i := 0; i < len(GPUID); i++ {
					if nip13.Difficulty(GPUID[i]) >= 21 {
						logrus.Info("new Event ID:", GPUID[i])
					}
				}

				fmt.Println("GPU ID:", GPUID)
				// }(ranSerial)

				ranSerial = []string{}
				continue
			}
		}
	},
}

var CreateEventCmd = &cli.Command{
	Name:  "createEvent",
	Usage: "createEvent for create evnt",
	Flags: []cli.Flag{
		&cli.StringFlag{
			Name:  "rpcurl",
			Usage: "trb rpc url",
			Value: "https://arb1.arbitrum.io/rpc",
		},
		&cli.StringFlag{
			Name:  "noscriptionWss",
			Usage: "noscription wss url",
			Value: "wss://report-worker-2.noscription.org",
		},
		&cli.StringFlag{
			Name:  "pk",
			Usage: "private key",
			Value: "x",
		},
		&cli.IntFlag{
			Name:  "num",
			Usage: "num times",
			Value: 1000,
		},
		&cli.StringFlag{
			Name:       "loglevel",
			Usage:      "log level",
			Value:      "info",
			HasBeenSet: false,
		},
	},
	Action: func(cctx *cli.Context) error {

		if cctx.String("loglevel") == "debug" {
			logrus.SetLevel(logrus.DebugLevel)
		}

		eMan := utils.NewEventMan(cctx.Int("num"))

		go eMan.Run()

		eMan.HashCalculate()
		return nil
	},
}
