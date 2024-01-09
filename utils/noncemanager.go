package utils

import (
	"crypto/rand"
	"sync"
	"time"

	"nostr/logger"
)

type NonceMan struct {
	NonceChan chan string // 通道
}

func NewNonceMan(nonch chan string) *NonceMan {
	return &NonceMan{
		NonceChan: nonch,
	}
}

func (n *NonceMan) GenerateRandomString(length int) (string, error) {
	charset := "abcdefghijklmnopqrstuvwxyz0123456789" // 字符集
	b := make([]byte, length)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}

	for i := 0; i < length; i++ {
		b[i] = charset[int(b[i])%len(charset)]
	}

	return string(b), nil
}

func (n *NonceMan) GenNonceRun(wg *sync.WaitGroup) {

	for {

		nonce, err := n.GenerateRandomString(10)
		if err != nil {

			logger.GLogger.Info("generate random string error:", err)
			time.Sleep(1 * time.Second)
			continue
		}
		//check NonceChan is full or not
		logger.GLogger.Info("nonce:", nonce)
		n.NonceChan <- nonce
	}

	wg.Done()
}
