package utils

import (
	"context"
	"nostr/logger"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/ethclient"
)

type ConnectManager struct {
	Ethclient []*ethclient.Client
	BlockChan chan ChanType

	// LatestBlockInfo BlockInfoWithMux
	BInfo        BlockInfo
	FlagBlockNum uint64
}

type BlockInfoWithMux struct {
	sync.RWMutex
	Binfo BlockInfo
}
type BlockInfo struct {
	BlockNumber uint64
	BlockHash   string
}

func NewConnectManager(commonChan chan ChanType) *ConnectManager {

	return &ConnectManager{
		Ethclient: make([]*ethclient.Client, 0),
		BlockChan: commonChan,
	}
}

func (c *ConnectManager) Connect() {
	for _, url := range ArbRpcUrls {
		client, err := ethclient.Dial(url)
		if err != nil {
			continue
		}
		c.Ethclient = append(c.Ethclient, client)
	}
}

func (c *ConnectManager) GetBlockInfo(wg *sync.WaitGroup) {
	for _, client := range c.Ethclient {

		// wg.Add(1)
		// go func(wg *sync.WaitGroup) {

		for {
			header, err := client.HeaderByNumber(context.Background(), nil)
			if err != nil {
				logger.GLogger.Error("get BlockNumber error:", err)
				time.Sleep(1 * time.Second)
				continue
			}
			// block, err := client.BlockByNumber(context.Background(), big.NewInt(int64(blockNumber)))
			// if err != nil {
			// 	logger.GLogger.Error("get block hash error:", err)
			// 	time.Sleep(1 * time.Second)
			// 	continue
			// }
			// c.BlockChan <- &BlockInfo{
			// 	BlockNumber: header.Number.Uint64(),
			// 	BlockHash:   header.Hash().String(),
			// }
			c.BlockChan <- ChanType{
				Datatype: "block",
				Data: &BlockInfo{
					BlockNumber: header.Number.Uint64(),
					BlockHash:   header.Hash().String(),
				},
			}
		}

		wg.Done()

		// }(&wg)

	}

	wg.Wait()

}
