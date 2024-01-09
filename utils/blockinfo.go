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
	BInfo          BlockInfo
	FlagBlockNum   uint64
	FirstBlockInfo BlockInfo
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
		FirstBlockInfo: BlockInfo{
			BlockNumber: 18967110,
			BlockHash:   "0xbfb5f7164f0d09460add41aadc9235b9c32eff8123660ebffc99c31c0d7f82d6",
		},
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
				Data: BlockInfo{
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
