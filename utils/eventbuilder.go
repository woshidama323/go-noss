package utils

import (
	"nostr/logger"
	"runtime"
	"strconv"
	"sync"

	"github.com/nbd-wtf/go-nostr"
	"github.com/nbd-wtf/go-nostr/nip13"
)

type EventMan struct {
	NonceMan   *NonceMan
	ConnectMan *ConnectManager
	EventIDMan *EventIDMan
	Num        int
	CommonChan chan ChanType

	UnsignedEvent nostr.Event
}

// func NewEventMan(nonceMan NonceMan, connectMan ConnectManager, preEventMan EventIDMan, num int) *EventMan {
func NewEventMan(num int) *EventMan {

	commonChan := make(chan ChanType, 1000)
	nonceMan := NewNonceMan(commonChan)
	connectMan := NewConnectManager(commonChan)
	preEventMan := NewEventIDMan(commonChan)

	return &EventMan{
		NonceMan:   nonceMan,
		ConnectMan: connectMan,
		EventIDMan: preEventMan,
		Num:        num,
		CommonChan: commonChan,
	}
}

func (e *EventMan) Run() {

	var wg sync.WaitGroup
	//start a gorountine for generate random string continuously

	wg.Add(1)
	go e.NonceMan.GenNonceRun(&wg)

	e.ConnectMan.Connect()

	go e.ConnectMan.GetBlockInfo(&wg)

	wg.Add(1)
	go e.EventIDMan.GetPreviousID(&wg, "wss://report-worker-2.noscription.org")

	wg.Wait()
}

func (e *EventMan) AssembleBaseEvent(newNonce, newBlockHash, newPreviousID string, blockNumber int64) nostr.Event {

	// nonce := ""
	// blockNumber := 0
	// blockHash := ""
	// previousID := ""

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
	event.Tags = event.Tags.AppendUnique(nostr.Tag{"e", newPreviousID, WSURL, "reply"})
	event.Tags = event.Tags.AppendUnique(nostr.Tag{"seq_witness", strconv.FormatInt(blockNumber, 10), newBlockHash})
	event.Tags = event.Tags.AppendUnique(nostr.Tag{"nonce", newNonce, "21"})

	return event
}

func (e *EventMan) HashCalculate() {
	// runtime.LockOSThread()
	forHash := []nostr.Event{}
	forHashString := []string{}

	nonce := e.NonceMan.Firstnonce
	blockNumber := e.ConnectMan.FirstBlockInfo.BlockNumber
	blockHash := e.ConnectMan.FirstBlockInfo.BlockHash
	previousID := e.EventIDMan.FirstID

	for comData := range e.CommonChan {

		// logger.GLogger.Info("comData:", comData)
		if comData.Datatype == "nonce" {
			nonce = comData.Data.(string)
		} else if comData.Datatype == "block" {
			blockNumber = uint64(comData.Data.(BlockInfo).BlockNumber)
			blockHash = comData.Data.(BlockInfo).BlockHash
		} else if comData.Datatype == "previousid" {
			previousID = comData.Data.(string)
		}
		if nonce == "" || blockNumber == 0 || blockHash == "" || previousID == "" {
			continue
		}

		ev := e.AssembleBaseEvent(nonce, blockHash, previousID, int64(blockNumber))
		forHash = append(forHash, ev)
		forHashString = append(forHashString, string(ev.Serialize()))
		if len(forHashString) >= e.Num {

			logger.GLogger.Debugln("len(forHashString) > e.Num", len(forHashString))
			hashGPU := HashStringsNew(forHashString)
			logger.GLogger.Debugln("hashGPU:", hashGPU[0])

			// runtime.LockOSThread()
			//verify hash
			// go func(input []string) {
			for i := 0; i < len(hashGPU); i++ {
				logger.GLogger.Debugln("hashGPU[i]:", hashGPU[i])
				if nip13.Difficulty(hashGPU[i]) >= 21 {

					logger.GLogger.Info("new Event ID:", hashGPU[i])
					forHash[i].ID = hashGPU[i]
					forHash[i].Sign("710155b5a9e39097669893d132b0a34b7302e78f2a9d75fcd304bf7951eeb878")
					logger.GLogger.Info("new Event ID:", forHash[i].GetID())

					forHash = []nostr.Event{}
					//send event to noscription
				}
			}

			// time.Sleep(3 * time.Second)
			// }(hashGPU)

			forHashString = []string{}
		}

	}

	runtime.UnlockOSThread()
}
