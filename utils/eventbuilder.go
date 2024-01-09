package utils

import (
	"nostr/logger"
	"sync"

	"github.com/nbd-wtf/go-nostr"
)

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
	for nonce := range e.NonceChan {
		// fmt.Println("nonce:", nonce)

		logger.GLogger.Info("nonce:", nonce)

		//assemble event

		forHash = append(forHash, e.AssembleBaseEvent(nonce))

		if len(forHash) > e.Num {

			logger.GLogger.Info("start GPU hash =================")

		}

	}
}
