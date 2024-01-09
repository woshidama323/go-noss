package utils

import (
	"encoding/json"
	"fmt"
	"log"
	"nostr/types"
	"sync"
)

type PreviousIDMux struct {
	sync.RWMutex
	PreviesId string
}
type EventIDMan struct {
	PreviesIdChan chan ChanType
	PreID         PreviousIDMux
	FirstID       string
}

func NewEventIDMan(commonChan chan ChanType) *EventIDMan {

	return &EventIDMan{
		PreviesIdChan: commonChan,
		FirstID:       "0000004023b58ef98f88dc696269e429cbf3c9062ae1267f4f11a016e2640dbf",
	}
}

// func (g *EventIDMan) HashCalculate(dataForHash []string) {

// 	// for x := range Hsh
// 	GPUID := HashStrings(dataForHash)
// 	//verify hash
// 	for i := 0; i < len(GPUID); i++ {
// 		if nip13.Difficulty(GPUID[i]) >= 21 {
// 			logger.LogInfo("new Event ID:")
// 			//send event to noscription
// 		}
// 	}
// 	fmt.Println("GPU ID:", GPUID)
// }

func (g *EventIDMan) GetPreviousID(wg *sync.WaitGroup, url string) {

	//每隔三秒种获取一次最新的messageId
	c, err := ConnectToWSS(url)
	if err != nil {
		panic(err)
	}
	defer c.Close()

	for {
		_, message, err := c.ReadMessage()
		if err != nil {
			log.Println("read:", err)
			break
		}

		var messageDecode types.Message
		if err := json.Unmarshal(message, &messageDecode); err != nil {
			fmt.Println(err)
			continue
		}

		// g.PreviesIdChan <- messageDecode.EventId
		g.PreviesIdChan <- ChanType{
			Datatype: "previousid",
			Data:     messageDecode.EventId,
		}
	}

	wg.Done()

}

// func (g *EventIDMan) GetPreviousIDFromChan() {

// 	forDup := ""
// 	for id := range g.PreviesIdChan {
// 		logger.GLogger.Info("get previous id:", id)

// 		if forDup == id {
// 			continue
// 		} else {
// 			g.PreID.Lock()
// 			g.PreID.PreviesId = id
// 			g.PreID.Unlock()
// 			forDup = id

// 		}
// 	}
// }
