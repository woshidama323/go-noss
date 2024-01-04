package cmd

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/urfave/cli/v2"
)

var DaemonCmd = &cli.Command{
	Name:  "daemon",
	Usage: "run backend",
	Flags: []cli.Flag{
		&cli.StringFlag{
			Name:  "type",
			Usage: "specify key type to create",
			Value: "bls",
		},
	},
	Action: func(cctx *cli.Context) error {
		wssAddr := "wss://report-worker-2.noscription.org"
		// relayUrl := "wss://relay.noscription.org/"
		ctx := context.Background()

		var err error

		client, err := ethclient.Dial(arbRpcUrl)
		if err != nil {
			log.Fatalf("无法连接到Arbitrum节点: %v", err)
		}

		c, err := connectToWSS(wssAddr)
		if err != nil {
			panic(err)
		}
		defer c.Close()

		// initialize an empty cancel function

		// get block
		go func() {
			for {
				header, err := client.HeaderByNumber(context.Background(), nil)
				if err != nil {
					log.Fatalf("无法获取最新区块号: %v", err)
				}
				if header.Number.Uint64() >= blockNumber {
					hash = header.Hash().Hex()
					blockNumber = header.Number.Uint64()
				}
			}
		}()

		go func() {
			for {
				_, message, err := c.ReadMessage()
				if err != nil {
					log.Println("read:", err)
					break
				}

				var messageDecode Message
				if err := json.Unmarshal(message, &messageDecode); err != nil {
					fmt.Println(err)
					continue
				}
				messageId = messageDecode.EventId
			}

		}()

		atomic.StoreInt32(&currentWorkers, 0)
		// 初始化一个取消上下文和它的取消函数
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// 监听blockNumber和messageId变化
		go func() {
			for {
				select {
				case <-ctx.Done(): // 如果上下文被取消，则退出协程
					return
				default:
					if atomic.LoadInt32(&currentWorkers) < int32(numberOfWorkers) {
						atomic.AddInt32(&currentWorkers, 1) // 增加工作者数量
						go func(bn uint64, mid string) {
							defer atomic.AddInt32(&currentWorkers, -1) // 完成后减少工作者数量
							mine(ctx, mid, client)
						}(blockNumber, messageId)
					}
				}
			}
		}()

		select {}
		return nil
	},
}

func generateRandomString(length int) (string, error) {
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

func Generate(event nostr.Event, targetDifficulty int) (nostr.Event, error) {
	tag := nostr.Tag{"nonce", "", strconv.Itoa(targetDifficulty)}
	event.Tags = append(event.Tags, tag)
	start := time.Now()
	for {
		nonce, err := generateRandomString(10)
		if err != nil {
			fmt.Println(err)
		}
		// fmt.Println("nonce: ", nonce)
		tag[1] = nonce
		event.CreatedAt = nostr.Now()
		event.Tags[len(event.Tags)-1] = tag
		newID := event.GetID()

		if nip13.Difficulty(newID) >= targetDifficulty {
			fmt.Println("new Event ID:", newID)
			// fmt.Print(time.Since(start))
			event.ID = newID
			return event, nil
		}
		if time.Since(start) >= 1*time.Second {
			fmt.Println("timeout here")
			return event, ErrGenerateTimeout
		}
	}
}

type Message struct {
	EventId string `json:"eventId"`
}

type EV struct {
	Sig       string          `json:"sig"`
	Id        string          `json:"id"`
	Kind      int             `json:"kind"`
	CreatedAt nostr.Timestamp `json:"created_at"`
	Tags      nostr.Tags      `json:"tags"`
	Content   string          `json:"content"`
	PubKey    string          `json:"pubkey"`
}

func mine(ctx context.Context, messageId string, client *ethclient.Client) {

	replayUrl := "wss://relay.noscription.org/"
	difficulty := 21

	// Create a channel to signal the finding of a valid nonce
	foundEvent := make(chan nostr.Event, 1)
	// Create a channel to signal all workers to stop
	content := `{"p":"nrc-20","op":"mint","tick":"noss","amt":"10"}`
	startTime := time.Now()

	ev := nostr.Event{
		Content:   content,
		CreatedAt: nostr.Now(),
		ID:        "",
		Kind:      nostr.KindTextNote,
		PubKey:    pk,
		Sig:       "",
		Tags:      nil,
	}
	ev.Tags = ev.Tags.AppendUnique(nostr.Tag{"p", "9be107b0d7218c67b4954ee3e6bd9e4dba06ef937a93f684e42f730a0c3d053c"})
	ev.Tags = ev.Tags.AppendUnique(nostr.Tag{"e", "51ed7939a984edee863bfbb2e66fdc80436b000a8ddca442d83e6a2bf1636a95", replayUrl, "root"})
	ev.Tags = ev.Tags.AppendUnique(nostr.Tag{"e", messageId, replayUrl, "reply"})
	ev.Tags = ev.Tags.AppendUnique(nostr.Tag{"seq_witness", strconv.Itoa(int(blockNumber)), hash})
	// Start multiple worker goroutines
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				evCopy := ev
				evCopy, err := Generate(evCopy, difficulty)
				if err != nil {
					// fmt.Println(err)
					atomic.AddInt32(&currentWorkers, -1)
					return
				}

				fmt.Println("evCopy: ", evCopy)

				foundEvent <- evCopy
			}
		}
	}()

	select {
	case evNew := <-foundEvent:

		fmt.Println("evNew: ", evNew.ID)
		evNew.Sign(sk)
		fmt.Println("evNew after: ", evNew.ID)
		evNewInstance := EV{
			Sig:       evNew.Sig,
			Id:        evNew.ID,
			Kind:      evNew.Kind,
			CreatedAt: evNew.CreatedAt,
			Tags:      evNew.Tags,
			Content:   evNew.Content,
			PubKey:    evNew.PubKey,
		}
		// 将ev转为Json格式
		eventJSON, err := json.Marshal(evNewInstance)
		if err != nil {
			log.Fatal(err)
		}

		wrapper := map[string]json.RawMessage{
			"event": eventJSON,
		}

		// 将包装后的对象序列化成JSON
		wrapperJSON, err := json.MarshalIndent(wrapper, "", "  ") // 使用MarshalIndent美化输出
		if err != nil {
			log.Fatalf("Error marshaling wrapper: %v", err)
		}

		url := "https://api-worker.noscription.org/inscribe/postEvent"
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(wrapperJSON))
		if err != nil {
			log.Fatalf("Error creating request: %v", err)
		}

		// 设置HTTP Header
		req.Header.Set("Content-Type", "application/json")

		// 发送请求
		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			log.Fatalf("Error sending request: %v", err)
		}
		defer resp.Body.Close()

		fmt.Println("Response Status:", resp.Status)
		spendTime := time.Since(startTime)
		// fmt.Println("Response Body:", string(body))
		fmt.Println(nostr.Now().Time(), "spend: ", spendTime, "!!!!!!!!!!!!!!!!!!!!!published to:", evNewInstance.Id)
		atomic.StoreInt32(&nonceFound, 0)
	case <-ctx.Done():
		fmt.Print("done")
	}

}

func connectToWSS(url string) (*websocket.Conn, error) {
	var conn *websocket.Conn
	var err error
	headers := http.Header{}
	headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0")
	headers.Add("Origin", "https://noscription.org")
	headers.Add("Host", "report-worker-2.noscription.org")
	for {
		// 使用gorilla/websocket库建立连接
		conn, _, err = websocket.DefaultDialer.Dial(url, headers)
		fmt.Println("Connecting to wss")
		if err != nil {
			// 连接失败，打印错误并等待一段时间后重试
			fmt.Println("Error connecting to WebSocket:", err)
			// time.Sleep(1 * time.Second) // 5秒重试间隔
			continue
		}
		// 连接成功，退出循环
		break
	}
	return conn, nil
}
