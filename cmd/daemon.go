package cmd

import (
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"github.com/ethereum/go-ethereum/core/types"
	"os"

	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/gorilla/websocket"
	"github.com/nbd-wtf/go-nostr"
	"github.com/nbd-wtf/go-nostr/nip13"

	"github.com/joho/godotenv"
	"github.com/urfave/cli/v2"
	"github.com/sirupsen/logrus"
)

var sk string
var pk string
var numberOfWorkers int
var nonceFound int32 = 0
var blockNumber uint64
var hash string
var messageId string
var currentWorkers int32
var arbRpcUrls []string
var (
	ErrDifficultyTooLow = errors.New("nip13: insufficient difficulty")
	ErrGenerateTimeout  = errors.New("nip13: generating proof of work took too long")
)

func init() {

	logrus.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true, // Enable timestamp
		ForceColors:   true, // Force colored output even when stdout is not a terminal

	})

	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	sk = os.Getenv("sk")
	pk = os.Getenv("pk")
	numberOfWorkers, _ = strconv.Atoi(os.Getenv("numberOfWorkers"))
	arbRpcUrls = []string{
		"https://arbitrum.llamarpc.com",
		"https://rpc.arb1.arbitrum.gateway.fm",
		"https://api.zan.top/node/v1/arb/one/public",
		"https://arbitrum.meowrpc.com",
		"https://arb-pokt.nodies.app",
		"https://arbitrum.blockpi.network/v1/rpc/public",
		"https://arbitrum-one.publicnode.com",
		"https://arbitrum-one.public.blastapi.io",
		"https://arbitrum.drpc.org",
		"https://arb1.arbitrum.io/rpc",
		"https://endpoints.omniatech.io/v1/arbitrum/one/public",
		"https://1rpc.io/arb",
		"https://rpc.ankr.com/arbitrum",
		"https://arbitrum.api.onfinality.io/public",
		"wss://arbitrum-one.publicnode.com",
		"https://arb-mainnet-public.unifra.io",
		"https://arb-mainnet.g.alchemy.com/v2/demo",
		"https://arbitrum.getblock.io/api_key/mainnet",
	}
}

var DaemonCmd = &cli.Command{
	Name:  "daemon",
	Usage: "run backend",
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
	},
	Action: func(cctx *cli.Context) error {
		// wssAddr := "wss://report-worker-2.noscription.org"
		// relayUrl := "wss://relay.noscription.org/"

		//
		ctx := cctx.Context

		c, err := connectToWSS(cctx.String("noscriptionWss"))
		if err != nil {
			panic(err)
		}
		defer c.Close()

		// 初始化所有的链接
		clients := make([]*ethclient.Client, len(arbRpcUrls))
		checkClient := func() {
			for k, c := range clients {
				if c != nil {
					continue
				}
				rurl := arbRpcUrls[k]
				client, err := ethclient.Dial(rurl)
				if err != nil {
					log.Printf("无法连接到Arbitrum节点(%s): %v", rurl, err)
					continue
				}

				clients[k] = client
			}
		}

		// 先初始化一下所有的rpc
		go func() {
			for {
				checkClient()
				time.Sleep(10 * time.Second)
			}
		}()

		count := 0
		getClinet := func() *ethclient.Client {
			countOld := count
			for {
				count++
				count = count % len(arbRpcUrls)
				if count == countOld {
					return nil
				}

				client := clients[count]
				if client == nil {
					continue
				}
				//fmt.Println("获得client: ", count)

				return client
			}
		}

		ch := make(chan *types.Header, 5)
		defer close(ch)

		// get block
		go func() {
			for {
				go func() {
					client := getClinet()
					if client == nil {
						return
					}
					header, err := client.HeaderByNumber(context.Background(), nil)
					if err != nil {
						client = nil
						fmt.Println("无法获取最新区块号")
						return
					}
					fmt.Println("获取最新区块号")
					ch <- header

				}()
				time.Sleep(500 * time.Millisecond)
			}
		}()

		go func() {
			for header := range ch {
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
					fmt.Println("exit now")
					return
				default:
					if atomic.LoadInt32(&currentWorkers) < int32(numberOfWorkers) {
						atomic.AddInt32(&currentWorkers, 1) // 增加工作者数量
						go func(bn uint64, mid string) {
							defer atomic.AddInt32(&currentWorkers, -1) // 完成后减少工作者数量
							mine(ctx, mid, getClinet())
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

		logrus.Info("evNew: ", evNew.ID)
		tmpID := evNew.ID
		evNew.Sign(sk)
		logrus.Info("evNew after: ", evNew.ID)
		evNewInstance := EV{
			Sig:       evNew.Sig,
			Id:        tmpID,
			Kind:      evNew.Kind,
			CreatedAt: evNew.CreatedAt,
			Tags:      evNew.Tags,
			Content:   evNew.Content,
			PubKey:    evNew.PubKey,
		}

		fmt.Printf("evNewInstance: %+v\n", evNewInstance)
		
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

		req.Header.Set("authority", "api-worker.noscription.org")
		req.Header.Set("accept", "application/json, text/plain, */*")
		req.Header.Set("accept-language", "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7")
		req.Header.Set("content-type", "application/json")
		req.Header.Set("origin", "https://noscription.org")
		req.Header.Set("referer", "https://noscription.org/")
	
		req.Header.Set("user-agent", "Mozilla/5.0 (Macintosh;c Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
		req.Header.Set("x-gorgon", "ac880d214ba2fb1d4ee63a47f2e26f52917f04b25955c218bf78f6b4f94bc85b")
	  

		for name, values := range req.Header {
			// Header中的值是一个字符串切片
			for _, value := range values {
				logrus.Infof("%v: %v\n", name, value)
				
			}
		}

		logrus.Info(string(wrapperJSON))

		
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
