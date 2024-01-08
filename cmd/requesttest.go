package cmd

import (
	"bytes"
	"io/ioutil"
	"os"

	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"

	"github.com/joho/godotenv"
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli/v2"
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

var TestCmd = &cli.Command{
	Name:  "test",
	Usage: "test request",
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
		type Event struct {
			Sig       string     `json:"sig"`
			Id        string     `json:"id"`
			Kind      int        `json:"kind"`
			CreatedAt int        `json:"created_at"`
			Tags      [][]string `json:"tags"`
			Content   string     `json:"content"`
			PubKey    string     `json:"pubkey"`
		}

		ev := Event{
			Sig:       "a6e539ff34fc1fe1bd7bcfbc72d4cfa1f32e0f4fb4c37c9d3a89fdafb306f3a382617a9c308078212ce8ce4555c93789ed3390c818e47f839550100150558eb5",
			Id:        "0000004023b58ef98f88dc696269e429cbf3c9062ae1267f4f11a016e2640dbf",
			Kind:      1,
			CreatedAt: 1704575164,
			Tags:      [][]string{{"p", "9be107b0d7218c67b4954ee3e6bd9e4dba06ef937a93f684e42f730a0c3d053c"}, {"e", "51ed7939a984edee863bfbb2e66fdc80436b000a8ddca442d83e6a2bf1636a95", "wss://relay.noscription.org/", "root"}, {"e", "00000571cb4fabce45e915cce67ec468051d550847f166137fd8aea8615bcd8c", "wss://relay.noscription.org/", "reply"}, {"seq_witness", "167798002", "0xec4bb82180016b3e050cc9c5deceb672e360bfb251e7543c6323348d1505d99e"}, {"nonce", "ctlqrejf99", "21"}},
			Content:   "{\"p\":\"nrc-20\",\"op\":\"mint\",\"tick\":\"noss\",\"amt\":\"10\"}",
			PubKey:    "66313c9225464c64e8cbab0d48b16a9b5a25f206e00bb79371b684743aa9d288",
		}

		// 包装event到一个外部结构体
		wrappedEvent := map[string]Event{
			"event": ev,
		}

		// 序列化为JSON
		jsonData, err := json.Marshal(wrappedEvent)
		if err != nil {
			log.Fatal("Error marshaling JSON: ", err)
		}

		// 执行HTTP请求
		url := "https://api-worker.noscription.org/inscribe/postEvent"
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
		if err != nil {
			log.Fatal("Error creating request: ", err)
		}

		// 设置HTTP Header

		req.Header.Set("authority", "api-worker.noscription.org")
		req.Header.Set("accept", "application/json, text/plain, */*")
		req.Header.Set("accept-language", "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7")
		req.Header.Set("content-type", "application/json")
		req.Header.Set("origin", "https://noscription.org")
		req.Header.Set("referer", "https://noscription.org/")
		req.Header.Set("sec-ch-ua", `"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"`)
		req.Header.Set("sec-ch-ua-mobile", "?0")
		req.Header.Set("sec-ch-ua-platform", `"macOS"`)
		req.Header.Set("sec-fetch-dest", "empty")
		req.Header.Set("sec-fetch-mode", "cors")
		req.Header.Set("sec-fetch-site", "same-site")
		req.Header.Set("user-agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
		req.Header.Set("x-gorgon", "ac880d214ba2fb1d4ee63a47f2e26f52917f04b25955c218bf78f6b4f94bc85b")
		req.Header.Set("cache-control", "no-cache")
		req.Header.Set("pragma", "no-cache")
		for name, values := range req.Header {
			// Header中的值是一个字符串切片
			for _, value := range values {
				logrus.Infof("%v: %v\n", name, value)

			}
		}

		// 发送请求
		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			log.Fatalf("Error sending request: %v", err)
		}
		defer resp.Body.Close()

		fmt.Println("Response Status:", resp.Status)
		// 读取响应体
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			log.Printf("Error reading response body: %v", err)
			return err
		}

		// 打印响应体
		fmt.Println("Response Body:", string(body))
		return nil
	},
}
