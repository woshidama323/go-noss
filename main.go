package main

import (
	"log"
	"os"

	"github.com/urfave/cli/v2"

	"github.com/woshidama323/go-noss/cmd"

	"log"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

func init() {

	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	sk = os.Getenv("sk")
	pk = os.Getenv("pk")
	numberOfWorkers, _ = strconv.Atoi(os.Getenv("numberOfWorkers"))
	arbRpcUrl = os.Getenv("arbRpcUrl")
}

func main() {
	app := cli.NewApp()
	app.Name = "bnbot"
	app.Usage = "bnbot"
	app.Version = "0.0.1"
	app.Commands = []*cli.Command{
		cmd.KeysCmd,
		cmd.DaemonCmd,
	}
	// run the app
	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
