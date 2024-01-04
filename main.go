package main

import (
	"log"
	"os"

	"github.com/urfave/cli/v2"

	"nostr/cmd"
)

func main() {
	app := cli.NewApp()
	app.Name = "nostrbot"
	app.Usage = "nostrbot"
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
