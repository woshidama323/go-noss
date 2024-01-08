package cmd

import (
	"github.com/nbd-wtf/go-nostr"
	"github.com/urfave/cli/v2"

	"fmt"

	"github.com/nbd-wtf/go-nostr/nip19"
)

var White []string

var KeysCmd = &cli.Command{
	Name:  "keys",
	Usage: "Manage keys",
	Subcommands: []*cli.Command{
		CreateKeyCmd,
	},
}

var CreateKeyCmd = &cli.Command{
	Name:      "create",
	Usage:     "Create a key",
	ArgsUsage: "[name]",
	Flags: []cli.Flag{
		&cli.StringFlag{
			Name:  "type",
			Usage: "specify key type to create",
			Value: "bls",
		},
	},

	Action: func(cctx *cli.Context) error {
		sk := nostr.GeneratePrivateKey()
		pk, _ := nostr.GetPublicKey(sk)
		nsec, _ := nip19.EncodePrivateKey(sk)
		npub, _ := nip19.EncodePublicKey(pk)

		fmt.Println("sk:", sk)
		fmt.Println("pk:", pk)
		fmt.Println(nsec)
		fmt.Println(npub)
		return nil
	},
}
