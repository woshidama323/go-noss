package cmd

import (
	"github.com/urfave/cli/v2"
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
		return nil
	},
}
