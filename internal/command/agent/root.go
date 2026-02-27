package agent

import "github.com/urfave/cli/v2"

func Root() *cli.Command {
	return &cli.Command{
		Name:  "agent",
		Usage: "Agent related commands",
		Flags: []cli.Flag{},
		Subcommands: []*cli.Command{
			Do(),
			A2A(),
		},
	}
}
