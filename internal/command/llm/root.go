package llm

import "github.com/urfave/cli/v2"

func Root() *cli.Command {
	return &cli.Command{
		Name:  "llm",
		Usage: "LLMs related commands",
		Flags: []cli.Flag{},
		Subcommands: []*cli.Command{
			Generate(),
		},
	}
}
