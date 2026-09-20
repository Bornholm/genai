package command

import (
	"fmt"
	"log/slog"
	"os"
	"sort"

	"github.com/bornholm/genai/internal/logx"
	"github.com/bornholm/genai/llm/provider/plugin"
	"github.com/hashicorp/go-hclog"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

func Main(name string, version string, usage string, commands ...*cli.Command) {
	app := &cli.App{
		Name:     name,
		Usage:    usage,
		Commands: commands,
		Version:  version,
		Before: func(ctx *cli.Context) error {
			workdir := ctx.String("workdir")
			// Switch to new working directory if defined
			if workdir != "" {
				if err := os.Chdir(workdir); err != nil {
					return errors.Wrap(err, "could not change working directory")
				}
			}

			if pluginDir := ctx.String("plugin-dir"); pluginDir != "" {
				plugin.SetSearchDir(pluginDir)
			}

			logLevel := ctx.String("log-level")
			slogLevel := slog.LevelWarn
			plugin.SetLogLevel(hclog.Warn)

			switch logLevel {
			case "debug":
				slogLevel = slog.LevelDebug
				plugin.SetLogLevel(hclog.Debug)
			case "info":
				slogLevel = slog.LevelInfo
				plugin.SetLogLevel(hclog.Info)
			case "warn":
				slogLevel = slog.LevelWarn
			case "error":
				slogLevel = slog.LevelError
				plugin.SetLogLevel(hclog.Error)
			}

			logger := slog.New(logx.ContextHandler{
				Handler: slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
					Level: slogLevel,
				}),
			})
			slog.SetDefault(logger)

			return nil
		},
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "workdir",
				Value:   "",
				EnvVars: []string{"GENAI_WORKDIR"},
				Usage:   "The working directory",
			},
			&cli.BoolFlag{
				Name:    "debug",
				EnvVars: []string{"GENAI_DEBUG"},
				Usage:   "Enable debug mode",
			},
			&cli.StringFlag{
				Name:    "log-level",
				EnvVars: []string{"GENAI_LOG_LEVEL"},
				Usage:   "Set logging level",
				Value:   "info",
			},
			&cli.StringFlag{
				Name:    "plugin-dir",
				EnvVars: []string{plugin.SearchDirEnv},
				Usage:   "Directory holding provider plugins (genai-provider-<name>). Setting it enables plugins: an unknown provider name then runs a binary from there, trusted like genai itself. A value in a .env file is ignored: use this flag, the environment or the YAML config",
			},
		},
	}

	app.ExitErrHandler = func(ctx *cli.Context, err error) {
		if err == nil {
			return
		}

		debug := ctx.Bool("debug")

		if !debug {
			slog.ErrorContext(ctx.Context, err.Error())
		} else {
			slog.ErrorContext(ctx.Context, fmt.Sprintf("%+v", err))
		}
	}

	sort.Sort(cli.FlagsByName(app.Flags))
	sort.Sort(cli.CommandsByName(app.Commands))

	err := app.Run(os.Args)
	plugin.CleanupClients()
	if err != nil {
		os.Exit(1)
	}
}
