package proxy

import (
	"strings"

	"github.com/bornholm/genai/internal/command/common"
	"github.com/bornholm/genai/proxy"
	"github.com/bornholm/genai/proxy/hooks/filter"
	"github.com/bornholm/genai/proxy/hooks/logging"
	"github.com/bornholm/genai/proxy/hooks/router"
	"github.com/bornholm/genai/proxy/hooks/usage"
	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"

	_ "github.com/bornholm/genai/llm/provider/all"
)

func Root() *cli.Command {
	return &cli.Command{
		Name:  "proxy",
		Usage: "OpenAI-compatible proxy server",
		Subcommands: []*cli.Command{
			Serve(),
		},
	}
}

func Serve() *cli.Command {
	return &cli.Command{
		Name:  "serve",
		Usage: "Start the OpenAI-compatible proxy server",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "proxy-addr",
				Usage:   "Address to listen on",
				Value:   ":8080",
				EnvVars: []string{"PROXY_ADDR"},
			},
			&cli.StringFlag{
				Name:    "proxy-auth-mode",
				Usage:   "Authentication mode: bearer or header",
				Value:   "bearer",
				EnvVars: []string{"PROXY_AUTH_MODE"},
			},
			&cli.StringFlag{
				Name:    "proxy-auth-header",
				Usage:   "Header name for identity extraction when --proxy-auth-mode=header",
				Value:   "X-User-ID",
				EnvVars: []string{"PROXY_AUTH_HEADER"},
			},
			&cli.StringSliceFlag{
				Name:    "proxy-route",
				Usage:   "Route mapping: model=provider:actual_model (repeatable)",
				EnvVars: []string{"PROXY_ROUTES"},
			},
			&cli.StringSliceFlag{
				Name:    "proxy-route-weight",
				Usage:   "Weighted route: model=provider:actual_model:weight (repeatable)",
				EnvVars: []string{"PROXY_ROUTE_WEIGHTS"},
			},
			&cli.StringSliceFlag{
				Name:    "proxy-filter-keywords",
				Usage:   "Comma-separated keywords to block in requests",
				EnvVars: []string{"PROXY_FILTER_KEYWORDS"},
			},
			&cli.IntFlag{
				Name:    "proxy-quota-daily-tokens",
				Usage:   "Maximum total tokens per user per day (0 = disabled)",
				EnvVars: []string{"PROXY_QUOTA_DAILY_TOKENS"},
			},
			&cli.IntFlag{
				Name:    "proxy-quota-daily-requests",
				Usage:   "Maximum requests per user per day (0 = disabled)",
				EnvVars: []string{"PROXY_QUOTA_DAILY_REQUESTS"},
			},
			// LLM client flags for the default backend
			&cli.StringFlag{
				Name:      "env-file",
				Usage:     "Environment file path for default LLM backend",
				EnvVars:   []string{"GENAI_LLM_ENV_FILE"},
				Value:     ".env",
				TakesFile: true,
			},
			&cli.StringFlag{
				Name:    "env-prefix",
				Usage:   "Environment variable prefix for default LLM backend",
				EnvVars: []string{"GENAI_LLM_ENV_PREFIX"},
				Value:   "GENAI_",
			},
		},
		Action: func(cliCtx *cli.Context) error {
			ctx := cliCtx.Context

			var opts []proxy.OptionFunc

			// Address
			opts = append(opts, proxy.WithAddr(cliCtx.String("proxy-addr")))

			// Auth extractor
			authMode := cliCtx.String("proxy-auth-mode")
			switch authMode {
			case "header":
				opts = append(opts, proxy.WithAuthExtractor(
					proxy.HeaderExtractor(cliCtx.String("proxy-auth-header")),
				))
			default:
				opts = append(opts, proxy.WithAuthExtractor(proxy.BearerTokenExtractor()))
			}

			// Logging hook — pre-request priority 0 (first), post-response priority 200 (last)
			opts = append(opts, proxy.WithHook(logging.New(nil, 0)))

			// Usage store (always created; needed by tracker and quota)
			store := usage.NewInMemoryUsageStore()

			// Usage tracker (post-response, priority 100)
			opts = append(opts, proxy.WithHook(usage.NewUsageTracker(store, 100)))

			// Quota enforcer (pre-request, priority 10)
			dailyTokens := cliCtx.Int("proxy-quota-daily-tokens")
			dailyRequests := cliCtx.Int("proxy-quota-daily-requests")
			if dailyTokens > 0 || dailyRequests > 0 {
				quotaCfg := usage.QuotaConfig{
					MaxTokensPerDay:   dailyTokens,
					MaxRequestsPerDay: dailyRequests,
				}
				opts = append(opts, proxy.WithHook(
					usage.NewQuotaEnforcer(store, map[string]usage.QuotaConfig{"*": quotaCfg}, 10),
				))
			}

			// Content filter (pre-request, priority 20)
			keywordArgs := cliCtx.StringSlice("proxy-filter-keywords")
			if len(keywordArgs) > 0 {
				var keywords []string
				for _, arg := range keywordArgs {
					for _, kw := range strings.Split(arg, ",") {
						kw = strings.TrimSpace(kw)
						if kw != "" {
							keywords = append(keywords, kw)
						}
					}
				}
				if len(keywords) > 0 {
					opts = append(opts, proxy.WithHook(
						filter.NewContentFilter(20, filter.NewKeywordRule(keywords...)),
					))
				}
			}

			// Static router (priority 50)
			staticRoutes := parseStaticRoutes(cliCtx.StringSlice("proxy-route"))
			if len(staticRoutes) > 0 {
				// Build per-model clients from env configuration
				routeMap := make(map[string]router.Route, len(staticRoutes))
				for proxyModel, target := range staticRoutes {
					envPrefix := cliCtx.String("env-prefix")
					envFile := cliCtx.String("env-file")
					client, err := common.NewResilientClient(ctx, envPrefix, envFile, nil)
					if err != nil {
						return errors.Wrapf(err, "could not create client for route %s→%s", proxyModel, target)
					}
					routeMap[proxyModel] = router.Route{Client: client, Model: target}
				}
				opts = append(opts, proxy.WithHook(router.NewStaticRouter(routeMap, 50)))
			}

			// Default fallback client
			envPrefix := cliCtx.String("env-prefix")
			envFile := cliCtx.String("env-file")
			defaultClient, err := common.NewResilientClient(ctx, envPrefix, envFile, nil)
			if err == nil {
				opts = append(opts, proxy.WithDefaultClient(defaultClient))
			}

			server := proxy.NewServer(opts...)
			return server.ListenAndServe()
		},
	}
}

// parseStaticRoutes parses "--proxy-route model=actual_model" flags.
func parseStaticRoutes(args []string) map[string]string {
	routes := make(map[string]string)
	for _, arg := range args {
		parts := strings.SplitN(arg, "=", 2)
		if len(parts) != 2 {
			continue
		}
		routes[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
	}
	return routes
}
