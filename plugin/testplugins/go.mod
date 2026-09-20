// Module holding the plugin binaries the tests build. They live outside
// plugin/sdk so that the SDK, which external plugin authors depend on, does
// not carry the dependencies of the providers these fixtures wrap.
module github.com/bornholm/genai/plugin/testplugins

go 1.25.5

require (
	github.com/bornholm/genai v0.42.0
	github.com/bornholm/genai/plugin/sdk v0.0.0
	github.com/pkg/errors v0.9.1
)

require (
	github.com/RealAlexandreAI/json-repair v0.0.14 // indirect
	github.com/caarlos0/env/v11 v11.3.1 // indirect
	github.com/fatih/color v1.18.0 // indirect
	github.com/golang/protobuf v1.5.4 // indirect
	github.com/hashicorp/go-hclog v1.6.3 // indirect
	github.com/hashicorp/go-plugin v1.8.0 // indirect
	github.com/hashicorp/yamux v0.1.2 // indirect
	github.com/mattn/go-colorable v0.1.14 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/oklog/run v1.1.0 // indirect
	github.com/openai/openai-go v0.1.0-beta.10 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	golang.org/x/net v0.52.0 // indirect
	golang.org/x/sys v0.42.0 // indirect
	golang.org/x/text v0.35.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260311181403-84a4fc48630c // indirect
	google.golang.org/grpc v1.79.2 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

// Fixtures of this repository: always built against the checked-out sources.
replace (
	github.com/bornholm/genai => ../..
	github.com/bornholm/genai/plugin/sdk => ../sdk
)
