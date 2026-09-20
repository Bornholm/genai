module github.com/bornholm/genai/plugin/sdk

go 1.25.5

// The require below points at a commit of the root module rather than a
// tag: the sdk needs llm/provider/plugin/codec, which no tag carries yet.
// Bump it to the tag right after the root module is tagged, then tag
// plugin/sdk itself.
require (
	github.com/bornholm/genai v0.42.1-0.20260920115138-8839905c2655
	github.com/caarlos0/env/v11 v11.3.1
	github.com/hashicorp/go-hclog v1.6.3
	github.com/hashicorp/go-plugin v1.8.0
	github.com/pkg/errors v0.9.1
	google.golang.org/grpc v1.79.2
)

require (
	github.com/RealAlexandreAI/json-repair v0.0.14 // indirect
	github.com/fatih/color v1.18.0 // indirect
	github.com/golang/protobuf v1.5.4 // indirect
	github.com/hashicorp/yamux v0.1.2 // indirect
	github.com/mattn/go-colorable v0.1.14 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/oklog/run v1.1.0 // indirect
	golang.org/x/net v0.52.0 // indirect
	golang.org/x/sys v0.42.0 // indirect
	golang.org/x/text v0.35.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260311181403-84a4fc48630c // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)
