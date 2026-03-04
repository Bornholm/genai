GORELEASER_ARGS ?= --snapshot --clean

build:
	CGO_ENABLED=0 go build -o bin/genai ./cmd/genai

release:
	goreleaser $(GORELEASER_ARGS)
