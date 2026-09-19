GORELEASER_ARGS ?= --snapshot --clean

SHELL := /bin/bash

build:
	CGO_ENABLED=0 go build -o bin/genai ./cmd/genai

build-wasm:
	CGO_ENABLED=0 GOOS=js GOARCH=wasm go build -o bin/genai.wasm ./wasm

watch: tools/modd/bin/modd
	tools/modd/bin/modd

run-with-env: .env
	( set -o allexport && source .env && set +o allexport && $(value CMD))

.env:
	cp .env.dist .env

release:
	goreleaser $(GORELEASER_ARGS)

test:
	$(MAKE) run-with-env CMD="go test -v ./..."

tools/modd/bin/modd:
	mkdir -p tools/modd/bin
	GOBIN=$(PWD)/tools/modd/bin go install github.com/cortesi/modd/cmd/modd@latest
PROTOC_GEN_GO_VERSION ?= v1.36.11
PROTOC_GEN_GO_GRPC_VERSION ?= v1.6.2

tools/proto/bin/protoc-gen-go:
	mkdir -p tools/proto/bin
	GOBIN=$(PWD)/tools/proto/bin go install google.golang.org/protobuf/cmd/protoc-gen-go@$(PROTOC_GEN_GO_VERSION)

tools/proto/bin/protoc-gen-go-grpc:
	mkdir -p tools/proto/bin
	GOBIN=$(PWD)/tools/proto/bin go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@$(PROTOC_GEN_GO_GRPC_VERSION)

proto: tools/proto/bin/protoc-gen-go tools/proto/bin/protoc-gen-go-grpc
	PATH=$(PWD)/tools/proto/bin:$$PATH protoc \
		-I llm/provider/plugin/proto \
		--go_out=llm/provider/plugin/proto --go_opt=paths=source_relative \
		--go-grpc_out=llm/provider/plugin/proto --go-grpc_opt=paths=source_relative \
		llm/provider/plugin/proto/genai/plugin/v1/provider.proto
