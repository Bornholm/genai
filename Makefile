GORELEASER_ARGS ?= --snapshot --clean

SHELL := /bin/bash

build:
	CGO_ENABLED=0 go build -o bin/genai ./cmd/genai

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