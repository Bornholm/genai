GORELEASER_ARGS ?= --snapshot --clean

release:
	goreleaser $(GORELEASER_ARGS)
