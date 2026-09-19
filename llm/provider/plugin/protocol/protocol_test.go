package protocol

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"testing"
)

// TestProtoChecksum ties a change of provider.proto to a review of
// ProtocolVersion: the constant has to be updated by hand, next to the
// version, whenever the file changes.
func TestProtoChecksum(t *testing.T) {
	data, err := os.ReadFile("../proto/genai/plugin/v1/provider.proto")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != ProtoChecksum {
		t.Errorf("provider.proto changed: update ProtoChecksum to %q and bump ProtocolVersion if the change is incompatible", got)
	}
}
