package openai

import (
	"encoding/json"

	"github.com/openai/openai-go/packages/resp"
)

// costCurrency is the currency OpenAI compatible gateways report costs in.
// OpenRouter and LiteLLM both bill in US dollars; no gateway seen so far
// reports the currency alongside the amount.
const costCurrency = "USD"

// extraCost reads the non standard "cost" field gateways add to the usage
// object. A missing, malformed or negative value reports no cost at all
// rather than a wrong one: downstream, an absent cost falls back to an
// estimate, while a zero would be taken for a free call.
func extraCost(fields map[string]resp.Field) (float64, bool) {
	field, found := fields["cost"]
	if !found {
		return 0, false
	}

	var cost float64
	if err := json.Unmarshal([]byte(field.Raw()), &cost); err != nil {
		return 0, false
	}
	if cost < 0 {
		return 0, false
	}

	return cost, true
}
