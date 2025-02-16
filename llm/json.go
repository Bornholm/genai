package llm

import (
	"encoding/json"
	"log"
	"regexp"

	jsonrepair "github.com/RealAlexandreAI/json-repair"
	"github.com/pkg/errors"
)

var jsonBlockRegExp = regexp.MustCompile(`(?mis)\{.*\}`)

func ParseJSON[T any](message Message) ([]T, error) {
	var items []T

	jsonBlocks := jsonBlockRegExp.FindAllString(message.Content(), -1)

	for _, b := range jsonBlocks {
		var t T

		repaired, err := jsonrepair.RepairJSON(b)
		if err != nil {
			log.Printf("[ERROR] %+v", errors.Wrapf(err, "could not repair json: %s", b))
			continue
		}

		if err := json.Unmarshal([]byte(repaired), &t); err != nil {
			log.Printf("[ERROR] %+v", errors.Wrapf(err, "invalid json: %s", b))
			continue
		}

		items = append(items, t)
	}

	return items, nil
}
