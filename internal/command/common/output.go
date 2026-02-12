package common

import (
	"fmt"
	"os"

	"github.com/pkg/errors"
	"github.com/urfave/cli/v2"
)

func WriteToOutput(ctx cli.Context, param string, content string) error {
	// Output the response
	if outputPath := ctx.String(param); outputPath != "" {
		err := os.WriteFile(outputPath, []byte(content), 0644)
		if err != nil {
			return errors.Wrap(err, "failed to write output file")
		}
	} else {
		fmt.Print(content)
	}

	return nil
}
