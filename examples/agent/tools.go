package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

func getFrenchAddress(ctx context.Context, params map[string]any) (string, error) {
	postalAddress, err := llm.ToolParam[string](params, "postal_address")
	if err != nil {
		return "", errors.WithStack(err)
	}

	slog.InfoContext(ctx, "retrieving french address location", slog.String("postalAddress", postalAddress))

	url, err := url.Parse("https://data.geopf.fr/geocodage/search")
	if err != nil {
		return "", errors.WithStack(err)
	}

	query := url.Query()
	query.Set("q", postalAddress)
	query.Set("limit", "1")
	query.Set("index", "address")
	query.Set("returntruegeometry", "false")
	url.RawQuery = query.Encode()

	res, err := http.Get(url.String())
	if err != nil {
		return "", errors.WithStack(err)
	}

	defer res.Body.Close()

	decoder := json.NewDecoder(res.Body)

	var payload struct {
		Features []struct {
			Geometry struct {
				Coordinates []float64 `json:"coordinates"`
			} `json:"geometry"`
		} `json:"features"`
	}
	if err := decoder.Decode(&payload); err != nil {
		return "", errors.WithStack(err)
	}

	result := fmt.Sprintf(`Coordinates for "%s":
		Longitude: %v
		Latitude: %v
	`, postalAddress, payload.Features[0].Geometry.Coordinates[0], payload.Features[0].Geometry.Coordinates[1])

	return result, nil
}

func getWeather(ctx context.Context, params map[string]any) (string, error) {
	longitude, err := llm.ToolParam[float64](params, "longitude")
	if err != nil {
		return "", errors.WithStack(err)
	}

	latitude, err := llm.ToolParam[float64](params, "latitude")
	if err != nil {
		return "", errors.WithStack(err)
	}

	slog.InfoContext(ctx, "retrieving weather for location", slog.Float64("longitude", longitude), slog.Float64("latitude", latitude))

	url, err := url.Parse("https://api.open-meteo.com/v1/forecast")

	query := url.Query()
	query.Set("current", "apparent_temperature,precipitation,cloud_cover,wind_speed_10m,temperature_2m,wind_direction_10m,rain,showers,relative_humidity_2m")
	query.Set("longitude", fmt.Sprintf("%v", longitude))
	query.Set("latitude", fmt.Sprintf("%v", latitude))
	url.RawQuery = query.Encode()

	res, err := http.Get(url.String())
	if err != nil {
		return "", errors.WithStack(err)
	}

	defer res.Body.Close()

	decoder := json.NewDecoder(res.Body)

	var payload struct {
		Current struct {
			ApparentTemperature float64 `json:"apparent_temperature"`
			WindSpeed10M        float64 `json:"wind_speed_10m"`
			CloudCover          float64 `json:"cloud_cover"`
			Rain                float64 `json:"rain"`
			Showers             float64 `json:"showers"`
			RelativeHumidity    float64 `json:"relative_humidity_2m"`
		} `json:"current"`
	}
	if err := decoder.Decode(&payload); err != nil {
		return "", errors.WithStack(err)
	}

	result := fmt.Sprintf(`
	Current weather at %v, %v:

	- Apparent Temperature: %v°C
	- Wind speed: %vkm/h
	- Cloud cover: %v%%
	- Rain: %v%%
	- Showers: %v%%
	- Humidity: %v%%
	
	`,
		latitude, longitude,
		payload.Current.ApparentTemperature,
		payload.Current.WindSpeed10M,
		payload.Current.CloudCover,
		payload.Current.Rain,
		payload.Current.Showers,
		payload.Current.RelativeHumidity,
	)

	return result, nil
}
