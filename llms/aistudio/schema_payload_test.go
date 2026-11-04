package aistudio_test

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	llmberjack "github.com/checkmarble/llmberjack"
	"github.com/checkmarble/llmberjack/llms/aistudio"
	"github.com/invopop/jsonschema"
	"github.com/stretchr/testify/require"
)

type schemaPayloadTransport func(*http.Request) (*http.Response, error)

func (f schemaPayloadTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestResponseSchemaPayload(t *testing.T) {
	type Output struct {
		Reply string `json:"reply"`
	}
	override := *jsonschema.Reflect(struct {
		Findings []string `json:"findings"`
	}{})
	for _, tc := range []struct {
		name            string
		typed, override bool
		wantSchema      string
	}{
		{name: "untyped override", override: true, wantSchema: `{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false,"properties":{"findings":{"type":"array","items":{"type":"string"}}},"required":["findings"]}`},
		{name: "override wins over generated schema", typed: true, override: true, wantSchema: `{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false,"properties":{"findings":{"type":"array","items":{"type":"string"}}},"required":["findings"]}`},
		{name: "no schema"},
		{name: "generated schema without override", typed: true, wantSchema: `{"$id":"https://github.com/checkmarble/llmberjack/llms/aistudio_test/output","$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false,"properties":{"reply":{"type":"string"}},"required":["reply"]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			const rawText = "malformed JSON retained for diagnostics"
			client := &http.Client{Transport: schemaPayloadTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				require.Equal(t, http.MethodPost, r.Method)
				require.Contains(t, r.URL.Path, "/models/test-model:generateContent")
				var payload struct {
					GenerationConfig map[string]json.RawMessage `json:"generationConfig"`
				}
				require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
				if tc.wantSchema == "" {
					require.NotContains(t, payload.GenerationConfig, "responseMimeType")
					require.NotContains(t, payload.GenerationConfig, "responseJsonSchema")
				} else {
					require.JSONEq(t, `"application/json"`, string(payload.GenerationConfig["responseMimeType"]))
					require.JSONEq(t, tc.wantSchema, string(payload.GenerationConfig["responseJsonSchema"]))
				}
				return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(`{"candidates":[{"content":{"role":"model","parts":[{"text":"` + rawText + `"}]},"finishReason":"STOP"}]}`)), Request: r}, nil
			})}
			provider, err := aistudio.New(aistudio.WithApiKey("test-key"))
			require.NoError(t, err)
			llm, err := llmberjack.New(llmberjack.WithDefaultProvider(provider), llmberjack.WithHttpClient(client))
			require.NoError(t, err)
			if tc.typed {
				req := llmberjack.NewRequest[Output]().WithModel("test-model").WithText(llmberjack.RoleUser, "test prompt")
				if tc.override {
					req = req.OverrideResponseSchema(override)
				}
				response, err := req.Do(t.Context(), llm)
				require.NoError(t, err)
				candidate, err := response.Candidate(0)
				require.NoError(t, err)
				require.Equal(t, rawText, candidate.Text)
			} else {
				req := llmberjack.NewUntypedRequest().WithModel("test-model").WithText(llmberjack.RoleUser, "test prompt")
				if tc.override {
					req = req.OverrideResponseSchema(override)
				}
				response, err := req.Do(t.Context(), llm)
				require.NoError(t, err)
				text, err := response.Get(0)
				require.NoError(t, err)
				require.Equal(t, rawText, text)
			}
			require.Equal(t, 1, calls)
		})
	}
}
