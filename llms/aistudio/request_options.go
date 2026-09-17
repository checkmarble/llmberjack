package aistudio

type ThinkingConfig struct {
	IncludeThoughts bool
	ThinkingLevel   string
}

type RequestOptions struct {
	GoogleSearch *bool
	TopK         *float64
	Thinking     *ThinkingConfig
}

func (RequestOptions) ProviderRequestOptions() {}
