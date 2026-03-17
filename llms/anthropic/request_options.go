package anthropic

// RequestOptions contains Anthropic-specific request configuration.
type RequestOptions struct {
	// BudgetTokens is the maximum number of tokens to spend on extended thinking.
	// Only applicable with models that support extended thinking.
	BudgetTokens *int
}

func (RequestOptions) ProviderRequestOptions() {}
