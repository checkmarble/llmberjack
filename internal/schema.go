package internal

import "github.com/invopop/jsonschema"

type Schema struct {
	Name        string
	Description string
	Schema      jsonschema.Schema
}

func GenerateSchema[S any]() jsonschema.Schema {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            false,
	}

	jsonSchema := reflector.Reflect(new(S))

	return *jsonSchema
}
