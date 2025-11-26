package adk

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"google.golang.org/genai"
)

// JSONSchemaConverter implements SchemaConverter using JSON marshal/unmarshal.
// This approach is more maintainable (automatically adapts to genai.Schema changes)
// but approximately 5.6x slower than manual conversion. Recommended for development/debugging.
type JSONSchemaConverter struct{}

// NewJSONSchemaConverter creates a new JSON schema converter (slower but maintainable).
func NewJSONSchemaConverter() SchemaConverter {
	return &JSONSchemaConverter{}
}

// Convert transforms a genai.Schema into a map suitable for Fantasy providers using JSON round-trip.
// Still requires normalization for Bug #5 fix (MCP strict JSON schema compatibility).
func (c *JSONSchemaConverter) Convert(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}

	slog.Default().Debug("JSON schema conversion", "schema_type", schema.Type)

	// Marshal schema to JSON bytes
	jsonBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	// Unmarshal JSON bytes to map
	var result map[string]any
	if err := json.Unmarshal(jsonBytes, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal schema: %w", err)
	}

	// Lowercase type field to match manual conversion behavior (per JSON Schema spec)
	lowercaseTypeField(result)

	// CRITICAL: Normalize arrays for Bug #5 fix (MCP strict JSON schema compatibility)
	normalizeSchemaArrays(result)

	return result, nil
}

// lowercaseTypeField recursively lowercases all "type" fields in the schema map.
// This ensures consistency with the manual conversion and JSON Schema spec.
func lowercaseTypeField(schema map[string]any) {
	if typeVal, ok := schema["type"]; ok {
		if typeStr, ok := typeVal.(string); ok {
			schema["type"] = strings.ToLower(typeStr)
		}
	}

	// Recursively lowercase in nested properties
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, prop := range props {
			if propSchema, ok := prop.(map[string]any); ok {
				lowercaseTypeField(propSchema)
			}
		}
	}

	// Lowercase in array items schema
	if items, ok := schema["items"].(map[string]any); ok {
		lowercaseTypeField(items)
	}

	// Recursively lowercase in anyOf schemas
	if anyOf, ok := schema["anyOf"].([]interface{}); ok {
		for _, item := range anyOf {
			if subSchema, ok := item.(map[string]any); ok {
				lowercaseTypeField(subSchema)
			}
		}
	}
}
