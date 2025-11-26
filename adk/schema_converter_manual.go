package adk

import (
	"fmt"
	"log/slog"
	"strings"

	"google.golang.org/genai"
)

// ManualSchemaConverter implements SchemaConverter using manual field-by-field copy.
// This is the default and fastest approach (~5.6x faster than JSON conversion) but
// requires manual updates when genai.Schema struct changes.
type ManualSchemaConverter struct{}

// NewManualSchemaConverter creates a new manual schema converter (default, recommended for production).
func NewManualSchemaConverter() SchemaConverter {
	return &ManualSchemaConverter{}
}

// Convert transforms a genai.Schema into a map suitable for Fantasy providers using manual field copy.
func (c *ManualSchemaConverter) Convert(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}

	slog.Default().Debug("Manual schema conversion", "schema_type", schema.Type)
	result := make(map[string]any)

	if schema.Type != "" {
		result["type"] = strings.ToLower(string(schema.Type))
	}

	if schema.Description != "" {
		result["description"] = schema.Description
	}

	if schema.Title != "" {
		result["title"] = schema.Title
	}

	if len(schema.Properties) > 0 {
		props := make(map[string]any)
		for key, val := range schema.Properties {
			propMap, err := c.Convert(val)
			if err != nil {
				return nil, fmt.Errorf("failed to convert property %q: %w", key, err)
			}
			props[key] = propMap
		}
		result["properties"] = props
	}

	if schema.Items != nil {
		items, err := c.Convert(schema.Items)
		if err != nil {
			return nil, fmt.Errorf("failed to convert items: %w", err)
		}
		result["items"] = items
	}

	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	if schema.Format != "" {
		result["format"] = schema.Format
	}

	if schema.Pattern != "" {
		result["pattern"] = schema.Pattern
	}

	if schema.Minimum != nil {
		result["minimum"] = *schema.Minimum
	}

	if schema.Maximum != nil {
		result["maximum"] = *schema.Maximum
	}

	if schema.MinLength != nil {
		result["minLength"] = *schema.MinLength
	}

	if schema.MaxLength != nil {
		result["maxLength"] = *schema.MaxLength
	}

	if schema.MinItems != nil {
		result["minItems"] = *schema.MinItems
	}

	if schema.MaxItems != nil {
		result["maxItems"] = *schema.MaxItems
	}

	if schema.MinProperties != nil {
		result["minProperties"] = *schema.MinProperties
	}

	if schema.MaxProperties != nil {
		result["maxProperties"] = *schema.MaxProperties
	}

	if schema.Nullable != nil {
		result["nullable"] = *schema.Nullable
	}

	if schema.Default != nil {
		result["default"] = schema.Default
	}

	if schema.Example != nil {
		result["example"] = schema.Example
	}

	if len(schema.PropertyOrdering) > 0 {
		result["propertyOrdering"] = schema.PropertyOrdering
	}

	if len(schema.AnyOf) > 0 {
		anyOf := make([]any, len(schema.AnyOf))
		for i, s := range schema.AnyOf {
			schemaMap, err := c.Convert(s)
			if err != nil {
				return nil, fmt.Errorf("failed to convert anyOf[%d]: %w", i, err)
			}
			anyOf[i] = schemaMap
		}
		result["anyOf"] = anyOf
	}

	return result, nil
}
