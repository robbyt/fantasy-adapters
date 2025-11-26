package adk

import "google.golang.org/genai"

// SchemaConverter converts genai.Schema to map[string]any for Fantasy provider compatibility.
// Different implementations trade off performance vs maintainability:
//   - ManualSchemaConverter: Fast field-by-field copy (default, ~5.6x faster)
//   - JSONSchemaConverter: JSON marshal/unmarshal (maintainable, auto-adapts to schema changes)
type SchemaConverter interface {
	// Convert transforms a genai.Schema into a map suitable for Fantasy providers.
	// Both implementations apply Bug #5 normalization for MCP compatibility.
	Convert(schema *genai.Schema) (map[string]any, error)
}
