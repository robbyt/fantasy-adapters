package adk

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

// TestJSONSchemaConverter_Bug5Compatibility verifies that JSON conversion maintains Bug #5 fix.
// Bug #5: Array fields (required, enum, propertyOrdering) must be []string for MCP compatibility.
func TestJSONSchemaConverter_Bug5Compatibility(t *testing.T) {
	schema := &genai.Schema{
		Type:        genai.TypeObject,
		Description: "Test schema",
		Properties: map[string]*genai.Schema{
			"name": {
				Type:        genai.TypeString,
				Description: "Name field",
			},
			"status": {
				Type:        genai.TypeString,
				Description: "Status field",
				Enum:        []string{"active", "inactive", "pending"},
			},
		},
		Required:         []string{"name", "status"},
		PropertyOrdering: []string{"name", "status"},
	}

	converter := NewJSONSchemaConverter()
	result, err := converter.Convert(schema)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify required field is []string (not []interface{})
	required, ok := result["required"]
	require.True(t, ok, "required field must exist")
	requiredSlice, ok := required.([]string)
	require.True(t, ok, "required must be []string, got %T (Bug #5 fix)", required)
	assert.Equal(t, []string{"name", "status"}, requiredSlice)

	// Verify enum in nested property is []string
	props, ok := result["properties"].(map[string]any)
	require.True(t, ok, "properties must be map")
	statusProp, ok := props["status"].(map[string]any)
	require.True(t, ok, "status property must be map")
	enum, ok := statusProp["enum"]
	require.True(t, ok, "enum field must exist")
	enumSlice, ok := enum.([]string)
	require.True(t, ok, "enum must be []string, got %T (Bug #5 fix)", enum)
	assert.Equal(t, []string{"active", "inactive", "pending"}, enumSlice)

	// Verify propertyOrdering is []string
	propOrder, ok := result["propertyOrdering"]
	require.True(t, ok, "propertyOrdering field must exist")
	propOrderSlice, ok := propOrder.([]string)
	require.True(t, ok, "propertyOrdering must be []string, got %T (Bug #5 fix)", propOrder)
	assert.Equal(t, []string{"name", "status"}, propOrderSlice)
}

func BenchmarkJSONSchemaConverter(b *testing.B) {
	schema := getBenchmarkSchema()
	converter := NewJSONSchemaConverter()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := converter.Convert(schema)
		if err != nil {
			b.Fatal(err)
		}
	}
}
