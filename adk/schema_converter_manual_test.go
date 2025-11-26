package adk

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestManualSchemaConverter_NilSchema(t *testing.T) {
	converter := NewManualSchemaConverter()
	result, err := converter.Convert(nil)
	require.NoError(t, err)
	assert.Nil(t, result)
}

func TestManualSchemaConverter_BasicTypes(t *testing.T) {
	tests := []struct {
		name     string
		schema   *genai.Schema
		expected map[string]any
	}{
		{
			name:     "string type",
			schema:   &genai.Schema{Type: "STRING", Description: "A string value"},
			expected: map[string]any{"type": "string", "description": "A string value"},
		},
		{
			name:     "number type with format",
			schema:   &genai.Schema{Type: "NUMBER", Format: "float"},
			expected: map[string]any{"type": "number", "format": "float"},
		},
		{
			name:     "boolean type",
			schema:   &genai.Schema{Type: "BOOLEAN"},
			expected: map[string]any{"type": "boolean"},
		},
		{
			name:     "integer with min/max",
			schema:   &genai.Schema{Type: "INTEGER", Minimum: genai.Ptr(1.0), Maximum: genai.Ptr(100.0)},
			expected: map[string]any{"type": "integer", "minimum": 1.0, "maximum": 100.0},
		},
	}

	converter := NewManualSchemaConverter()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := converter.Convert(tt.schema)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestManualSchemaConverter_ObjectWithProperties(t *testing.T) {
	schema := objectSchema().
		withDescription("A person object").
		withStringProp("name", "Person's name").
		withIntegerProp("age", "Person's age").
		withRequired("name").
		build()

	converter := NewManualSchemaConverter()
	result, err := converter.Convert(schema)
	require.NoError(t, err)
	assert.Equal(t, "object", result["type"])
	assert.Equal(t, "A person object", result["description"])
	assert.Contains(t, result, "properties")
	assert.Contains(t, result, "required")
	assert.Equal(t, []string{"name"}, result["required"])

	props, ok := result["properties"].(map[string]any)
	require.True(t, ok)
	assert.Contains(t, props, "name")
	assert.Contains(t, props, "age")
}

func TestManualSchemaConverter_ArrayWithItems(t *testing.T) {
	schema := &genai.Schema{
		Type: "ARRAY",
		Items: &genai.Schema{
			Type:        "STRING",
			Description: "String item",
		},
	}

	converter := NewManualSchemaConverter()
	result, err := converter.Convert(schema)
	require.NoError(t, err)
	assert.Equal(t, "array", result["type"])
	assert.Contains(t, result, "items")

	items, ok := result["items"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "string", items["type"])
	assert.Equal(t, "String item", items["description"])
}

func TestManualSchemaConverter_WithEnum(t *testing.T) {
	schema := &genai.Schema{
		Type: "STRING",
		Enum: []string{"red", "green", "blue"},
	}

	converter := NewManualSchemaConverter()
	result, err := converter.Convert(schema)
	require.NoError(t, err)
	assert.Equal(t, "string", result["type"])
	assert.Equal(t, []string{"red", "green", "blue"}, result["enum"])
}

func TestManualSchemaConverter_NestedObjects(t *testing.T) {
	schema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"address": {
				Type: "OBJECT",
				Properties: map[string]*genai.Schema{
					"street": {Type: "STRING"},
					"city":   {Type: "STRING"},
				},
			},
		},
	}

	converter := NewManualSchemaConverter()
	result, err := converter.Convert(schema)
	require.NoError(t, err)

	props, ok := result["properties"].(map[string]any)
	require.True(t, ok)

	address, ok := props["address"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "object", address["type"])

	addressProps, ok := address["properties"].(map[string]any)
	require.True(t, ok)
	assert.Contains(t, addressProps, "street")
	assert.Contains(t, addressProps, "city")
}

func BenchmarkManualSchemaConverter(b *testing.B) {
	schema := getBenchmarkSchema()
	converter := NewManualSchemaConverter()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := converter.Convert(schema)
		if err != nil {
			b.Fatal(err)
		}
	}
}
