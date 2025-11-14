package adk

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/azure"
	"charm.land/fantasy/providers/bedrock"
	"charm.land/fantasy/providers/google"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	"charm.land/fantasy/providers/openrouter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type MockLanguageModel struct {
	mock.Mock
}

func (m *MockLanguageModel) Provider() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockLanguageModel) Model() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockLanguageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	args := m.Called(ctx, call)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*fantasy.Response), args.Error(1)
}

func (m *MockLanguageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	args := m.Called(ctx, call)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(fantasy.StreamResponse), args.Error(1)
}

// Test helper functions

type streamBuilder struct {
	parts []fantasy.StreamPart
}

func newStreamBuilder() *streamBuilder {
	return &streamBuilder{parts: []fantasy.StreamPart{}}
}

func (sb *streamBuilder) addTextStart(id string) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type: fantasy.StreamPartTypeTextStart,
		ID:   id,
	})
	return sb
}

func (sb *streamBuilder) addTextDelta(id, delta string) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type:  fantasy.StreamPartTypeTextDelta,
		ID:    id,
		Delta: delta,
	})
	return sb
}

func (sb *streamBuilder) addTextEnd(id string) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type: fantasy.StreamPartTypeTextEnd,
		ID:   id,
	})
	return sb
}

func (sb *streamBuilder) addToolStart(id, name string) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type:         fantasy.StreamPartTypeToolInputStart,
		ID:           id,
		ToolCallName: name,
	})
	return sb
}

func (sb *streamBuilder) addToolDelta(id, input string) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type:          fantasy.StreamPartTypeToolInputDelta,
		ID:            id,
		ToolCallInput: input,
	})
	return sb
}

func (sb *streamBuilder) addToolEnd(id string) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type: fantasy.StreamPartTypeToolInputEnd,
		ID:   id,
	})
	return sb
}

func (sb *streamBuilder) addReasoningStart(id string) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type: fantasy.StreamPartTypeReasoningStart,
		ID:   id,
	})
	return sb
}

func (sb *streamBuilder) addReasoningDelta(id, delta string) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type:  fantasy.StreamPartTypeReasoningDelta,
		ID:    id,
		Delta: delta,
	})
	return sb
}

func (sb *streamBuilder) addReasoningEnd(id string) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type: fantasy.StreamPartTypeReasoningEnd,
		ID:   id,
	})
	return sb
}

func (sb *streamBuilder) addFinish(reason fantasy.FinishReason, usage fantasy.Usage) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type:         fantasy.StreamPartTypeFinish,
		FinishReason: reason,
		Usage:        usage,
	})
	return sb
}

func (sb *streamBuilder) addError(err error) *streamBuilder {
	sb.parts = append(sb.parts, fantasy.StreamPart{
		Type:  fantasy.StreamPartTypeError,
		Error: err,
	})
	return sb
}

func (sb *streamBuilder) build() fantasy.StreamResponse {
	parts := sb.parts
	return func(yield func(fantasy.StreamPart) bool) {
		for _, part := range parts {
			if !yield(part) {
				return
			}
		}
	}
}

func collectResponses(t *testing.T, iter func(func(*model.LLMResponse, error) bool)) []*model.LLMResponse {
	t.Helper()
	var responses []*model.LLMResponse
	for resp, err := range iter {
		require.NoError(t, err)
		responses = append(responses, resp)
	}
	return responses
}

// Request builder helper

type requestBuilder struct {
	req *model.LLMRequest
}

func newRequest() *requestBuilder {
	return &requestBuilder{
		req: &model.LLMRequest{
			Model:    "test/model",
			Contents: []*genai.Content{},
			Config:   &genai.GenerateContentConfig{},
		},
	}
}

func (rb *requestBuilder) withUserText(text string) *requestBuilder {
	rb.req.Contents = append(rb.req.Contents, &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: text}},
	})
	return rb
}

func (rb *requestBuilder) build() *model.LLMRequest {
	return rb.req
}

// Response builder helper

func newResponseWithUsage(text string, inputTokens, outputTokens int64) *fantasy.Response {
	return &fantasy.Response{
		Content: []fantasy.Content{fantasy.TextContent{Text: text}},
		Usage: fantasy.Usage{
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			TotalTokens:  inputTokens + outputTokens,
		},
		FinishReason: fantasy.FinishReasonStop,
	}
}

// Mock setup helpers

func mockGenerate(m *MockLanguageModel, response *fantasy.Response, err error) *MockLanguageModel {
	m.On("Generate", mock.Anything, mock.Anything).Return(response, err)
	return m
}

func mockStream(m *MockLanguageModel, stream fantasy.StreamResponse, err error) *MockLanguageModel {
	m.On("Stream", mock.Anything, mock.Anything).Return(stream, err)
	return m
}

// Assertion helpers

func assertUsage(t *testing.T, resp *model.LLMResponse, prompt, candidates, total int32) {
	t.Helper()
	require.NotNil(t, resp.UsageMetadata)
	assert.Equal(t, prompt, resp.UsageMetadata.PromptTokenCount)
	assert.Equal(t, candidates, resp.UsageMetadata.CandidatesTokenCount)
	assert.Equal(t, total, resp.UsageMetadata.TotalTokenCount)
}

func assertToolCall(t *testing.T, part *genai.Part, id, name string) {
	t.Helper()
	require.NotNil(t, part.FunctionCall)
	assert.Equal(t, id, part.FunctionCall.ID)
	assert.Equal(t, name, part.FunctionCall.Name)
}

// Schema builder helper

type schemaBuilder struct {
	schema *genai.Schema
}

func objectSchema() *schemaBuilder {
	return &schemaBuilder{
		schema: &genai.Schema{
			Type:       "OBJECT",
			Properties: make(map[string]*genai.Schema),
		},
	}
}

func (sb *schemaBuilder) withStringProp(name, desc string) *schemaBuilder {
	sb.schema.Properties[name] = &genai.Schema{Type: "STRING", Description: desc}
	return sb
}

func (sb *schemaBuilder) withIntegerProp(name, desc string) *schemaBuilder {
	sb.schema.Properties[name] = &genai.Schema{Type: "INTEGER", Description: desc}
	return sb
}

func (sb *schemaBuilder) withDescription(desc string) *schemaBuilder {
	sb.schema.Description = desc
	return sb
}

func (sb *schemaBuilder) withRequired(fields ...string) *schemaBuilder {
	sb.schema.Required = fields
	return sb
}

func (sb *schemaBuilder) build() *genai.Schema {
	return sb.schema
}

func TestNewAdapter(t *testing.T) {
	m := new(MockLanguageModel)

	adapter := NewAdapter(m)
	require.NotNil(t, adapter)
	require.IsType(t, &Adapter{}, adapter)

	adapterImpl := adapter.(*Adapter)
	assert.Equal(t, m, adapterImpl.model)
}

func TestAdapter_Implements_ModelLLM_Interface(t *testing.T) {
	m := new(MockLanguageModel)

	adapter := NewAdapter(m)

	assert.Implements(t, (*model.LLM)(nil), adapter)
}

func TestAdapter_Name(t *testing.T) {
	m := new(MockLanguageModel)
	m.On("Provider").Return("test-provider")
	m.On("Model").Return("test-model")
	defer m.AssertExpectations(t)

	adapter := &Adapter{model: m}
	name := adapter.Name()

	assert.Equal(t, "test-provider/test-model", name)
}

func TestAdapter_GenerateContent_NonStreaming(t *testing.T) {
	expectedResponse := newResponseWithUsage("test response", 10, 20)
	expectedResponse.FinishReason = fantasy.FinishReasonStop

	m := mockGenerate(new(MockLanguageModel), expectedResponse, nil)
	defer m.AssertExpectations(t)

	adapter := &Adapter{model: m}
	req := newRequest().withUserText("hello").build()

	ctx := t.Context()
	iter := adapter.GenerateContent(ctx, req, false)

	var responses []*model.LLMResponse
	var errs []error
	for resp, err := range iter {
		responses = append(responses, resp)
		errs = append(errs, err)
	}

	require.Len(t, responses, 1)
	require.NoError(t, errs[0])

	resp := responses[0]
	require.NotNil(t, resp.Content)
	require.Len(t, resp.Content.Parts, 1)

	assert.Equal(t, "test response", resp.Content.Parts[0].Text)
	assert.True(t, resp.TurnComplete)
	assert.False(t, resp.Partial)

	assertUsage(t, resp, 10, 20, 30)
	assert.Equal(t, genai.FinishReasonStop, resp.FinishReason)
}

func TestAdapter_GenerateContent_Streaming(t *testing.T) {
	stream := newStreamBuilder().
		addTextStart("0").
		addTextDelta("0", "test").
		addTextEnd("0").
		addFinish(fantasy.FinishReasonStop, fantasy.Usage{
			InputTokens:  10,
			OutputTokens: 4,
			TotalTokens:  14,
		}).
		build()

	m := mockStream(new(MockLanguageModel), stream, nil)
	defer m.AssertExpectations(t)

	adapter := &Adapter{model: m}
	req := newRequest().withUserText("hello").build()

	ctx := t.Context()
	iter := adapter.GenerateContent(ctx, req, true)

	var responses []*model.LLMResponse
	for resp, err := range iter {
		require.NoError(t, err)
		responses = append(responses, resp)
	}

	require.NotEmpty(t, responses)

	finalResp := responses[len(responses)-1]
	assert.True(t, finalResp.TurnComplete)
	assertUsage(t, finalResp, 10, 4, 14)
}

func TestLlmRequestToFantasyCall_Basic(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.9)
	topK := float32(40)

	req := &model.LLMRequest{
		Model: "test-model",
		Config: &genai.GenerateContentConfig{
			Temperature:     &temp,
			TopP:            &topP,
			TopK:            &topK,
			MaxOutputTokens: 100,
		},
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "hello"},
				},
			},
		},
	}

	call, err := llmRequestToFantasyCall(req)
	require.NoError(t, err)

	require.NotNil(t, call.Temperature)
	assert.InDelta(t, 0.7, *call.Temperature, 0.0001)

	require.NotNil(t, call.TopP)
	assert.InDelta(t, 0.9, *call.TopP, 0.0001)

	require.NotNil(t, call.TopK)
	assert.Equal(t, int64(40), *call.TopK)

	require.NotNil(t, call.MaxOutputTokens)
	assert.Equal(t, int64(100), *call.MaxOutputTokens)

	require.Len(t, call.Prompt, 1)
}

func TestLlmRequestToFantasyCall_UnsupportedFeatures(t *testing.T) {
	tests := []struct {
		name   string
		config *genai.GenerateContentConfig
		errMsg string
	}{
		{
			name: "safety settings",
			config: &genai.GenerateContentConfig{
				SafetySettings: []*genai.SafetySetting{{}},
			},
			errMsg: "safety settings not supported",
		},
		{
			name: "response MIME type",
			config: &genai.GenerateContentConfig{
				ResponseMIMEType: "application/json",
			},
			errMsg: "response MIME type not supported",
		},
		{
			name: "response schema",
			config: &genai.GenerateContentConfig{
				ResponseSchema: &genai.Schema{},
			},
			errMsg: "response schema not supported",
		},
		{
			name: "cached content",
			config: &genai.GenerateContentConfig{
				CachedContent: "cached-id",
			},
			errMsg: "cached content not supported",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &model.LLMRequest{
				Config: tt.config,
			}

			_, err := llmRequestToFantasyCall(req)
			require.Error(t, err)
			assert.ErrorContains(t, err, tt.errMsg)
		})
	}
}

func TestGenaiContentToFantasyMessage(t *testing.T) {
	tests := []struct {
		name        string
		content     *genai.Content
		expectRole  fantasy.MessageRole
		expectParts int
		expectErr   bool
	}{
		{
			name: "text part",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "hello"},
				},
			},
			expectRole:  fantasy.MessageRoleUser,
			expectParts: 1,
		},
		{
			name: "thought part",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "thinking...", Thought: true},
				},
			},
			expectRole:  fantasy.MessageRoleAssistant,
			expectParts: 1,
		},
		{
			name: "inline data",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{Data: []byte("data"), MIMEType: "image/png"}},
				},
			},
			expectRole:  fantasy.MessageRoleUser,
			expectParts: 1,
		},
		{
			name: "file data URI",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{FileData: &genai.FileData{FileURI: "gs://bucket/file"}},
				},
			},
			expectRole: fantasy.MessageRoleUser,
			expectErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg, err := genaiContentToFantasyMessage(tt.content)

			if tt.expectErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Len(t, msg.Content, tt.expectParts)
			assert.Equal(t, tt.expectRole, msg.Role)
		})
	}
}

func TestFantasyResponseToLLM(t *testing.T) {
	resp := &fantasy.Response{
		Content: []fantasy.Content{
			fantasy.TextContent{Text: "hello"},
			fantasy.ToolCallContent{
				ToolCallID: "call-123",
				ToolName:   "test-tool",
			},
			fantasy.ReasoningContent{Text: "thinking"},
		},
		Usage: fantasy.Usage{
			InputTokens:     10,
			OutputTokens:    20,
			TotalTokens:     30,
			CacheReadTokens: 5,
		},
		FinishReason: fantasy.FinishReasonStop,
	}

	llmResp := fantasyResponseToLLM(resp)

	require.NotNil(t, llmResp.Content)
	require.Len(t, llmResp.Content.Parts, 3)

	assert.Equal(t, "hello", llmResp.Content.Parts[0].Text)

	require.NotNil(t, llmResp.Content.Parts[1].FunctionCall)

	assert.Equal(t, "thinking", llmResp.Content.Parts[2].Text)
	assert.True(t, llmResp.Content.Parts[2].Thought)

	require.NotNil(t, llmResp.UsageMetadata)
	assert.Equal(t, int32(10), llmResp.UsageMetadata.PromptTokenCount)

	assert.True(t, llmResp.TurnComplete)
	assert.False(t, llmResp.Partial)
	assert.Equal(t, genai.FinishReasonStop, llmResp.FinishReason)
}

func TestFantasyStreamToLLM(t *testing.T) {
	stream := newStreamBuilder().
		addTextStart("0").
		addTextDelta("0", "hello").
		addTextDelta("0", " world").
		addTextEnd("0").
		addFinish(fantasy.FinishReasonStop, fantasy.Usage{
			InputTokens:  5,
			OutputTokens: 2,
			TotalTokens:  7,
		}).
		build()

	iter := fantasyStreamToLLM(stream)
	responses := collectResponses(t, iter)

	require.NotEmpty(t, responses)
	finalResp := responses[len(responses)-1]
	assert.True(t, finalResp.TurnComplete)
	require.NotNil(t, finalResp.UsageMetadata)
}

func TestFantasyStreamToLLM_WithReasoning(t *testing.T) {
	stream := newStreamBuilder().
		addReasoningStart("0").
		addReasoningDelta("0", "thinking...").
		addReasoningEnd("0").
		addFinish(fantasy.FinishReasonStop, fantasy.Usage{}).
		build()

	iter := fantasyStreamToLLM(stream)

	for resp, err := range iter {
		require.NoError(t, err)
		if resp.TurnComplete {
			break
		}
	}
}

func TestFantasyStreamToLLM_WithError(t *testing.T) {
	testErr := errors.New("test error")
	stream := newStreamBuilder().
		addError(testErr).
		build()

	iter := fantasyStreamToLLM(stream)

	var gotError error
	for resp, err := range iter {
		gotError = err
		if resp != nil && resp.ErrorCode != "" {
			assert.Equal(t, "ERROR", resp.ErrorCode)
			assert.Equal(t, testErr.Error(), resp.ErrorMessage)
		}
	}

	require.Error(t, gotError)
}

func TestFantasyFinishReasonToGenai(t *testing.T) {
	tests := []struct {
		fantasy fantasy.FinishReason
		genai   genai.FinishReason
	}{
		{fantasy.FinishReasonStop, genai.FinishReasonStop},
		{fantasy.FinishReasonLength, genai.FinishReasonMaxTokens},
		{fantasy.FinishReasonContentFilter, genai.FinishReasonSafety},
		{fantasy.FinishReasonToolCalls, genai.FinishReasonStop},
		{fantasy.FinishReasonError, genai.FinishReasonOther},
		{fantasy.FinishReasonOther, genai.FinishReasonUnspecified},
		{fantasy.FinishReasonUnknown, genai.FinishReasonUnspecified},
	}

	for _, tt := range tests {
		t.Run(string(tt.fantasy), func(t *testing.T) {
			result := fantasyFinishReasonToGenai(tt.fantasy)
			assert.Equal(t, tt.genai, result)
		})
	}
}

func TestGenaiToolsToFantasyTools(t *testing.T) {
	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        "test-function",
					Description: "test description",
				},
			},
		},
	}

	fantasyTools, err := genaiToolsToFantasyTools(tools)
	require.NoError(t, err)
	require.Len(t, fantasyTools, 1)

	ft, ok := fantasyTools[0].(fantasy.FunctionTool)
	require.True(t, ok)
	assert.Equal(t, "test-function", ft.Name)
}

func TestGenaiToolsToFantasyTools_UnsupportedTools(t *testing.T) {
	tools := []*genai.Tool{
		{
			Retrieval: &genai.Retrieval{},
		},
		{
			CodeExecution: &genai.ToolCodeExecution{},
		},
	}

	_, err := genaiToolsToFantasyTools(tools)
	require.Error(t, err)
}

func TestLlmRequestToFantasyCall_SystemInstruction(t *testing.T) {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant"},
				},
			},
		},
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "hello"},
				},
			},
		},
	}

	call, err := llmRequestToFantasyCall(req)
	require.NoError(t, err)
	require.Len(t, call.Prompt, 2)
	assert.Equal(t, fantasy.MessageRoleSystem, call.Prompt[0].Role)
}

func TestLlmRequestToFantasyCall_ToolConfigAndFiltering(t *testing.T) {
	tests := []struct {
		name           string
		toolNames      []string // Tools to define
		config         *genai.FunctionCallingConfig
		wantChoice     *fantasy.ToolChoice
		wantToolCount  int    // Expected number of tools after filtering
		wantErr        bool   // Whether error is expected
		wantErrContain string // Substring to check in error message
	}{
		// Basic mode tests
		{
			name:          "AUTO mode",
			config:        &genai.FunctionCallingConfig{Mode: "AUTO"},
			wantChoice:    ptrToolChoice(fantasy.ToolChoiceAuto),
			wantToolCount: 0,
		},
		{
			name:          "ANY mode",
			config:        &genai.FunctionCallingConfig{Mode: "ANY"},
			wantChoice:    ptrToolChoice(fantasy.ToolChoiceRequired),
			wantToolCount: 0,
		},
		{
			name:          "NONE mode",
			config:        &genai.FunctionCallingConfig{Mode: "NONE"},
			wantChoice:    ptrToolChoice(fantasy.ToolChoiceNone),
			wantToolCount: 0,
		},
		{
			name:           "VALIDATED mode",
			config:         &genai.FunctionCallingConfig{Mode: "VALIDATED"},
			wantErr:        true,
			wantErrContain: "validated tool mode not supported",
		},
		{
			name:           "Unknown mode",
			config:         &genai.FunctionCallingConfig{Mode: "UNKNOWN_MODE"},
			wantErr:        true,
			wantErrContain: "unsupported tool calling mode",
		},

		// Single allowed function tests
		{
			name:          "single allowed function no mode",
			toolNames:     []string{"test-func"},
			config:        &genai.FunctionCallingConfig{AllowedFunctionNames: []string{"test-func"}},
			wantChoice:    ptrToolChoice(fantasy.ToolChoice("test-func")),
			wantToolCount: 1,
		},
		{
			name:          "AUTO mode with single allowed function",
			toolNames:     []string{"test-func"},
			config:        &genai.FunctionCallingConfig{Mode: "AUTO", AllowedFunctionNames: []string{"test-func"}},
			wantChoice:    ptrToolChoice(fantasy.ToolChoice("test-func")),
			wantToolCount: 1,
		},
		{
			name:          "ANY mode with single allowed function",
			toolNames:     []string{"func1", "func2"},
			config:        &genai.FunctionCallingConfig{Mode: "ANY", AllowedFunctionNames: []string{"func2"}},
			wantChoice:    ptrToolChoice(fantasy.ToolChoice("func2")),
			wantToolCount: 1,
		},
		{
			name:          "NONE mode with single allowed function",
			toolNames:     []string{"test-func"},
			config:        &genai.FunctionCallingConfig{Mode: "NONE", AllowedFunctionNames: []string{"test-func"}},
			wantChoice:    ptrToolChoice(fantasy.ToolChoiceNone),
			wantToolCount: 1,
		},

		// Multiple allowed functions tests
		{
			name:          "AUTO mode with multiple allowed functions",
			toolNames:     []string{"func1", "func2"},
			config:        &genai.FunctionCallingConfig{Mode: "AUTO", AllowedFunctionNames: []string{"func1", "func2"}},
			wantChoice:    ptrToolChoice(fantasy.ToolChoiceAuto),
			wantToolCount: 2,
		},
		{
			name:          "ANY mode with multiple allowed functions",
			toolNames:     []string{"func1", "func2"},
			config:        &genai.FunctionCallingConfig{Mode: "ANY", AllowedFunctionNames: []string{"func1", "func2"}},
			wantChoice:    ptrToolChoice(fantasy.ToolChoiceRequired),
			wantToolCount: 2,
		},

		// Filtering tests
		{
			name:          "filters tools to allowed subset",
			toolNames:     []string{"func1", "func2", "func3", "func4"},
			config:        &genai.FunctionCallingConfig{Mode: "AUTO", AllowedFunctionNames: []string{"func2", "func4"}},
			wantChoice:    ptrToolChoice(fantasy.ToolChoiceAuto),
			wantToolCount: 2,
		},
		{
			name:           "non-existent allowed function",
			toolNames:      []string{"func1", "func2"},
			config:         &genai.FunctionCallingConfig{Mode: "AUTO", AllowedFunctionNames: []string{"func1", "func_nonexistent"}},
			wantErr:        true,
			wantErrContain: "func_nonexistent",
		},
		{
			name:           "all allowed functions non-existent",
			toolNames:      []string{"func1", "func2"},
			config:         &genai.FunctionCallingConfig{Mode: "AUTO", AllowedFunctionNames: []string{"func_bad1", "func_bad2"}},
			wantErr:        true,
			wantErrContain: "not found in tools list",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &genai.GenerateContentConfig{
				ToolConfig: &genai.ToolConfig{
					FunctionCallingConfig: tt.config,
				},
			}

			// Add tools if specified
			if len(tt.toolNames) > 0 {
				tools := make([]*genai.FunctionDeclaration, len(tt.toolNames))
				for i, name := range tt.toolNames {
					tools[i] = &genai.FunctionDeclaration{
						Name:        name,
						Description: "Test function " + name,
					}
				}
				config.Tools = []*genai.Tool{
					{FunctionDeclarations: tools},
				}
			}

			req := &model.LLMRequest{Config: config}

			call, err := llmRequestToFantasyCall(req)
			if tt.wantErr {
				require.Error(t, err)
				if tt.wantErrContain != "" {
					assert.Contains(t, err.Error(), tt.wantErrContain)
				}
				return
			}

			require.NoError(t, err)

			// Check ToolChoice
			if tt.wantChoice != nil {
				require.NotNil(t, call.ToolChoice)
				assert.Equal(t, *tt.wantChoice, *call.ToolChoice)
			}

			// Check tool count after filtering
			assert.Len(t, call.Tools, tt.wantToolCount)

			// Verify filtered tools contain only allowed functions
			if len(tt.config.AllowedFunctionNames) > 0 && !tt.wantErr {
				allowedMap := make(map[string]bool)
				for _, name := range tt.config.AllowedFunctionNames {
					allowedMap[name] = true
				}
				for _, tool := range call.Tools {
					if ft, ok := tool.(fantasy.FunctionTool); ok {
						assert.True(t, allowedMap[ft.Name], "tool %q should be in allowed list", ft.Name)
					}
				}
			}
		})
	}
}

func TestGenaiContentToFantasyMessage_FunctionCall(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				FunctionCall: &genai.FunctionCall{
					ID:   "call-123",
					Name: "test-func",
				},
			},
		},
	}

	msg, err := genaiContentToFantasyMessage(content)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, fantasy.MessageRoleAssistant, msg.Role)

	toolCall, ok := msg.Content[0].(fantasy.ToolCallPart)
	require.True(t, ok)
	assert.Equal(t, "call-123", toolCall.ToolCallID)
}

func TestGenaiContentToFantasyMessage_FunctionResponse(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					ID:   "call-123",
					Name: "test-func",
				},
			},
		},
	}

	msg, err := genaiContentToFantasyMessage(content)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, fantasy.MessageRoleTool, msg.Role)

	toolResult, ok := msg.Content[0].(fantasy.ToolResultPart)
	require.True(t, ok)
	assert.Equal(t, "call-123", toolResult.ToolCallID)
}

func TestGenaiContentToFantasyMessage_UnsupportedPartTypes(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
	}{
		{
			name: "ExecutableCode",
			part: &genai.Part{ExecutableCode: &genai.ExecutableCode{}},
		},
		{
			name: "VideoMetadata",
			part: &genai.Part{VideoMetadata: &genai.VideoMetadata{}},
		},
		{
			name: "CodeExecutionResult",
			part: &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content := &genai.Content{Parts: []*genai.Part{tt.part}}
			_, err := genaiContentToFantasyMessage(content)
			require.Error(t, err)
		})
	}
}

func TestAdapter_GenerateContent_RequestError(t *testing.T) {
	m := new(MockLanguageModel)

	adapter := &Adapter{model: m}
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			SafetySettings: []*genai.SafetySetting{{}},
		},
	}

	ctx := t.Context()
	iter := adapter.GenerateContent(ctx, req, false)

	var gotErr error
	for _, err := range iter {
		gotErr = err
	}

	require.Error(t, gotErr)
}

func TestAdapter_GenerateContent_GenerateError(t *testing.T) {
	testErr := errors.New("generate error")

	m := mockGenerate(new(MockLanguageModel), nil, testErr)
	defer m.AssertExpectations(t)

	adapter := &Adapter{model: m}
	req := newRequest().withUserText("hello").build()

	ctx := t.Context()
	iter := adapter.GenerateContent(ctx, req, false)

	var gotErr error
	for _, err := range iter {
		gotErr = err
	}

	require.Error(t, gotErr)
	assert.ErrorIs(t, gotErr, testErr)
}

func TestAdapter_GenerateContent_StreamError(t *testing.T) {
	testErr := errors.New("stream error")

	m := mockStream(new(MockLanguageModel), nil, testErr)
	defer m.AssertExpectations(t)

	adapter := &Adapter{model: m}
	req := newRequest().withUserText("hello").build()

	ctx := t.Context()
	iter := adapter.GenerateContent(ctx, req, true)

	var gotErr error
	for _, err := range iter {
		gotErr = err
	}

	require.Error(t, gotErr)
	assert.ErrorIs(t, gotErr, testErr)
}

func TestFantasyStreamToLLM_ToolCalls(t *testing.T) {
	stream := newStreamBuilder().
		addToolStart("call-123", "test-func").
		addToolDelta("call-123", `{"arg":"value"}`).
		addToolEnd("call-123").
		addFinish(fantasy.FinishReasonToolCalls, fantasy.Usage{}).
		build()

	iter := fantasyStreamToLLM(stream)
	responses := collectResponses(t, iter)

	require.NotEmpty(t, responses)
}

func TestFantasyFinishReasonToGenai_Default(t *testing.T) {
	result := fantasyFinishReasonToGenai(fantasy.FinishReason("invalid"))
	assert.Equal(t, genai.FinishReasonUnspecified, result)
}

func ptrToolChoice(tc fantasy.ToolChoice) *fantasy.ToolChoice {
	return &tc
}

func TestLlmRequestToFantasyCall_PresenceAndFrequencyPenalty(t *testing.T) {
	pp := float32(0.5)
	fp := float32(0.3)

	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			PresencePenalty:  &pp,
			FrequencyPenalty: &fp,
		},
	}

	call, err := llmRequestToFantasyCall(req)
	require.NoError(t, err)

	require.NotNil(t, call.PresencePenalty)
	assert.InDelta(t, 0.5, *call.PresencePenalty, 0.0001)

	require.NotNil(t, call.FrequencyPenalty)
	assert.InDelta(t, 0.3, *call.FrequencyPenalty, 0.0001)
}

func TestLlmRequestToFantasyCall_Tools(t *testing.T) {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "test-tool",
							Description: "test description",
						},
					},
				},
			},
		},
	}

	call, err := llmRequestToFantasyCall(req)
	require.NoError(t, err)
	require.Len(t, call.Tools, 1)
}

func TestLlmRequestToFantasyCall_ToolChoiceValidated(t *testing.T) {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			ToolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: "VALIDATED",
				},
			},
		},
	}

	_, err := llmRequestToFantasyCall(req)
	require.Error(t, err)
}

func TestLlmRequestToFantasyCall_RoleMapping(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "assistant message"},
				},
			},
		},
	}

	call, err := llmRequestToFantasyCall(req)
	require.NoError(t, err)
	require.Len(t, call.Prompt, 1)
	assert.Equal(t, fantasy.MessageRoleAssistant, call.Prompt[0].Role)
}

func TestFantasyStreamToLLM_ToolCall(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolCall,
			ID:           "call-123",
			ToolCallName: "test-func",
		})
		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonStop,
			Usage:        fantasy.Usage{},
		})
	}

	iter := fantasyStreamToLLM(stream)

	for _, err := range iter {
		require.NoError(t, err)
	}
}

func TestGenaiToolsToFantasyTools_WithParameters(t *testing.T) {
	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        "test-function",
					Description: "test description",
					Parameters:  &genai.Schema{Type: "OBJECT"},
				},
			},
		},
	}

	fantasyTools, err := genaiToolsToFantasyTools(tools)
	require.NoError(t, err)
	require.Len(t, fantasyTools, 1)

	ft, ok := fantasyTools[0].(fantasy.FunctionTool)
	require.True(t, ok)
	assert.Equal(t, "test-function", ft.Name)
	assert.Equal(t, "test description", ft.Description)
	require.NotNil(t, ft.InputSchema)
	assert.Equal(t, "object", ft.InputSchema["type"])
}

func TestGenaiToolsToFantasyTools_ParameterPropertiesPreserved(t *testing.T) {
	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        "get_weather",
					Description: "Get weather for a location",
					Parameters: &genai.Schema{
						Type: "OBJECT",
						Properties: map[string]*genai.Schema{
							"location": {
								Type:        "STRING",
								Description: "City name",
							},
							"units": {
								Type:        "STRING",
								Description: "Temperature units",
								Enum:        []string{"celsius", "fahrenheit"},
							},
						},
						Required: []string{"location"},
					},
				},
			},
		},
	}

	fantasyTools, err := genaiToolsToFantasyTools(tools)
	require.NoError(t, err)
	require.Len(t, fantasyTools, 1)

	ft, ok := fantasyTools[0].(fantasy.FunctionTool)
	require.True(t, ok)
	assert.Equal(t, "get_weather", ft.Name)
	assert.Equal(t, "Get weather for a location", ft.Description)

	require.NotNil(t, ft.InputSchema)
	assert.Equal(t, "object", ft.InputSchema["type"])

	assert.Contains(t, ft.InputSchema, "properties", "InputSchema should contain properties")
	props, ok := ft.InputSchema["properties"].(map[string]any)
	require.True(t, ok, "properties should be a map")

	assert.Contains(t, props, "location", "Should have location property")
	assert.Contains(t, props, "units", "Should have units property")

	location, ok := props["location"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "string", location["type"])
	assert.Equal(t, "City name", location["description"])

	units, ok := props["units"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "string", units["type"])
	assert.Equal(t, "Temperature units", units["description"])
	assert.Equal(t, []string{"celsius", "fahrenheit"}, units["enum"])

	assert.Contains(t, ft.InputSchema, "required", "InputSchema should contain required fields")
	assert.Equal(t, []string{"location"}, ft.InputSchema["required"])
}

func TestGenaiToolsToFantasyTools_SystemInformationTool(t *testing.T) {
	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        "system_information",
					Description: "Get system information about the host including OS and architecture",
					Parameters: &genai.Schema{
						Type: "OBJECT",
						Properties: map[string]*genai.Schema{
							"detailed": {
								Type:        "BOOLEAN",
								Description: "Whether to include detailed information",
							},
						},
						Required: []string{"detailed"},
					},
				},
			},
		},
	}

	fantasyTools, err := genaiToolsToFantasyTools(tools)
	require.NoError(t, err)
	require.Len(t, fantasyTools, 1)

	ft, ok := fantasyTools[0].(fantasy.FunctionTool)
	require.True(t, ok)
	assert.Equal(t, "system_information", ft.Name)

	require.NotNil(t, ft.InputSchema)

	jsonBytes, err := json.MarshalIndent(ft.InputSchema, "", "  ")
	require.NoError(t, err)
	t.Logf("InputSchema JSON:\n%s", string(jsonBytes))

	assert.Equal(t, "object", ft.InputSchema["type"])
	assert.Contains(t, ft.InputSchema, "properties")
	assert.Contains(t, ft.InputSchema, "required")

	props, ok := ft.InputSchema["properties"].(map[string]any)
	require.True(t, ok, "properties should be a map")
	assert.Contains(t, props, "detailed")

	detailed, ok := props["detailed"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "boolean", detailed["type"])
	assert.Equal(t, "Whether to include detailed information", detailed["description"])

	assert.Equal(t, []string{"detailed"}, ft.InputSchema["required"])
}

func TestGenaiToolsToFantasyTools_ParametersJsonSchemaAsMap(t *testing.T) {
	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        "geocode_city",
					Description: "Convert US city name to geographic coordinates",
					ParametersJsonSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"city_name": map[string]any{
								"type":        "string",
								"description": "US city name",
							},
						},
						"required": []any{"city_name"},
					},
				},
			},
		},
	}

	fantasyTools, err := genaiToolsToFantasyTools(tools)
	require.NoError(t, err)
	require.Len(t, fantasyTools, 1)

	ft, ok := fantasyTools[0].(fantasy.FunctionTool)
	require.True(t, ok)
	assert.Equal(t, "geocode_city", ft.Name)
	assert.Equal(t, "Convert US city name to geographic coordinates", ft.Description)

	require.NotNil(t, ft.InputSchema)
	assert.Equal(t, "object", ft.InputSchema["type"])

	assert.Contains(t, ft.InputSchema, "properties")
	props, ok := ft.InputSchema["properties"].(map[string]any)
	require.True(t, ok)

	assert.Contains(t, props, "city_name")
	cityName, ok := props["city_name"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "string", cityName["type"])
	assert.Equal(t, "US city name", cityName["description"])

	assert.Contains(t, ft.InputSchema, "required")
}

func TestGenaiToolsToFantasyTools_ParametersJsonSchemaAsBytes(t *testing.T) {
	schemaJSON := []byte(`{
		"type": "object",
		"properties": {
			"latitude": {
				"type": "number",
				"description": "Latitude coordinate"
			},
			"longitude": {
				"type": "number",
				"description": "Longitude coordinate"
			}
		},
		"required": ["latitude", "longitude"]
	}`)

	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:                 "get_weather",
					Description:          "Get weather forecast for coordinates",
					ParametersJsonSchema: schemaJSON,
				},
			},
		},
	}

	fantasyTools, err := genaiToolsToFantasyTools(tools)
	require.NoError(t, err)
	require.Len(t, fantasyTools, 1)

	ft, ok := fantasyTools[0].(fantasy.FunctionTool)
	require.True(t, ok)
	assert.Equal(t, "get_weather", ft.Name)

	require.NotNil(t, ft.InputSchema)
	assert.Equal(t, "object", ft.InputSchema["type"])

	props, ok := ft.InputSchema["properties"].(map[string]any)
	require.True(t, ok)
	assert.Contains(t, props, "latitude")
	assert.Contains(t, props, "longitude")

	lat, ok := props["latitude"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "number", lat["type"])
	assert.Equal(t, "Latitude coordinate", lat["description"])

	lon, ok := props["longitude"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "number", lon["type"])
	assert.Equal(t, "Longitude coordinate", lon["description"])

	required, ok := ft.InputSchema["required"].([]any)
	require.True(t, ok)
	assert.Len(t, required, 2)
}

func TestGenaiToolsToFantasyTools_BothParametersAndJsonSchema(t *testing.T) {
	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        "test_tool",
					Description: "Test tool with Parameters field",
					Parameters: &genai.Schema{
						Type: "object",
						Properties: map[string]*genai.Schema{
							"param1": {
								Type:        "string",
								Description: "From Parameters field",
							},
						},
					},
					ParametersJsonSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"param2": map[string]any{
								"type":        "string",
								"description": "From ParametersJsonSchema field",
							},
						},
					},
				},
			},
		},
	}

	fantasyTools, err := genaiToolsToFantasyTools(tools)
	require.NoError(t, err)
	require.Len(t, fantasyTools, 1)

	ft, ok := fantasyTools[0].(fantasy.FunctionTool)
	require.True(t, ok)

	props, ok := ft.InputSchema["properties"].(map[string]any)
	require.True(t, ok)
	assert.Contains(t, props, "param1", "Should use Parameters field when both are present")
	assert.NotContains(t, props, "param2", "Should prefer Parameters over ParametersJsonSchema")
}

func TestSchemaToMap_NilSchema(t *testing.T) {
	result, err := schemaToMap(nil)
	require.NoError(t, err)
	assert.Nil(t, result)
}

func TestSchemaToMap_BasicTypes(t *testing.T) {
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

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := schemaToMap(tt.schema)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestSchemaToMap_ObjectWithProperties(t *testing.T) {
	schema := objectSchema().
		withDescription("A person object").
		withStringProp("name", "Person's name").
		withIntegerProp("age", "Person's age").
		withRequired("name").
		build()

	result, err := schemaToMap(schema)
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

func TestSchemaToMap_ArrayWithItems(t *testing.T) {
	schema := &genai.Schema{
		Type: "ARRAY",
		Items: &genai.Schema{
			Type:        "STRING",
			Description: "String item",
		},
	}

	result, err := schemaToMap(schema)
	require.NoError(t, err)
	assert.Equal(t, "array", result["type"])
	assert.Contains(t, result, "items")

	items, ok := result["items"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "string", items["type"])
	assert.Equal(t, "String item", items["description"])
}

func TestSchemaToMap_WithEnum(t *testing.T) {
	schema := &genai.Schema{
		Type: "STRING",
		Enum: []string{"red", "green", "blue"},
	}

	result, err := schemaToMap(schema)
	require.NoError(t, err)
	assert.Equal(t, "string", result["type"])
	assert.Equal(t, []string{"red", "green", "blue"}, result["enum"])
}

func TestSchemaToMap_NestedObjects(t *testing.T) {
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

	result, err := schemaToMap(schema)
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

func TestGenaiContentToFantasyMessage_FunctionCallSerialization(t *testing.T) {
	content := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{
				FunctionCall: &genai.FunctionCall{
					ID:   "call-123",
					Name: "get_weather",
					Args: map[string]any{
						"location": "San Francisco",
						"unit":     "celsius",
					},
				},
			},
		},
	}

	msg, err := genaiContentToFantasyMessage(content)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, fantasy.MessageRoleUser, msg.Role)

	toolCall, ok := msg.Content[0].(fantasy.ToolCallPart)
	require.True(t, ok)
	assert.Equal(t, "call-123", toolCall.ToolCallID)
	assert.Equal(t, "get_weather", toolCall.ToolName)

	var args map[string]any
	err = json.Unmarshal([]byte(toolCall.Input), &args)
	require.NoError(t, err)
	assert.Equal(t, "San Francisco", args["location"])
	assert.Equal(t, "celsius", args["unit"])
}

func TestGenaiContentToFantasyMessage_FunctionCallEmptyArgs(t *testing.T) {
	content := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{
				FunctionCall: &genai.FunctionCall{
					ID:   "call-456",
					Name: "get_time",
					Args: nil,
				},
			},
		},
	}

	msg, err := genaiContentToFantasyMessage(content)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, fantasy.MessageRoleUser, msg.Role)

	toolCall, ok := msg.Content[0].(fantasy.ToolCallPart)
	require.True(t, ok)
	assert.Equal(t, "call-456", toolCall.ToolCallID)
	assert.Equal(t, "get_time", toolCall.ToolName)
	assert.Empty(t, toolCall.Input)
}

func TestGenaiContentToFantasyMessage_FunctionResponseSerialization(t *testing.T) {
	content := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					ID:   "call-789",
					Name: "get_weather",
					Response: map[string]any{
						"temperature": 72,
						"condition":   "sunny",
						"humidity":    65,
					},
				},
			},
		},
	}

	msg, err := genaiContentToFantasyMessage(content)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, fantasy.MessageRoleTool, msg.Role)

	toolResult, ok := msg.Content[0].(fantasy.ToolResultPart)
	require.True(t, ok)
	assert.Equal(t, "call-789", toolResult.ToolCallID)

	textOutput, ok := toolResult.Output.(fantasy.ToolResultOutputContentText)
	require.True(t, ok)

	var response map[string]any
	err = json.Unmarshal([]byte(textOutput.Text), &response)
	require.NoError(t, err)
	assert.InDelta(t, 72.0, response["temperature"], 0.001)
	assert.Equal(t, "sunny", response["condition"])
	assert.InDelta(t, 65.0, response["humidity"], 0.001)
}

func TestGenaiContentToFantasyMessage_FunctionResponseEmptyResponse(t *testing.T) {
	content := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					ID:       "call-999",
					Name:     "get_status",
					Response: nil,
				},
			},
		},
	}

	msg, err := genaiContentToFantasyMessage(content)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, fantasy.MessageRoleTool, msg.Role)

	toolResult, ok := msg.Content[0].(fantasy.ToolResultPart)
	require.True(t, ok)
	assert.Equal(t, "call-999", toolResult.ToolCallID)

	textOutput, ok := toolResult.Output.(fantasy.ToolResultOutputContentText)
	require.True(t, ok)
	assert.Empty(t, textOutput.Text)
}

func TestFantasyResponseToLLM_ToolCallDeserialization(t *testing.T) {
	resp := &fantasy.Response{
		Content: []fantasy.Content{
			fantasy.ToolCallContent{
				ToolCallID: "call-abc",
				ToolName:   "calculate",
				Input:      `{"operation":"add","numbers":[1,2,3]}`,
			},
		},
	}

	llmResp := fantasyResponseToLLM(resp)
	require.NotNil(t, llmResp.Content)
	require.Len(t, llmResp.Content.Parts, 1)

	part := llmResp.Content.Parts[0]
	assertToolCall(t, part, "call-abc", "calculate")
	assert.Equal(t, "add", part.FunctionCall.Args["operation"])
	assert.Equal(t, []any{float64(1), float64(2), float64(3)}, part.FunctionCall.Args["numbers"])
}

func TestFantasyResponseToLLM_ToolCallEmptyInput(t *testing.T) {
	resp := &fantasy.Response{
		Content: []fantasy.Content{
			fantasy.ToolCallContent{
				ToolCallID: "call-def",
				ToolName:   "get_status",
				Input:      "",
			},
		},
	}

	llmResp := fantasyResponseToLLM(resp)
	require.NotNil(t, llmResp.Content)
	require.Len(t, llmResp.Content.Parts, 1)

	part := llmResp.Content.Parts[0]
	assertToolCall(t, part, "call-def", "get_status")
	assert.Empty(t, part.FunctionCall.Args)
}

func TestFantasyResponseToLLM_ToolCallInvalidJSON(t *testing.T) {
	resp := &fantasy.Response{
		Content: []fantasy.Content{
			fantasy.ToolCallContent{
				ToolCallID: "call-ghi",
				ToolName:   "parse_data",
				Input:      `{invalid json}`,
			},
		},
	}

	llmResp := fantasyResponseToLLM(resp)
	require.NotNil(t, llmResp)
	assert.Equal(t, ErrorCodeUnmarshal, llmResp.ErrorCode)
	assert.Contains(t, llmResp.ErrorMessage, "failed to unmarshal tool call input")
}

func TestFantasyStreamToLLM_ToolInputAccumulation(t *testing.T) {
	stream := newStreamBuilder().
		addToolStart("call-xyz", "search").
		addToolDelta("call-xyz", `{"query":`).
		addToolDelta("call-xyz", `"golang best practices"`).
		addToolDelta("call-xyz", `,"limit":10}`).
		addToolEnd("call-xyz").
		addFinish(fantasy.FinishReasonToolCalls, fantasy.Usage{}).
		build()

	iter := fantasyStreamToLLM(stream)

	var foundToolCall bool
	for resp, err := range iter {
		require.NoError(t, err)
		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil && part.FunctionCall.Name == "search" {
					foundToolCall = true
					assert.Equal(t, "call-xyz", part.FunctionCall.ID)
					assert.Equal(t, "search", part.FunctionCall.Name)
					assert.Equal(t, "golang best practices", part.FunctionCall.Args["query"])
					assert.InDelta(t, 10.0, part.FunctionCall.Args["limit"], 0.001)
				}
			}
		}
	}

	assert.True(t, foundToolCall, "Expected to find accumulated tool call in stream")
}

func TestFantasyStreamToLLM_ToolInputNestedObjects(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-nested",
			ToolCallName: "configure",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-nested",
			ToolCallInput: `{"config":{"timeout":`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-nested",
			ToolCallInput: `30,"retries":3},`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-nested",
			ToolCallInput: `"users":[{"id":1,`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-nested",
			ToolCallInput: `"name":"Alice","active":true}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-nested",
			ToolCallInput: `]}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-nested",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonToolCalls,
			Usage:        fantasy.Usage{},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	var foundToolCall bool
	for resp, err := range iter {
		require.NoError(t, err)
		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil && part.FunctionCall.Name == "configure" {
					foundToolCall = true
					assert.Equal(t, "call-nested", part.FunctionCall.ID)
					assert.Equal(t, "configure", part.FunctionCall.Name)

					// Marshal the Args back to JSON and compare with expected JSON
					argsJSON, err := json.Marshal(part.FunctionCall.Args)
					require.NoError(t, err)

					expectedJSON := `{
						"config": {
							"timeout": 30,
							"retries": 3
						},
						"users": [
							{
								"id": 1,
								"name": "Alice",
								"active": true
							}
						]
					}`
					assert.JSONEq(t, expectedJSON, string(argsJSON))
				}
			}
		}
	}

	assert.True(t, foundToolCall, "Expected to find accumulated tool call with nested objects in stream")
}

func TestFantasyStreamToLLM_MultipleToolCallsAccumulation(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-1",
			ToolCallName: "geocode_city",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-1",
			ToolCallInput: `{"city":"San `,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-1",
			ToolCallInput: `Francisco, CA"}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-1",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-2",
			ToolCallName: "get_weather",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-2",
			ToolCallInput: `{"latitude":37.7749,`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-2",
			ToolCallInput: `"longitude":-122.4194}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-2",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonToolCalls,
			Usage:        fantasy.Usage{},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	foundCalls := make(map[string]map[string]any)
	for resp, err := range iter {
		require.NoError(t, err)
		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					foundCalls[part.FunctionCall.Name] = part.FunctionCall.Args
					t.Logf("Found tool call: %s with args: %+v",
						part.FunctionCall.Name, part.FunctionCall.Args)
				}
			}
		}
	}

	require.Len(t, foundCalls, 2, "Should find 2 tool calls")

	require.Contains(t, foundCalls, "geocode_city", "Should find geocode_city tool call")
	geocodeArgs := foundCalls["geocode_city"]
	assert.Equal(t, "San Francisco, CA", geocodeArgs["city"],
		"Geocode city parameter should be 'San Francisco, CA'")

	require.Contains(t, foundCalls, "get_weather", "Should find get_weather tool call")
	weatherArgs := foundCalls["get_weather"]
	assert.InDelta(t, 37.7749, weatherArgs["latitude"], 0.0001,
		"Weather latitude should be 37.7749")
	assert.InDelta(t, -122.4194, weatherArgs["longitude"], 0.0001,
		"Weather longitude should be -122.4194")

	assert.NotContains(t, geocodeArgs, "latitude",
		"Geocode args should not contain weather parameters")
	assert.NotContains(t, weatherArgs, "city",
		"Weather args should not contain geocode parameters")
}

func TestFantasyStreamToLLM_ToolInputInvalidJSON(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-bad",
			ToolCallName: "bad_tool",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-bad",
			ToolCallInput: `{invalid`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-bad",
			ToolCallInput: ` json}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-bad",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonToolCalls,
			Usage:        fantasy.Usage{},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	var foundError bool
	for resp, err := range iter {
		if err != nil && resp != nil && resp.ErrorCode == ErrorCodeUnmarshal {
			foundError = true
			assert.Contains(t, resp.ErrorMessage, "failed to unmarshal tool input")
			continue
		}
		if err != nil {
			continue
		}
	}

	assert.True(t, foundError, "Expected to find unmarshal error for invalid JSON")
}

func TestProviders_ImplementModelLLM(t *testing.T) {
	tests := []struct {
		name    string
		envVars map[string]string
		setup   func(ctx context.Context) (fantasy.LanguageModel, error)
		model   string
	}{
		{
			name:    "Anthropic",
			envVars: map[string]string{"ANTHROPIC_API_KEY": ""},
			model:   "claude-3-5-sonnet-20241022",
			setup: func(ctx context.Context) (fantasy.LanguageModel, error) {
				provider, err := anthropic.New(anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
				if err != nil {
					return nil, err
				}
				return provider.LanguageModel(ctx, "claude-3-5-sonnet-20241022")
			},
		},
		{
			name:    "OpenAI",
			envVars: map[string]string{"OPENAI_API_KEY": ""},
			model:   "gpt-4o",
			setup: func(ctx context.Context) (fantasy.LanguageModel, error) {
				provider, err := openai.New(openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
				if err != nil {
					return nil, err
				}
				return provider.LanguageModel(ctx, "gpt-4o")
			},
		},
		{
			name:    "Google",
			envVars: map[string]string{"GOOGLE_API_KEY": ""},
			model:   "gemini-2.0-flash-exp",
			setup: func(ctx context.Context) (fantasy.LanguageModel, error) {
				provider, err := google.New(google.WithGeminiAPIKey(os.Getenv("GOOGLE_API_KEY")))
				if err != nil {
					return nil, err
				}
				return provider.LanguageModel(ctx, "gemini-2.0-flash-exp")
			},
		},
		{
			name:    "Azure",
			envVars: map[string]string{"AZURE_OPENAI_API_KEY": "", "AZURE_OPENAI_BASE_URL": ""},
			model:   "gpt-4o",
			setup: func(ctx context.Context) (fantasy.LanguageModel, error) {
				provider, err := azure.New(
					azure.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
					azure.WithBaseURL(os.Getenv("AZURE_OPENAI_BASE_URL")),
				)
				if err != nil {
					return nil, err
				}
				return provider.LanguageModel(ctx, "gpt-4o")
			},
		},
		{
			name:    "Bedrock",
			envVars: map[string]string{"AWS_REGION": ""},
			model:   "anthropic.claude-3-5-sonnet-20241022-v2:0",
			setup: func(ctx context.Context) (fantasy.LanguageModel, error) {
				provider, err := bedrock.New()
				if err != nil {
					return nil, err
				}
				return provider.LanguageModel(ctx, "anthropic.claude-3-5-sonnet-20241022-v2:0")
			},
		},
		{
			name:    "OpenRouter",
			envVars: map[string]string{"OPENROUTER_API_KEY": ""},
			model:   "anthropic/claude-3.5-sonnet",
			setup: func(ctx context.Context) (fantasy.LanguageModel, error) {
				provider, err := openrouter.New(openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")))
				if err != nil {
					return nil, err
				}
				return provider.LanguageModel(ctx, "anthropic/claude-3.5-sonnet")
			},
		},
		{
			name:    "OpenAICompat",
			envVars: map[string]string{"OPENAI_COMPAT_API_KEY": "", "OPENAI_COMPAT_BASE_URL": ""},
			model:   "model-name",
			setup: func(ctx context.Context) (fantasy.LanguageModel, error) {
				provider, err := openaicompat.New(
					openaicompat.WithAPIKey(os.Getenv("OPENAI_COMPAT_API_KEY")),
					openaicompat.WithBaseURL(os.Getenv("OPENAI_COMPAT_BASE_URL")),
				)
				if err != nil {
					return nil, err
				}
				return provider.LanguageModel(ctx, "model-name")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for envKey := range tt.envVars {
				if os.Getenv(envKey) == "" {
					t.Skipf("%s not set", envKey)
				}
			}

			ctx := t.Context()
			languageModel, err := tt.setup(ctx)
			require.NoError(t, err)

			adapter := NewAdapter(languageModel)
			assert.Implements(t, (*model.LLM)(nil), adapter)
		})
	}
}

func TestLlmRequestToFantasyCall_ToolResultsGetToolRole(t *testing.T) {
	req := &model.LLMRequest{
		Model: "test-model",
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "What's the weather in NYC?"},
				},
			},
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "I'll get the weather for you."},
					{FunctionCall: &genai.FunctionCall{
						ID:   "call-123",
						Name: "get_weather",
						Args: map[string]any{"city": "NYC"},
					}},
				},
			},
			{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:   "call-123",
						Name: "get_weather",
						Response: map[string]any{
							"temperature": 72,
							"conditions":  "sunny",
						},
					}},
				},
			},
		},
	}

	call, err := llmRequestToFantasyCall(req)
	require.NoError(t, err)

	require.Len(t, call.Prompt, 3, "Expected 3 messages in prompt")

	assert.Equal(t, fantasy.MessageRoleUser, call.Prompt[0].Role,
		"First message should be user role")

	assert.Equal(t, fantasy.MessageRoleAssistant, call.Prompt[1].Role,
		"Second message should be assistant role")

	assert.Equal(t, fantasy.MessageRoleTool, call.Prompt[2].Role,
		"Tool results should have MessageRoleTool, not MessageRoleUser")

	require.Len(t, call.Prompt[2].Content, 1, "Expected 1 content part in tool result")
	toolResult, ok := call.Prompt[2].Content[0].(fantasy.ToolResultPart)
	require.True(t, ok, "Expected ToolResultPart")
	assert.Equal(t, "call-123", toolResult.ToolCallID)

	output, ok := toolResult.Output.(fantasy.ToolResultOutputContentText)
	require.True(t, ok, "Expected ToolResultOutputContentText")
	assert.Contains(t, output.Text, "temperature")
	assert.Contains(t, output.Text, "72")
}

func TestGenaiContentToFantasyMessage_MixedTextAndFunctionResponse(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{Text: "Here's the result:"},
			{FunctionResponse: &genai.FunctionResponse{
				ID:   "call-123",
				Name: "get_data",
				Response: map[string]any{
					"value": 42,
				},
			}},
		},
	}

	msg, err := genaiContentToFantasyMessage(content)
	require.NoError(t, err)
	assert.Equal(t, fantasy.MessageRoleTool, msg.Role, "Mixed content with FunctionResponse should get MessageRoleTool")
	require.Len(t, msg.Content, 2, "Expected 2 content parts")

	textPart, ok := msg.Content[0].(fantasy.TextPart)
	require.True(t, ok)
	assert.Equal(t, "Here's the result:", textPart.Text)

	toolResult, ok := msg.Content[1].(fantasy.ToolResultPart)
	require.True(t, ok)
	assert.Equal(t, "call-123", toolResult.ToolCallID)
}

func TestGenaiContentToFantasyMessage_MultipleFunctionResponses(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{
				ID:   "call-1",
				Name: "func1",
				Response: map[string]any{
					"result": "first",
				},
			}},
			{FunctionResponse: &genai.FunctionResponse{
				ID:   "call-2",
				Name: "func2",
				Response: map[string]any{
					"result": "second",
				},
			}},
		},
	}

	msg, err := genaiContentToFantasyMessage(content)
	require.NoError(t, err)
	assert.Equal(t, fantasy.MessageRoleTool, msg.Role, "Multiple FunctionResponses should get MessageRoleTool")
	require.Len(t, msg.Content, 2, "Expected 2 tool result parts")

	toolResult1, ok := msg.Content[0].(fantasy.ToolResultPart)
	require.True(t, ok)
	assert.Equal(t, "call-1", toolResult1.ToolCallID)

	toolResult2, ok := msg.Content[1].(fantasy.ToolResultPart)
	require.True(t, ok)
	assert.Equal(t, "call-2", toolResult2.ToolCallID)
}

func TestGenaiContentToFantasyMessage_FunctionCallAndResponseMixed(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{
				ID:   "call-1",
				Name: "func1",
			}},
			{FunctionResponse: &genai.FunctionResponse{
				ID:   "call-2",
				Name: "func2",
			}},
		},
	}

	_, err := genaiContentToFantasyMessage(content)
	require.Error(t, err, "Content with both FunctionCall and FunctionResponse should error")
	assert.Contains(t, err.Error(), "cannot contain both FunctionCall and FunctionResponse")
}

func TestFantasyStreamToLLM_TextDeltasAreNotAccumulated(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextStart,
			ID:   "0",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "I'll ",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "get ",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "the ",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "weather",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextEnd,
			ID:   "0",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonStop,
			Usage: fantasy.Usage{
				InputTokens:  10,
				OutputTokens: 4,
				TotalTokens:  14,
			},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	var responses []*model.LLMResponse
	for resp, err := range iter {
		require.NoError(t, err)
		responses = append(responses, resp)
	}

	require.NotEmpty(t, responses)

	// Verify each partial response contains ONLY the delta text, not accumulated
	// This test SHOULD FAIL with current implementation, proving Bug #1 exists
	var partialResponses []*model.LLMResponse
	var finalResponse *model.LLMResponse

	for _, resp := range responses {
		if resp.TurnComplete {
			finalResponse = resp
		} else if resp.Partial {
			partialResponses = append(partialResponses, resp)
		}
	}

	require.Len(t, partialResponses, 4, "Expected 4 partial responses for 4 deltas")

	// Each partial should contain ONLY the delta, not accumulated text
	assert.Equal(t, "I'll ", partialResponses[0].Content.Parts[0].Text,
		"First delta should contain ONLY 'I'll ', not accumulated text")

	assert.Equal(t, "get ", partialResponses[1].Content.Parts[0].Text,
		"Second delta should contain ONLY 'get ', not 'I'll get '")

	assert.Equal(t, "the ", partialResponses[2].Content.Parts[0].Text,
		"Third delta should contain ONLY 'the ', not 'I'll get the '")

	assert.Equal(t, "weather", partialResponses[3].Content.Parts[0].Text,
		"Fourth delta should contain ONLY 'weather', not 'I'll get the weather'")

	// Final response should contain the complete accumulated text
	require.NotNil(t, finalResponse, "Expected final response with TurnComplete=true")
	require.NotNil(t, finalResponse.Content)
	require.Len(t, finalResponse.Content.Parts, 1)
	assert.Equal(t, "I'll get the weather", finalResponse.Content.Parts[0].Text,
		"Final response should contain complete accumulated text")
}

func TestFantasyStreamToLLM_PartialFlagCorrect(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextStart,
			ID:   "0",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "Hello",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: " world",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextEnd,
			ID:   "0",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonStop,
			Usage:        fantasy.Usage{},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	var responses []*model.LLMResponse
	for resp, err := range iter {
		require.NoError(t, err)
		responses = append(responses, resp)
	}

	require.Len(t, responses, 3, "Expected 3 responses: 2 deltas + 1 final")

	// First delta response
	assert.True(t, responses[0].Partial, "Delta response should have Partial=true")
	assert.False(t, responses[0].TurnComplete, "Delta response should have TurnComplete=false")
	assert.Empty(t, responses[0].ErrorCode, "Delta response should have no error")

	// Second delta response
	assert.True(t, responses[1].Partial, "Second delta should have Partial=true")
	assert.False(t, responses[1].TurnComplete, "Second delta should have TurnComplete=false")

	// Final response
	assert.False(t, responses[2].Partial, "Final response should have Partial=false")
	assert.True(t, responses[2].TurnComplete, "Final response should have TurnComplete=true")
	assert.Equal(t, genai.FinishReasonStop, responses[2].FinishReason)
	require.NotNil(t, responses[2].UsageMetadata, "Final response should have usage metadata")
}

func TestFantasyStreamToLLM_ToolParametersNotLost(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-geocode",
			ToolCallName: "geocode_city",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-geocode",
			ToolCallInput: `{"city":"New York City, NY","unit":"c`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-geocode",
			ToolCallInput: `elsius","forecast_days":7,"include_hour`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-geocode",
			ToolCallInput: `ly":true}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-geocode",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonToolCalls,
			Usage:        fantasy.Usage{},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	var foundToolCall bool
	var toolCallArgs map[string]any
	for resp, err := range iter {
		require.NoError(t, err)
		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil && part.FunctionCall.Name == "geocode_city" {
					foundToolCall = true
					toolCallArgs = part.FunctionCall.Args

					t.Logf("Tool call ID: %s", part.FunctionCall.ID)
					t.Logf("Tool call Name: %s", part.FunctionCall.Name)
					t.Logf("Tool call Args: %+v", part.FunctionCall.Args)

					// Verify all parameters are present and correct
					assert.Equal(t, "call-geocode", part.FunctionCall.ID,
						"Tool call ID should match")
					assert.Equal(t, "geocode_city", part.FunctionCall.Name,
						"Tool call name should match")

					// Check each parameter
					require.Contains(t, toolCallArgs, "city",
						"Args should contain 'city' parameter")
					assert.Equal(t, "New York City, NY", toolCallArgs["city"],
						"City parameter should be 'New York City, NY'")

					require.Contains(t, toolCallArgs, "unit",
						"Args should contain 'unit' parameter")
					assert.Equal(t, "celsius", toolCallArgs["unit"],
						"Unit parameter should be 'celsius'")

					require.Contains(t, toolCallArgs, "forecast_days",
						"Args should contain 'forecast_days' parameter")
					assert.InDelta(t, 7.0, toolCallArgs["forecast_days"], 0.001,
						"Forecast days should be 7")

					require.Contains(t, toolCallArgs, "include_hourly",
						"Args should contain 'include_hourly' parameter")
					assert.Equal(t, true, toolCallArgs["include_hourly"],
						"Include hourly should be true")
				}
			}
		}
	}

	assert.True(t, foundToolCall, "Should find tool call in stream")
	assert.Len(t, toolCallArgs, 4, "Should have all 4 parameters")
}

func TestFantasyStreamToLLM_CompleteToolCallFlow(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		// Text response: "I'll get the weather for you."
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextStart,
			ID:   "0",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "I'll get ",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "the weather ",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "for you.",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextEnd,
			ID:   "0",
		}) {
			return
		}

		// Tool call: geocode_city
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-geocode",
			ToolCallName: "geocode_city",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-geocode",
			ToolCallInput: `{"city":"New `,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-geocode",
			ToolCallInput: `York City, NY"}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-geocode",
		}) {
			return
		}

		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonToolCalls,
			Usage: fantasy.Usage{
				InputTokens:  15,
				OutputTokens: 25,
				TotalTokens:  40,
			},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	var textResponses []string
	var toolCalls []*genai.FunctionCall
	var finalResponse *model.LLMResponse

	for resp, err := range iter {
		require.NoError(t, err)

		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					textResponses = append(textResponses, part.Text)
					t.Logf("Text delta: %q", part.Text)
				}
				if part.FunctionCall != nil {
					toolCalls = append(toolCalls, part.FunctionCall)
					t.Logf("Tool call: %s with args: %+v",
						part.FunctionCall.Name, part.FunctionCall.Args)
				}
			}
		}

		if resp.TurnComplete {
			finalResponse = resp
		}
	}

	// Verify complete conversation flow
	require.NotNil(t, finalResponse, "Should have final response")

	// Verify tool call was found with correct parameters
	require.Len(t, toolCalls, 1, "Should have exactly 1 tool call")
	assert.Equal(t, "call-geocode", toolCalls[0].ID)
	assert.Equal(t, "geocode_city", toolCalls[0].Name)

	require.Contains(t, toolCalls[0].Args, "city",
		"Tool call should have 'city' parameter")
	assert.Equal(t, "New York City, NY", toolCalls[0].Args["city"],
		"City parameter should match exactly")

	// Verify final response has usage metadata
	require.NotNil(t, finalResponse.UsageMetadata)
	assert.Equal(t, int32(15), finalResponse.UsageMetadata.PromptTokenCount)
	assert.Equal(t, int32(25), finalResponse.UsageMetadata.CandidatesTokenCount)
	assert.Equal(t, int32(40), finalResponse.UsageMetadata.TotalTokenCount)

	assert.Equal(t, genai.FinishReasonStop, finalResponse.FinishReason)

	t.Logf("Complete conversation flow test passed:")
	t.Logf("  - Text responses: %d", len(textResponses))
	t.Logf("  - Tool calls: %d", len(toolCalls))
	t.Logf("  - Final response present: %v", finalResponse != nil)
}

func TestFantasyStreamToLLM_ToolCallTimingDiagnostic(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextStart,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			Delta: "I'll call a tool.",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextEnd,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-123",
			ToolCallName: "test_tool",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-123",
			ToolCallInput: `{"arg":"val`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-123",
			ToolCallInput: `ue"}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-123",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonToolCalls,
			Usage:        fantasy.Usage{},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	type event struct {
		index        int
		hasText      bool
		textContent  string
		hasToolCall  bool
		toolCallName string
		partial      bool
		turnComplete bool
	}

	var events []event
	eventIndex := 0

	for resp, err := range iter {
		require.NoError(t, err)

		ev := event{
			index:        eventIndex,
			partial:      resp.Partial,
			turnComplete: resp.TurnComplete,
		}

		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					ev.hasText = true
					ev.textContent = part.Text
				}
				if part.FunctionCall != nil {
					ev.hasToolCall = true
					ev.toolCallName = part.FunctionCall.Name
				}
			}
		}

		events = append(events, ev)
		eventIndex++
	}

	t.Logf("Tool call timing diagnostic results:")
	t.Logf("Total events yielded: %d", len(events))
	for _, ev := range events {
		t.Logf("  Event[%d]: Text=%v (%q), ToolCall=%v (%q), Partial=%v, TurnComplete=%v",
			ev.index, ev.hasText, ev.textContent, ev.hasToolCall, ev.toolCallName, ev.partial, ev.turnComplete)
	}

	require.GreaterOrEqual(t, len(events), 2, "Should have at least 2 events (text delta + final)")

	var foundToolBeforeTurnComplete bool
	var foundToolInFinal bool

	for _, ev := range events {
		if ev.hasToolCall {
			if !ev.turnComplete {
				foundToolBeforeTurnComplete = true
				t.Logf("Found tool call BEFORE TurnComplete at event %d (Partial=%v, TurnComplete=%v)",
					ev.index, ev.partial, ev.turnComplete)
			}
			if ev.turnComplete {
				foundToolInFinal = true
				t.Logf("Found tool call in FINAL response at event %d", ev.index)
			}
		}
	}

	t.Logf("")
	t.Logf("Summary:")
	t.Logf("  Tool call with TurnComplete=false (correct): %v", foundToolBeforeTurnComplete)
	t.Logf("  Tool call with TurnComplete=true (incorrect): %v", foundToolInFinal)

	require.True(t, foundToolBeforeTurnComplete,
		"Tool call MUST be yielded with TurnComplete=false so ADK can execute it")
	assert.False(t, foundToolInFinal,
		"Tool call should NOT be in TurnComplete=true response (should be filtered out)")
}

func TestFantasyStreamToLLM_ToolCallsWithoutTurnComplete(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-456",
			ToolCallName: "immediate_tool",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-456",
			ToolCallInput: `{"param":"value"}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-456",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonToolCalls,
			Usage:        fantasy.Usage{},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	var responses []*model.LLMResponse
	for resp, err := range iter {
		require.NoError(t, err)
		responses = append(responses, resp)
	}

	require.GreaterOrEqual(t, len(responses), 1, "Should have at least final response")

	var toolCallResponse *model.LLMResponse
	for _, resp := range responses {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					toolCallResponse = resp
					t.Logf("Found tool call in response: Partial=%v, TurnComplete=%v",
						resp.Partial, resp.TurnComplete)
					t.Logf("  Tool: %s", part.FunctionCall.Name)
					t.Logf("  Args: %+v", part.FunctionCall.Args)
					break
				}
			}
		}
	}

	require.NotNil(t, toolCallResponse, "Tool call must be in some response")

	assert.False(t, toolCallResponse.TurnComplete,
		"Tool call MUST have TurnComplete=false so ADK can execute it before ending the turn")

	assert.Contains(t, toolCallResponse.Content.Parts[0].FunctionCall.Args, "param",
		"Tool should have parameters")
	assert.Equal(t, "value", toolCallResponse.Content.Parts[0].FunctionCall.Args["param"])
}

func TestFantasyStreamToLLM_ToolCallRequiredFields(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-789",
			ToolCallName: "check_fields",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-789",
			ToolCallInput: `{"field1":"val1","field2":42}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-789",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonToolCalls,
			Usage:        fantasy.Usage{},
		}) {
			return
		}
	}

	iter := fantasyStreamToLLM(stream)

	var toolCall *genai.FunctionCall
	for resp, err := range iter {
		require.NoError(t, err)
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					toolCall = part.FunctionCall
					break
				}
			}
		}
	}

	require.NotNil(t, toolCall, "Tool call must exist")

	assert.NotEmpty(t, toolCall.ID, "Tool call must have ID")
	assert.Equal(t, "call-789", toolCall.ID, "Tool call ID must match")

	assert.NotEmpty(t, toolCall.Name, "Tool call must have Name")
	assert.Equal(t, "check_fields", toolCall.Name, "Tool call Name must match")

	require.NotNil(t, toolCall.Args, "Tool call must have Args")
	require.IsType(t, map[string]any{}, toolCall.Args, "Args must be map[string]any")

	assert.Contains(t, toolCall.Args, "field1", "Args must contain field1")
	assert.Contains(t, toolCall.Args, "field2", "Args must contain field2")
	assert.Equal(t, "val1", toolCall.Args["field1"])
	assert.InDelta(t, 42.0, toolCall.Args["field2"], 0.001)

	t.Logf("Tool call structure verification:")
	t.Logf("  ID: %s ✓", toolCall.ID)
	t.Logf("  Name: %s ✓", toolCall.Name)
	t.Logf("  Args type: %T ✓", toolCall.Args)
	t.Logf("  Args content: %+v ✓", toolCall.Args)
}
