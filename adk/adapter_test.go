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
	m := new(MockLanguageModel)

	expectedResponse := &fantasy.Response{
		Content: []fantasy.Content{
			fantasy.TextContent{Text: "test response"},
		},
		Usage: fantasy.Usage{
			InputTokens:  10,
			OutputTokens: 20,
			TotalTokens:  30,
		},
		FinishReason: fantasy.FinishReasonStop,
	}

	m.On("Generate", mock.Anything, mock.MatchedBy(func(call fantasy.Call) bool {
		return len(call.Prompt) == 1 &&
			call.Prompt[0].Role == fantasy.MessageRoleUser &&
			len(call.Prompt[0].Content) == 1
	})).Return(expectedResponse, nil)
	defer m.AssertExpectations(t)

	adapter := &Adapter{model: m}
	req := &model.LLMRequest{
		Model: "test/model",
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "hello"},
				},
			},
		},
	}

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

	require.NotNil(t, resp.UsageMetadata)
	assert.Equal(t, int32(10), resp.UsageMetadata.PromptTokenCount)
	assert.Equal(t, int32(20), resp.UsageMetadata.CandidatesTokenCount)
	assert.Equal(t, int32(30), resp.UsageMetadata.TotalTokenCount)
	assert.Equal(t, genai.FinishReasonStop, resp.FinishReason)
}

func TestAdapter_GenerateContent_Streaming(t *testing.T) {
	m := new(MockLanguageModel)

	streamFunc := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextStart,
			ID:   "0",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "test",
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

	m.On("Stream", mock.Anything, mock.MatchedBy(func(call fantasy.Call) bool {
		return len(call.Prompt) == 1 &&
			call.Prompt[0].Role == fantasy.MessageRoleUser
	})).Return(fantasy.StreamResponse(streamFunc), nil)
	defer m.AssertExpectations(t)

	adapter := &Adapter{model: m}
	req := &model.LLMRequest{
		Model: "test/model",
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "hello"},
				},
			},
		},
	}

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
	require.NotNil(t, finalResp.UsageMetadata)
	assert.Equal(t, int32(10), finalResp.UsageMetadata.PromptTokenCount)
	assert.Equal(t, int32(4), finalResp.UsageMetadata.CandidatesTokenCount)
	assert.Equal(t, int32(14), finalResp.UsageMetadata.TotalTokenCount)
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
		role        fantasy.MessageRole
		expectParts int
		expectErr   bool
	}{
		{
			name: "text part",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "hello"},
				},
			},
			role:        fantasy.MessageRoleUser,
			expectParts: 1,
		},
		{
			name: "thought part",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "thinking...", Thought: true},
				},
			},
			role:        fantasy.MessageRoleAssistant,
			expectParts: 1,
		},
		{
			name: "inline data",
			content: &genai.Content{
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{Data: []byte("data"), MIMEType: "image/png"}},
				},
			},
			role:        fantasy.MessageRoleUser,
			expectParts: 1,
		},
		{
			name: "file data URI",
			content: &genai.Content{
				Parts: []*genai.Part{
					{FileData: &genai.FileData{FileURI: "gs://bucket/file"}},
				},
			},
			role:      fantasy.MessageRoleUser,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg, err := genaiContentToFantasyMessage(tt.content, tt.role)

			if tt.expectErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Len(t, msg.Content, tt.expectParts)
			assert.Equal(t, tt.role, msg.Role)
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
			Delta: "hello",
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
			Usage: fantasy.Usage{
				InputTokens:  5,
				OutputTokens: 2,
				TotalTokens:  7,
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

	finalResp := responses[len(responses)-1]
	assert.True(t, finalResp.TurnComplete)
	require.NotNil(t, finalResp.UsageMetadata)
}

func TestFantasyStreamToLLM_WithReasoning(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeReasoningStart,
			ID:   "0",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeReasoningDelta,
			ID:    "0",
			Delta: "thinking...",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeReasoningEnd,
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

	for resp, err := range iter {
		require.NoError(t, err)
		if resp.TurnComplete {
			break
		}
	}
}

func TestFantasyStreamToLLM_WithError(t *testing.T) {
	testErr := errors.New("test error")
	stream := func(yield func(fantasy.StreamPart) bool) {
		yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeError,
			Error: testErr,
		})
	}

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
		Parts: []*genai.Part{
			{
				FunctionCall: &genai.FunctionCall{
					ID:   "call-123",
					Name: "test-func",
				},
			},
		},
	}

	msg, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleAssistant)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)

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

	msg, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)

	toolResult, ok := msg.Content[0].(fantasy.ToolResultPart)
	require.True(t, ok)
	assert.Equal(t, "call-123", toolResult.ToolCallID)
}

func TestGenaiContentToFantasyMessage_ExecutableCode(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{ExecutableCode: &genai.ExecutableCode{}},
		},
	}

	_, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	require.Error(t, err)
}

func TestGenaiContentToFantasyMessage_VideoMetadata(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{VideoMetadata: &genai.VideoMetadata{}},
		},
	}

	_, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	require.Error(t, err)
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
	m := new(MockLanguageModel)

	m.On("Generate", mock.Anything, mock.Anything).Return(nil, testErr)
	defer m.AssertExpectations(t)

	adapter := &Adapter{model: m}
	req := &model.LLMRequest{
		Model: "test/model",
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "hello"},
				},
			},
		},
	}

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
	m := new(MockLanguageModel)

	m.On("Stream", mock.Anything, mock.Anything).Return(nil, testErr)
	defer m.AssertExpectations(t)

	adapter := &Adapter{model: m}
	req := &model.LLMRequest{
		Model: "test/model",
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "hello"},
				},
			},
		},
	}

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
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-123",
			ToolCallName: "test-func",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-123",
			ToolCallInput: `{"arg":"value"}`,
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

	var responses []*model.LLMResponse
	for resp, err := range iter {
		require.NoError(t, err)
		responses = append(responses, resp)
	}

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

func TestGenaiContentToFantasyMessage_CodeExecutionResult(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{CodeExecutionResult: &genai.CodeExecutionResult{}},
		},
	}

	_, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	require.Error(t, err)
}

func TestFantasyStreamToLLM_ToolCall(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolCall,
			ID:           "call-123",
			ToolCallName: "test-func",
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
	assert.Equal(t, "OBJECT", ft.InputSchema["type"])
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
			expected: map[string]any{"type": "STRING", "description": "A string value"},
		},
		{
			name:     "number type with format",
			schema:   &genai.Schema{Type: "NUMBER", Format: "float"},
			expected: map[string]any{"type": "NUMBER", "format": "float"},
		},
		{
			name:     "boolean type",
			schema:   &genai.Schema{Type: "BOOLEAN"},
			expected: map[string]any{"type": "BOOLEAN"},
		},
		{
			name:     "integer with min/max",
			schema:   &genai.Schema{Type: "INTEGER", Minimum: genai.Ptr(1.0), Maximum: genai.Ptr(100.0)},
			expected: map[string]any{"type": "INTEGER", "minimum": 1.0, "maximum": 100.0},
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
	schema := &genai.Schema{
		Type:        "OBJECT",
		Description: "A person object",
		Properties: map[string]*genai.Schema{
			"name": {Type: "STRING", Description: "Person's name"},
			"age":  {Type: "INTEGER", Description: "Person's age"},
		},
		Required: []string{"name"},
	}

	result, err := schemaToMap(schema)
	require.NoError(t, err)
	assert.Equal(t, "OBJECT", result["type"])
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
	assert.Equal(t, "ARRAY", result["type"])
	assert.Contains(t, result, "items")

	items, ok := result["items"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "STRING", items["type"])
	assert.Equal(t, "String item", items["description"])
}

func TestSchemaToMap_WithEnum(t *testing.T) {
	schema := &genai.Schema{
		Type: "STRING",
		Enum: []string{"red", "green", "blue"},
	}

	result, err := schemaToMap(schema)
	require.NoError(t, err)
	assert.Equal(t, "STRING", result["type"])
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
	assert.Equal(t, "OBJECT", address["type"])

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

	msg, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)

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

	msg, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)

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

	msg, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)

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

	msg, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)

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
	require.NotNil(t, part.FunctionCall)
	assert.Equal(t, "call-abc", part.FunctionCall.ID)
	assert.Equal(t, "calculate", part.FunctionCall.Name)
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
	require.NotNil(t, part.FunctionCall)
	assert.Equal(t, "call-def", part.FunctionCall.ID)
	assert.Equal(t, "get_status", part.FunctionCall.Name)
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
	stream := func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-xyz",
			ToolCallName: "search",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-xyz",
			ToolCallInput: `{"query":`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-xyz",
			ToolCallInput: `"golang best practices"`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-xyz",
			ToolCallInput: `,"limit":10}`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-xyz",
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
			ToolCallName: "tool1",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-1",
			ToolCallInput: `{"arg1":`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-1",
			ToolCallInput: `"value1"}`,
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
			ToolCallName: "tool2",
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-2",
			ToolCallInput: `{"arg2":`,
		}) {
			return
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-2",
			ToolCallInput: `"value2"}`,
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

	foundCalls := make(map[string]bool)
	for resp, err := range iter {
		require.NoError(t, err)
		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					foundCalls[part.FunctionCall.Name] = true
					switch part.FunctionCall.Name {
					case "tool1":
						assert.Equal(t, "value1", part.FunctionCall.Args["arg1"])
					case "tool2":
						assert.Equal(t, "value2", part.FunctionCall.Args["arg2"])
					}
				}
			}
		}
	}

	assert.True(t, foundCalls["tool1"], "Expected to find tool1 call")
	assert.True(t, foundCalls["tool2"], "Expected to find tool2 call")
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

func TestAnthropicProvider_Implements_ModelLLM(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := t.Context()
	provider, err := anthropic.New(anthropic.WithAPIKey(apiKey))
	require.NoError(t, err)

	languageModel, err := provider.LanguageModel(ctx, "claude-3-5-sonnet-20241022")
	require.NoError(t, err)

	adapter := NewAdapter(languageModel)
	assert.Implements(t, (*model.LLM)(nil), adapter)
}

func TestOpenAIProvider_Implements_ModelLLM(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	ctx := t.Context()
	provider, err := openai.New(openai.WithAPIKey(apiKey))
	require.NoError(t, err)

	languageModel, err := provider.LanguageModel(ctx, "gpt-4o")
	require.NoError(t, err)

	adapter := NewAdapter(languageModel)
	assert.Implements(t, (*model.LLM)(nil), adapter)
}

func TestGoogleProvider_Implements_ModelLLM(t *testing.T) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}

	ctx := t.Context()
	provider, err := google.New(google.WithGeminiAPIKey(apiKey))
	require.NoError(t, err)

	languageModel, err := provider.LanguageModel(ctx, "gemini-2.0-flash-exp")
	require.NoError(t, err)

	adapter := NewAdapter(languageModel)
	assert.Implements(t, (*model.LLM)(nil), adapter)
}

func TestAzureProvider_Implements_ModelLLM(t *testing.T) {
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	baseURL := os.Getenv("AZURE_OPENAI_BASE_URL")
	if apiKey == "" || baseURL == "" {
		t.Skip("AZURE_OPENAI_API_KEY or AZURE_OPENAI_BASE_URL not set")
	}

	ctx := t.Context()
	provider, err := azure.New(
		azure.WithAPIKey(apiKey),
		azure.WithBaseURL(baseURL),
	)
	require.NoError(t, err)

	languageModel, err := provider.LanguageModel(ctx, "gpt-4o")
	require.NoError(t, err)

	adapter := NewAdapter(languageModel)
	assert.Implements(t, (*model.LLM)(nil), adapter)
}

func TestBedrockProvider_Implements_ModelLLM(t *testing.T) {
	region := os.Getenv("AWS_REGION")
	if region == "" {
		t.Skip("AWS_REGION not set")
	}

	ctx := t.Context()
	provider, err := bedrock.New()
	require.NoError(t, err)

	languageModel, err := provider.LanguageModel(ctx, "anthropic.claude-3-5-sonnet-20241022-v2:0")
	require.NoError(t, err)

	adapter := NewAdapter(languageModel)
	assert.Implements(t, (*model.LLM)(nil), adapter)
}

func TestOpenRouterProvider_Implements_ModelLLM(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY not set")
	}

	ctx := t.Context()
	provider, err := openrouter.New(openrouter.WithAPIKey(apiKey))
	require.NoError(t, err)

	languageModel, err := provider.LanguageModel(ctx, "anthropic/claude-3.5-sonnet")
	require.NoError(t, err)

	adapter := NewAdapter(languageModel)
	assert.Implements(t, (*model.LLM)(nil), adapter)
}

func TestOpenAICompatProvider_Implements_ModelLLM(t *testing.T) {
	apiKey := os.Getenv("OPENAI_COMPAT_API_KEY")
	baseURL := os.Getenv("OPENAI_COMPAT_BASE_URL")
	if apiKey == "" || baseURL == "" {
		t.Skip("OPENAI_COMPAT_API_KEY or OPENAI_COMPAT_BASE_URL not set")
	}

	ctx := t.Context()
	provider, err := openaicompat.New(
		openaicompat.WithAPIKey(apiKey),
		openaicompat.WithBaseURL(baseURL),
	)
	require.NoError(t, err)

	languageModel, err := provider.LanguageModel(ctx, "model-name")
	require.NoError(t, err)

	adapter := NewAdapter(languageModel)
	assert.Implements(t, (*model.LLM)(nil), adapter)
}
