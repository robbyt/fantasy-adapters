package adk

import (
	"context"
	"errors"
	"testing"

	"charm.land/fantasy"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type mockLanguageModel struct {
	provider     string
	modelID      string
	generateFunc func(ctx context.Context, call fantasy.Call) (*fantasy.Response, error)
	streamFunc   func(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error)
}

func (m *mockLanguageModel) Provider() string {
	return m.provider
}

func (m *mockLanguageModel) Model() string {
	return m.modelID
}

func (m *mockLanguageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, call)
	}
	return &fantasy.Response{
		Content: []fantasy.Content{
			fantasy.TextContent{Text: "test response"},
		},
		Usage: fantasy.Usage{
			InputTokens:  10,
			OutputTokens: 20,
			TotalTokens:  30,
		},
		FinishReason: fantasy.FinishReasonStop,
	}, nil
}

func (m *mockLanguageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	if m.streamFunc != nil {
		return m.streamFunc(ctx, call)
	}
	return func(yield func(fantasy.StreamPart) bool) {
		yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextStart,
			ID:   "0",
		})
		yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "test",
		})
		yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextEnd,
			ID:   "0",
		})
		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonStop,
			Usage: fantasy.Usage{
				InputTokens:  10,
				OutputTokens: 4,
				TotalTokens:  14,
			},
		})
	}, nil
}

func TestNewAdapter(t *testing.T) {
	mock := &mockLanguageModel{
		provider: "test-provider",
		modelID:  "test-model",
	}

	adapter := NewAdapter(mock)
	if adapter == nil {
		t.Fatal("NewAdapter returned nil")
	}

	adapterImpl, ok := adapter.(*Adapter)
	if !ok {
		t.Fatal("NewAdapter did not return *Adapter")
	}

	if adapterImpl.model != mock {
		t.Error("Adapter model not set correctly")
	}
}

func TestAdapter_Name(t *testing.T) {
	mock := &mockLanguageModel{
		provider: "test-provider",
		modelID:  "test-model",
	}

	adapter := &Adapter{model: mock}
	name := adapter.Name()

	expected := "test-provider/test-model"
	if name != expected {
		t.Errorf("Name() = %q, want %q", name, expected)
	}
}

func TestAdapter_GenerateContent_NonStreaming(t *testing.T) {
	mock := &mockLanguageModel{
		provider: "test",
		modelID:  "model",
	}

	adapter := &Adapter{model: mock}
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

	ctx := context.Background()
	iter := adapter.GenerateContent(ctx, req, false)

	var responses []*model.LLMResponse
	var errs []error
	for resp, err := range iter {
		responses = append(responses, resp)
		errs = append(errs, err)
	}

	if len(responses) != 1 {
		t.Fatalf("expected 1 response, got %d", len(responses))
	}

	if errs[0] != nil {
		t.Errorf("unexpected error: %v", errs[0])
	}

	resp := responses[0]
	if resp.Content == nil {
		t.Fatal("response Content is nil")
	}

	if len(resp.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(resp.Content.Parts))
	}

	if resp.Content.Parts[0].Text != "test response" {
		t.Errorf("unexpected text: %q", resp.Content.Parts[0].Text)
	}

	if !resp.TurnComplete {
		t.Error("TurnComplete should be true")
	}

	if resp.Partial {
		t.Error("Partial should be false")
	}
}

func TestAdapter_GenerateContent_Streaming(t *testing.T) {
	mock := &mockLanguageModel{
		provider: "test",
		modelID:  "model",
	}

	adapter := &Adapter{model: mock}
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

	ctx := context.Background()
	iter := adapter.GenerateContent(ctx, req, true)

	var responses []*model.LLMResponse
	for resp, err := range iter {
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		responses = append(responses, resp)
	}

	if len(responses) < 1 {
		t.Fatal("expected at least 1 response")
	}

	finalResp := responses[len(responses)-1]
	if !finalResp.TurnComplete {
		t.Error("final response TurnComplete should be true")
	}
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
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if call.Temperature == nil {
		t.Fatal("Temperature is nil")
	}
	const epsilon = 0.0001
	if *call.Temperature < 0.7-epsilon || *call.Temperature > 0.7+epsilon {
		t.Errorf("Temperature = %f, want ~0.7", *call.Temperature)
	}

	if call.TopP == nil {
		t.Fatal("TopP is nil")
	}
	if *call.TopP < 0.9-epsilon || *call.TopP > 0.9+epsilon {
		t.Errorf("TopP = %f, want ~0.9", *call.TopP)
	}

	if call.TopK == nil || *call.TopK != 40 {
		t.Error("TopK not set correctly")
	}

	if call.MaxOutputTokens == nil || *call.MaxOutputTokens != 100 {
		t.Error("MaxOutputTokens not set correctly")
	}

	if len(call.Prompt) != 1 {
		t.Fatalf("expected 1 message, got %d", len(call.Prompt))
	}
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
			if err == nil {
				t.Fatal("expected error, got nil")
			}

			if !errors.Is(err, errors.New(tt.errMsg)) && err.Error() != tt.errMsg {
				t.Errorf("expected error containing %q, got %v", tt.errMsg, err)
			}
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
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(msg.Content) != tt.expectParts {
				t.Errorf("expected %d parts, got %d", tt.expectParts, len(msg.Content))
			}

			if msg.Role != tt.role {
				t.Errorf("expected role %v, got %v", tt.role, msg.Role)
			}
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

	if llmResp.Content == nil {
		t.Fatal("Content is nil")
	}

	if len(llmResp.Content.Parts) != 3 {
		t.Fatalf("expected 3 parts, got %d", len(llmResp.Content.Parts))
	}

	if llmResp.Content.Parts[0].Text != "hello" {
		t.Error("text part not converted correctly")
	}

	if llmResp.Content.Parts[1].FunctionCall == nil {
		t.Fatal("function call part is nil")
	}

	if llmResp.Content.Parts[2].Text != "thinking" || !llmResp.Content.Parts[2].Thought {
		t.Error("reasoning part not converted correctly")
	}

	if llmResp.UsageMetadata == nil {
		t.Fatal("UsageMetadata is nil")
	}

	if llmResp.UsageMetadata.PromptTokenCount != 10 {
		t.Errorf("PromptTokenCount = %d, want 10", llmResp.UsageMetadata.PromptTokenCount)
	}

	if !llmResp.TurnComplete {
		t.Error("TurnComplete should be true")
	}

	if llmResp.Partial {
		t.Error("Partial should be false")
	}

	if llmResp.FinishReason != genai.FinishReasonStop {
		t.Errorf("FinishReason = %v, want %v", llmResp.FinishReason, genai.FinishReasonStop)
	}
}

func TestFantasyStreamToLLM(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextStart,
			ID:   "0",
		})
		yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: "hello",
		})
		yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			ID:    "0",
			Delta: " world",
		})
		yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeTextEnd,
			ID:   "0",
		})
		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonStop,
			Usage: fantasy.Usage{
				InputTokens:  5,
				OutputTokens: 2,
				TotalTokens:  7,
			},
		})
	}

	iter := fantasyStreamToLLM(stream)

	var responses []*model.LLMResponse
	for resp, err := range iter {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		responses = append(responses, resp)
	}

	if len(responses) < 1 {
		t.Fatal("expected at least 1 response")
	}

	finalResp := responses[len(responses)-1]
	if !finalResp.TurnComplete {
		t.Error("final response should have TurnComplete=true")
	}

	if finalResp.UsageMetadata == nil {
		t.Fatal("final response UsageMetadata is nil")
	}
}

func TestFantasyStreamToLLM_WithReasoning(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeReasoningStart,
			ID:   "0",
		})
		yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeReasoningDelta,
			ID:    "0",
			Delta: "thinking...",
		})
		yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeReasoningEnd,
			ID:   "0",
		})
		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonStop,
			Usage:        fantasy.Usage{},
		})
	}

	iter := fantasyStreamToLLM(stream)

	for resp, err := range iter {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
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
			if resp.ErrorCode != "ERROR" {
				t.Errorf("ErrorCode = %q, want ERROR", resp.ErrorCode)
			}
			if resp.ErrorMessage != testErr.Error() {
				t.Errorf("ErrorMessage = %q, want %q", resp.ErrorMessage, testErr.Error())
			}
		}
	}

	if gotError == nil {
		t.Error("expected error, got nil")
	}
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
			if result != tt.genai {
				t.Errorf("fantasyFinishReasonToGenai(%q) = %v, want %v", tt.fantasy, result, tt.genai)
			}
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
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(fantasyTools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(fantasyTools))
	}

	ft, ok := fantasyTools[0].(fantasy.FunctionTool)
	if !ok {
		t.Fatal("tool is not FunctionTool")
	}

	if ft.Name != "test-function" {
		t.Errorf("Name = %q, want test-function", ft.Name)
	}
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
	if err == nil {
		t.Fatal("expected error for unsupported tool, got nil")
	}
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
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(call.Prompt) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(call.Prompt))
	}

	if call.Prompt[0].Role != fantasy.MessageRoleSystem {
		t.Error("first message should be system")
	}
}

func TestLlmRequestToFantasyCall_ToolChoice(t *testing.T) {
	tests := []struct {
		name       string
		config     *genai.FunctionCallingConfig
		wantChoice *fantasy.ToolChoice
		wantErr    bool
	}{
		{
			name: "AUTO mode",
			config: &genai.FunctionCallingConfig{
				Mode: "AUTO",
			},
			wantChoice: ptrToolChoice(fantasy.ToolChoiceAuto),
		},
		{
			name: "ANY mode",
			config: &genai.FunctionCallingConfig{
				Mode: "ANY",
			},
			wantChoice: ptrToolChoice(fantasy.ToolChoiceRequired),
		},
		{
			name: "NONE mode",
			config: &genai.FunctionCallingConfig{
				Mode: "NONE",
			},
			wantChoice: ptrToolChoice(fantasy.ToolChoiceNone),
		},
		{
			name: "single allowed function",
			config: &genai.FunctionCallingConfig{
				AllowedFunctionNames: []string{"test-func"},
			},
			wantChoice: ptrToolChoice(fantasy.ToolChoice("test-func")),
		},
		{
			name: "multiple allowed functions",
			config: &genai.FunctionCallingConfig{
				AllowedFunctionNames: []string{"func1", "func2"},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: tt.config,
					},
				},
			}

			call, err := llmRequestToFantasyCall(req)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.wantChoice != nil {
				if call.ToolChoice == nil {
					t.Fatal("ToolChoice is nil")
				}
				if *call.ToolChoice != *tt.wantChoice {
					t.Errorf("ToolChoice = %v, want %v", *call.ToolChoice, *tt.wantChoice)
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
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(msg.Content) != 1 {
		t.Fatalf("expected 1 part, got %d", len(msg.Content))
	}

	toolCall, ok := msg.Content[0].(fantasy.ToolCallPart)
	if !ok {
		t.Fatal("part is not ToolCallPart")
	}

	if toolCall.ToolCallID != "call-123" {
		t.Errorf("ToolCallID = %q, want call-123", toolCall.ToolCallID)
	}
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
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(msg.Content) != 1 {
		t.Fatalf("expected 1 part, got %d", len(msg.Content))
	}

	toolResult, ok := msg.Content[0].(fantasy.ToolResultPart)
	if !ok {
		t.Fatal("part is not ToolResultPart")
	}

	if toolResult.ToolCallID != "call-123" {
		t.Errorf("ToolCallID = %q, want call-123", toolResult.ToolCallID)
	}
}

func TestGenaiContentToFantasyMessage_ExecutableCode(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{ExecutableCode: &genai.ExecutableCode{}},
		},
	}

	_, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	if err == nil {
		t.Fatal("expected error for executable code, got nil")
	}
}

func TestGenaiContentToFantasyMessage_VideoMetadata(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{VideoMetadata: &genai.VideoMetadata{}},
		},
	}

	_, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	if err == nil {
		t.Fatal("expected error for video metadata, got nil")
	}
}

func TestAdapter_GenerateContent_RequestError(t *testing.T) {
	mock := &mockLanguageModel{
		provider: "test",
		modelID:  "model",
	}

	adapter := &Adapter{model: mock}
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			SafetySettings: []*genai.SafetySetting{{}},
		},
	}

	ctx := context.Background()
	iter := adapter.GenerateContent(ctx, req, false)

	var gotErr error
	for _, err := range iter {
		gotErr = err
	}

	if gotErr == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestAdapter_GenerateContent_GenerateError(t *testing.T) {
	testErr := errors.New("generate error")
	mock := &mockLanguageModel{
		provider: "test",
		modelID:  "model",
		generateFunc: func(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
			return nil, testErr
		},
	}

	adapter := &Adapter{model: mock}
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

	ctx := context.Background()
	iter := adapter.GenerateContent(ctx, req, false)

	var gotErr error
	for _, err := range iter {
		gotErr = err
	}

	if gotErr == nil {
		t.Fatal("expected error, got nil")
	}

	if !errors.Is(gotErr, testErr) {
		t.Errorf("expected error %v, got %v", testErr, gotErr)
	}
}

func TestAdapter_GenerateContent_StreamError(t *testing.T) {
	testErr := errors.New("stream error")
	mock := &mockLanguageModel{
		provider: "test",
		modelID:  "model",
		streamFunc: func(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
			return nil, testErr
		},
	}

	adapter := &Adapter{model: mock}
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

	ctx := context.Background()
	iter := adapter.GenerateContent(ctx, req, true)

	var gotErr error
	for _, err := range iter {
		gotErr = err
	}

	if gotErr == nil {
		t.Fatal("expected error, got nil")
	}

	if !errors.Is(gotErr, testErr) {
		t.Errorf("expected error %v, got %v", testErr, gotErr)
	}
}

func TestFantasyStreamToLLM_ToolCalls(t *testing.T) {
	stream := func(yield func(fantasy.StreamPart) bool) {
		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           "call-123",
			ToolCallName: "test-func",
		})
		yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolInputDelta,
			ID:            "call-123",
			ToolCallInput: `{"arg":"value"}`,
		})
		yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   "call-123",
		})
		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonToolCalls,
			Usage:        fantasy.Usage{},
		})
	}

	iter := fantasyStreamToLLM(stream)

	var responses []*model.LLMResponse
	for resp, err := range iter {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		responses = append(responses, resp)
	}

	if len(responses) < 1 {
		t.Fatal("expected at least 1 response")
	}
}

func TestFantasyFinishReasonToGenai_Default(t *testing.T) {
	result := fantasyFinishReasonToGenai(fantasy.FinishReason("invalid"))
	if result != genai.FinishReasonUnspecified {
		t.Errorf("unexpected finish reason: %v", result)
	}
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
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if call.PresencePenalty == nil {
		t.Fatal("PresencePenalty is nil")
	}
	const epsilon = 0.0001
	if *call.PresencePenalty < 0.5-epsilon || *call.PresencePenalty > 0.5+epsilon {
		t.Errorf("PresencePenalty = %f, want ~0.5", *call.PresencePenalty)
	}

	if call.FrequencyPenalty == nil {
		t.Fatal("FrequencyPenalty is nil")
	}
	if *call.FrequencyPenalty < 0.3-epsilon || *call.FrequencyPenalty > 0.3+epsilon {
		t.Errorf("FrequencyPenalty = %f, want ~0.3", *call.FrequencyPenalty)
	}
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
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(call.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(call.Tools))
	}
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
	if err == nil {
		t.Fatal("expected error for VALIDATED mode, got nil")
	}
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
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(call.Prompt) != 1 {
		t.Fatalf("expected 1 message, got %d", len(call.Prompt))
	}

	if call.Prompt[0].Role != fantasy.MessageRoleAssistant {
		t.Errorf("expected MessageRoleAssistant, got %v", call.Prompt[0].Role)
	}
}

func TestGenaiContentToFantasyMessage_CodeExecutionResult(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{CodeExecutionResult: &genai.CodeExecutionResult{}},
		},
	}

	_, err := genaiContentToFantasyMessage(content, fantasy.MessageRoleUser)
	if err == nil {
		t.Fatal("expected error for code execution result, got nil")
	}
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
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
}

func TestGenaiToolsToFantasyTools_WithParameters(t *testing.T) {
	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        "test-function",
					Description: "test description",
					Parameters:  &genai.Schema{Type: "object"},
				},
			},
		},
	}

	_, err := genaiToolsToFantasyTools(tools)
	if err == nil {
		t.Fatal("expected error for schema conversion, got nil")
	}
}
