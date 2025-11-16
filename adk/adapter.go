// Package adk provides an adapter to use fantasy providers with the Google ADK.
//
// The adapter implements google.golang.org/adk/model.LLM interface, allowing any
// Fantasy provider (Anthropic, OpenAI, Google, Azure, etc.) to be used with ADK agents.
//
// # Key Conversions
//
// ADK -> Fantasy:
//   - LLMRequest.Config (Temperature, TopP, TopK, MaxOutputTokens) -> fantasy.Call parameters
//   - LLMRequest.Config.SystemInstruction -> fantasy.Message with MessageRoleSystem
//   - LLMRequest.Contents -> fantasy.Prompt messages
//   - genai.Tool.FunctionDeclarations -> fantasy.FunctionTool
//   - genai.Part.Text -> fantasy.TextPart
//   - genai.Part.InlineData -> fantasy.FilePart
//   - genai.Part.FunctionCall -> fantasy.ToolCallPart
//   - genai.Part.FunctionResponse -> fantasy.ToolResultPart
//   - genai.Part with Thought:true -> fantasy.ReasoningPart
//
// Fantasy -> ADK:
//   - fantasy.Response.Content -> genai.Content.Parts
//   - fantasy.TextContent -> genai.Part{Text}
//   - fantasy.ToolCallContent -> genai.Part{FunctionCall}
//   - fantasy.ReasoningContent -> genai.Part{Text, Thought:true}
//   - fantasy.Usage -> LLMResponse.UsageMetadata
//   - fantasy.FinishReason -> genai.FinishReason (see fantasyFinishReasonToGenai)
package adk

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"strings"

	"charm.land/fantasy"
	anthropic "charm.land/fantasy/providers/anthropic"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Adapter wraps a fantasy.LanguageModel to implement the Google ADK model.LLM interface.
type Adapter struct {
	model           fantasy.LanguageModel
	schemaConverter SchemaConverter
}

const (
	ErrorCodeGeneric   = "ERROR"
	ErrorCodeUnmarshal = "UNMARSHAL_ERROR"
	RoleModel          = "model"

	ToolModeAuto      = "AUTO"
	ToolModeAny       = "ANY"
	ToolModeNone      = "NONE"
	ToolModeValidated = "VALIDATED"
)

// NewAdapter creates a new ADK adapter for the given fantasy language model.
// Uses manual schema conversion by default for optimal performance (~5.6x faster than JSON).
func NewAdapter(m fantasy.LanguageModel) model.LLM {
	return &Adapter{
		model:           m,
		schemaConverter: NewManualSchemaConverter(),
	}
}

// NewAdapterWithSchemaConverter creates an Adapter with a custom schema converter.
// Use NewJSONSchemaConverter() for maintainability at the cost of performance (~5.6x slower).
// Example: NewAdapterWithSchemaConverter(model, NewJSONSchemaConverter())
func NewAdapterWithSchemaConverter(m fantasy.LanguageModel, converter SchemaConverter) model.LLM {
	return &Adapter{
		model:           m,
		schemaConverter: converter,
	}
}

// Name implements model.LLM.
func (a *Adapter) Name() string {
	return fmt.Sprintf("%s/%s", a.model.Provider(), a.model.Model())
}

// GenerateContent implements model.LLM.
//
// Returns an iterator of LLMResponse structs. All fields are populated:
//   - Content: Maps fantasy content types (Text, ToolCall, Reasoning) to genai.Content
//   - UsageMetadata: Token counts from fantasy.Usage
//   - FinishReason: Mapped from fantasy finish reasons (stop->STOP, length->MAX_TOKENS, etc.)
//   - Partial: true during streaming text deltas, false on completion
//   - TurnComplete: false during streaming, true on finish or error
//   - ErrorCode/ErrorMessage: Populated when errors occur
//   - CitationMetadata, GroundingMetadata, LogprobsResult: nil (not supported by fantasy)
//   - CustomMetadata: nil (no fantasy equivalent)
//   - Interrupted: false (not tracked)
//   - AvgLogprobs: 0 (not supported by fantasy)
func (a *Adapter) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	call, err := a.llmRequestToFantasyCall(req)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, err)
		}
	}

	if stream {
		streamResp, err := a.model.Stream(ctx, call)
		if err != nil {
			return func(yield func(*model.LLMResponse, error) bool) {
				yield(nil, err)
			}
		}
		return fantasyStreamToLLM(streamResp)
	}

	resp, err := a.model.Generate(ctx, call)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, err)
		}
	}

	llmResp := fantasyResponseToLLM(resp)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(llmResp, nil)
	}
}

// applyLLMConfig applies configuration parameters from ADK's GenerateContentConfig to a fantasy.Call.
// Returns errors for unsupported configuration options.
func applyLLMConfig(call *fantasy.Call, config *genai.GenerateContentConfig) []error {
	if config == nil {
		return nil
	}

	var errs []error

	// Apply scalar configuration parameters
	if config.Temperature != nil {
		temp := float64(*config.Temperature)
		call.Temperature = &temp
	}
	if config.TopP != nil {
		topP := float64(*config.TopP)
		call.TopP = &topP
	}
	if config.TopK != nil {
		topK := int64(*config.TopK)
		call.TopK = &topK
	}
	if config.MaxOutputTokens > 0 {
		tokens := int64(config.MaxOutputTokens)
		call.MaxOutputTokens = &tokens
	}
	if config.PresencePenalty != nil {
		pp := float64(*config.PresencePenalty)
		call.PresencePenalty = &pp
	}
	if config.FrequencyPenalty != nil {
		fp := float64(*config.FrequencyPenalty)
		call.FrequencyPenalty = &fp
	}

	// Detect unsupported features
	if len(config.SafetySettings) > 0 {
		errs = append(errs, errors.New("safety settings not supported"))
	}
	if config.ResponseMIMEType != "" {
		errs = append(errs, errors.New("response MIME type not supported"))
	}
	if config.ResponseSchema != nil || config.ResponseJsonSchema != nil {
		errs = append(errs, errors.New("response schema not supported"))
	}
	if config.ThinkingConfig != nil {
		errs = append(errs, errors.New("thinking config not supported (use provider-specific options)"))
	}
	if config.CachedContent != "" {
		errs = append(errs, errors.New("cached content not supported"))
	}

	return errs
}

// extractToolNames builds a map of tool names from a list of fantasy.Tool.
// Only FunctionTool types are included in the map.
func extractToolNames(tools []fantasy.Tool) map[string]bool {
	names := make(map[string]bool)
	for _, tool := range tools {
		if ft, ok := tool.(fantasy.FunctionTool); ok {
			names[ft.Name] = true
		}
	}
	return names
}

// filterToolsByNames returns a new slice containing only the tools whose names
// are in the allowed list. Order is preserved based on the original tools slice.
func filterToolsByNames(tools []fantasy.Tool, allowed []string) []fantasy.Tool {
	allowedSet := make(map[string]bool)
	for _, name := range allowed {
		allowedSet[name] = true
	}

	filtered := make([]fantasy.Tool, 0, len(allowed))
	for _, tool := range tools {
		if ft, ok := tool.(fantasy.FunctionTool); ok {
			if allowedSet[ft.Name] {
				filtered = append(filtered, tool)
			}
		}
	}
	return filtered
}

// getToolChoiceForMode converts ADK's tool calling mode to a fantasy.ToolChoice.
// Special case: if exactly one function is allowed and mode is not NONE, that specific
// function is returned as the choice.
func getToolChoiceForMode(mode string, allowedNames []string) (*fantasy.ToolChoice, error) {
	// Single allowed function forces that tool (unless mode is None)
	if len(allowedNames) == 1 && mode != ToolModeNone {
		tc := fantasy.ToolChoice(allowedNames[0])
		return &tc, nil
	}

	switch mode {
	case ToolModeAuto:
		tc := fantasy.ToolChoiceAuto
		return &tc, nil
	case ToolModeAny:
		tc := fantasy.ToolChoiceRequired
		return &tc, nil
	case ToolModeNone:
		tc := fantasy.ToolChoiceNone
		return &tc, nil
	case ToolModeValidated:
		return nil, errors.New("validated tool mode not supported")
	case "":
		return nil, nil
	default:
		return nil, fmt.Errorf("unsupported tool calling mode: %q", mode)
	}
}

// applyTools converts ADK tools and tool configuration to fantasy.Call tools and tool choice.
// Handles tool filtering by AllowedFunctionNames and tool choice mode selection.
func (a *Adapter) applyTools(call *fantasy.Call, tools []*genai.Tool, toolConfig *genai.ToolConfig) []error {
	var errs []error

	if len(tools) > 0 {
		fantasyTools, err := a.genaiToolsToFantasyTools(tools)
		if err != nil {
			errs = append(errs, fmt.Errorf("tools: %w", err))
		} else {
			call.Tools = fantasyTools
		}
	}

	if toolConfig != nil && toolConfig.FunctionCallingConfig != nil {
		fc := toolConfig.FunctionCallingConfig

		// Filter tools by AllowedFunctionNames if specified
		if len(fc.AllowedFunctionNames) > 0 {
			// Validate that all allowed names exist in the tools list
			toolNames := extractToolNames(call.Tools)
			for _, allowedName := range fc.AllowedFunctionNames {
				if !toolNames[allowedName] {
					errs = append(errs, fmt.Errorf("allowed function %q not found in tools list", allowedName))
				}
			}

			// Filter tools to only include allowed ones
			call.Tools = filterToolsByNames(call.Tools, fc.AllowedFunctionNames)
		}

		// Set ToolChoice based on Mode
		toolChoice, err := getToolChoiceForMode(string(fc.Mode), fc.AllowedFunctionNames)
		if err != nil {
			errs = append(errs, err)
		} else if toolChoice != nil {
			call.ToolChoice = toolChoice
		}
	}

	return errs
}

// llmRequestToFantasyCall converts an ADK LLMRequest to a fantasy.Call.
//
// Converts:
//   - Config.Temperature/TopP/TopK/MaxOutputTokens -> Call parameters
//   - Config.PresencePenalty/FrequencyPenalty -> Call parameters
//   - Config.SystemInstruction -> System message in Prompt
//   - Contents -> Prompt messages
//   - Config.Tools -> Call.Tools (filtered by AllowedFunctionNames if specified)
//   - Config.ToolConfig.FunctionCallingConfig.Mode -> Call.ToolChoice
//   - Config.ToolConfig.FunctionCallingConfig.AllowedFunctionNames -> filters Call.Tools
//
// AllowedFunctionNames handling:
//   - Validates all allowed names exist in the tools list (returns error if not)
//   - Filters Call.Tools to only include allowed functions
//   - Single allowed function + Mode=AUTO/ANY -> forces that specific tool
//   - Multiple allowed functions + Mode=AUTO -> ToolChoiceAuto
//   - Multiple allowed functions + Mode=ANY -> ToolChoiceRequired
//
// Returns errors for unsupported ADK features:
//   - SafetySettings, ResponseMIMEType, ResponseSchema
//   - ThinkingConfig (use provider-specific options instead)
//   - CachedContent, FileData URIs
//   - Retrieval/code execution tools
//   - AllowedFunctionNames containing non-existent function names
func (a *Adapter) llmRequestToFantasyCall(req *model.LLMRequest) (fantasy.Call, error) {
	var call fantasy.Call
	var errs []error

	if req.Config != nil {
		// Apply configuration parameters
		errs = append(errs, applyLLMConfig(&call, req.Config)...)

		// Handle system instruction
		if req.Config.SystemInstruction != nil {
			systemMsg, err := genaiSystemInstructionToFantasyMessage(req.Config.SystemInstruction)
			if err != nil {
				errs = append(errs, fmt.Errorf("system instruction: %w", err))
			} else {
				call.Prompt = append(call.Prompt, systemMsg)
			}
		}

		// Apply tools and tool configuration
		errs = append(errs, a.applyTools(&call, req.Config.Tools, req.Config.ToolConfig)...)
	}

	for _, content := range req.Contents {
		msg, err := genaiContentToFantasyMessage(content)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		call.Prompt = append(call.Prompt, msg)
	}

	return call, errors.Join(errs...)
}

// genaiContentToFantasyMessage converts genai.Content to fantasy.Message.
//
// Determines the message role based on content:
//   - Content with FunctionResponse parts -> MessageRoleTool
//   - Content with Role="model" -> MessageRoleAssistant
//   - Otherwise -> MessageRoleUser
//
// Maps genai.Part types to fantasy message parts:
//   - Part.Text with Thought:true -> ReasoningPart
//   - Part.Text -> TextPart
//   - Part.InlineData -> FilePart
//   - Part.FunctionCall -> ToolCallPart
//   - Part.FunctionResponse -> ToolResultPart
//
// Returns errors for unsupported part types:
//   - FileData (URI references)
//   - ExecutableCode, CodeExecutionResult
//   - VideoMetadata
//
// Returns error if content contains both FunctionCall and FunctionResponse parts.
func genaiContentToFantasyMessage(content *genai.Content) (fantasy.Message, error) {
	var msg fantasy.Message
	var errs []error

	// Determine role based on content
	hasFunctionCall := false
	hasFunctionResponse := false
	for _, part := range content.Parts {
		if part.FunctionCall != nil {
			hasFunctionCall = true
		}
		if part.FunctionResponse != nil {
			hasFunctionResponse = true
		}
	}

	if hasFunctionCall && hasFunctionResponse {
		return msg, errors.New("content cannot contain both FunctionCall and FunctionResponse parts")
	}

	if hasFunctionResponse {
		msg.Role = fantasy.MessageRoleTool
	} else if content.Role == RoleModel {
		msg.Role = fantasy.MessageRoleAssistant
	} else {
		msg.Role = fantasy.MessageRoleUser
	}

	for _, part := range content.Parts {
		if part.Text != "" && part.Thought {
			reasoningPart := fantasy.ReasoningPart{Text: part.Text}
			// Preserve signature for multi-turn context
			if len(part.ThoughtSignature) > 0 {
				reasoningPart.ProviderOptions = fantasy.ProviderOptions{
					"anthropic": &anthropic.ReasoningOptionMetadata{
						Signature: string(part.ThoughtSignature),
					},
				}
			}
			msg.Content = append(msg.Content, reasoningPart)
		} else if part.Text != "" {
			msg.Content = append(msg.Content, fantasy.TextPart{Text: part.Text})
		} else if part.InlineData != nil {
			msg.Content = append(msg.Content, fantasy.FilePart{
				Data:      part.InlineData.Data,
				MediaType: part.InlineData.MIMEType,
			})
		} else if part.FileData != nil {
			errs = append(errs, errors.New("file data (URI references) not supported"))
		} else if part.FunctionCall != nil {
			input := ""
			if part.FunctionCall.Args != nil {
				jsonBytes, err := json.Marshal(part.FunctionCall.Args)
				if err != nil {
					errs = append(errs, fmt.Errorf("failed to marshal function call args: %w", err))
				} else {
					input = string(jsonBytes)
				}
			}
			msg.Content = append(msg.Content, fantasy.ToolCallPart{
				ToolCallID: part.FunctionCall.ID,
				ToolName:   part.FunctionCall.Name,
				Input:      input,
			})
		} else if part.FunctionResponse != nil {
			output := fantasy.ToolResultOutputContentText{Text: ""}
			if part.FunctionResponse.Response != nil {
				jsonBytes, err := json.Marshal(part.FunctionResponse.Response)
				if err != nil {
					errs = append(errs, fmt.Errorf("failed to marshal function response: %w", err))
				} else {
					output = fantasy.ToolResultOutputContentText{Text: string(jsonBytes)}
				}
			}
			msg.Content = append(msg.Content, fantasy.ToolResultPart{
				ToolCallID: part.FunctionResponse.ID,
				Output:     output,
			})
		} else if part.ExecutableCode != nil || part.CodeExecutionResult != nil {
			errs = append(errs, errors.New("code execution not supported"))
		} else if part.VideoMetadata != nil {
			errs = append(errs, errors.New("video metadata not supported"))
		}
	}

	return msg, errors.Join(errs...)
}

// genaiSystemInstructionToFantasyMessage converts a genai.Content to a fantasy.Message
// with MessageRoleSystem role, regardless of the content's role field.
// This is used specifically for system instructions.
func genaiSystemInstructionToFantasyMessage(content *genai.Content) (fantasy.Message, error) {
	msg, err := genaiContentToFantasyMessage(content)
	if err != nil {
		return msg, err
	}
	msg.Role = fantasy.MessageRoleSystem
	return msg, nil
}

// normalizeStringArrayField converts a []interface{} field to []string for the given field name.
// This is necessary for JSON Schema fields that must be string arrays (required, enum, propertyOrdering).
func normalizeStringArrayField(schema map[string]any, fieldName string) {
	if field, ok := schema[fieldName]; ok && field != nil {
		if fieldSlice, ok := field.([]interface{}); ok {
			strSlice := make([]string, 0, len(fieldSlice))
			for _, item := range fieldSlice {
				if str, ok := item.(string); ok {
					strSlice = append(strSlice, str)
				}
			}
			schema[fieldName] = strSlice
		}
	}
}

// normalizeSchemaArrays converts []interface{} to []string for schema fields
// that must be string arrays per JSON Schema spec (required, enum, propertyOrdering).
// Also recursively normalizes nested schemas in properties, items, and anyOf.
// This handles type mismatches from JSON marshal/unmarshal round-trips.
func normalizeSchemaArrays(schema map[string]any) {
	// Normalize string array fields
	normalizeStringArrayField(schema, "required")
	normalizeStringArrayField(schema, "enum")
	normalizeStringArrayField(schema, "propertyOrdering")

	// Recursively normalize nested properties
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, prop := range props {
			if propSchema, ok := prop.(map[string]any); ok {
				normalizeSchemaArrays(propSchema)
			}
		}
	}

	// Normalize array items schema
	if items, ok := schema["items"].(map[string]any); ok {
		normalizeSchemaArrays(items)
	}

	// Recursively normalize anyOf schemas
	if anyOf, ok := schema["anyOf"].([]interface{}); ok {
		for _, item := range anyOf {
			if subSchema, ok := item.(map[string]any); ok {
				normalizeSchemaArrays(subSchema)
			}
		}
	}
}

func (a *Adapter) genaiToolsToFantasyTools(tools []*genai.Tool) ([]fantasy.Tool, error) {
	var fantasyTools []fantasy.Tool
	var errs []error

	slog.Default().Debug("ADK->Fantasy tool conversion starting", "tool_count", len(tools))

	for _, tool := range tools {
		if len(tool.FunctionDeclarations) > 0 {
			for _, fn := range tool.FunctionDeclarations {
				slog.Default().Debug("Converting tool schema",
					"tool_name", fn.Name,
					"description", fn.Description,
					"has_parameters", fn.Parameters != nil,
					"has_json_schema", fn.ParametersJsonSchema != nil)

				params := make(map[string]any)
				if fn.Parameters != nil {
					slog.Default().Debug("Using genai.Schema Parameters", "tool_name", fn.Name)
					schemaMap, err := a.schemaConverter.Convert(fn.Parameters)
					if err != nil {
						errs = append(errs, fmt.Errorf("failed to convert schema for function %q: %w", fn.Name, err))
					} else if schemaMap != nil {
						params = schemaMap
					}
				} else if fn.ParametersJsonSchema != nil {
					slog.Default().Debug("Using ParametersJsonSchema",
						"tool_name", fn.Name,
						"schema_type", fmt.Sprintf("%T", fn.ParametersJsonSchema))
					switch v := fn.ParametersJsonSchema.(type) {
					case map[string]any:
						slog.Default().Debug("ParametersJsonSchema is map",
							"tool_name", fn.Name,
							"raw_schema", v)
						params = v
					case []byte:
						if err := json.Unmarshal(v, &params); err != nil {
							errs = append(errs, fmt.Errorf("failed to unmarshal ParametersJsonSchema for function %q: %w", fn.Name, err))
						}
					default:
						jsonBytes, err := json.Marshal(v)
						if err != nil {
							errs = append(errs, fmt.Errorf("failed to marshal ParametersJsonSchema for function %q: %w", fn.Name, err))
						} else if err := json.Unmarshal(jsonBytes, &params); err != nil {
							errs = append(errs, fmt.Errorf("failed to unmarshal ParametersJsonSchema for function %q: %w", fn.Name, err))
						}
					}
					// Normalize array fields to ensure []string type for strict API compatibility
					normalizeSchemaArrays(params)
				}
				ft := fantasy.FunctionTool{
					Name:        fn.Name,
					Description: fn.Description,
					InputSchema: params,
				}
				slog.Default().Debug("Fantasy tool created",
					"name", ft.Name,
					"description", ft.Description,
					"schema", ft.InputSchema)
				fantasyTools = append(fantasyTools, ft)
			}
		}

		if tool.Retrieval != nil || tool.GoogleSearchRetrieval != nil {
			errs = append(errs, errors.New("retrieval tools not supported"))
		}
		if tool.CodeExecution != nil {
			errs = append(errs, errors.New("code execution not supported"))
		}
	}

	return fantasyTools, errors.Join(errs...)
}

// fantasyResponseToLLM converts fantasy.Response to model.LLMResponse.
//
// Maps fantasy content types to genai.Content.Parts:
//   - TextContent -> Part{Text}
//   - ToolCallContent -> Part{FunctionCall}
//   - ReasoningContent -> Part{Text, Thought:true}
//
// All LLMResponse fields are explicitly populated:
//   - Content, UsageMetadata, FinishReason: mapped from fantasy
//   - Partial: false, TurnComplete: true
//   - CitationMetadata, GroundingMetadata, LogprobsResult: nil (not supported)
//   - CustomMetadata: nil, Interrupted: false, AvgLogprobs: 0
func fantasyResponseToLLM(resp *fantasy.Response) *model.LLMResponse {
	slog.Default().Debug("Fantasy->ADK response conversion",
		"content_count", len(resp.Content),
		"finish_reason", resp.FinishReason)

	llmResp := &model.LLMResponse{
		Content: &genai.Content{
			Role:  RoleModel,
			Parts: make([]*genai.Part, 0, len(resp.Content)),
		},
		CitationMetadata:  nil,
		GroundingMetadata: nil,
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        int32(resp.Usage.InputTokens),
			CandidatesTokenCount:    int32(resp.Usage.OutputTokens),
			TotalTokenCount:         int32(resp.Usage.TotalTokens),
			CachedContentTokenCount: int32(resp.Usage.CacheReadTokens),
		},
		CustomMetadata: nil,
		LogprobsResult: nil,
		Partial:        false,
		TurnComplete:   true,
		Interrupted:    false,
		ErrorCode:      "",
		ErrorMessage:   "",
		FinishReason:   fantasyFinishReasonToGenai(resp.FinishReason),
		AvgLogprobs:    0,
	}

	for _, content := range resp.Content {
		switch c := content.(type) {
		case fantasy.TextContent:
			slog.Default().Debug("Converting TextContent", "text_length", len(c.Text))
			llmResp.Content.Parts = append(llmResp.Content.Parts, &genai.Part{
				Text: c.Text,
			})
		case fantasy.ToolCallContent:
			slog.Default().Debug("Converting ToolCallContent",
				"tool_name", c.ToolName,
				"tool_call_id", c.ToolCallID,
				"input_length", len(c.Input))
			args := make(map[string]any)
			if c.Input != "" {
				if err := json.Unmarshal([]byte(c.Input), &args); err != nil {
					slog.Default().Error("Failed to unmarshal tool input",
						"tool_name", c.ToolName,
						"input", c.Input,
						"error", err)
					llmResp.ErrorCode = ErrorCodeUnmarshal
					llmResp.ErrorMessage = fmt.Sprintf("failed to unmarshal tool call input: %v", err)
				}
			}
			slog.Default().Debug("Tool call args parsed", "tool_name", c.ToolName, "args", args)
			llmResp.Content.Parts = append(llmResp.Content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   c.ToolCallID,
					Name: c.ToolName,
					Args: args,
				},
			})
		case fantasy.ReasoningContent:
			part := &genai.Part{
				Text:    c.Text,
				Thought: true,
			}
			llmResp.Content.Parts = append(llmResp.Content.Parts, part)
		}
	}

	return llmResp
}

// streamProcessor manages state during streaming response processing.
// It accumulates content parts and yields responses at appropriate times.
type streamProcessor struct {
	currentContent       *genai.Content
	currentPart          *genai.Part
	partIndex            int
	toolInputAccumulator map[string]*strings.Builder
	yieldedToolIDs       map[string]bool
}

// newStreamProcessor creates a new stream processor with initialized state.
func newStreamProcessor() *streamProcessor {
	return &streamProcessor{
		toolInputAccumulator: make(map[string]*strings.Builder),
		yieldedToolIDs:       make(map[string]bool),
	}
}

// handleError processes error stream parts.
func (p *streamProcessor) handleError(part fantasy.StreamPart, yield func(*model.LLMResponse, error) bool) bool {
	errMsg := ""
	errCode := ""
	if part.Error != nil {
		errMsg = part.Error.Error()
		errCode = ErrorCodeGeneric
	}
	slog.Default().Error("STREAM: Error received",
		"error", errMsg,
		"error_code", errCode)
	return yield(&model.LLMResponse{
		Content:           nil,
		CitationMetadata:  nil,
		GroundingMetadata: nil,
		UsageMetadata:     nil,
		CustomMetadata:    nil,
		LogprobsResult:    nil,
		Partial:           false,
		TurnComplete:      true,
		Interrupted:       false,
		ErrorCode:         errCode,
		ErrorMessage:      errMsg,
		FinishReason:      genai.FinishReasonOther,
		AvgLogprobs:       0,
	}, part.Error)
}

// handleTextStart processes text start stream parts.
func (p *streamProcessor) handleTextStart() {
	slog.Default().Debug("STREAM: Text start")
	if p.currentContent == nil {
		p.currentContent = &genai.Content{Role: RoleModel, Parts: []*genai.Part{}}
	}
	p.currentPart = &genai.Part{Text: ""}
	p.currentContent.Parts = append(p.currentContent.Parts, p.currentPart)
}

// handleTextDelta processes text delta stream parts and yields partial responses.
func (p *streamProcessor) handleTextDelta(part fantasy.StreamPart, yield func(*model.LLMResponse, error) bool) bool {
	if p.currentPart != nil {
		p.currentPart.Text += part.Delta
		deltaContent := &genai.Content{
			Role: RoleModel,
			Parts: []*genai.Part{
				{Text: part.Delta},
			},
		}
		return yield(&model.LLMResponse{
			Content:           deltaContent,
			CitationMetadata:  nil,
			GroundingMetadata: nil,
			UsageMetadata:     nil,
			CustomMetadata:    nil,
			LogprobsResult:    nil,
			Partial:           true,
			TurnComplete:      false,
			Interrupted:       false,
			ErrorCode:         "",
			ErrorMessage:      "",
			FinishReason:      "",
			AvgLogprobs:       0,
		}, nil)
	}
	return true
}

// handleTextEnd processes text end stream parts.
func (p *streamProcessor) handleTextEnd() {
	p.currentPart = nil
	p.partIndex++
}

// handleToolInputStart processes tool input start stream parts.
func (p *streamProcessor) handleToolInputStart(part fantasy.StreamPart) {
	slog.Default().Debug("STREAM: Tool input start",
		"tool_name", part.ToolCallName,
		"tool_id", part.ID)
	if p.currentContent == nil {
		p.currentContent = &genai.Content{Role: RoleModel, Parts: []*genai.Part{}}
	}
	p.currentPart = &genai.Part{
		FunctionCall: &genai.FunctionCall{
			ID:   part.ID,
			Name: part.ToolCallName,
			Args: make(map[string]any),
		},
	}
	p.currentContent.Parts = append(p.currentContent.Parts, p.currentPart)
	p.toolInputAccumulator[part.ID] = &strings.Builder{}
}

// handleToolInputDelta processes tool input delta stream parts.
func (p *streamProcessor) handleToolInputDelta(part fantasy.StreamPart) {
	if builder, ok := p.toolInputAccumulator[part.ID]; ok {
		// Different providers use different fields:
		// - Anthropic uses ToolCallInput
		// - OpenAI/Google use Delta (consistent with TextDelta, ReasoningDelta)
		if part.Delta != "" {
			builder.WriteString(part.Delta)
		} else if part.ToolCallInput != "" {
			builder.WriteString(part.ToolCallInput)
		}
	}
}

// handleToolInputEnd processes tool input end stream parts and yields tool call responses.
func (p *streamProcessor) handleToolInputEnd(part fantasy.StreamPart, yield func(*model.LLMResponse, error) bool) bool {
	builder, ok := p.toolInputAccumulator[part.ID]
	if !ok {
		slog.Default().Warn("STREAM: Tool input end for unknown ID", "tool_id", part.ID)
		return true
	}

	if p.currentPart != nil && p.currentPart.FunctionCall != nil {
		var jsonStr string

		// Check if we accumulated deltas (Anthropic streaming pattern)
		if builder.Len() > 0 {
			jsonStr = builder.String()
		} else if part.ToolCallInput != "" {
			// No deltas, use direct input (OpenAI/Google pattern)
			// OpenAI and Google send the full arguments in the ToolCall event
			// instead of streaming them as deltas
			jsonStr = part.ToolCallInput
		}

		slog.Default().Debug("STREAM: Tool input end",
			"tool_name", p.currentPart.FunctionCall.Name,
			"tool_id", part.ID,
			"input_json", jsonStr)
		if jsonStr != "" {
			if err := json.Unmarshal([]byte(jsonStr), &p.currentPart.FunctionCall.Args); err != nil {
				slog.Default().Error("STREAM: Failed to unmarshal tool input",
					"tool_name", p.currentPart.FunctionCall.Name,
					"input", jsonStr,
					"error", err)
				return yield(&model.LLMResponse{
					Content:           nil,
					CitationMetadata:  nil,
					GroundingMetadata: nil,
					UsageMetadata:     nil,
					CustomMetadata:    nil,
					LogprobsResult:    nil,
					Partial:           false,
					TurnComplete:      true,
					Interrupted:       false,
					ErrorCode:         "UNMARSHAL_ERROR",
					ErrorMessage:      fmt.Sprintf("failed to unmarshal tool input: %v", err),
					FinishReason:      genai.FinishReasonOther,
					AvgLogprobs:       0,
				}, err)
			}
		}
	}

	toolCallContent := &genai.Content{
		Role:  RoleModel,
		Parts: []*genai.Part{p.currentPart},
	}
	if !yield(&model.LLMResponse{
		Content:           toolCallContent,
		CitationMetadata:  nil,
		GroundingMetadata: nil,
		UsageMetadata:     nil,
		CustomMetadata:    nil,
		LogprobsResult:    nil,
		Partial:           false,
		TurnComplete:      false,
		Interrupted:       false,
		ErrorCode:         "",
		ErrorMessage:      "",
		FinishReason:      "",
		AvgLogprobs:       0,
	}, nil) {
		return false
	}

	p.yieldedToolIDs[part.ID] = true
	delete(p.toolInputAccumulator, part.ID)
	p.currentPart = nil
	p.partIndex++
	return true
}

// handleReasoningStart processes reasoning start stream parts.
func (p *streamProcessor) handleReasoningStart() {
	slog.Default().Debug("STREAM: Reasoning start")
	if p.currentContent == nil {
		p.currentContent = &genai.Content{Role: RoleModel, Parts: []*genai.Part{}}
	}
	p.currentPart = &genai.Part{Text: "", Thought: true}
	p.currentContent.Parts = append(p.currentContent.Parts, p.currentPart)
}

// handleReasoningDelta processes reasoning delta stream parts and yields partial responses.
func (p *streamProcessor) handleReasoningDelta(part fantasy.StreamPart, yield func(*model.LLMResponse, error) bool) bool {
	if p.currentPart != nil {
		p.currentPart.Text += part.Delta
		deltaContent := &genai.Content{
			Role: RoleModel,
			Parts: []*genai.Part{
				{Text: part.Delta, Thought: true},
			},
		}
		return yield(&model.LLMResponse{
			Content:           deltaContent,
			CitationMetadata:  nil,
			GroundingMetadata: nil,
			UsageMetadata:     nil,
			CustomMetadata:    nil,
			LogprobsResult:    nil,
			Partial:           true,
			TurnComplete:      false,
			Interrupted:       false,
			ErrorCode:         "",
			ErrorMessage:      "",
			FinishReason:      "",
			AvgLogprobs:       0,
		}, nil)
	}
	return true
}

// handleReasoningEnd processes reasoning end stream parts.
func (p *streamProcessor) handleReasoningEnd(part fantasy.StreamPart) {
	// Extract and preserve signature for multi-turn context
	if p.currentPart != nil {
		if metadata, ok := part.ProviderMetadata["anthropic"]; ok {
			if reasoningMeta, ok := metadata.(*anthropic.ReasoningOptionMetadata); ok {
				if reasoningMeta.Signature != "" {
					p.currentPart.ThoughtSignature = []byte(reasoningMeta.Signature)
				}
			}
		}
	}
	p.currentPart = nil
	p.partIndex++
}

// handleFinish processes finish stream parts and yields the final response.
func (p *streamProcessor) handleFinish(part fantasy.StreamPart, yield func(*model.LLMResponse, error) bool) bool {
	slog.Default().Debug("STREAM: Finish received",
		"finish_reason", part.FinishReason,
		"input_tokens", part.Usage.InputTokens,
		"output_tokens", part.Usage.OutputTokens)
	finalContent := p.currentContent
	if p.currentContent != nil && len(p.currentContent.Parts) > 0 {
		filteredParts := make([]*genai.Part, 0)
		for _, part := range p.currentContent.Parts {
			// Filter already-yielded tool calls
			if part.FunctionCall != nil {
				if p.yieldedToolIDs[part.FunctionCall.ID] {
					continue
				}
			}
			// Filter reasoning Parts (already yielded as deltas during streaming)
			if part.Thought {
				continue
			}
			filteredParts = append(filteredParts, part)
		}
		if len(filteredParts) > 0 {
			finalContent = &genai.Content{Role: RoleModel, Parts: filteredParts}
		} else {
			finalContent = nil
		}
	}

	return yield(&model.LLMResponse{
		Content:           finalContent,
		CitationMetadata:  nil,
		GroundingMetadata: nil,
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        int32(part.Usage.InputTokens),
			CandidatesTokenCount:    int32(part.Usage.OutputTokens),
			TotalTokenCount:         int32(part.Usage.TotalTokens),
			CachedContentTokenCount: int32(part.Usage.CacheReadTokens),
		},
		CustomMetadata: nil,
		LogprobsResult: nil,
		Partial:        false,
		TurnComplete:   true,
		Interrupted:    false,
		ErrorCode:      "",
		ErrorMessage:   "",
		FinishReason:   fantasyFinishReasonToGenai(part.FinishReason),
		AvgLogprobs:    0,
	}, nil)
}

// fantasyStreamToLLM converts fantasy.StreamResponse to an ADK response iterator.
//
// Accumulates streaming parts into genai.Content and yields LLMResponse on deltas and completion:
//   - TextDelta: yields with Partial:true, TurnComplete:false
//   - ReasoningDelta: accumulates into Part{Thought:true}
//   - ToolInput: accumulates into Part{FunctionCall}
//   - Finish: yields final response with usage and FinishReason
//   - Error: yields response with ErrorCode and ErrorMessage
func fantasyStreamToLLM(stream fantasy.StreamResponse) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		slog.Default().Debug("STREAM: Starting to process fantasy stream")
		processor := newStreamProcessor()

		for part := range stream {
			switch part.Type {
			case fantasy.StreamPartTypeError:
				if !processor.handleError(part, yield) {
					return
				}
			case fantasy.StreamPartTypeTextStart:
				processor.handleTextStart()
			case fantasy.StreamPartTypeTextDelta:
				if !processor.handleTextDelta(part, yield) {
					return
				}
			case fantasy.StreamPartTypeTextEnd:
				processor.handleTextEnd()
			case fantasy.StreamPartTypeToolInputStart:
				processor.handleToolInputStart(part)
			case fantasy.StreamPartTypeToolInputDelta:
				processor.handleToolInputDelta(part)
			case fantasy.StreamPartTypeToolInputEnd, fantasy.StreamPartTypeToolCall:
				if !processor.handleToolInputEnd(part, yield) {
					return
				}
			case fantasy.StreamPartTypeReasoningStart:
				processor.handleReasoningStart()
			case fantasy.StreamPartTypeReasoningDelta:
				if !processor.handleReasoningDelta(part, yield) {
					return
				}
			case fantasy.StreamPartTypeReasoningEnd:
				processor.handleReasoningEnd(part)
			case fantasy.StreamPartTypeFinish:
				if !processor.handleFinish(part, yield) {
					return
				}
			}
		}
	}
}

// fantasyFinishReasonToGenai maps fantasy finish reasons to genai finish reasons.
//
// Mapping:
//   - stop → STOP
//   - length → MAX_TOKENS
//   - content-filter → SAFETY
//   - tool-calls → STOP
//   - error → OTHER
//   - other/unknown → FINISH_REASON_UNSPECIFIED
func fantasyFinishReasonToGenai(reason fantasy.FinishReason) genai.FinishReason {
	switch reason {
	case fantasy.FinishReasonStop:
		return genai.FinishReasonStop
	case fantasy.FinishReasonLength:
		return genai.FinishReasonMaxTokens
	case fantasy.FinishReasonContentFilter:
		return genai.FinishReasonSafety
	case fantasy.FinishReasonToolCalls:
		return genai.FinishReasonStop
	case fantasy.FinishReasonError:
		return genai.FinishReasonOther
	case fantasy.FinishReasonOther, fantasy.FinishReasonUnknown:
		return genai.FinishReasonUnspecified
	default:
		return genai.FinishReasonUnspecified
	}
}
