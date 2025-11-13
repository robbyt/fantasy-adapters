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
	"strings"

	"charm.land/fantasy"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Adapter wraps a fantasy.LanguageModel to implement the Google ADK model.LLM interface.
type Adapter struct {
	model fantasy.LanguageModel
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
func NewAdapter(m fantasy.LanguageModel) model.LLM {
	return &Adapter{model: m}
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
	call, err := llmRequestToFantasyCall(req)
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
func llmRequestToFantasyCall(req *model.LLMRequest) (fantasy.Call, error) {
	var call fantasy.Call
	var errs []error

	if req.Config != nil {
		if req.Config.Temperature != nil {
			temp := float64(*req.Config.Temperature)
			call.Temperature = &temp
		}
		if req.Config.TopP != nil {
			topP := float64(*req.Config.TopP)
			call.TopP = &topP
		}
		if req.Config.TopK != nil {
			topK := int64(*req.Config.TopK)
			call.TopK = &topK
		}
		if req.Config.MaxOutputTokens > 0 {
			tokens := int64(req.Config.MaxOutputTokens)
			call.MaxOutputTokens = &tokens
		}
		if req.Config.PresencePenalty != nil {
			pp := float64(*req.Config.PresencePenalty)
			call.PresencePenalty = &pp
		}
		if req.Config.FrequencyPenalty != nil {
			fp := float64(*req.Config.FrequencyPenalty)
			call.FrequencyPenalty = &fp
		}

		if len(req.Config.SafetySettings) > 0 {
			errs = append(errs, errors.New("safety settings not supported"))
		}
		if req.Config.ResponseMIMEType != "" {
			errs = append(errs, errors.New("response MIME type not supported"))
		}
		if req.Config.ResponseSchema != nil || req.Config.ResponseJsonSchema != nil {
			errs = append(errs, errors.New("response schema not supported"))
		}
		if req.Config.ThinkingConfig != nil {
			errs = append(errs, errors.New("thinking config not supported (use provider-specific options)"))
		}
		if req.Config.CachedContent != "" {
			errs = append(errs, errors.New("cached content not supported"))
		}

		if req.Config.SystemInstruction != nil {
			systemMsg, err := genaiSystemInstructionToFantasyMessage(req.Config.SystemInstruction)
			if err != nil {
				errs = append(errs, fmt.Errorf("system instruction: %w", err))
			} else {
				call.Prompt = append(call.Prompt, systemMsg)
			}
		}

		if len(req.Config.Tools) > 0 {
			tools, err := genaiToolsToFantasyTools(req.Config.Tools)
			if err != nil {
				errs = append(errs, fmt.Errorf("tools: %w", err))
			} else {
				call.Tools = tools
			}
		}

		if req.Config.ToolConfig != nil && req.Config.ToolConfig.FunctionCallingConfig != nil {
			fc := req.Config.ToolConfig.FunctionCallingConfig

			// Filter tools by AllowedFunctionNames if specified
			if len(fc.AllowedFunctionNames) > 0 {
				// Validate that all allowed names exist in the tools list
				toolNames := make(map[string]bool)
				for _, tool := range call.Tools {
					if ft, ok := tool.(fantasy.FunctionTool); ok {
						toolNames[ft.Name] = true
					}
				}

				for _, allowedName := range fc.AllowedFunctionNames {
					if !toolNames[allowedName] {
						errs = append(errs, fmt.Errorf("allowed function %q not found in tools list", allowedName))
					}
				}

				// Filter tools to only include allowed ones
				filteredTools := make([]fantasy.Tool, 0, len(fc.AllowedFunctionNames))
				for _, tool := range call.Tools {
					if ft, ok := tool.(fantasy.FunctionTool); ok {
						for _, allowedName := range fc.AllowedFunctionNames {
							if ft.Name == allowedName {
								filteredTools = append(filteredTools, tool)
								break
							}
						}
					}
				}
				call.Tools = filteredTools
			}

			// Set ToolChoice based on Mode
			switch fc.Mode {
			case ToolModeAuto:
				tc := fantasy.ToolChoiceAuto
				call.ToolChoice = &tc
			case ToolModeAny:
				tc := fantasy.ToolChoiceRequired
				call.ToolChoice = &tc
			case ToolModeNone:
				tc := fantasy.ToolChoiceNone
				call.ToolChoice = &tc
			case ToolModeValidated:
				errs = append(errs, errors.New("validated tool mode not supported"))
			case "":
				// No mode specified, allow specific function if only one
			default:
				if fc.Mode != "" {
					errs = append(errs, fmt.Errorf("unsupported tool calling mode: %q", fc.Mode))
				}
			}

			// Override with specific function if exactly one is allowed
			if len(fc.AllowedFunctionNames) == 1 && fc.Mode != ToolModeNone {
				tc := fantasy.ToolChoice(fc.AllowedFunctionNames[0])
				call.ToolChoice = &tc
			}
		}
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
			msg.Content = append(msg.Content, fantasy.ReasoningPart{Text: part.Text})
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

func schemaToMap(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}

	result := make(map[string]any)

	if schema.Type != "" {
		result["type"] = fmt.Sprintf("%v", schema.Type)
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
			propMap, err := schemaToMap(val)
			if err != nil {
				return nil, fmt.Errorf("failed to convert property %q: %w", key, err)
			}
			props[key] = propMap
		}
		result["properties"] = props
	}

	if schema.Items != nil {
		items, err := schemaToMap(schema.Items)
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
			schemaMap, err := schemaToMap(s)
			if err != nil {
				return nil, fmt.Errorf("failed to convert anyOf[%d]: %w", i, err)
			}
			anyOf[i] = schemaMap
		}
		result["anyOf"] = anyOf
	}

	return result, nil
}

func genaiToolsToFantasyTools(tools []*genai.Tool) ([]fantasy.Tool, error) {
	var fantasyTools []fantasy.Tool
	var errs []error

	for _, tool := range tools {
		if len(tool.FunctionDeclarations) > 0 {
			for _, fn := range tool.FunctionDeclarations {
				params := make(map[string]any)
				if fn.Parameters != nil {
					schemaMap, err := schemaToMap(fn.Parameters)
					if err != nil {
						errs = append(errs, fmt.Errorf("failed to convert schema for function %q: %w", fn.Name, err))
					} else if schemaMap != nil {
						params = schemaMap
					}
				}
				fantasyTools = append(fantasyTools, fantasy.FunctionTool{
					Name:        fn.Name,
					Description: fn.Description,
					InputSchema: params,
				})
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
			llmResp.Content.Parts = append(llmResp.Content.Parts, &genai.Part{
				Text: c.Text,
			})
		case fantasy.ToolCallContent:
			args := make(map[string]any)
			if c.Input != "" {
				if err := json.Unmarshal([]byte(c.Input), &args); err != nil {
					llmResp.ErrorCode = ErrorCodeUnmarshal
					llmResp.ErrorMessage = fmt.Sprintf("failed to unmarshal tool call input: %v", err)
				}
			}
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
		var currentContent *genai.Content
		var currentPart *genai.Part
		var partIndex int
		toolInputAccumulator := make(map[string]*strings.Builder)

		for part := range stream {
			switch part.Type {
			case fantasy.StreamPartTypeError:
				errMsg := ""
				errCode := ""
				if part.Error != nil {
					errMsg = part.Error.Error()
					errCode = ErrorCodeGeneric
				}
				if !yield(&model.LLMResponse{
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
				}, part.Error) {
					return
				}

			case fantasy.StreamPartTypeTextStart:
				if currentContent == nil {
					currentContent = &genai.Content{Role: RoleModel, Parts: []*genai.Part{}}
				}
				currentPart = &genai.Part{Text: ""}
				currentContent.Parts = append(currentContent.Parts, currentPart)

			case fantasy.StreamPartTypeTextDelta:
				if currentPart != nil {
					currentPart.Text += part.Delta
					if !yield(&model.LLMResponse{
						Content:           currentContent,
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
					}, nil) {
						return
					}
				}

			case fantasy.StreamPartTypeTextEnd:
				currentPart = nil
				partIndex++

			case fantasy.StreamPartTypeToolInputStart:
				if currentContent == nil {
					currentContent = &genai.Content{Role: RoleModel, Parts: []*genai.Part{}}
				}
				currentPart = &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   part.ID,
						Name: part.ToolCallName,
						Args: make(map[string]any),
					},
				}
				currentContent.Parts = append(currentContent.Parts, currentPart)
				toolInputAccumulator[part.ID] = &strings.Builder{}

			case fantasy.StreamPartTypeToolInputDelta:
				if builder, ok := toolInputAccumulator[part.ID]; ok {
					builder.WriteString(part.ToolCallInput)
				}

			case fantasy.StreamPartTypeToolInputEnd, fantasy.StreamPartTypeToolCall:
				if builder, ok := toolInputAccumulator[part.ID]; ok {
					if currentPart != nil && currentPart.FunctionCall != nil {
						jsonStr := builder.String()
						if jsonStr != "" {
							if err := json.Unmarshal([]byte(jsonStr), &currentPart.FunctionCall.Args); err != nil {
								if !yield(&model.LLMResponse{
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
								}, err) {
									return
								}
							}
						}
					}
					delete(toolInputAccumulator, part.ID)
				}
				currentPart = nil
				partIndex++

			case fantasy.StreamPartTypeReasoningStart:
				if currentContent == nil {
					currentContent = &genai.Content{Role: RoleModel, Parts: []*genai.Part{}}
				}
				currentPart = &genai.Part{Text: "", Thought: true}
				currentContent.Parts = append(currentContent.Parts, currentPart)

			case fantasy.StreamPartTypeReasoningDelta:
				if currentPart != nil {
					currentPart.Text += part.Delta
				}

			case fantasy.StreamPartTypeReasoningEnd:
				currentPart = nil
				partIndex++

			case fantasy.StreamPartTypeFinish:
				if !yield(&model.LLMResponse{
					Content:           currentContent,
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
				}, nil) {
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
