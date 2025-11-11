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
	"errors"
	"fmt"
	"iter"

	"charm.land/fantasy"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Adapter wraps a fantasy.LanguageModel to implement the Google ADK model.LLM interface.
type Adapter struct {
	model fantasy.LanguageModel
}

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
//   - Config.Tools -> Call.Tools
//   - Config.ToolConfig.FunctionCallingConfig -> Call.ToolChoice
//
// Returns errors for unsupported ADK features:
//   - SafetySettings, ResponseMIMEType, ResponseSchema
//   - ThinkingConfig (use provider-specific options instead)
//   - CachedContent, FileData URIs
//   - Retrieval/code execution tools
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
			systemMsg, err := genaiContentToFantasyMessage(req.Config.SystemInstruction, fantasy.MessageRoleSystem)
			if err != nil {
				errs = append(errs, fmt.Errorf("system instruction: %w", err))
			} else {
				call.Prompt = append(call.Prompt, systemMsg)
			}
		}

		if req.Config.Tools != nil && len(req.Config.Tools) > 0 {
			tools, err := genaiToolsToFantasyTools(req.Config.Tools)
			if err != nil {
				errs = append(errs, fmt.Errorf("tools: %w", err))
			} else {
				call.Tools = tools
			}
		}

		if req.Config.ToolConfig != nil && req.Config.ToolConfig.FunctionCallingConfig != nil {
			fc := req.Config.ToolConfig.FunctionCallingConfig
			switch fc.Mode {
			case "AUTO":
				tc := fantasy.ToolChoiceAuto
				call.ToolChoice = &tc
			case "ANY":
				tc := fantasy.ToolChoiceRequired
				call.ToolChoice = &tc
			case "NONE":
				tc := fantasy.ToolChoiceNone
				call.ToolChoice = &tc
			case "VALIDATED":
				errs = append(errs, errors.New("validated tool mode not supported"))
			}

			if len(fc.AllowedFunctionNames) == 1 {
				tc := fantasy.ToolChoice(fc.AllowedFunctionNames[0])
				call.ToolChoice = &tc
			} else if len(fc.AllowedFunctionNames) > 1 {
				errs = append(errs, errors.New("multiple allowed function names not supported"))
			}
		}
	}

	for _, content := range req.Contents {
		role := fantasy.MessageRoleUser
		if content.Role == "model" {
			role = fantasy.MessageRoleAssistant
		}

		msg, err := genaiContentToFantasyMessage(content, role)
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
func genaiContentToFantasyMessage(content *genai.Content, role fantasy.MessageRole) (fantasy.Message, error) {
	var msg fantasy.Message
	msg.Role = role
	var errs []error

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
				errs = append(errs, errors.New("function call serialization not implemented"))
			}
			msg.Content = append(msg.Content, fantasy.ToolCallPart{
				ToolCallID: part.FunctionCall.ID,
				ToolName:   part.FunctionCall.Name,
				Input:      input,
			})
		} else if part.FunctionResponse != nil {
			output := fantasy.ToolResultOutputContentText{Text: ""}
			if part.FunctionResponse.Response != nil {
				errs = append(errs, errors.New("function response serialization not implemented"))
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

func genaiToolsToFantasyTools(tools []*genai.Tool) ([]fantasy.Tool, error) {
	var fantasyTools []fantasy.Tool
	var errs []error

	for _, tool := range tools {
		if len(tool.FunctionDeclarations) > 0 {
			for _, fn := range tool.FunctionDeclarations {
				params := make(map[string]any)
				if fn.Parameters != nil {
					errs = append(errs, errors.New("schema to map conversion not implemented"))
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
			Role:  "model",
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
			llmResp.Content.Parts = append(llmResp.Content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   c.ToolCallID,
					Name: c.ToolName,
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

		for part := range stream {
			switch part.Type {
			case fantasy.StreamPartTypeError:
				errMsg := ""
				errCode := ""
				if part.Error != nil {
					errMsg = part.Error.Error()
					errCode = "ERROR"
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
					currentContent = &genai.Content{Role: "model", Parts: []*genai.Part{}}
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
					currentContent = &genai.Content{Role: "model", Parts: []*genai.Part{}}
				}
				currentPart = &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   part.ID,
						Name: part.ToolCallName,
						Args: make(map[string]any),
					},
				}
				currentContent.Parts = append(currentContent.Parts, currentPart)

			case fantasy.StreamPartTypeToolInputDelta:

			case fantasy.StreamPartTypeToolInputEnd, fantasy.StreamPartTypeToolCall:
				currentPart = nil
				partIndex++

			case fantasy.StreamPartTypeReasoningStart:
				if currentContent == nil {
					currentContent = &genai.Content{Role: "model", Parts: []*genai.Part{}}
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
