# Fantasy ADK Adapter

This package provides an adapter to use Fantasy AI providers with the [Google ADK (Agent Development Kit)](https://google.golang.org/adk).

## Installation

```bash
go get github.com/robbyt/fantasy-adapters/adk
```

## Usage

The adapter allows you to use any Fantasy provider (Anthropic, OpenAI, Google, Azure, etc.) with the ADK's `model.LLM` interface.

```go
package main

import (
    "context"
    "log"
    "os"

    "charm.land/fantasy"
    "github.com/robbyt/fantasy-adapters/adk"
    "charm.land/fantasy/providers/anthropic"
    "google.golang.org/adk/model"
    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()

    // Create a Fantasy provider
    provider, err := anthropic.New(
        anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Get a language model
    model, err := provider.LanguageModel(ctx, "claude-3-5-sonnet-20241022")
    if err != nil {
        log.Fatal(err)
    }

    // Wrap it in the ADK adapter
    llm := adk.NewAdapter(model)

    // Use the wrapped model with the ADK
    req := &model.LLMRequest{
        Model: llm.Name(),
        Contents: []*genai.Content{
            {
                Role: "user",
                Parts: []*genai.Part{
                    {Text: "Hello, world!"},
                },
            },
        },
    }

    for resp, err := range llm.GenerateContent(ctx, req, false) {
        if err != nil {
            log.Fatal(err)
        }
        // Process response
    }
}
```

## Supported Features

- ✅ Text generation (streaming and non-streaming)
- ✅ Temperature, TopP, TopK
- ✅ MaxOutputTokens
- ✅ System instructions
- ✅ Tool/function calling (with JSON serialization)
- ✅ Tool choice modes (auto, required, none, specific tool)
- ✅ Multi-turn conversations
- ✅ Image inputs (inline data)
- ✅ Token usage tracking

### Tool/Function Calling

The adapter supports tool calling with the following implementation:
- Function arguments are serialized as JSON strings for transport
- Function responses are serialized as JSON strings
- Streaming tool input accumulation is fully supported
- Tool parameters use JSON Schema format (OpenAPI 3.0 compatible)

## Unsupported Features

The following ADK features are not supported and will return errors:

- Safety settings
- Response MIME type constraints
- Response schemas (structured output)
- Cached content references
- File data (URI references)
- Retrieval tools (Google Search, Vertex AI Search)
- Code execution
- Video metadata

For provider-specific features (like Anthropic's extended thinking), use the Fantasy provider options directly instead of ADK's configuration.

## Multi-Provider Support

The adapter works with all Fantasy providers:

```go
// Anthropic
anthropicProvider, _ := anthropic.New(anthropic.WithAPIKey(key))
claudeModel, _ := anthropicProvider.LanguageModel(ctx, "claude-3-5-sonnet-20241022")
claudeLLM := adk.NewAdapter(claudeModel)

// OpenAI
openaiProvider, _ := openai.New(openai.WithAPIKey(key))
gptModel, _ := openaiProvider.LanguageModel(ctx, "gpt-4o")
gptLLM := adk.NewAdapter(gptModel)

// Google
googleProvider, _ := google.New(google.WithAPIKey(key))
geminiModel, _ := googleProvider.LanguageModel(ctx, "gemini-2.0-flash-exp")
geminiLLM := adk.NewAdapter(geminiModel)
```

## Examples

See the [examples](./examples) directory for complete working examples.
