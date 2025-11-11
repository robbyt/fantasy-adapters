# Fantasy Adapters

Adapters for using [Fantasy](https://charm.land/fantasy) with various AI frameworks and SDKs.

Fantasy is a unified interface for AI providers (Anthropic, OpenAI, Google, Azure, etc.). These adapters allow you to use Fantasy providers with other frameworks that have their own model interfaces.

## Available Adapters

### [ADK Adapter](./adk)

Adapter for using Fantasy providers with the [Google ADK (Agent Development Kit)](https://google.golang.org/adk).

```bash
go get github.com/robbyt/fantasy-adapters/adk
```

```go
import (
    "github.com/robbyt/fantasy-adapters/adk"
    "charm.land/fantasy/providers/anthropic"
)

provider, _ := anthropic.New(anthropic.WithAPIKey(apiKey))
model, _ := provider.LanguageModel(ctx, "claude-3-5-sonnet-20241022")
llm := adk.NewAdapter(model)

// Use llm with Google ADK
```

See the [ADK adapter documentation](./adk/README.md) for details on supported features.

## Usage

Each adapter is a separate Go module that can be imported independently. Check the adapter's directory for installation instructions and usage examples.

## Contributing

Contributions are welcome. When adding new adapters, follow the existing structure:
- Create a new directory for the adapter
- Include a README.md with installation and usage instructions
- Add tests and examples

## License

[MIT](LICENSE)
