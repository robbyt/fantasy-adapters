package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/openai"
	adapter "github.com/robbyt/fantasy-adapters/adk"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	anthropicProvider, err := anthropic.New(
		anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
	)
	if err != nil {
		log.Fatal(err)
	}

	openaiProvider, err := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		log.Fatal(err)
	}

	claudeModel, err := anthropicProvider.LanguageModel(ctx, "claude-3-5-sonnet-20241022")
	if err != nil {
		log.Fatal(err)
	}

	gptModel, err := openaiProvider.LanguageModel(ctx, "gpt-4o-mini")
	if err != nil {
		log.Fatal(err)
	}

	claudeLLM := adapter.NewAdapter(claudeModel)
	gptLLM := adapter.NewAdapter(gptModel)

	fmt.Println("Using Claude via ADK adapter:")
	if err := generateWithADK(ctx, claudeLLM); err != nil {
		log.Fatal(err)
	}

	fmt.Println("\nUsing GPT via ADK adapter:")
	if err := generateWithADK(ctx, gptLLM); err != nil {
		log.Fatal(err)
	}
}

func generateWithADK(ctx context.Context, llm model.LLM) error {
	req := &model.LLMRequest{
		Model: llm.Name(),
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Write a haiku about Go programming."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 100,
			Temperature:     ptrFloat32(0.7),
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return err
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Println(part.Text)
				}
			}
		}
	}

	return nil
}

func ptrFloat32(f float32) *float32 {
	return &f
}
