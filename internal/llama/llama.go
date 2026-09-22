package llama

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// PatchGrammar constrains a completion to the patch schema of internal/patch.
//
//go:embed patch.gbnf
var PatchGrammar string

// ErrTruncated means the completion stopped on its token limit. The grammar
// holds the shape only while the model is still writing: a completion cut
// short is an unfinished object, whatever the grammar said.
var ErrTruncated = errors.New("the completion reached its token limit before the end")

// maxErrorBody bounds how much of a failed response is kept in the error.
const maxErrorBody = 1 << 10

// Client talks to the two llama-server processes of the lgtm-brain pod. One
// process serves one model, so completion and embedding have one URL each.
type Client struct {
	// CompletionURL is the base URL of llama-coder, such as http://lgtm-brain:8081.
	CompletionURL string
	// EmbeddingURL is the base URL of llama-embed, such as http://lgtm-brain:8082.
	EmbeddingURL string
	// EmbeddingDimensions is the length every vector must have. A different
	// length means another model answered, and its vectors cannot be compared
	// with the ones the incident memory already holds.
	EmbeddingDimensions int
	// HTTPClient sends the requests; http.DefaultClient when nil. Cancellation
	// comes from the context of each call, not from a client timeout.
	HTTPClient *http.Client
}

// Request is one completion.
type Request struct {
	System string
	Prompt string
	// Grammar is a GBNF grammar, such as PatchGrammar. Empty means none, and
	// then nothing holds the model to any format.
	Grammar string
	// MaxTokens bounds the answer. It is required: a model constrained to a
	// grammar that allows an endless string would otherwise fill its context.
	MaxTokens int
	// Temperature 0 picks the most likely token every time.
	Temperature float64
}

// Completion is what the model answered, with what it cost.
type Completion struct {
	Text             string
	PromptTokens     int
	CompletionTokens int
	// TokensPerSecond is the generation speed llama-server measured.
	TokensPerSecond float64
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	// Grammar is an extension of llama-server to the OpenAI schema.
	Grammar string `json:"grammar,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message      chatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Timings struct {
		PredictedPerSecond float64 `json:"predicted_per_second"`
	} `json:"timings"`
}

// Complete asks llama-coder for one completion.
func (c *Client) Complete(ctx context.Context, request Request) (Completion, error) {
	if request.MaxTokens <= 0 {
		return Completion{}, errors.New("a completion needs a token limit")
	}
	messages := []chatMessage{{Role: "user", Content: request.Prompt}}
	if request.System != "" {
		messages = append([]chatMessage{{Role: "system", Content: request.System}}, messages...)
	}

	var response chatResponse
	err := c.post(ctx, c.CompletionURL+"/v1/chat/completions", chatRequest{
		Messages:    messages,
		MaxTokens:   request.MaxTokens,
		Temperature: request.Temperature,
		Grammar:     request.Grammar,
	}, &response)
	if err != nil {
		return Completion{}, err
	}
	if len(response.Choices) != 1 {
		return Completion{}, fmt.Errorf("llama-coder answered %d choices, not one",
			len(response.Choices))
	}

	choice := response.Choices[0]
	completion := Completion{
		Text:             choice.Message.Content,
		PromptTokens:     response.Usage.PromptTokens,
		CompletionTokens: response.Usage.CompletionTokens,
		TokensPerSecond:  response.Timings.PredictedPerSecond,
	}
	if choice.FinishReason == "length" {
		return completion, ErrTruncated
	}
	return completion, nil
}

type embeddingRequest struct {
	Input string `json:"input"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed asks llama-embed for the vector of text.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	var response embeddingResponse
	err := c.post(ctx, c.EmbeddingURL+"/v1/embeddings", embeddingRequest{Input: text}, &response)
	if err != nil {
		return nil, err
	}
	if len(response.Data) != 1 {
		return nil, fmt.Errorf("llama-embed answered %d vectors, not one", len(response.Data))
	}
	vector := response.Data[0].Embedding
	if len(vector) != c.EmbeddingDimensions {
		return nil, fmt.Errorf("llama-embed answered a vector of %d dimensions, and the memory "+
			"holds vectors of %d", len(vector), c.EmbeddingDimensions)
	}
	return vector, nil
}

func (c *Client) post(ctx context.Context, url string, body, into any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		// llama-server explains a refused grammar or an overflowing prompt in the
		// body, and that explanation is the whole diagnosis.
		detail, err := io.ReadAll(io.LimitReader(response.Body, maxErrorBody))
		if err != nil {
			return fmt.Errorf("%s answered %s", url, response.Status)
		}
		return fmt.Errorf("%s answered %s: %s", url, response.Status,
			strings.TrimSpace(string(detail)))
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("could not read the answer of %s: %w", url, err)
	}
	return nil
}
