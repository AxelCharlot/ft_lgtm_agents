package llama

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeServer answers every request with status and body, and keeps the last
// request it decoded.
func fakeServer(t *testing.T, status int, body string) (*httptest.Server, *map[string]any) {
	t.Helper()
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Errorf("the request is not JSON: %v", err)
		}
		received["_path"] = r.URL.Path
		w.WriteHeader(status)
		if _, err := w.Write([]byte(body)); err != nil {
			t.Error(err)
		}
	}))
	t.Cleanup(server.Close)
	return server, &received
}

const chatAnswer = `{"choices": [{"message": {"role": "assistant", "content": "{}"},
	"finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 3},
	"timings": {"predicted_per_second": 11.5}}`

func TestCompleteSendsTheGrammarAndReadsTheAnswer(t *testing.T) {
	server, received := fakeServer(t, http.StatusOK, chatAnswer)
	client := &Client{CompletionURL: server.URL}

	completion, err := client.Complete(context.Background(), Request{
		System: "repair", Prompt: "the pod is OOMKilled", Grammar: PatchGrammar,
		MaxTokens: 512, Temperature: 0.2,
	})
	if err != nil {
		t.Fatal(err)
	}

	want := Completion{Text: "{}", PromptTokens: 12, CompletionTokens: 3, TokensPerSecond: 11.5}
	if completion != want {
		t.Errorf("Complete = %+v, want %+v", completion, want)
	}
	request := *received
	if request["_path"] != "/v1/chat/completions" || request["grammar"] != PatchGrammar ||
		request["max_tokens"] != 512.0 || request["temperature"] != 0.2 {
		t.Errorf("the request was %v", request)
	}
	messages, _ := request["messages"].([]any)
	if len(messages) != 2 || !strings.Contains(messages[0].(map[string]any)["role"].(string), "system") {
		t.Errorf("the messages were %v, want the system message first", messages)
	}
}

func TestCompleteReportsATruncatedAnswer(t *testing.T) {
	server, _ := fakeServer(t, http.StatusOK, strings.Replace(chatAnswer, `"stop"`, `"length"`, 1))
	client := &Client{CompletionURL: server.URL}

	completion, err := client.Complete(context.Background(), Request{Prompt: "p", MaxTokens: 8})
	if !errors.Is(err, ErrTruncated) {
		t.Fatalf("Complete = %v, want ErrTruncated", err)
	}
	if completion.Text != "{}" {
		t.Errorf("the truncated text was lost: %+v", completion)
	}
}

func TestCompleteRefuses(t *testing.T) {
	cases := map[string]struct {
		status  int
		body    string
		request Request
		want    string
	}{
		"no token limit": {http.StatusOK, chatAnswer, Request{Prompt: "p"}, "token limit"},
		"a refused grammar": {http.StatusBadRequest,
			`{"error": {"message": "failed to parse grammar"}}`,
			Request{Prompt: "p", MaxTokens: 8}, "failed to parse grammar"},
		"an answer that is not JSON": {http.StatusOK, "<html>", Request{Prompt: "p", MaxTokens: 8},
			"could not read"},
		"no choice": {http.StatusOK, `{"choices": []}`, Request{Prompt: "p", MaxTokens: 8},
			"0 choices"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			server, _ := fakeServer(t, test.status, test.body)
			client := &Client{CompletionURL: server.URL}
			_, err := client.Complete(context.Background(), test.request)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Errorf("Complete = %v, want an error containing %q", err, test.want)
			}
		})
	}
}

func TestEmbed(t *testing.T) {
	cases := map[string]struct {
		body string
		want string
	}{
		"the expected dimension": {`{"data": [{"embedding": [0.1, 0.2, 0.3]}]}`, ""},
		"another dimension":      {`{"data": [{"embedding": [0.1, 0.2]}]}`, "2 dimensions"},
		"no vector":              {`{"data": []}`, "0 vectors"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			server, received := fakeServer(t, http.StatusOK, test.body)
			client := &Client{EmbeddingURL: server.URL, EmbeddingDimensions: 3}

			vector, err := client.Embed(context.Background(), "the backend cannot reach kubo")
			if test.want != "" {
				if err == nil || !strings.Contains(err.Error(), test.want) {
					t.Errorf("Embed = %v, want an error containing %q", err, test.want)
				}
				return
			}
			if err != nil || len(vector) != 3 {
				t.Fatalf("Embed = %v, %v", vector, err)
			}
			if (*received)["_path"] != "/v1/embeddings" ||
				(*received)["input"] != "the backend cannot reach kubo" {
				t.Errorf("the request was %v", *received)
			}
		})
	}
}

func TestACancelledCallStops(t *testing.T) {
	server, _ := fakeServer(t, http.StatusOK, chatAnswer)
	client := &Client{CompletionURL: server.URL}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := client.Complete(ctx, Request{Prompt: "p", MaxTokens: 8}); !errors.Is(err, context.Canceled) {
		t.Errorf("Complete = %v, want context.Canceled", err)
	}
}
