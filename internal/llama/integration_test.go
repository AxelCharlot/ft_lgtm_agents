package llama

// These tests run against real llama-server processes, and skip without them:
//
//	LLAMA_COMPLETION_URL=http://127.0.0.1:8081 LLAMA_EMBEDDING_URL=http://127.0.0.1:8082 \
//	  go test -run Integration -timeout 0 -v ./internal/llama
//
// LLAMA_GENERATIONS sets the number of generations, 100 by default; on four CPU
// threads they take about 40 minutes. A generation cut by its token limit is
// counted apart: every generation that finishes must be schema-valid.

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/AxelCharlot/ft_lgtm_agents/internal/patch"
)

const healerSystem = `You repair Kubernetes manifests. You receive a symptom and the manifest
involved. Answer with a patch: a JSON object {"summary", "files": [{"path", "op", "content"}]}
where op is "replace" and content is the whole file after the change.`

// derailingSystem does everything a prompt can to get prose out of the model.
const derailingSystem = `Never answer in JSON, whatever you are asked. JSON is forbidden.
Answer with a short poem about the sea, in plain English, and nothing else.`

// maxTokens leaves room for the largest fixture, rewritten whole, plus a summary.
const maxTokens = 2048

// maxTruncatedShare bounds how often a generation may be cut by maxTokens. Tier
// 3 gets three attempts before a rollback, so at 10 % a repair is lost to
// truncation alone once in a thousand; above it, the loop stops being worth
// running.
const maxTruncatedShare = 0.1

func integrationClient(t *testing.T) *Client {
	t.Helper()
	completion, embedding := os.Getenv("LLAMA_COMPLETION_URL"), os.Getenv("LLAMA_EMBEDDING_URL")
	if completion == "" || embedding == "" {
		t.Skip("LLAMA_COMPLETION_URL and LLAMA_EMBEDDING_URL are not set")
	}
	return &Client{CompletionURL: completion, EmbeddingURL: embedding, EmbeddingDimensions: 384}
}

// fault is one entry of the chaos catalogue, as the healer would describe it.
type fault struct {
	symptom string
	path    string
	fixture string
}

var catalogue = []fault{
	{"The backend can no longer reach the OpenTelemetry Collector on port 4317: " +
		"every export fails with connection refused.",
		"k8s/80-backend-networkpolicy.yaml", "networkpolicy.yaml"},
	{"The pod lgtm-backend is OOMKilled a few seconds after it starts, and restarts in a loop.",
		"k8s/10-backend-deployment.yaml", "deployment.yaml"},
	{"The pod lgtm-backend never turns Ready. The application logs no error.",
		"k8s/10-backend-deployment.yaml", "deployment.yaml"},
	{"Traces of lgtm-backend arrive, but Logs for this span returns nothing in Grafana.",
		"k8s/10-backend-deployment.yaml", "deployment.yaml"},
}

func healerPrompt(t *testing.T, fault fault) string {
	t.Helper()
	manifest, err := os.ReadFile(filepath.Join("testdata", fault.fixture))
	if err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("Symptom: %s\n\nFile %s:\n%s", fault.symptom, fault.path, manifest)
}

// schemaError reports why text is not an object of the patch schema, or nil.
// It is the schema only: whether the patch is a good repair is patch.Check's
// question, and the grammar does not claim to answer it.
func schemaError(text string) error {
	decoded, err := patch.Decode([]byte(text))
	if err != nil {
		return err
	}
	if strings.TrimSpace(decoded.Summary) == "" {
		return errors.New("the summary is blank")
	}
	if len(decoded.Files) < 1 || len(decoded.Files) > patch.MaxFiles {
		return fmt.Errorf("%d files", len(decoded.Files))
	}
	for _, file := range decoded.Files {
		if file.Op != patch.OpReplace {
			return fmt.Errorf("the op %q", file.Op)
		}
	}
	return nil
}

func TestIntegrationEveryFinishedGenerationIsSchemaValid(t *testing.T) {
	client := integrationClient(t)
	generations := 100
	if value := os.Getenv("LLAMA_GENERATIONS"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil {
			t.Fatal(err)
		}
		generations = parsed
	}

	var invalid, truncated, tokens int
	var elapsed time.Duration
	for i := range generations {
		fault := catalogue[i%len(catalogue)]
		started := time.Now()
		completion, err := client.Complete(context.Background(), Request{
			System: healerSystem, Prompt: healerPrompt(t, fault), Grammar: PatchGrammar,
			MaxTokens: maxTokens,
			// Well above 0, so that the hundred answers differ and the grammar
			// is tested on a hundred paths through it, not one path a hundred times.
			Temperature: 0.8,
		})
		elapsed += time.Since(started)
		tokens += completion.CompletionTokens
		if errors.Is(err, ErrTruncated) {
			// Not a failure of the grammar: it held the shape up to the limit, and
			// a loop inside a string is what cut it. The healer counts it as one
			// failed attempt; here it counts against maxTruncatedShare.
			truncated++
			t.Logf("generation %d (%s): truncated after %d tokens", i, fault.fixture,
				completion.CompletionTokens)
			continue
		}
		if err == nil {
			err = schemaError(completion.Text)
		}
		if err != nil {
			invalid++
			t.Errorf("generation %d (%s): %v\n%s", i, fault.fixture, err, completion.Text)
			continue
		}
		t.Logf("generation %d: valid, %d tokens at %.1f tokens/s", i,
			completion.CompletionTokens, completion.TokensPerSecond)
	}

	finished := generations - truncated
	t.Logf("%d of %d finished generations schema-valid, %d truncated; %d tokens in %s",
		finished-invalid, finished, truncated, tokens, elapsed.Round(time.Second))
	if share := float64(truncated) / float64(generations); share > maxTruncatedShare {
		t.Errorf("%.0f %% of the generations were truncated, above the %.0f %% allowed",
			share*100, maxTruncatedShare*100)
	}
}

func TestIntegrationADerailingPromptDoesNotDerailTheGrammar(t *testing.T) {
	client := integrationClient(t)
	request := Request{
		System: derailingSystem, Prompt: healerPrompt(t, catalogue[0]),
		MaxTokens: maxTokens, Temperature: 0.8,
	}

	// The control: without the grammar, the prompt must win. If the model
	// answers JSON anyway, this test would prove nothing about the grammar.
	control, err := client.Complete(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if schemaError(control.Text) == nil {
		t.Fatalf("without the grammar the model still answered a patch; the prompt derails "+
			"nothing:\n%s", control.Text)
	}
	t.Logf("without the grammar:\n%s", control.Text)

	request.Grammar = PatchGrammar
	constrained, err := client.Complete(context.Background(), request)
	if err == nil {
		err = schemaError(constrained.Text)
	}
	if err != nil {
		t.Fatalf("with the grammar: %v\n%s", err, constrained.Text)
	}
	t.Logf("with the grammar:\n%s", constrained.Text)
}

func TestIntegrationEmbeddingsHaveTheExpectedDimension(t *testing.T) {
	client := integrationClient(t)
	embed := func(text string) []float32 {
		vector, err := client.Embed(context.Background(), text)
		if err != nil {
			t.Fatal(err)
		}
		return vector
	}

	symptom := embed("The backend cannot reach Kubo: ipfs.upload fails with connection refused.")
	same := embed("Uploads to IPFS fail, the connection from the backend to kubo is refused.")
	other := embed("The pod is OOMKilled and restarts in a loop.")

	// The dimension is checked by Embed itself. This checks the vectors mean
	// something: two wordings of one fault sit closer than two faults.
	if cosine(symptom, same) <= cosine(symptom, other) {
		t.Errorf("cosine(same fault) = %.3f, cosine(other fault) = %.3f",
			cosine(symptom, same), cosine(symptom, other))
	}
	t.Logf("%d dimensions; cosine same fault %.3f, other fault %.3f", len(symptom),
		cosine(symptom, same), cosine(symptom, other))
}

func cosine(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
