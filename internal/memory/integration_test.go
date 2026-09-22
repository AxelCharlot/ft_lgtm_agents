package memory

// This test runs against a real llama-embed, and skips without it:
//
//	LLAMA_EMBEDDING_URL=http://127.0.0.1:8082 go test -run Integration -v ./internal/memory

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/AxelCharlot/ft_lgtm_agents/internal/llama"
)

// labelled is the hand-checked set: three past incidents per fault of the
// catalogue, worded the way the healer words a symptom — an alert, an event, a
// log line — and never twice the same way.
var labelled = map[string][]string{
	"networkpolicy": {
		"Alert IPFSUploadFailing: ipfs.upload spans end in error, dial tcp kubo:5001: " +
			"connection timed out.",
		"The backend logs: could not pin the source, the request to the Kubo API timed out.",
		"Every run shares nothing: the upload to IPFS fails and the share link stays empty.",
	},
	"memory": {
		"Event OOMKilled on pod lgtm-backend, container backend, exit code 137.",
		"Alert PodCrashLooping: lgtm-backend restarted 5 times in 3 minutes, last state " +
			"terminated with reason OOMKilled.",
		"The backend container is killed by the kernel as soon as it compiles a program; " +
			"memory usage hits the limit.",
	},
	"readiness": {
		"Alert DeploymentNotReady: lgtm-backend has 0 of 1 replicas ready for 5 minutes.",
		"Event Unhealthy on pod lgtm-backend: Readiness probe failed: HTTP probe failed " +
			"with statuscode 404.",
		"The pod lgtm-backend runs without restarting and logs no error, yet it never " +
			"becomes Ready and the Service has no endpoint.",
	},
	"servicename": {
		"Grafana: Logs for this span returns no line for traces of lgtm-backend.",
		"Loki holds no log line with service_name lgtm-backend since the last rollout, " +
			"while Tempo keeps receiving its traces.",
		"The trace to logs link is empty: the service name on the logs no longer matches " +
			"the one the datasource queries.",
	},
}

// queries are new wordings, one per fault, none of them in the memory.
var queries = map[string]string{
	"networkpolicy": "Uploads to IPFS time out: the backend cannot open a connection to " +
		"kubo on port 5001.",
	"memory": "Container backend of lgtm-backend was OOMKilled again and the pod is in " +
		"CrashLoopBackOff.",
	"readiness": "lgtm-backend stays 0/1 Ready; the readiness probe keeps failing although " +
		"the process is healthy.",
	"servicename": "Traces arrive in Tempo but the logs of those spans cannot be found " +
		"in Loki.",
}

func TestIntegrationRetrievalMatchesTheHandLabels(t *testing.T) {
	url := os.Getenv("LLAMA_EMBEDDING_URL")
	if url == "" {
		t.Skip("LLAMA_EMBEDDING_URL is not set")
	}
	client := &llama.Client{EmbeddingURL: url, EmbeddingDimensions: 384}
	embed := func(text string) []float32 {
		vector, err := client.Embed(context.Background(), text)
		if err != nil {
			t.Fatal(err)
		}
		return vector
	}

	path := filepath.Join(t.TempDir(), "memory.json")
	memory, err := Open(path, 384)
	if err != nil {
		t.Fatal(err)
	}
	for fault, symptoms := range labelled {
		for _, symptom := range symptoms {
			stored := entry(symptom, embed(symptom)...)
			stored.Diagnosis = fault
			if err := memory.Add(stored); err != nil {
				t.Fatal(err)
			}
		}
	}

	// Through a restart, so the vectors searched are the ones read back from
	// the file.
	memory, err = Open(path, 384)
	if err != nil {
		t.Fatal(err)
	}
	var right, total int
	for fault, query := range queries {
		matches, err := memory.Nearest(embed(query), 3)
		if err != nil {
			t.Fatal(err)
		}
		// The nearest example must be the right fault. Ranks 2 and 3 are
		// counted, not required: faults that share their words, such as two ways
		// a pod of lgtm-backend fails, sit close for the embedding model too, and
		// that is a property of the model, not of the scan — the hand-computed
		// test proves the scan returns the exact nearest entries.
		if matches[0].Entry.Diagnosis != fault {
			t.Errorf("%s: the nearest entry is %s (%.3f): %s", fault, matches[0].Entry.Diagnosis,
				matches[0].Similarity, matches[0].Entry.Symptom)
		}
		for rank, match := range matches {
			total++
			if match.Entry.Diagnosis == fault {
				right++
			}
			t.Logf("%s: rank %d %s %.3f", fault, rank+1, match.Entry.Diagnosis, match.Similarity)
		}
	}
	t.Logf("%d of %d examples in the top 3 are the right fault", right, total)
}
