package memory

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"

	"github.com/AxelCharlot/ft_lgtm_agents/internal/patch"
)

func entry(symptom string, embedding ...float32) Entry {
	return Entry{
		Time:      time.Date(2026, 9, 22, 10, 0, 0, 0, time.UTC),
		Symptom:   symptom,
		Embedding: embedding,
		Diagnosis: "the diagnosis of " + symptom,
		Patch: patch.Patch{Summary: "repair " + symptom, Files: []patch.File{
			{Path: "k8s/10-backend-deployment.yaml", Op: patch.OpReplace, Content: "kind: Deployment\n"},
		}},
		Outcome: OutcomeRepaired,
	}
}

// handChecked is the set every cosine of which can be worked out by hand
// against the query (3, 4, 0), whose norm is 5.
var handChecked = []struct {
	entry Entry
	// similarity is the cosine with the query, computed by hand.
	similarity float64
}{
	{entry("a", 1, 0, 0), 0.6},   // 3 / 5
	{entry("b", 0, 1, 0), 0.8},   // 4 / 5
	{entry("c", 0, 0, 1), 0},     // orthogonal
	{entry("d", -1, 0, 0), -0.6}, // -3 / 5
	{entry("e", 6, 8, 0), 1},     // the query, doubled: the norm does not count
	{entry("f", 4, 3, 0), 0.96},  // (12 + 12) / (5 × 5)
	{entry("g", 0, 2, 0), 0.8},   // the same direction as b, added after it
}

var query = []float32{3, 4, 0}

func openFilled(t *testing.T, path string) *Memory {
	t.Helper()
	memory, err := Open(path, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range handChecked {
		if err := memory.Add(item.entry); err != nil {
			t.Fatal(err)
		}
	}
	return memory
}

func symptoms(matches []Match) string {
	names := make([]string, len(matches))
	for i, match := range matches {
		names[i] = match.Entry.Symptom
	}
	return strings.Join(names, " ")
}

func TestNearestMatchesTheHandCheckedOrder(t *testing.T) {
	memory := openFilled(t, filepath.Join(t.TempDir(), "memory.json"))

	cases := map[int]string{
		3: "e f b",
		// b and g tie at 0.8; b was added first, so b comes first.
		4:  "e f b g",
		10: "e f b g a c d",
		0:  "",
		-1: "",
	}
	for k, want := range cases {
		matches, err := memory.Nearest(query, k)
		if err != nil {
			t.Fatal(err)
		}
		if got := symptoms(matches); got != want {
			t.Errorf("Nearest(k=%d) = %q, want %q", k, got, want)
		}
	}

	matches, err := memory.Nearest(query, len(handChecked))
	if err != nil {
		t.Fatal(err)
	}
	for _, match := range matches {
		for _, item := range handChecked {
			if item.entry.Symptom == match.Entry.Symptom &&
				math.Abs(item.similarity-match.Similarity) > 1e-9 {
				t.Errorf("similarity of %s = %v, computed by hand %v",
					match.Entry.Symptom, match.Similarity, item.similarity)
			}
		}
	}
}

func TestEqualDistancesKeepTheInsertionOrder(t *testing.T) {
	memory, err := Open(filepath.Join(t.TempDir(), "memory.json"), 3)
	if err != nil {
		t.Fatal(err)
	}
	// Five directions, ten entries each, added interleaved. The sort has to
	// move entries across one another, so an unstable sort shows; with one
	// value, or with a handful of entries, the standard library happens to keep
	// the order anyway.
	directions := [][]float32{{3, 4, 0}, {4, 3, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}}
	want := make([][]string, len(directions))
	for i := range 50 {
		direction := i % len(directions)
		name := strconv.Itoa(i)
		want[direction] = append(want[direction], name)
		if err := memory.Add(entry(name, directions[direction]...)); err != nil {
			t.Fatal(err)
		}
	}
	matches, err := memory.Nearest(query, 50)
	if err != nil {
		t.Fatal(err)
	}
	// The directions come in the order of their cosine with the query, 1, 0.96,
	// 0.8, 0.6 and 0, which is the order they are listed in.
	var ordered []string
	for _, names := range want {
		ordered = append(ordered, names...)
	}
	if got := symptoms(matches); got != strings.Join(ordered, " ") {
		t.Errorf("Nearest = %q\nwant      %q", got, strings.Join(ordered, " "))
	}
}

func TestNearestOnAnEmptyMemory(t *testing.T) {
	memory, err := Open(filepath.Join(t.TempDir(), "memory.json"), 3)
	if err != nil {
		t.Fatal(err)
	}
	matches, err := memory.Nearest(query, 3)
	if err != nil || len(matches) != 0 {
		t.Errorf("Nearest = %v, %v, want nothing", matches, err)
	}
}

func TestNearestRefusesAQueryItCannotCompare(t *testing.T) {
	memory := openFilled(t, filepath.Join(t.TempDir(), "memory.json"))
	for name, query := range map[string][]float32{
		"another dimension": {1, 0},
		"a zero vector":     {0, 0, 0},
	} {
		if _, err := memory.Nearest(query, 3); err == nil {
			t.Errorf("Nearest accepted %s", name)
		}
	}
}

func TestAddRefuses(t *testing.T) {
	unknown := entry("x", 1, 0, 0)
	unknown.Outcome = "merged"
	cases := map[string]Entry{
		"another dimension":  entry("x", 1, 0),
		"a zero vector":      entry("x", 0, 0, 0),
		"a NaN":              entry("x", float32(math.NaN()), 0, 0),
		"an unknown outcome": unknown,
	}
	for name, refused := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "memory.json")
			memory, err := Open(path, 3)
			if err != nil {
				t.Fatal(err)
			}
			if err := memory.Add(refused); err == nil {
				t.Fatal("Add accepted it")
			}
			if memory.Len() != 0 {
				t.Errorf("Len = %d after a refusal", memory.Len())
			}
			if _, err := os.Stat(path); !os.IsNotExist(err) {
				t.Errorf("a refused entry wrote the file: %v", err)
			}
		})
	}
}

func TestTheMemorySurvivesARestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.json")
	before := openFilled(t, path)
	wantMatches, err := before.Nearest(query, 3)
	if err != nil {
		t.Fatal(err)
	}

	after, err := Open(path, 3)
	if err != nil {
		t.Fatal(err)
	}
	if after.Len() != len(handChecked) {
		t.Fatalf("Len = %d after the restart, want %d", after.Len(), len(handChecked))
	}
	gotMatches, err := after.Nearest(query, 3)
	if err != nil {
		t.Fatal(err)
	}
	// Every field, not just the order: the diagnosis and the patch are what the
	// model is shown.
	if !reflect.DeepEqual(gotMatches, wantMatches) {
		t.Errorf("after the restart:\n%+v\nbefore:\n%+v", gotMatches, wantMatches)
	}

	// No temporary file left behind next to the memory.
	files, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 {
		t.Errorf("the directory holds %d files, want the memory alone", len(files))
	}
}

func TestOpenRefusesAFileItCannotTrust(t *testing.T) {
	cases := map[string]string{
		"a damaged file":  `{"version": 1, "dimensions": 3, "entries": [`,
		"another version": `{"version": 2, "dimensions": 3, "entries": []}`,
		"another model":   `{"version": 1, "dimensions": 384, "entries": []}`,
		"an entry that lies": `{"version": 1, "dimensions": 3, "entries": [{"embedding": [1, 0],
			"outcome": "repaired"}]}`,
	}
	for name, content := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "memory.json")
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := Open(path, 3); err == nil {
				t.Error("Open accepted it")
			}
		})
	}
}

func TestAFailedWriteKeepsNothing(t *testing.T) {
	// A directory that does not exist: the temporary file cannot be created.
	memory, err := Open(filepath.Join(t.TempDir(), "missing", "memory.json"), 3)
	if err != nil {
		t.Fatal(err)
	}
	if err := memory.Add(entry("a", 1, 0, 0)); err == nil {
		t.Fatal("Add reported no error")
	}
	if memory.Len() != 0 {
		t.Errorf("Len = %d: the memory holds an entry the file does not", memory.Len())
	}
}

func TestTheMemoryOwnsItsVectors(t *testing.T) {
	memory, err := Open(filepath.Join(t.TempDir(), "memory.json"), 3)
	if err != nil {
		t.Fatal(err)
	}
	added := entry("a", 1, 0, 0)
	if err := memory.Add(added); err != nil {
		t.Fatal(err)
	}
	added.Embedding[0] = -1
	added.Patch.Files[0].Content = "kind: Evil\n"

	matches, err := memory.Nearest([]float32{1, 0, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	matches[0].Entry.Embedding[0] = -1
	matches[0].Entry.Patch.Files[0].Content = "kind: Evil\n"

	again, err := memory.Nearest([]float32{1, 0, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if again[0].Similarity != 1 || again[0].Entry.Patch.Files[0].Content != "kind: Deployment\n" {
		t.Errorf("a caller changed the memory: %+v", again[0])
	}
}

func TestTheGaugeReportsTheRealCount(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.json")
	scenarios := []struct {
		name  string
		open  func(t *testing.T) *Memory
		count int64
	}{
		{"seven entries added", func(t *testing.T) *Memory { return openFilled(t, path) }, 7},
		{"the same seven reloaded, and one more", func(t *testing.T) *Memory {
			memory, err := Open(path, 3)
			if err != nil {
				t.Fatal(err)
			}
			if err := memory.Add(entry("h", 1, 1, 1)); err != nil {
				t.Fatal(err)
			}
			return memory
		}, 8},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			reader := sdkmetric.NewManualReader()
			provider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
			t.Cleanup(func() {
				if err := provider.Shutdown(context.Background()); err != nil {
					t.Error(err)
				}
			})

			memory := scenario.open(t)
			if err := memory.Observe(provider.Meter("test")); err != nil {
				t.Fatal(err)
			}
			var collected metricdata.ResourceMetrics
			if err := reader.Collect(context.Background(), &collected); err != nil {
				t.Fatal(err)
			}
			if got := gaugeValue(t, collected, "selfheal_memory_entries"); got != scenario.count {
				t.Errorf("selfheal_memory_entries = %d, want %d", got, scenario.count)
			}
		})
	}
}

func gaugeValue(t *testing.T, collected metricdata.ResourceMetrics, name string) int64 {
	t.Helper()
	for _, scope := range collected.ScopeMetrics {
		for _, instrument := range scope.Metrics {
			if instrument.Name != name {
				continue
			}
			gauge, ok := instrument.Data.(metricdata.Gauge[int64])
			if !ok || len(gauge.DataPoints) != 1 {
				t.Fatalf("%s is %T with the wrong shape", name, instrument.Data)
			}
			return gauge.DataPoints[0].Value
		}
	}
	t.Fatalf("%s was not collected", name)
	return 0
}
