package memory

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"time"

	"go.opentelemetry.io/otel/metric"

	"github.com/AxelCharlot/ft_lgtm_agents/internal/patch"
)

// Outcome is how a loop ended.
type Outcome string

const (
	// OutcomeRepaired means the pull request merged and the symptom cleared.
	OutcomeRepaired Outcome = "repaired"
	// OutcomeVetoed means the reviewer refused the pull request.
	OutcomeVetoed Outcome = "vetoed"
	// OutcomeRolledBack means the pull request merged and the symptom stayed, so
	// the merge was reverted.
	OutcomeRolledBack Outcome = "rolled_back"
)

// Entry is one closed loop. Failures are kept too: a patch that was vetoed is
// as useful an example to the model as one that repaired.
type Entry struct {
	Time time.Time `json:"time"`
	// Symptom is the text that was embedded, kept because it is what the model
	// is shown as the example.
	Symptom   string      `json:"symptom"`
	Embedding []float32   `json:"embedding"`
	Diagnosis string      `json:"diagnosis"`
	Patch     patch.Patch `json:"patch"`
	Outcome   Outcome     `json:"outcome"`
}

// Match is an entry and how close it is to the query, from -1 to 1.
type Match struct {
	Entry      Entry
	Similarity float64
}

// fileVersion changes when the file format does, so that an old file is
// refused rather than misread.
const fileVersion = 1

type file struct {
	Version    int     `json:"version"`
	Dimensions int     `json:"dimensions"`
	Entries    []Entry `json:"entries"`
}

// Memory holds every past incident and finds the nearest ones to a symptom.
//
// The search is an exhaustive cosine scan over a slice. An approximate index
// such as HNSW only pays above roughly ten thousand vectors; this memory will
// hold a few hundred, and a scan of those takes well under a millisecond.
type Memory struct {
	path       string
	dimensions int

	mu      sync.Mutex
	entries []Entry
	// norms[i] is the norm of entries[i].Embedding, kept so that a search
	// computes it once per entry and not once per entry per query.
	norms []float64
}

// Open loads the memory from path, or starts an empty one when the file does
// not exist yet.
//
// A file that exists and cannot be read stops the start instead: an agent that
// started with an empty memory over a damaged file would overwrite it at its
// first loop, and every past incident would be gone without a word.
func Open(path string, dimensions int) (*Memory, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("the dimension must be positive, not %d", dimensions)
	}
	memory := &Memory{path: path, dimensions: dimensions}

	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return memory, nil
	}
	if err != nil {
		return nil, fmt.Errorf("could not read the memory: %w", err)
	}

	var stored file
	if err := json.Unmarshal(data, &stored); err != nil {
		return nil, fmt.Errorf("the memory at %s is damaged: %w", path, err)
	}
	if stored.Version != fileVersion {
		return nil, fmt.Errorf("the memory at %s is version %d, and this agent reads version %d",
			path, stored.Version, fileVersion)
	}
	// Vectors of another model cannot be compared with the ones this agent
	// produces: the cosine would be computed, and would mean nothing.
	if stored.Dimensions != dimensions {
		return nil, fmt.Errorf("the memory at %s holds vectors of %d dimensions, not %d",
			path, stored.Dimensions, dimensions)
	}
	for i, entry := range stored.Entries {
		norm, err := memory.validate(entry)
		if err != nil {
			return nil, fmt.Errorf("the entry %d of %s: %w", i, path, err)
		}
		memory.entries = append(memory.entries, entry)
		memory.norms = append(memory.norms, norm)
	}
	return memory, nil
}

// Add records a closed loop and writes the memory to disk before it returns.
// When the write fails, the entry is not kept: what the agent searches is
// always what the file holds.
func (m *Memory) Add(entry Entry) error {
	norm, err := m.validate(entry)
	if err != nil {
		return err
	}
	// The caller keeps its own slices; a change to them later must not reach
	// the memory, which would then differ from its file.
	entry.Embedding = slices.Clone(entry.Embedding)
	entry.Patch.Files = slices.Clone(entry.Patch.Files)

	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, entry)
	m.norms = append(m.norms, norm)
	if err := m.save(); err != nil {
		m.entries = m.entries[:len(m.entries)-1]
		m.norms = m.norms[:len(m.norms)-1]
		return err
	}
	return nil
}

// Nearest returns the k entries closest to query, the closest first. Two
// entries at the same distance come in the order they were added.
func (m *Memory) Nearest(query []float32, k int) ([]Match, error) {
	if len(query) != m.dimensions {
		return nil, fmt.Errorf("the query has %d dimensions, not %d", len(query), m.dimensions)
	}
	queryNorm := norm(query)
	if queryNorm == 0 {
		return nil, errors.New("the query is a zero vector, which is at no distance from anything")
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	matches := make([]Match, len(m.entries))
	for i, entry := range m.entries {
		matches[i] = Match{
			Entry:      entry,
			Similarity: dot(query, entry.Embedding) / (queryNorm * m.norms[i]),
		}
	}
	// Stable, so equal similarities keep the insertion order.
	slices.SortStableFunc(matches, func(a, b Match) int {
		return cmp.Compare(b.Similarity, a.Similarity)
	})
	nearest := matches[:max(0, min(k, len(matches)))]
	// The entries share their slices with the memory; a caller that edits an
	// example before showing it to the model must not edit the memory.
	for i := range nearest {
		nearest[i].Entry.Embedding = slices.Clone(nearest[i].Entry.Embedding)
		nearest[i].Entry.Patch.Files = slices.Clone(nearest[i].Entry.Patch.Files)
	}
	return nearest, nil
}

// Len is the number of entries.
func (m *Memory) Len() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.entries)
}

// Observe reports Len as the gauge selfheal_memory_entries on meter. The gauge
// is read at every collection, so it cannot drift from the real count.
func (m *Memory) Observe(meter metric.Meter) error {
	_, err := meter.Int64ObservableGauge("selfheal_memory_entries",
		metric.WithDescription("Closed loops the incident memory holds."),
		metric.WithUnit("{entry}"),
		metric.WithInt64Callback(func(_ context.Context, observer metric.Int64Observer) error {
			observer.Observe(int64(m.Len()))
			return nil
		}))
	return err
}

// validate checks an entry and returns the norm of its embedding.
func (m *Memory) validate(entry Entry) (float64, error) {
	if len(entry.Embedding) != m.dimensions {
		return 0, fmt.Errorf("the embedding has %d dimensions, not %d",
			len(entry.Embedding), m.dimensions)
	}
	switch entry.Outcome {
	case OutcomeRepaired, OutcomeVetoed, OutcomeRolledBack:
	default:
		return 0, fmt.Errorf("the outcome %q does not exist", entry.Outcome)
	}
	entryNorm := norm(entry.Embedding)
	// A zero vector, or one holding NaN, would give every search a NaN, and a
	// NaN sorts nowhere in particular.
	if entryNorm == 0 || math.IsNaN(entryNorm) || math.IsInf(entryNorm, 0) {
		return 0, fmt.Errorf("the embedding has no usable norm (%v)", entryNorm)
	}
	return entryNorm, nil
}

// save writes the whole memory to a new file and renames it over the old one.
// A rename within one directory is atomic, so a crash in the middle leaves the
// previous file, never half of a new one.
func (m *Memory) save() error {
	data, err := json.Marshal(file{Version: fileVersion, Dimensions: m.dimensions,
		Entries: m.entries})
	if err != nil {
		return err
	}

	directory := filepath.Dir(m.path)
	temporary, err := os.CreateTemp(directory, filepath.Base(m.path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("could not write the memory: %w", err)
	}
	if err := writeAndSync(temporary, data); err != nil {
		return errors.Join(fmt.Errorf("could not write the memory: %w", err),
			os.Remove(temporary.Name()))
	}
	if err := os.Rename(temporary.Name(), m.path); err != nil {
		return errors.Join(fmt.Errorf("could not replace the memory: %w", err),
			os.Remove(temporary.Name()))
	}
	// The rename lives in the directory: until the directory is synced, a power
	// cut can still bring back the old name.
	return syncDirectory(directory)
}

func writeAndSync(temporary *os.File, data []byte) error {
	if _, err := temporary.Write(data); err != nil {
		return errors.Join(err, temporary.Close())
	}
	if err := temporary.Sync(); err != nil {
		return errors.Join(err, temporary.Close())
	}
	return temporary.Close()
}

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("could not sync the memory directory: %w", err)
	}
	if err := directory.Sync(); err != nil {
		return errors.Join(fmt.Errorf("could not sync the memory directory: %w", err),
			directory.Close())
	}
	return directory.Close()
}

func dot(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func norm(vector []float32) float64 {
	return math.Sqrt(dot(vector, vector))
}
