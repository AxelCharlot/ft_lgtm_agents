package patch

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
)

// Op is what a patch does to one file.
type Op string

// OpReplace replaces the whole content of a file that already exists. It is the
// only operation: creating or deleting a file adds or removes a resource, and no
// entry of the allowed list can cover that.
const OpReplace Op = "replace"

// File is one file of a patch, with its complete content after the change.
type File struct {
	Path    string `json:"path"`
	Op      Op     `json:"op"`
	Content string `json:"content"`
}

// Patch is a repair as the healer proposes it. Nothing may be done with it
// until Check has turned it into a Checked.
type Patch struct {
	Summary string `json:"summary"`
	Files   []File `json:"files"`
}

// Rule names the rule that refused a patch. The values are stable: they are
// meant to label a metric.
type Rule string

// The rules, in the order Check applies them.
const (
	RuleSchema    Rule = "schema"
	RuleFileCount Rule = "file_count"
	RulePath      Rule = "path"
	RuleEmpty     Rule = "empty"
	RuleShrink    Rule = "shrink"
	RuleYAML      Rule = "yaml"
	RuleUnchanged Rule = "unchanged"
	RuleField     Rule = "field"
)

// Refusal is the error Check and Decode return when a patch breaks a rule. Any
// other error means the check itself could not run.
type Refusal struct {
	Rule Rule
	// Path is the file that broke the rule, or empty when the rule is about the
	// patch as a whole.
	Path   string
	Reason string
}

func (r *Refusal) Error() string {
	if r.Path == "" {
		return fmt.Sprintf("patch refused by rule %s: %s", r.Rule, r.Reason)
	}
	return fmt.Sprintf("patch refused by rule %s on %s: %s", r.Rule, r.Path, r.Reason)
}

func refuse(rule Rule, path, format string, args ...any) *Refusal {
	return &Refusal{Rule: rule, Path: path, Reason: fmt.Sprintf(format, args...)}
}

// Decode reads a patch as the model writes it.
//
// A field the schema does not name is refused rather than ignored: the grammar
// already constrains the shape, so an extra field means the output is not what
// it claims to be.
func Decode(data []byte) (Patch, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	var patch Patch
	if err := decoder.Decode(&patch); err != nil {
		return Patch{}, refuse(RuleSchema, "", "the patch is not JSON of the expected shape: %v", err)
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return Patch{}, refuse(RuleSchema, "", "the patch is followed by more data")
	}
	return patch, nil
}

// Checked is a patch that passed every rule.
//
// Check is the only function that returns one with files in it, so a GitHub
// client that takes a Checked cannot be handed a patch that skipped the rules.
// The zero value holds no file, and a client must refuse it.
type Checked struct {
	summary string
	files   []File
}

// Summary is the one-line description of the repair.
func (c Checked) Summary() string {
	return c.summary
}

// Files returns a copy of the files, so no caller can change a checked patch
// after the fact.
func (c Checked) Files() []File {
	return slices.Clone(c.files)
}
