package patch

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v3"
)

// MaxFiles is the most files one patch may replace. Every fault of the
// catalogue lives in one field of one file; plan.md B.5 fixes the bound.
const MaxFiles = 3

// maxShrink is the largest share of its bytes a file may lose in one patch,
// from plan.md B.5. A model that answers with part of a file shrinks it far
// beyond this, and this is the rule that notices.
const maxShrink = 0.6

// Check applies every rule to patch, and returns it as a Checked only when all
// of them pass.
//
// current returns a file as it is on the branch the pull request targets, and
// an error wrapping fs.ErrNotExist when there is none. A refusal is a *Refusal;
// any other error means current failed and nothing was decided.
func Check(patch Patch, policy *Policy,
	current func(name string) ([]byte, error)) (Checked, error) {
	if strings.TrimSpace(patch.Summary) == "" {
		return Checked{}, refuse(RuleSchema, "", "the patch has no summary")
	}
	if len(patch.Files) == 0 {
		return Checked{}, refuse(RuleSchema, "", "the patch replaces no file")
	}
	if len(patch.Files) > MaxFiles {
		return Checked{}, refuse(RuleFileCount, "",
			"the patch replaces %d files, and the limit is %d", len(patch.Files), MaxFiles)
	}

	seen := map[string]bool{}
	for _, file := range patch.Files {
		if file.Op != OpReplace {
			return Checked{}, refuse(RuleSchema, file.Path,
				"the operation %q does not exist; the only one is %q", file.Op, OpReplace)
		}
		if seen[file.Path] {
			return Checked{}, refuse(RuleSchema, file.Path, "the patch replaces this file twice")
		}
		seen[file.Path] = true

		if err := checkFile(file, policy, current); err != nil {
			return Checked{}, err
		}
	}
	return Checked{summary: patch.Summary, files: slices.Clone(patch.Files)}, nil
}

func checkFile(file File, policy *Policy, current func(name string) ([]byte, error)) error {
	// A path that is not already clean could name one file two ways, and only
	// one of them would match the allowed list.
	if !fs.ValidPath(file.Path) || file.Path == "." {
		return refuse(RulePath, file.Path, "the path is not a clean path inside the repository")
	}
	if !policy.allowsPath(file.Path) {
		return refuse(RulePath, file.Path, "the allowed list does not include this file")
	}

	before, err := current(file.Path)
	if errors.Is(err, fs.ErrNotExist) {
		return refuse(RulePath, file.Path, "the file does not exist, and a patch only replaces files")
	}
	if err != nil {
		return fmt.Errorf("could not read %s: %w", file.Path, err)
	}
	after := []byte(file.Content)

	if len(bytes.TrimSpace(before)) > 0 && len(bytes.TrimSpace(after)) == 0 {
		return refuse(RuleEmpty, file.Path, "the patch empties the file")
	}
	if float64(len(after)) < (1-maxShrink)*float64(len(before)) {
		return refuse(RuleShrink, file.Path, "the file shrinks from %d to %d bytes, "+
			"more than the %.0f %% allowed", len(before), len(after), maxShrink*100)
	}
	return checkFields(file.Path, before, after, policy)
}

// checkFields refuses the file unless every field the patch changes is on the
// allowed list for the kind of its resource.
func checkFields(name string, before, after []byte, policy *Policy) error {
	// The current file is read with the same parser, so a file the repository
	// already holds but cannot be parsed is refused too: nothing can say what a
	// patch to it changes.
	oldDocuments, err := parseDocuments(before)
	if err != nil {
		return refuse(RuleYAML, name, "the current file is not valid YAML: %v", err)
	}
	newDocuments, err := parseDocuments(after)
	if err != nil {
		return refuse(RuleYAML, name, "the patched file is not valid YAML: %v", err)
	}
	if len(oldDocuments) != len(newDocuments) {
		return refuse(RuleField, name, "the patch changes the number of resources from %d to %d",
			len(oldDocuments), len(newDocuments))
	}

	changedAny := false
	for i := range oldDocuments {
		var changed [][]string
		diff(oldDocuments[i], newDocuments[i], nil, &changed)
		// The kind comes from the current resource. A patch that changes the kind
		// changes the field "kind", which no entry allows.
		kind := kindOf(oldDocuments[i])
		for _, field := range changed {
			if !policy.allowsField(kind, field) {
				return refuse(RuleField, name, "the field %s of the %s is not on the allowed list",
					renderPath(field), describeKind(kind))
			}
		}
		changedAny = changedAny || len(changed) > 0
	}
	if !changedAny {
		// Comments and layout only. A pull request that changes no field repairs
		// nothing, and Argo CD would have nothing to apply.
		return refuse(RuleUnchanged, name, "the patch changes no field of the file")
	}
	return nil
}

func parseDocuments(data []byte) ([]any, error) {
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	var documents []any
	for {
		var document any
		err := decoder.Decode(&document)
		if errors.Is(err, io.EOF) {
			return documents, nil
		}
		if err != nil {
			return nil, err
		}
		// An empty document between two separators is no resource.
		if document != nil {
			documents = append(documents, document)
		}
	}
}

func kindOf(document any) string {
	resource, ok := document.(map[string]any)
	if !ok {
		return ""
	}
	kind, _ := resource["kind"].(string)
	return kind
}

func describeKind(kind string) string {
	if kind == "" {
		return "resource with no kind"
	}
	return kind
}

// diff appends to changed the path of every field that differs between before
// and after, as deep as the two trees keep the same shape.
func diff(before, after any, at []string, changed *[][]string) {
	switch old := before.(type) {
	case map[string]any:
		updated, ok := after.(map[string]any)
		if !ok {
			*changed = append(*changed, at)
			return
		}
		keys := make([]string, 0, len(old)+len(updated))
		for key := range old {
			keys = append(keys, key)
		}
		for key := range updated {
			if _, found := old[key]; !found {
				keys = append(keys, key)
			}
		}
		// Sorted, so the first field a refusal names is the same on every run.
		slices.Sort(keys)
		for _, key := range keys {
			diffChild(old, updated, key, key, at, changed)
		}
	case []any:
		updated, ok := after.([]any)
		if !ok {
			*changed = append(*changed, at)
			return
		}
		diffList(old, updated, at, changed)
	default:
		if !reflect.DeepEqual(before, after) {
			*changed = append(*changed, at)
		}
	}
}

// diffList compares two lists item by item. Items that all carry a name, as
// containers, ports and environment variables do, are matched by that name, so
// adding one item does not read as a change to every item after it. Other
// lists are matched by position, and a change of length changes the list.
func diffList(before, after []any, at []string, changed *[][]string) {
	oldByName, oldNamed := itemsByName(before)
	newByName, newNamed := itemsByName(after)
	if oldNamed && newNamed {
		names := make([]string, 0, len(oldByName)+len(newByName))
		for name := range oldByName {
			names = append(names, name)
		}
		for name := range newByName {
			if _, found := oldByName[name]; !found {
				names = append(names, name)
			}
		}
		slices.Sort(names)
		for _, name := range names {
			diffChild(oldByName, newByName, name, "[name="+name+"]", at, changed)
		}
		return
	}

	if len(before) != len(after) {
		*changed = append(*changed, at)
		return
	}
	for i := range before {
		diff(before[i], after[i], child(at, "["+strconv.Itoa(i)+"]"), changed)
	}
}

// diffChild compares the entry key of two maps, and records segment as changed
// when only one side has it.
func diffChild(before, after map[string]any, key, segment string,
	at []string, changed *[][]string) {
	oldValue, inOld := before[key]
	newValue, inNew := after[key]
	if inOld != inNew {
		*changed = append(*changed, child(at, segment))
		return
	}
	diff(oldValue, newValue, child(at, segment), changed)
}

// itemsByName indexes a list by the name of its items. It reports false when
// one item has no name, or two items share one.
func itemsByName(items []any) (map[string]any, bool) {
	byName := make(map[string]any, len(items))
	for _, item := range items {
		fields, ok := item.(map[string]any)
		if !ok {
			return nil, false
		}
		name, ok := fields["name"].(string)
		if !ok || name == "" {
			return nil, false
		}
		if _, taken := byName[name]; taken {
			return nil, false
		}
		byName[name] = item
	}
	return byName, true
}

// child returns a new path; appending to at in place would let two siblings
// share, and overwrite, one backing array.
func child(at []string, segment string) []string {
	return append(slices.Clip(at), segment)
}

func renderPath(segments []string) string {
	if len(segments) == 0 {
		return "(the whole document)"
	}
	var rendered strings.Builder
	for i, segment := range segments {
		if i > 0 && !strings.HasPrefix(segment, "[") {
			rendered.WriteByte('.')
		}
		rendered.WriteString(segment)
	}
	return rendered.String()
}
