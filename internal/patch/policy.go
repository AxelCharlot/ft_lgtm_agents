package patch

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"path"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v3"
)

// Policy is the allowed list: which files a patch may replace, and which fields
// of which kind of resource it may change inside them.
//
// It is data, read by ParsePolicy, and never a list in the code. The catalogue
// of faults grows; the code that enforces the list does not change with it.
type Policy struct {
	paths  []string
	fields map[string][][]string
}

type policyFile struct {
	// Paths are path.Match patterns, relative to the root of ft_lgtm_gitops.
	Paths  []string `yaml:"paths"`
	Fields []struct {
		Kind string `yaml:"kind"`
		Path string `yaml:"path"`
	} `yaml:"fields"`
}

// ParsePolicy reads an allowed list.
//
// A field path is written like spec.template.spec.containers[*].env[name=X].value:
// [*] matches any item of a list, [name=X] the item whose name is X, and [0] the
// first item of a list whose items have no name. A path allows everything below
// it. A key that contains a dot cannot be written, and so can never be allowed.
func ParsePolicy(data []byte) (*Policy, error) {
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	// A misspelt key would otherwise read as an empty list, and an empty list
	// refuses everything without saying why.
	decoder.KnownFields(true)

	var file policyFile
	if err := decoder.Decode(&file); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, errors.New("the policy is empty")
		}
		return nil, fmt.Errorf("could not read the policy: %w", err)
	}
	if len(file.Paths) == 0 {
		return nil, errors.New("the policy allows no path")
	}
	for _, pattern := range file.Paths {
		if _, err := path.Match(pattern, ""); err != nil {
			return nil, fmt.Errorf("the path pattern %q is not valid: %w", pattern, err)
		}
	}

	policy := &Policy{paths: file.Paths, fields: map[string][][]string{}}
	for _, field := range file.Fields {
		if field.Kind == "" {
			return nil, fmt.Errorf("the field %q names no kind", field.Path)
		}
		segments, err := parseFieldPath(field.Path)
		if err != nil {
			return nil, fmt.Errorf("the field of kind %s: %w", field.Kind, err)
		}
		policy.fields[field.Kind] = append(policy.fields[field.Kind], segments)
	}
	return policy, nil
}

// parseFieldPath splits a field path into its segments: one per key, and one
// per list selector, brackets included.
func parseFieldPath(text string) ([]string, error) {
	if text == "" {
		return nil, errors.New("the field path is empty")
	}
	var segments []string
	for rest := text; rest != ""; {
		if rest[0] == '[' {
			end := strings.IndexByte(rest, ']')
			if end < 0 {
				return nil, fmt.Errorf("the field path %q has an unclosed bracket", text)
			}
			selector := rest[:end+1]
			if !validSelector(selector) {
				return nil, fmt.Errorf("the field path %q has the selector %s; "+
					"only [*], [name=X] and [N] exist", text, selector)
			}
			segments = append(segments, selector)
			rest = rest[end+1:]
		} else {
			end := strings.IndexAny(rest, ".[")
			if end < 0 {
				end = len(rest)
			}
			if end == 0 {
				return nil, fmt.Errorf("the field path %q has an empty key", text)
			}
			segments = append(segments, rest[:end])
			rest = rest[end:]
		}
		if strings.HasPrefix(rest, ".") {
			rest = rest[1:]
			if rest == "" {
				return nil, fmt.Errorf("the field path %q ends with a dot", text)
			}
		}
	}
	return segments, nil
}

func validSelector(selector string) bool {
	inside := selector[1 : len(selector)-1]
	if inside == "*" {
		return true
	}
	if name, found := strings.CutPrefix(inside, "name="); found {
		return name != ""
	}
	index, err := strconv.Atoi(inside)
	return err == nil && index >= 0
}

// allowsPath reports whether a patch may replace the file at name.
func (p *Policy) allowsPath(name string) bool {
	for _, pattern := range p.paths {
		// The pattern was validated by ParsePolicy, so Match cannot fail here.
		if matched, _ := path.Match(pattern, name); matched {
			return true
		}
	}
	return false
}

// allowsField reports whether a change at changed, in a resource of the given
// kind, falls under one of the allowed field paths.
func (p *Policy) allowsField(kind string, changed []string) bool {
	for _, allowed := range p.fields[kind] {
		if covers(allowed, changed) {
			return true
		}
	}
	return false
}

// covers reports whether allowed is changed or one of its ancestors. A change
// above the allowed path — the whole list, when only one field of its items is
// allowed — is not covered.
func covers(allowed, changed []string) bool {
	if len(allowed) > len(changed) {
		return false
	}
	for i, segment := range allowed {
		if segment == "[*]" && strings.HasPrefix(changed[i], "[") {
			continue
		}
		if segment != changed[i] {
			return false
		}
	}
	return true
}
