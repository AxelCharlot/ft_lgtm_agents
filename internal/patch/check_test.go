package patch

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"
)

const (
	deploymentPath    = "k8s/10-backend-deployment.yaml"
	networkPolicyPath = "k8s/80-backend-networkpolicy.yaml"
)

func testPolicy(t *testing.T) *Policy {
	t.Helper()
	policy, err := ParsePolicy(readFixture(t, "policy.yaml"))
	if err != nil {
		t.Fatalf("ParsePolicy: %v", err)
	}
	return policy
}

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// repository serves the two fixtures under the paths they would have in
// ft_lgtm_gitops, and a selfheal.yml: a file outside the allowed paths must be
// refused for where it is, not because it happens to be missing.
func repository(t *testing.T) func(string) ([]byte, error) {
	files := map[string][]byte{
		deploymentPath:    readFixture(t, "deployment.yaml"),
		networkPolicyPath: readFixture(t, "networkpolicy.yaml"),
		"selfheal.yml":    []byte("enabled: true\n"),
	}
	return func(name string) ([]byte, error) {
		// Resolved as a file system would, so a path that climbs reaches the
		// file it climbs to.
		data, found := files[path.Clean(name)]
		if !found {
			return nil, fmt.Errorf("open %s: %w", name, fs.ErrNotExist)
		}
		return data, nil
	}
}

// edit returns the fixture with old replaced by new, and fails when old is not
// in it: a test whose edit silently misses checks the unchanged file.
func edit(t *testing.T, fixture, old, new string) string {
	t.Helper()
	text := string(readFixture(t, fixture))
	if strings.Count(text, old) != 1 {
		t.Fatalf("%q occurs %d times in %s, not once", old, strings.Count(text, old), fixture)
	}
	return strings.Replace(text, old, new, 1)
}

func replace(path, content string) File {
	return File{Path: path, Op: OpReplace, Content: content}
}

func onePatch(files ...File) Patch {
	return Patch{Summary: "raise the memory limit of the backend", Files: files}
}

func TestCheckAcceptsTheRepairsOfTheCatalogue(t *testing.T) {
	cases := map[string]func(t *testing.T) Patch{
		"fault 1: the NetworkPolicy lets the backend reach the Collector": func(t *testing.T) Patch {
			return onePatch(replace(networkPolicyPath, edit(t, "networkpolicy.yaml",
				"        - port: 5001\n",
				"        - port: 5001\n"+
					"    - to:\n"+
					"        - podSelector:\n"+
					"            matchLabels:\n"+
					"              app: otel-collector\n"+
					"      ports:\n"+
					"        - port: 4317\n")))
		},
		"fault 2: the memory limit goes back up": func(t *testing.T) Patch {
			return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
				"              memory: 1Gi", "              memory: 2Gi")))
		},
		"fault 3: the readiness probe points at the right path": func(t *testing.T) Patch {
			return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
				"              path: /healthz\n              port: http\n            periodSeconds: 5",
				"              path: /healthz\n              port: 8080\n            periodSeconds: 5")))
		},
		"fault 4: the service name is spelt right again": func(t *testing.T) Patch {
			return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
				"value: lgtm-backend\n", "value: lgtm-backend-fixed\n")))
		},
		"two files in one patch": func(t *testing.T) Patch {
			return onePatch(
				replace(deploymentPath, edit(t, "deployment.yaml",
					"              memory: 1Gi", "              memory: 2Gi")),
				replace(networkPolicyPath, edit(t, "networkpolicy.yaml",
					"port: 5001", "port: 5002")))
		},
		"a change of layout around an allowed change": func(t *testing.T) Patch {
			// Flow style and a comment change no field, so only the limit counts.
			return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
				"            limits:\n              memory: 1Gi",
				"            # raised after an OOMKilled\n            limits: {memory: 2Gi}")))
		},
	}
	for name, build := range cases {
		t.Run(name, func(t *testing.T) {
			patch := build(t)
			checked, err := Check(patch, testPolicy(t), repository(t))
			if err != nil {
				t.Fatalf("Check refused a valid repair: %v", err)
			}
			if checked.Summary() != patch.Summary || len(checked.Files()) != len(patch.Files) {
				t.Errorf("Checked = %q with %d files, want the patch unchanged",
					checked.Summary(), len(checked.Files()))
			}
		})
	}
}

func TestCheckRefuses(t *testing.T) {
	cases := []struct {
		name string
		// field, when set, must appear in the reason: the refusal has to name
		// what it refused, or nobody can act on it.
		field string
		rule  Rule
		patch func(t *testing.T) Patch
	}{
		{
			name: "a patch with no summary",
			rule: RuleSchema,
			patch: func(t *testing.T) Patch {
				return Patch{Files: []File{replace(deploymentPath, "")}}
			},
		},
		{
			name:  "a patch with no file",
			rule:  RuleSchema,
			patch: func(t *testing.T) Patch { return onePatch() },
		},
		{
			name: "an operation other than replace",
			rule: RuleSchema,
			patch: func(t *testing.T) Patch {
				return onePatch(File{Path: deploymentPath, Op: "delete"})
			},
		},
		{
			name: "the same file twice",
			rule: RuleSchema,
			patch: func(t *testing.T) Patch {
				content := edit(t, "deployment.yaml", "memory: 1Gi", "memory: 2Gi")
				return onePatch(replace(deploymentPath, content), replace(deploymentPath, content))
			},
		},
		{
			name: "more than three files",
			rule: RuleFileCount,
			patch: func(t *testing.T) Patch {
				return onePatch(replace("k8s/a.yaml", "a: 1"), replace("k8s/b.yaml", "b: 1"),
					replace("k8s/c.yaml", "c: 1"), replace("k8s/d.yaml", "d: 1"))
			},
		},
		{
			name:  "a path that climbs out of the repository",
			rule:  RulePath,
			patch: func(t *testing.T) Patch { return onePatch(replace("../k8s/a.yaml", "a: 1")) },
		},
		{
			name:  "an absolute path",
			rule:  RulePath,
			patch: func(t *testing.T) Patch { return onePatch(replace("/"+deploymentPath, "a: 1")) },
		},
		{
			name: "a path that is not clean",
			rule: RulePath,
			patch: func(t *testing.T) Patch {
				return onePatch(replace("k8s/./10-backend-deployment.yaml", "a: 1"))
			},
		},
		{
			name:  "the file of the hard bounds",
			rule:  RulePath,
			field: "the allowed list does not include this file",
			patch: func(t *testing.T) Patch {
				return onePatch(replace("selfheal.yml", "enabled: false"))
			},
		},
		{
			name: "a file that does not exist",
			rule: RulePath,
			patch: func(t *testing.T) Patch {
				return onePatch(replace("k8s/99-new.yaml", "kind: Deployment"))
			},
		},
		{
			name:  "an emptied file",
			rule:  RuleEmpty,
			patch: func(t *testing.T) Patch { return onePatch(replace(deploymentPath, " \n\n")) },
		},
		{
			name: "a file cut by more than 60 %",
			rule: RuleShrink,
			patch: func(t *testing.T) Patch {
				whole := string(readFixture(t, "deployment.yaml"))
				return onePatch(replace(deploymentPath, whole[:len(whole)*39/100]))
			},
		},
		{
			name: "a patched file that is not YAML",
			rule: RuleYAML,
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
					"memory: 1Gi", "memory: [1Gi")))
			},
		},
		{
			name: "a change of comments only",
			rule: RuleUnchanged,
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, "# repaired\n"+
					string(readFixture(t, "deployment.yaml"))))
			},
		},
		{
			name:  "a new image",
			rule:  RuleField,
			field: "spec.template.spec.containers[name=backend].image",
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
					"image: lgtm/backend:v1", "image: evil/backend:v1")))
			},
		},
		{
			name:  "a limit next to the allowed one",
			rule:  RuleField,
			field: "spec.template.spec.containers[name=backend].resources.limits.cpu",
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
					"              memory: 1Gi", "              memory: 1Gi\n              cpu: 100m")))
			},
		},
		{
			name:  "the liveness probe, sibling of the allowed readiness probe",
			rule:  RuleField,
			field: "spec.template.spec.containers[name=backend].livenessProbe.periodSeconds",
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
					"periodSeconds: 10", "periodSeconds: 60")))
			},
		},
		{
			name:  "another environment variable",
			rule:  RuleField,
			field: "env[name=OTEL_EXPORTER_OTLP_ENDPOINT].value",
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
					"otel-collector.lgtm.svc", "attacker.example")))
			},
		},
		{
			name:  "a new environment variable",
			rule:  RuleField,
			field: "env[name=DEBUG]",
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
					"          env:\n", "          env:\n            - name: DEBUG\n"+
						"              value: \"1\"\n")))
			},
		},
		{
			name:  "a renamed resource",
			rule:  RuleField,
			field: "metadata.name",
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
					"  name: lgtm-backend\n  namespace", "  name: other\n  namespace")))
			},
		},
		{
			name:  "a new kind",
			rule:  RuleField,
			field: "kind",
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
					"kind: Deployment", "kind: StatefulSet")))
			},
		},
		{
			name: "a second resource in the file",
			rule: RuleField,
			patch: func(t *testing.T) Patch {
				return onePatch(replace(deploymentPath, string(readFixture(t, "deployment.yaml"))+
					"---\n"+string(readFixture(t, "networkpolicy.yaml"))))
			},
		},
		{
			name:  "a field allowed for another kind",
			rule:  RuleField,
			field: "spec.podSelector",
			patch: func(t *testing.T) Patch {
				return onePatch(replace(networkPolicyPath, edit(t, "networkpolicy.yaml",
					"  podSelector:\n    matchLabels:\n      app: lgtm-backend",
					"  podSelector:\n    matchLabels:\n      app: anything")))
			},
		},
		{
			name: "a refused file next to an allowed one",
			rule: RuleField,
			patch: func(t *testing.T) Patch {
				return onePatch(
					replace(deploymentPath, edit(t, "deployment.yaml", "memory: 1Gi", "memory: 2Gi")),
					replace(networkPolicyPath, edit(t, "networkpolicy.yaml",
						"namespace: lgtm", "namespace: default")))
			},
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			checked, err := Check(test.patch(t), testPolicy(t), repository(t))

			var refusal *Refusal
			if !errors.As(err, &refusal) {
				t.Fatalf("Check = %v, want a refusal by rule %s", err, test.rule)
			}
			if refusal.Rule != test.rule {
				t.Errorf("refused by rule %s (%v), want %s", refusal.Rule, refusal, test.rule)
			}
			if !strings.Contains(refusal.Reason, test.field) {
				t.Errorf("the reason %q does not name %s", refusal.Reason, test.field)
			}
			// What a GitHub client would receive. It must carry nothing.
			if len(checked.Files()) != 0 || checked.Summary() != "" {
				t.Errorf("a refused patch came back with %d files", len(checked.Files()))
			}
		})
	}
}

func TestCheckRefusesAPathThatClimbsThroughAGlob(t *testing.T) {
	// The glob alone matches: * accepts "..". Only the check for a clean path
	// stands between this patch and selfheal.yml.
	policy, err := ParsePolicy([]byte("paths: [k8s/*/*.yml]\n"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = Check(onePatch(replace("k8s/../selfheal.yml", "enabled: false\n")),
		policy, repository(t))

	var refusal *Refusal
	if !errors.As(err, &refusal) || refusal.Rule != RulePath {
		t.Fatalf("Check = %v, want a refusal by rule %s", err, RulePath)
	}
}

func TestCheckReportsAFailedReadAsAnErrorNotARefusal(t *testing.T) {
	broken := func(string) ([]byte, error) { return nil, errors.New("the clone is gone") }
	_, err := Check(onePatch(replace(deploymentPath, "a: 1")), testPolicy(t), broken)

	var refusal *Refusal
	if err == nil || errors.As(err, &refusal) {
		t.Fatalf("Check = %v, want an error that is not a refusal", err)
	}
}

func TestCheckedCannotBeChangedAfterTheCheck(t *testing.T) {
	patch := onePatch(replace(deploymentPath, edit(t, "deployment.yaml",
		"memory: 1Gi", "memory: 2Gi")))
	checked, err := Check(patch, testPolicy(t), repository(t))
	if err != nil {
		t.Fatal(err)
	}

	patch.Files[0].Content = "kind: Evil"
	checked.Files()[0].Content = "kind: Evil"

	if strings.Contains(checked.Files()[0].Content, "Evil") {
		t.Error("a change made after Check reached the checked patch")
	}
}
