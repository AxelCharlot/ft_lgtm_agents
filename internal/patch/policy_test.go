package patch

import (
	"strings"
	"testing"
)

func TestParsePolicyRefuses(t *testing.T) {
	cases := map[string]string{
		"an empty file":        "",
		"no path":              "fields: []\n",
		"a misspelt key":       "paths: [k8s/*.yaml]\nfeilds: []\n",
		"a broken glob":        "paths: ['k8s/[.yaml']\n",
		"a field with no kind": "paths: [k8s/*.yaml]\nfields: [{path: spec}]\n",
		"an empty field path":  "paths: [k8s/*.yaml]\nfields: [{kind: Deployment, path: ''}]\n",
		"an empty key":         "paths: [k8s/*.yaml]\nfields: [{kind: Deployment, path: spec..replicas}]\n",
		"a trailing dot":       "paths: [k8s/*.yaml]\nfields: [{kind: Deployment, path: 'spec.'}]\n",
		"an unclosed bracket":  "paths: [k8s/*.yaml]\nfields: [{kind: Deployment, path: 'env[*'}]\n",
		"an unknown selector":  "paths: [k8s/*.yaml]\nfields: [{kind: Deployment, path: 'env[-1]'}]\n",
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParsePolicy([]byte(input)); err == nil {
				t.Error("ParsePolicy accepted it")
			}
		})
	}
}

func TestFieldPathsCoverTheirSubtreeAndNothingAbove(t *testing.T) {
	policy, err := ParsePolicy([]byte("paths: [k8s/*.yaml]\nfields:\n" +
		"  - {kind: Deployment, path: 'spec.containers[*].env[name=A].value'}\n" +
		"  - {kind: Deployment, path: 'spec.ports[0]'}\n"))
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]bool{
		"spec.containers.[name=x].env.[name=A].value":      true,
		"spec.containers.[name=x].env.[name=A].value.deep": true,
		"spec.containers.[name=x].env.[name=B].value":      false,
		"spec.containers.[name=x].env.[name=A]":            false,
		"spec.containers.[name=x].env":                     false,
		"spec.ports.[0].port":                              true,
		"spec.ports.[1].port":                              false,
		"spec.replicas":                                    false,
	}
	for changed, want := range cases {
		segments := strings.Split(changed, ".")
		if got := policy.allowsField("Deployment", segments); got != want {
			t.Errorf("allowsField(%s) = %v, want %v", changed, got, want)
		}
		if policy.allowsField("StatefulSet", segments) {
			t.Errorf("allowsField(%s) holds for a kind the policy does not name", changed)
		}
	}
}
