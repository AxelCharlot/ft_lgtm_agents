package patch

import (
	"errors"
	"testing"
)

func TestDecodeReadsTheSchema(t *testing.T) {
	patch, err := Decode([]byte(`{"summary": "raise the limit",
		"files": [{"path": "k8s/a.yaml", "op": "replace", "content": "a: 1\n"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	want := File{Path: "k8s/a.yaml", Op: OpReplace, Content: "a: 1\n"}
	if patch.Summary != "raise the limit" || len(patch.Files) != 1 || patch.Files[0] != want {
		t.Errorf("Decode = %+v", patch)
	}
}

func TestDecodeRefuses(t *testing.T) {
	cases := map[string]string{
		"text that is not JSON": `the fix is to raise the limit`,
		"a field the schema does not name": `{"summary": "s", "files": [],
			"command": "kubectl delete ns lgtm"}`,
		"a field of a file the schema does not name": `{"summary": "s",
			"files": [{"path": "k8s/a.yaml", "op": "replace", "content": "", "mode": "0777"}]}`,
		"a second object after the patch": `{"summary": "s", "files": []} {"summary": "t"}`,
		"a value of the wrong type":       `{"summary": ["s"], "files": []}`,
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Decode([]byte(input))
			var refusal *Refusal
			if !errors.As(err, &refusal) || refusal.Rule != RuleSchema {
				t.Errorf("Decode = %v, want a refusal by rule %s", err, RuleSchema)
			}
		})
	}
}
