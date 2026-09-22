// Package patch defines what a repair may change and refuses any patch that
// breaks those rules, before it reaches GitHub.
//
// A patch is structured JSON, never a diff: {summary, files: [{path, op,
// content}]}, where content is the whole file after the change. A diff has to
// apply against a file that may have moved since it was written; a whole file
// replaces it the same way every time.
//
// Every rule is a hard refusal. There is no warning level: a patch either passes
// all of them, or it opens nothing.
package patch
