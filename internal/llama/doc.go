// Package llama talks to llama.cpp: completions constrained by a grammar, so
// every answer parses, and embeddings for the incident memory.
//
// The grammar is the point. A model picks one token at a time out of about
// 150 000; the grammar removes, at every token, everything that could not lead
// to the schema. The model does not try to respect the format: it cannot leave
// it. What the grammar guarantees is the shape, never that the content is right.
package llama
