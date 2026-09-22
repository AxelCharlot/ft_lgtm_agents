// Package brain escalates a diagnosis from the cheapest tier to the costliest:
// tier 1 is a classifier, tier 2 a small fine-tuned model, tier 3 the full
// model. A tier answers only when it is confident; otherwise the next one tries.
package brain
