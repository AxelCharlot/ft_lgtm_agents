// Package memory keeps the past incidents and finds the nearest ones to a new
// symptom, by exhaustive cosine search over their embeddings. It persists to a
// single file.
//
// Incidents are indexed; manifests never are. The Kubernetes event names the
// pod, the pod names its Deployment, and the file is read by that name. A
// vector search over manifests would be strictly less reliable than the key
// lookup it would replace.
package memory
