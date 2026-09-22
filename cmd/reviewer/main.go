// Command reviewer reads a pull request of the healer against the live cluster
// state and approves or vetoes it.
//
// For now it only starts the telemetry and waits to be stopped. The loop
// arrives with its own issue; the telemetry comes first so that the first
// line of that loop is already traced.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/AxelCharlot/ft_lgtm/pkg/telemetry"
)

const (
	envServiceName = "OTEL_SERVICE_NAME"
	envEndpoint    = "OTEL_EXPORTER_OTLP_ENDPOINT"
)

func main() {
	if err := watch(); err != nil {
		fmt.Fprintf(os.Stderr, "lgtm-reviewer: %v\n", err)
		os.Exit(1)
	}
}

func watch() error {
	// Both are required, as they are for the backend: a default service name
	// would file every span under a name no dashboard queries, and nothing
	// would report it.
	serviceName, endpoint := os.Getenv(envServiceName), os.Getenv(envEndpoint)
	if serviceName == "" || endpoint == "" {
		return fmt.Errorf("%s and %s must both be set", envServiceName, envEndpoint)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	logger, flush, err := telemetry.Start(ctx, serviceName, endpoint)
	if err != nil {
		return err
	}
	logger.Info("started")
	<-ctx.Done()
	logger.Info("stopping")

	closing, cancel := context.WithTimeout(context.Background(), telemetry.ShutdownTimeout)
	defer cancel()
	// Reported, not returned: a Collector that is down loses the last spans, and
	// that is no reason to fail a stop that went well. On stderr, not through
	// logger, because the log exporter may be the very thing that failed.
	if err := flush(closing); err != nil {
		fmt.Fprintf(os.Stderr, "lgtm-reviewer: the telemetry did not flush: %v\n", err)
	}
	return nil
}
