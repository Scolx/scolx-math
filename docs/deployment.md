# Deployment Guide

This guide explains how to deploy the Scolx Math API to production environments using the provided Docker image and continuous integration pipeline.

## Prerequisites

- Docker 24 or later installed on the target host
- Access to a container registry (e.g., GitLab Container Registry)
- GitLab CI/CD configured for the project (see `.gitlab-ci.yml`)
- Optional: `docker compose` for multi-container deployments

## Building the Image

You can build the API image locally using the Dockerfile included in the repository:

```bash
docker build -t scolx-math:latest .
```

The Dockerfile installs system dependencies required for SymPy/SymEngine and exposes the application via Uvicorn.

## Running the Container

To run the API on port `8000`:

```bash
docker run --rm -p 8000:8000 scolx-math:latest
```

Expose a different port by adjusting the mapping:

```bash
docker run --rm -p 9000:8000 scolx-math:latest
```

### Environment Configuration

The default Uvicorn command listens on `0.0.0.0:8000`. Customize runtime settings via environment variables or command overrides.

Examples:

```bash
docker run --rm -e UVICORN_WORKERS=4 -p 8000:8000 scolx-math:latest uvicorn \
  scolx_math.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

You can also mount configuration files if needed:

```bash
docker run --rm -v $(pwd)/config.py:/app/config.py -p 8000:8000 scolx-math:latest
```

## Using Docker Compose

For deployments requiring TLS terminators, reverse proxies, or background workers, Docker Compose can orchestrate multiple services.

`docker-compose.yml` example:

```yaml
services:
  scolx-math:
    image: registry.cherkaoui.ch/scolx/scolx-math:latest
    ports:
      - "8000:8000"
    environment:
      UVICORN_WORKERS: "4"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/livez"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Bring the stack up with:

```bash
docker compose up -d
```

## Continuous Integration and Delivery

The GitLab pipeline already contains linting, testing, and image build stages.

- **Lint**: Runs Ruff on all Python sources.
- **Test**: Executes the pytest suite.
- **Build**: Builds and pushes a Docker image tagged `latest`.

The build stage depends on the test stage and uses Docker-in-Docker to publish to the project registry. Ensure the following variables are configured in GitLab:

- `CI_REGISTRY_USER`
- `CI_REGISTRY_PASSWORD`

Optionally, configure environment-specific deploy jobs that pull the published image and run it via `docker run` or `docker compose` on a target host.

## Health Checks and Monitoring

The API exposes a FastAPI application with automatic OpenAPI documentation and dedicated health check endpoints. After deployment:

- Verify the service responds: `curl http://<host>:8000/livez` (liveness probe)
- Check readiness: `curl http://<host>:8000/readyz` (readiness probe)
- Check startup status: `curl http://<host>:8000/startupz` (startup probe)
- Access Swagger UI at `http://<host>:8000/docs`
- Monitor logs using `docker logs` or your orchestration platform

## Scaling Considerations

- Increase Uvicorn worker count for CPU-bound workloads (`--workers <n>`)
- Place the container behind a reverse proxy (e.g., Nginx, Traefik) for SSL termination
- Use horizontal scaling with multiple container instances behind a load balancer when serving heavy traffic

## Backup and Recovery

The API is stateless and does not persist data. Redeploy by pulling the latest image and restarting the container:

```bash
docker pull registry.cherkaoui.ch/scolx/scolx-math:latest
docker compose up -d --force-recreate
```

In case of rollback, retain previous image tags and redeploy the desired version.
