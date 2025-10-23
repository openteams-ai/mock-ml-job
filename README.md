# Mock ML Job Simulator

A lightweight machine learning training job simulator designed for testing observability systems, OpenTelemetry sidecars, and metrics collection pipelines.

## Overview

This simulator generates realistic ML training metrics (loss, accuracy, GPU utilization, etc.) and writes them to a shared volume in JSON format. It's ideal for:

- Testing OpenTelemetry sidecar implementations
- Validating metrics collection and export pipelines
- Developing ML observability tools without running actual training jobs
- Load testing monitoring systems with realistic ML workload patterns

## Quick Start

### Using Docker

```bash
# Build the image
docker build -t mock-ml-job .

# Run with default configuration
docker run --rm mock-ml-job

# Run with custom configuration and volume mount
docker run --rm \
  -e TOTAL_EPOCHS=5 \
  -e MODEL_NAME=bert-base \
  -e WRITE_INTERVAL=5 \
  -v $(pwd)/metrics:/shared/metrics \
  mock-ml-job
```

### Using pre-built image from GitHub Container Registry

```bash
# Pull the latest image
docker pull ghcr.io/openteams-ai/mock-ml-job:latest

# Run it
docker run --rm ghcr.io/openteams-ai/mock-ml-job:latest
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulator
python training_simulator.py
```

## Configuration

Configure the simulator using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `JOB_ID` | Unique identifier for this training job | `training-job-001` |
| `MODEL_NAME` | Name of the model being trained | `resnet-50` |
| `DATASET` | Dataset name | `imagenet` |
| `METRICS_FILE_PATH` | Path to write metrics JSON | `/shared/metrics/current.json` |
| `WRITE_INTERVAL` | Seconds between metric updates | `10` |
| `TOTAL_EPOCHS` | Number of training epochs to simulate | `10` |
| `BATCHES_PER_EPOCH` | Batches per epoch | `100` |

## Generated Metrics

The simulator produces realistic training metrics with natural progression and variance:

- **training_loss**: Decreases from ~2.5 to ~0.15 over training
- **validation_loss**: Decreases slower, stays slightly higher than training loss
- **accuracy**: Increases from ~15% to ~94%
- **learning_rate**: Exponential decay (0.95^epoch)
- **gpu_utilization**: Stable around 92% with realistic variance
- **processing_time_ms**: ~245ms per batch with variance
- **samples_per_second**: Calculated throughput metric

### Metrics File Format

The metrics are written to a JSON file with the following structure:

```json
{
  "timestamp": "2025-10-23T10:30:45.123456+00:00",
  "job_metadata": {
    "job_id": "training-job-001",
    "model_name": "resnet-50",
    "dataset": "imagenet",
    "start_time": "2025-10-23T10:15:00.000000+00:00"
  },
  "training_metrics": {
    "epoch": 3,
    "batch_number": 45,
    "training_loss": 1.2345,
    "validation_loss": 1.3456,
    "accuracy": 0.6789,
    "learning_rate": 0.000857375,
    "gpu_utilization": 0.923,
    "processing_time_ms": 238,
    "samples_per_second": 134.45
  }
}
```

## Integration with Observability Systems

### Sidecar Pattern

The simulator is designed to run as the main container in a pod with an observability sidecar:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-with-sidecar
spec:
  containers:
  - name: training-simulator
    image: ghcr.io/openteams-ai/mock-ml-job:latest
    env:
    - name: TOTAL_EPOCHS
      value: "20"
    - name: WRITE_INTERVAL
      value: "5"
    volumeMounts:
    - name: metrics
      mountPath: /shared/metrics

  - name: metrics-exporter-sidecar
    image: your-sidecar-image:latest
    volumeMounts:
    - name: metrics
      mountPath: /shared/metrics
      readOnly: true

  volumes:
  - name: metrics
    emptyDir: {}
```

### Atomic Writes

The simulator uses an atomic write pattern (write to `.tmp` file, then rename) to ensure sidecars never read partial or corrupted JSON data.

## Example Use Cases

### Testing Different Training Scenarios

```bash
# Fast iteration testing (short epochs, quick updates)
docker run --rm \
  -e TOTAL_EPOCHS=3 \
  -e BATCHES_PER_EPOCH=20 \
  -e WRITE_INTERVAL=2 \
  mock-ml-job

# Long-running training simulation
docker run --rm \
  -e TOTAL_EPOCHS=100 \
  -e BATCHES_PER_EPOCH=500 \
  -e WRITE_INTERVAL=30 \
  mock-ml-job

# Custom model and dataset metadata
docker run --rm \
  -e MODEL_NAME=transformer-xl \
  -e DATASET=wikitext-103 \
  -e JOB_ID=exp-2025-001 \
  mock-ml-job
```

## Development

See [CLAUDE.md](CLAUDE.md) for development guidance.

## License

[Add your license here]
