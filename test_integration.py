"""
Integration tests for the Docker container.

These tests build and run the actual Docker container to verify it behaves
as documented in the README.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest


class TestDockerIntegration:
    """Integration tests for Docker container behavior."""

    @pytest.fixture(scope="class")
    def docker_image_name(self) -> str:
        """Return the Docker image name for testing."""
        return "mock-ml-job:test"

    @pytest.fixture(scope="class")
    def build_docker_image(self, docker_image_name: str) -> str:
        """Build the Docker image before running tests."""
        print(f"\nBuilding Docker image: {docker_image_name}")
        result = subprocess.run(
            ["docker", "build", "-t", docker_image_name, "."],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            pytest.fail(f"Docker build failed:\n{result.stderr}")

        return docker_image_name

    @pytest.fixture
    def temp_metrics_volume(self, tmp_path: Path) -> Path:
        """Create a temporary directory for metrics volume mount."""
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        return metrics_dir

    def test_docker_image_builds_successfully(self, build_docker_image: str) -> None:
        """Test that the Docker image builds without errors."""
        # Verify image exists
        result = subprocess.run(
            ["docker", "images", "-q", build_docker_image],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip(), f"Docker image {build_docker_image} not found"

    def test_container_runs_with_defaults(self, build_docker_image: str, temp_metrics_volume: Path) -> None:
        """Test that container runs with default configuration."""
        container_name = "test-ml-job-defaults"

        try:
            # Start container with very short training time
            result = subprocess.run(
                [
                    "docker", "run", "--rm", "--name", container_name,
                    "-e", "TOTAL_EPOCHS=1",
                    "-e", "BATCHES_PER_EPOCH=2",
                    "-e", "WRITE_INTERVAL=1",
                    "-v", f"{temp_metrics_volume}:/shared/metrics",
                    build_docker_image,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            # Should complete successfully
            assert result.returncode == 0, f"Container failed: {result.stderr}"

            # Check that output mentions completion
            assert "Training Complete" in result.stdout or "Training simulator stopped" in result.stdout

        finally:
            # Cleanup in case container is still running
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)

    def test_container_writes_metrics_file(self, build_docker_image: str, temp_metrics_volume: Path) -> None:
        """Test that container writes metrics to shared volume."""
        container_name = "test-ml-job-metrics"
        metrics_file = temp_metrics_volume / "current.json"

        try:
            # Run container in background
            subprocess.run(
                [
                    "docker", "run", "-d", "--name", container_name,
                    "-e", "TOTAL_EPOCHS=1",
                    "-e", "BATCHES_PER_EPOCH=3",
                    "-e", "WRITE_INTERVAL=1",
                    "-v", f"{temp_metrics_volume}:/shared/metrics",
                    build_docker_image,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Wait for metrics file to be created
            max_wait = 10
            for _ in range(max_wait):
                if metrics_file.exists():
                    break
                time.sleep(1)

            assert metrics_file.exists(), "Metrics file was not created"

            # Verify metrics file contains valid JSON
            with metrics_file.open() as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "job_metadata" in data
            assert "training_metrics" in data

        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)

    def test_container_respects_custom_config(self, build_docker_image: str, temp_metrics_volume: Path) -> None:
        """Test that container respects custom environment variables."""
        container_name = "test-ml-job-custom"
        metrics_file = temp_metrics_volume / "current.json"

        custom_job_id = "integration-test-123"
        custom_model = "test-transformer"
        custom_dataset = "test-data"

        try:
            # Run with custom configuration
            subprocess.run(
                [
                    "docker", "run", "-d", "--name", container_name,
                    "-e", f"JOB_ID={custom_job_id}",
                    "-e", f"MODEL_NAME={custom_model}",
                    "-e", f"DATASET={custom_dataset}",
                    "-e", "TOTAL_EPOCHS=1",
                    "-e", "BATCHES_PER_EPOCH=1",
                    "-e", "WRITE_INTERVAL=1",
                    "-v", f"{temp_metrics_volume}:/shared/metrics",
                    build_docker_image,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Wait for metrics
            time.sleep(3)

            assert metrics_file.exists(), "Metrics file was not created"

            with metrics_file.open() as f:
                data: dict[str, Any] = json.load(f)

            # Verify custom values are in the metrics
            assert data["job_metadata"]["job_id"] == custom_job_id
            assert data["job_metadata"]["model_name"] == custom_model
            assert data["job_metadata"]["dataset"] == custom_dataset

        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)

    def test_container_generates_expected_metrics(self, build_docker_image: str, temp_metrics_volume: Path) -> None:
        """Test that container generates all expected metric fields."""
        container_name = "test-ml-job-metrics-fields"
        metrics_file = temp_metrics_volume / "current.json"

        try:
            subprocess.run(
                [
                    "docker", "run", "-d", "--name", container_name,
                    "-e", "TOTAL_EPOCHS=1",
                    "-e", "BATCHES_PER_EPOCH=2",
                    "-e", "WRITE_INTERVAL=1",
                    "-v", f"{temp_metrics_volume}:/shared/metrics",
                    build_docker_image,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            time.sleep(3)

            with metrics_file.open() as f:
                data: dict[str, Any] = json.load(f)

            # Verify all expected fields exist
            training_metrics = data["training_metrics"]
            expected_fields = {
                "epoch",
                "batch_number",
                "training_loss",
                "validation_loss",
                "accuracy",
                "learning_rate",
                "gpu_utilization",
                "processing_time_ms",
                "samples_per_second",
            }

            assert set(training_metrics.keys()) == expected_fields, \
                f"Missing or extra fields in metrics: {set(training_metrics.keys())}"

            # Verify metric value ranges
            assert training_metrics["training_loss"] >= 0
            assert training_metrics["validation_loss"] >= 0
            assert 0.0 <= training_metrics["accuracy"] <= 1.0
            assert training_metrics["learning_rate"] > 0
            assert 0.0 <= training_metrics["gpu_utilization"] <= 1.0
            assert training_metrics["processing_time_ms"] >= 100
            assert training_metrics["samples_per_second"] >= 1.0

        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)

    def test_container_stops_gracefully_on_sigterm(self, build_docker_image: str, temp_metrics_volume: Path) -> None:
        """Test that container handles SIGTERM gracefully."""
        container_name = "test-ml-job-sigterm"

        try:
            # Start long-running container
            subprocess.run(
                [
                    "docker", "run", "-d", "--name", container_name,
                    "-e", "TOTAL_EPOCHS=100",
                    "-e", "BATCHES_PER_EPOCH=100",
                    "-e", "WRITE_INTERVAL=1",
                    "-v", f"{temp_metrics_volume}:/shared/metrics",
                    build_docker_image,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Wait for it to start
            time.sleep(2)

            # Stop container (sends SIGTERM)
            result = subprocess.run(
                ["docker", "stop", "-t", "5", container_name],
                capture_output=True,
                text=True,
                check=True,
            )

            # Should stop cleanly within timeout
            assert result.returncode == 0

            # Check logs for graceful shutdown message
            logs_result = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True,
                text=True,
                check=True,
            )

            # Should see shutdown or stopped message
            assert any(msg in logs_result.stdout for msg in [
                "shutting down gracefully",
                "Training simulator stopped",
                "Received signal"
            ]), "No graceful shutdown message in logs"

        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)

    def test_container_updates_metrics_multiple_times(self, build_docker_image: str, temp_metrics_volume: Path) -> None:
        """Test that container updates metrics file multiple times during execution."""
        container_name = "test-ml-job-updates"
        metrics_file = temp_metrics_volume / "current.json"

        try:
            subprocess.run(
                [
                    "docker", "run", "-d", "--name", container_name,
                    "-e", "TOTAL_EPOCHS=1",
                    "-e", "BATCHES_PER_EPOCH=5",
                    "-e", "WRITE_INTERVAL=1",
                    "-v", f"{temp_metrics_volume}:/shared/metrics",
                    build_docker_image,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Read initial metrics
            time.sleep(2)
            with metrics_file.open() as f:
                data1: dict[str, Any] = json.load(f)

            # Wait and read again
            time.sleep(2)
            with metrics_file.open() as f:
                data2: dict[str, Any] = json.load(f)

            # Batch number or epoch should have changed
            assert (
                data2["training_metrics"]["batch_number"] != data1["training_metrics"]["batch_number"]
                or data2["training_metrics"]["epoch"] != data1["training_metrics"]["epoch"]
            ), "Metrics were not updated"

        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)

    def test_container_runs_as_non_root(self, build_docker_image: str) -> None:
        """Test that container runs as non-root user."""
        container_name = "test-ml-job-user"

        try:
            # Check what user the process runs as
            result = subprocess.run(
                [
                    "docker", "run", "--rm", "--name", container_name,
                    "--entrypoint", "id",
                    build_docker_image,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Should be running as mluser (uid=1000)
            assert "uid=1000" in result.stdout, f"Container not running as mluser: {result.stdout}"
            assert "mluser" in result.stdout, f"Container not running as mluser: {result.stdout}"

        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
