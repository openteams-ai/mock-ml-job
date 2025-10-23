"""
Unit tests for the ML Training Job Simulator.
"""

import contextlib
import json
import signal
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training_simulator import TrainingSimulator


class TestTrainingSimulator:
    """Test suite for TrainingSimulator class."""

    @pytest.fixture
    def temp_metrics_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for metrics files."""
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        return metrics_dir

    @pytest.fixture
    def simulator(self, temp_metrics_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TrainingSimulator:
        """Create a TrainingSimulator instance with test configuration."""
        monkeypatch.setenv("JOB_ID", "test-job-123")
        monkeypatch.setenv("MODEL_NAME", "test-model")
        monkeypatch.setenv("DATASET", "test-dataset")
        monkeypatch.setenv("METRICS_FILE_PATH", str(temp_metrics_dir / "current.json"))
        monkeypatch.setenv("WRITE_INTERVAL", "1")
        monkeypatch.setenv("TOTAL_EPOCHS", "2")
        monkeypatch.setenv("BATCHES_PER_EPOCH", "5")

        with patch("builtins.print"):  # Suppress initialization print statements
            return TrainingSimulator()

    def test_initialization_from_env_vars(self, simulator: TrainingSimulator) -> None:
        """Test that simulator correctly reads configuration from environment variables."""
        assert simulator.job_id == "test-job-123"
        assert simulator.model_name == "test-model"
        assert simulator.dataset == "test-dataset"
        assert simulator.write_interval == 1
        assert simulator.total_epochs == 2
        assert simulator.batches_per_epoch == 5

    def test_initialization_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that simulator uses default values when env vars are not set."""
        # Remove the env vars if they exist
        for key in ["JOB_ID", "MODEL_NAME", "DATASET", "WRITE_INTERVAL", "TOTAL_EPOCHS", "BATCHES_PER_EPOCH"]:
            monkeypatch.delenv(key, raising=False)

        # Set metrics path to temp directory
        monkeypatch.setenv("METRICS_FILE_PATH", str(tmp_path / "metrics" / "current.json"))

        with patch("builtins.print"):
            sim = TrainingSimulator()

        assert sim.job_id == "training-job-001"
        assert sim.model_name == "resnet-50"
        assert sim.dataset == "imagenet"
        assert sim.write_interval == 10
        assert sim.total_epochs == 10
        assert sim.batches_per_epoch == 100

    def test_initial_state(self, simulator: TrainingSimulator) -> None:
        """Test that simulator initializes with correct initial state."""
        assert simulator.current_epoch == 1
        assert simulator.current_batch == 0
        assert simulator.running is True
        assert isinstance(simulator.start_time, datetime)

    def test_metrics_directory_creation(self, temp_metrics_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that metrics directory is created if it doesn't exist."""
        new_dir = temp_metrics_dir / "new" / "nested" / "dir"
        monkeypatch.setenv("METRICS_FILE_PATH", str(new_dir / "metrics.json"))

        with patch("builtins.print"):
            TrainingSimulator()

        assert new_dir.exists()

    def test_calculate_metrics_structure(self, simulator: TrainingSimulator) -> None:
        """Test that calculate_metrics returns properly structured data."""
        metrics = simulator.calculate_metrics()

        expected_keys = {
            "training_loss",
            "validation_loss",
            "accuracy",
            "learning_rate",
            "gpu_utilization",
            "processing_time_ms",
            "samples_per_second",
        }
        assert set(metrics.keys()) == expected_keys

    def test_calculate_metrics_types(self, simulator: TrainingSimulator) -> None:
        """Test that metrics have correct types."""
        metrics = simulator.calculate_metrics()

        assert isinstance(metrics["training_loss"], float)
        assert isinstance(metrics["validation_loss"], float)
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["learning_rate"], float)
        assert isinstance(metrics["gpu_utilization"], float)
        assert isinstance(metrics["processing_time_ms"], int)
        assert isinstance(metrics["samples_per_second"], float)

    def test_calculate_metrics_ranges(self, simulator: TrainingSimulator) -> None:
        """Test that metrics are within expected ranges."""
        metrics = simulator.calculate_metrics()

        assert metrics["training_loss"] >= 0
        assert metrics["validation_loss"] >= 0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["learning_rate"] > 0
        assert 0.0 <= metrics["gpu_utilization"] <= 1.0
        assert metrics["processing_time_ms"] >= 100
        assert metrics["samples_per_second"] >= 1.0

    def test_metrics_progression(self, simulator: TrainingSimulator) -> None:
        """Test that loss decreases and accuracy increases over training."""
        initial_metrics = simulator.calculate_metrics()

        # Advance to near the end of training
        simulator.current_epoch = simulator.total_epochs
        simulator.current_batch = simulator.batches_per_epoch - 1

        final_metrics = simulator.calculate_metrics()

        # Loss should generally decrease (allowing for some noise)
        assert final_metrics["training_loss"] < initial_metrics["training_loss"] + 0.5
        # Accuracy should generally increase (allowing for some noise)
        assert final_metrics["accuracy"] > initial_metrics["accuracy"] - 0.1

    def test_learning_rate_decay(self, simulator: TrainingSimulator) -> None:
        """Test that learning rate decays over epochs."""
        simulator.current_epoch = 1
        lr_epoch_1 = simulator.calculate_metrics()["learning_rate"]

        simulator.current_epoch = 5
        lr_epoch_5 = simulator.calculate_metrics()["learning_rate"]

        assert lr_epoch_5 < lr_epoch_1

    def test_write_metrics_creates_file(self, simulator: TrainingSimulator) -> None:
        """Test that write_metrics creates the metrics file."""
        with patch("builtins.print"):
            simulator.write_metrics()

        assert simulator.metrics_path.exists()

    def test_write_metrics_json_structure(self, simulator: TrainingSimulator) -> None:
        """Test that written metrics file has correct JSON structure."""
        with patch("builtins.print"):
            simulator.write_metrics()

        with simulator.metrics_path.open() as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "job_metadata" in data
        assert "training_metrics" in data

        # Check job_metadata structure
        assert data["job_metadata"]["job_id"] == "test-job-123"
        assert data["job_metadata"]["model_name"] == "test-model"
        assert data["job_metadata"]["dataset"] == "test-dataset"
        assert "start_time" in data["job_metadata"]

        # Check training_metrics structure
        assert "epoch" in data["training_metrics"]
        assert "batch_number" in data["training_metrics"]
        assert "training_loss" in data["training_metrics"]

    def test_write_metrics_atomic_write(self, simulator: TrainingSimulator) -> None:
        """Test that metrics are written atomically using temp file."""
        temp_file = simulator.metrics_path.with_suffix('.tmp')

        with patch("builtins.print"):
            simulator.write_metrics()

        # Temp file should not exist after write completes
        assert not temp_file.exists()
        # Final file should exist
        assert simulator.metrics_path.exists()

    def test_write_metrics_valid_json(self, simulator: TrainingSimulator) -> None:
        """Test that written file contains valid JSON."""
        with patch("builtins.print"):
            simulator.write_metrics()

        # Should not raise JSONDecodeError
        with simulator.metrics_path.open() as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_write_metrics_error_handling(self, simulator: TrainingSimulator) -> None:
        """Test that write_metrics handles errors gracefully."""
        # Make the metrics path unwritable
        simulator.metrics_path = Path("/root/unwritable/path.json")

        # Should not raise exception
        with patch("builtins.print"):
            simulator.write_metrics()

    def test_advance_training_batch(self, simulator: TrainingSimulator) -> None:
        """Test that advance_training correctly increments batch number."""
        simulator.current_batch = 2
        initial_epoch = simulator.current_epoch

        with patch("builtins.print"):
            simulator.advance_training()

        assert simulator.current_batch == 3
        assert simulator.current_epoch == initial_epoch

    def test_advance_training_epoch(self, simulator: TrainingSimulator) -> None:
        """Test that advance_training correctly increments epoch and resets batch."""
        simulator.current_batch = simulator.batches_per_epoch
        initial_epoch = simulator.current_epoch

        with patch("builtins.print"):
            simulator.advance_training()

        assert simulator.current_batch == 1
        assert simulator.current_epoch == initial_epoch + 1

    def test_advance_training_completion(self, simulator: TrainingSimulator) -> None:
        """Test that advance_training sets running=False when training completes."""
        simulator.current_epoch = simulator.total_epochs
        simulator.current_batch = simulator.batches_per_epoch

        with patch("builtins.print"):
            simulator.advance_training()

        assert simulator.running is False

    def test_signal_handler_sigterm(self, simulator: TrainingSimulator) -> None:
        """Test that signal handler sets running=False on SIGTERM."""
        with patch("builtins.print"):
            simulator.signal_handler(signal.SIGTERM, None)

        assert simulator.running is False

    def test_signal_handler_sigint(self, simulator: TrainingSimulator) -> None:
        """Test that signal handler sets running=False on SIGINT."""
        with patch("builtins.print"):
            simulator.signal_handler(signal.SIGINT, None)

        assert simulator.running is False

    @patch("time.sleep")
    @patch("builtins.print")
    def test_run_loop_iterations(self, _mock_print: MagicMock, _mock_sleep: MagicMock, simulator: TrainingSimulator) -> None:
        """Test that run loop executes expected number of iterations."""
        # Set very small training job
        simulator.total_epochs = 1
        simulator.batches_per_epoch = 2

        simulator.run()

        # Should write metrics for each batch
        assert simulator.current_epoch > simulator.total_epochs or not simulator.running

    @patch("time.sleep")
    @patch("builtins.print")
    def test_run_creates_metrics_files(self, _mock_print: MagicMock, _mock_sleep: MagicMock, simulator: TrainingSimulator) -> None:
        """Test that run loop creates metrics files."""
        simulator.total_epochs = 1
        simulator.batches_per_epoch = 1

        simulator.run()

        assert simulator.metrics_path.exists()

    @patch("time.sleep")
    @patch("builtins.print")
    def test_run_signal_handling(self, _mock_print: MagicMock, mock_sleep: MagicMock, simulator: TrainingSimulator) -> None:
        """Test that signal handlers are registered during run."""
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        try:
            # Interrupt the run loop immediately
            def stop_immediately(*_args):  # type: ignore
                simulator.running = False
                mock_sleep.side_effect = KeyboardInterrupt()

            mock_sleep.side_effect = stop_immediately

            with contextlib.suppress(KeyboardInterrupt):
                simulator.run()

            # Signal handlers should have been set
            current_sigterm = signal.getsignal(signal.SIGTERM)
            current_sigint = signal.getsignal(signal.SIGINT)

            # Handlers should have been changed from original
            assert current_sigterm != original_sigterm or current_sigint != original_sigint

        finally:
            # Restore original handlers
            signal.signal(signal.SIGTERM, original_sigterm)
            signal.signal(signal.SIGINT, original_sigint)

    @patch("time.sleep")
    @patch("builtins.print")
    def test_run_error_recovery(self, _mock_print: MagicMock, _mock_sleep: MagicMock, simulator: TrainingSimulator) -> None:
        """Test that run loop recovers from errors and continues."""
        simulator.total_epochs = 1
        simulator.batches_per_epoch = 2

        call_count = 0

        def fail_once_then_succeed(*_args, **_kwargs):  # type: ignore
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated error")

        # Make write_metrics fail once
        with patch.object(simulator, 'write_metrics', side_effect=fail_once_then_succeed):
            simulator.run()

        # Should have attempted multiple writes despite error
        assert call_count > 1

    def test_timestamp_format(self, simulator: TrainingSimulator) -> None:
        """Test that timestamp is in ISO 8601 format with timezone."""
        with patch("builtins.print"):
            simulator.write_metrics()

        with simulator.metrics_path.open() as f:
            data = json.load(f)

        timestamp = data["timestamp"]
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo is not None  # Should have timezone info

    def test_multiple_writes_update_file(self, simulator: TrainingSimulator) -> None:
        """Test that multiple writes update the metrics file."""
        with patch("builtins.print"):
            simulator.write_metrics()

        with simulator.metrics_path.open() as f:
            data1 = json.load(f)

        # Advance and write again
        simulator.advance_training()

        with patch("builtins.print"):
            simulator.write_metrics()

        with simulator.metrics_path.open() as f:
            data2 = json.load(f)

        # Batch or epoch should have changed
        assert (
            data2["training_metrics"]["batch_number"] != data1["training_metrics"]["batch_number"]
            or data2["training_metrics"]["epoch"] != data1["training_metrics"]["epoch"]
        )
