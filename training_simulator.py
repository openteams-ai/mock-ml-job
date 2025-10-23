#!/usr/bin/env python3
"""
ML Training Job Simulator

This script simulates a machine learning training job by generating
realistic training metrics and writing them to a shared volume.
The sidecar will read these metrics and export them via OpenTelemetry.
"""

import json
import time
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


class TrainingSimulator:
    """Simulates an ML training job with realistic metrics."""

    def __init__(self):
        self.job_id = os.getenv("JOB_ID", "training-job-001")
        self.model_name = os.getenv("MODEL_NAME", "resnet-50")
        self.dataset = os.getenv("DATASET", "imagenet")
        self.metrics_path = Path(os.getenv("METRICS_FILE_PATH", "/shared/metrics/current.json"))
        self.write_interval = int(os.getenv("WRITE_INTERVAL", "10"))
        self.total_epochs = int(os.getenv("TOTAL_EPOCHS", "10"))
        self.batches_per_epoch = int(os.getenv("BATCHES_PER_EPOCH", "100"))

        self.start_time = datetime.now(timezone.utc)
        self.current_epoch = 1
        self.current_batch = 0
        self.running = True

        # Training progression parameters
        self.initial_loss = 2.5
        self.target_loss = 0.15
        self.initial_val_loss = 2.8
        self.target_val_loss = 0.25
        self.initial_accuracy = 0.15
        self.target_accuracy = 0.94
        self.base_lr = 0.001

        # Ensure metrics directory exists
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Training Simulator initialized:")
        print(f"  Job ID: {self.job_id}")
        print(f"  Model: {self.model_name}")
        print(f"  Dataset: {self.dataset}")
        print(f"  Metrics path: {self.metrics_path}")
        print(f"  Write interval: {self.write_interval}s")
        print(f"  Total epochs: {self.total_epochs}")
        print(f"  Batches per epoch: {self.batches_per_epoch}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False

    def calculate_metrics(self):
        """Calculate realistic training metrics with progression and noise."""
        total_batches = self.total_epochs * self.batches_per_epoch
        progress = (self.current_epoch - 1) * self.batches_per_epoch + self.current_batch
        progress_ratio = progress / total_batches

        # Training loss decreases with noise
        training_loss = self.initial_loss * (1 - progress_ratio) + self.target_loss * progress_ratio
        training_loss += np.random.normal(0, 0.05)  # Add noise
        training_loss = max(0.01, training_loss)  # Ensure positive

        # Validation loss decreases slower with more noise
        val_loss = self.initial_val_loss * (1 - progress_ratio * 0.9) + self.target_val_loss * progress_ratio
        val_loss += np.random.normal(0, 0.08)
        val_loss = max(training_loss * 0.8, val_loss)  # Val loss usually higher than training

        # Accuracy increases
        accuracy = self.initial_accuracy * (1 - progress_ratio) + self.target_accuracy * progress_ratio
        accuracy += np.random.normal(0, 0.02)
        accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]

        # Learning rate with decay
        learning_rate = self.base_lr * (0.95 ** self.current_epoch)

        # GPU utilization (high with some variance)
        gpu_utilization = 0.92 + np.random.normal(0, 0.05)
        gpu_utilization = max(0.0, min(1.0, gpu_utilization))

        # Processing time (ms) varies by batch complexity
        base_processing_time = 245
        processing_time = int(base_processing_time + np.random.normal(0, 30))
        processing_time = max(100, processing_time)

        # Samples per second
        batch_size = 32
        samples_per_second = (batch_size / processing_time) * 1000
        samples_per_second = max(1.0, samples_per_second)

        return {
            "training_loss": round(training_loss, 4),
            "validation_loss": round(val_loss, 4),
            "accuracy": round(accuracy, 4),
            "learning_rate": learning_rate,
            "gpu_utilization": round(gpu_utilization, 3),
            "processing_time_ms": processing_time,
            "samples_per_second": round(samples_per_second, 2)
        }

    def write_metrics(self):
        """Write current metrics to shared file."""
        metrics = self.calculate_metrics()

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_metadata": {
                "job_id": self.job_id,
                "model_name": self.model_name,
                "dataset": self.dataset,
                "start_time": self.start_time.isoformat()
            },
            "training_metrics": {
                "epoch": self.current_epoch,
                "batch_number": self.current_batch,
                **metrics
            }
        }

        # Write atomically by writing to temp file then renaming
        temp_path = self.metrics_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.metrics_path)

            print(f"[Epoch {self.current_epoch}/{self.total_epochs}, "
                  f"Batch {self.current_batch}/{self.batches_per_epoch}] "
                  f"Loss: {metrics['training_loss']:.4f}, "
                  f"Val Loss: {metrics['validation_loss']:.4f}, "
                  f"Acc: {metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"Error writing metrics: {e}", file=sys.stderr)

    def advance_training(self):
        """Advance to next batch/epoch."""
        self.current_batch += 1

        if self.current_batch > self.batches_per_epoch:
            self.current_batch = 1
            self.current_epoch += 1
            print(f"\n--- Completed Epoch {self.current_epoch - 1} ---\n")

        if self.current_epoch > self.total_epochs:
            print(f"\n=== Training Complete! ===")
            print(f"Total time: {datetime.now(timezone.utc) - self.start_time}")
            self.running = False

    def run(self):
        """Main training loop."""
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        print(f"\n{'='*60}")
        print(f"Starting training for {self.model_name} on {self.dataset}")
        print(f"{'='*60}\n")

        while self.running and self.current_epoch <= self.total_epochs:
            try:
                self.write_metrics()
                self.advance_training()

                if self.running:
                    time.sleep(self.write_interval)

            except Exception as e:
                print(f"Error in training loop: {e}", file=sys.stderr)
                time.sleep(5)  # Back off on errors

        print("\nTraining simulator stopped.")


if __name__ == "__main__":
    simulator = TrainingSimulator()
    simulator.run()
