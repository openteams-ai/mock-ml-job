FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 mluser

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY training_simulator.py .

# Create directory for metrics output
RUN mkdir -p /shared/metrics && chown mluser:mluser /shared/metrics

# Switch to non-root user
USER mluser

# Run the training simulator
CMD ["python", "-u", "training_simulator.py"]
