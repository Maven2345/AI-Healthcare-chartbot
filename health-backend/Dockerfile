# Use an official Python runtime
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /workspace

# Copy all your repository files into the container
COPY . /workspace

# Install system dependencies needed for machine learning packages
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install your Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your voice agent pipeline
CMD ["python", "multilingual_agent.py", "start"]
