FROM python:3.12-slim

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ src/
COPY test_framework.py run_experiments.py ./

# Input/output volumes — mount at runtime:
#   docker run -v /path/to/logs:/app/data/input -v /path/to/results:/app/data/output ...
RUN mkdir -p data/input data/output

ENTRYPOINT ["python", "run_experiments.py"]
