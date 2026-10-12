# Start from a small official Python image that matches the version the
# model was trained under (anaconda env "ml" = Python 3.11).
FROM python:3.11-slim

# All following paths are relative to /app inside the image. This also makes
# "src.api" importable the same way as running from the project root locally.
WORKDIR /app

# Install dependencies BEFORE copying the code: this layer is cached, so
# editing source code doesn't re-run the slow pip install on every build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The application code and the one trained artifact the API loads at startup.
# No data/, no notebooks, no credentials - the model is NOT retrained here.
COPY src/ src/
COPY models/ercot_gb_dayahead.joblib models/ercot_gb_dayahead.joblib

# Documentation of the port uvicorn listens on (actual publishing happens
# with `docker run -p 8000:8000`).
EXPOSE 8000

# Bind to 0.0.0.0, not the default 127.0.0.1: inside a container, localhost
# means "this container only", and the published port connects from outside.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
