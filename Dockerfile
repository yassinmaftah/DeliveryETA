FROM python:3.12-slim

WORKDIR /workspace

# Install system utilities and build requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    streamlit==1.40.1 \
    pandas==2.2.3 \
    numpy==2.1.3 \
    scikit-learn==1.5.2 \
    scipy==1.14.1 \
    joblib==1.4.2 \
    matplotlib==3.9.2

COPY . /workspace

RUN chmod +x /workspace/scripts/entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/bin/bash", "/workspace/scripts/entrypoint.sh"]
