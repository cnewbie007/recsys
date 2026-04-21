FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch is already in the base image with the correct CUDA build.
# Reinstalling from PyPI would replace it with a CPU-only wheel, breaking CUDA.
RUN grep -v '^torch' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

COPY . .

ENV PYTHONPATH=/app

# Set WANDB_API_KEY as a RunPod secret environment variable.
# Override CMD per job, e.g.:
#   python -m two_tower.train --emb_size 64 --epochs 50
#   python -m factorization_machine.train --emb_size 64 --epochs 50
#   python -m collaborative_filtering.main
CMD ["bash"]
