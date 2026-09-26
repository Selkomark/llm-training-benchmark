#!/bin/sh
# Build the image and run the benchmark on the GPU.
#   ./run.sh                   default run (3 epochs)
#   ./run.sh --epochs 1        extra args go to train.py
set -eu
cd "$(dirname "$0")"
mkdir -p data outputs .cache/hf

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
docker compose build
docker compose run --rm train sh -c "python src/generate_data.py && python src/train.py $*"
