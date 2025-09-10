# syntax=docker/dockerfile:1
FROM nbianco/opensim-rl-base

# Set working directory
WORKDIR /opensim-rl

# Copy the repository contents
COPY ./deprl /opensim-rl/deprl
COPY ./environments /opensim-rl/environments
COPY ./models /opensim-rl/models
COPY ./tests /opensim-rl/tests
COPY __init__.py /opensim-rl/__init__.py
COPY train.py /opensim-rl/train.py
COPY eval.py /opensim-rl/eval.py
COPY print_file.py /opensim-rl/print_file.py

# Test installation
# RUN python3 train.py --env-id Gait3D --output-dir ./test_run --num-envs 4 --timesteps 25000
