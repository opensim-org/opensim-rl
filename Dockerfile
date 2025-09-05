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
COPY train_gait3d.py /opensim-rl/train_gait3d.py
COPY eval_gait3d.py /opensim-rl/eval_gait3d.py
COPY print_file.py /opensim-rl/print_file.py
COPY requirements.txt /opensim-rl/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Test installation
RUN python3 -m tests.test_envs
