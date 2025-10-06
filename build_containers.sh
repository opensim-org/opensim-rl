#!/bin/bash

# Build base image for AMD64
docker buildx build -f Dockerfile.base --platform linux/amd64 -t nbianco/opensim-rl-base .
docker push nbianco/opensim-rl-base

docker pull --platform linux/amd64 nbianco/opensim-rl-base

# Build main image for AMD64
docker buildx build -t nbianco/opensim-rl -f Dockerfile --platform linux/amd64 .
docker push nbianco/opensim-rl
