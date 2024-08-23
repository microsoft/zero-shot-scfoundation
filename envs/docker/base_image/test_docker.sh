#! /bin/bash

# This script is used to test the docker image built by the Dockerfile in the same directory.
# The docker image is built by the following command:
# docker build -t kzkedzierska/sc_foundation_evals[:tag] .

# The script runs the docker image and executes the test.py script in the container.
# The test.py script is a simple script that imports the sc_foundation_evals package and prints the version of the package.

docker run \
  --gpus all \
  -v "$(pwd)":/workspace kzkedzierska/sc_foundation_evals \
  python test.py
