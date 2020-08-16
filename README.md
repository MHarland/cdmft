# cdmft - Cluster Dynamical Mean-Field Theory

An implementation of the cluster dynamical mean-field theory in Python using [TRIQS](https://github.com/TRIQS/triqs).

## Installation

- Get [Docker](https://www.docker.com/)

- Get the cdmft Docker image by either `docker pull mharland/cdmft` or `git clone https://github.com/MHarland/cdmft.git && cd cdmft && docker build -t cdmft --build-arg email=YOUREMAIL .`

- Run tests `docker run --rm cdmft`

## Run

- Change into the directory in which you have the script to run and also in which the output shall be, e.g. `cd example`.

- Run `docker run --rm -v ${PWD}:/cdmft/run cdmft python ex_cdmft.py` or with MPI `docker run --rm -v ${PWD}:/cdmft/run cdmft mpirun --mca btl_vader_single_copy_mechanism none -np 8 python ex_cdmft.py`

The flag `--mca btl_vader_single_copy_mechanism none` is unfortunately required to prevent a shared memory issue of MPI.

## Develop the `cdmft` library

I suggest using a Docker container with a bind-mount including the code. In the same directory in which you ran the `git clone` command, run `docker run -it -v ${PWD}:/cdmft cdmft bash`. Then you share your code directories with the container and you can also run experiments in any subdirectories.
