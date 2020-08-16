FROM flatironinstitute/triqs:latest

ARG email
ARG version="2.0"
LABEL \
    maintainer=$email \
    version=$version \
    description="Cluster Dynamical Mean-Field Theory in Python using TRIQS"

RUN sudo mkdir /cdmft && \
    sudo chown -R triqs /cdmft && \
    mkdir /cdmft/run && \
    git clone https://github.com/MHarland/periodization.git /cdmft/periodization && \
    git clone https://github.com/MHarland/scstiffness.git /cdmft/scstiffness
COPY . /cdmft/cdmft
ENV PYTHONPATH="/cdmft/cdmft:/cdmft/periodization:/cdmft/scstiffness:${PYTHONPATH}"

WORKDIR /cdmft/run
CMD ["python", "/cdmft/cdmft/test/run_tests.py"]
