FROM ghcr.io/pyo3/maturin

WORKDIR /io/TP2

COPY requirements.txt .
COPY Makefile .

RUN make .venv PYTHON="python3.12" VENV=/venv

RUN git config --global --add safe.directory '*'

ENTRYPOINT ["make", "run", "PYTHON=python3.12", "VENV=/venv"]

