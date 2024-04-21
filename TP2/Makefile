
SHELL=/bin/bash

.venv: requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

install: .venv lib
	source .venv/bin/activate && maturin develop -m lib/Cargo.toml

install-release: .venv lib
	source .venv/bin/activate && maturin develop --release -m lib/Cargo.toml

clean:
	-@rm -r .venv
	-@cd lib && cargo clean


run: install
	source .venv/bin/activate && python run.py

run-release: install-release
	source .venv/bin/activate && python run.py

.PHONY: run run-release clean install-release install
