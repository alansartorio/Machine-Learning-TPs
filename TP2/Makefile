SHELL=/bin/bash

RUST_FILES=$(shell git ls-files lib)

VENV ?= .venv
PYTHON ?= python3
WHEELS ?= lib/target/wheels

WHEEL_OUTPUT=$(wildcard $(WHEELS)/tp2-*.whl)
ifeq ($(WHEEL_OUTPUT),)
WHEEL_OUTPUT := wheelfile
endif

$(info $$VENV is [${VENV}])
$(info $$WHEEL_OUTPUT is [${WHEEL_OUTPUT}])
$(info $$WHEELS is [${WHEELS}])

GRAPHS=out/single_tree_depth_2.dot
GRAPHS_OUT=$(GRAPHS:out/%.dot=plots/graphs/%.svg)

.venv: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

$(WHEEL_OUTPUT): $(RUST_FILES)
	VIRTUAL_ENV=$(VENV) maturin build \
				--release \
				--out $(WHEELS) \
				--skip-auditwheel \
				-i $(PYTHON) \
				-m lib/Cargo.toml

install: $(WHEEL_OUTPUT)
	source $(VENV)/bin/activate && pip install $(WHEELS)/tp2-*.whl

clean:
	-@rm -r $(VENV)
	-@cd lib && cargo clean


run: install
	source $(VENV)/bin/activate && $(PYTHON) $(SCRIPT)

run-docker:
	docker build --tag ml-tp2 . && \
		docker run --rm \
			-v $$(pwd)/.rustcache:/root/.cargo/registry \
			-v $$(pwd)/..:/io \
			-v $$(pwd)/docker-target:/io/TP2/lib/target \
			ml-tp2:latest \
			SCRIPT=$(SCRIPT)

dist/index.html: $(GRAPHS_OUT)

plots/graphs/%.svg: out/%.dot
	mkdir -p $(dir $@)
	dot $< -Tsvg > $@

-include ../SlideMakefile.mk

$(info $$GRAPHS_OUT is [${GRAPHS_OUT}])

.PHONY: .venv run run-docker clean install wheelfile
