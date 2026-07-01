# Nexus paper evaluation and build
ROOT := $(abspath ../..)
PYTHON := $(shell if [ -x "$(ROOT)/.venv/bin/python3" ]; then echo "$(ROOT)/.venv/bin/python3"; else echo python3; fi)
SCRIPTS := scripts
DATA := data
FIGURES := figures
TABLES := tables

export NEXUS_MODEL ?= /Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf
export PYTHONPATH := $(ROOT)/build:$(ROOT)/src:$(ROOT)/test
export PYTHONUNBUFFERED := 1
export NEXUS_EMBED_MODEL ?= /Users/mustafarslan/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5-GGUF/snapshots/0188c9bf409793f810680a5a431e7b899c46104c/nomic-embed-text-v1.5.f16.gguf

.PHONY: eval eval-fast pdf clean figures tables metrics

eval: eval-fast bench-cdf bench-waterfall bench-scalability bench-ablation figures tables metrics
	@echo "Paper evaluation complete. Artifacts in $(DATA)/, $(FIGURES)/, $(TABLES)/"

eval-fast: physics l0
	@echo "Fast eval steps done."

physics:
	cd $(SCRIPTS) && $(PYTHON) aggregate_physics_boundary.py

l0:
	cd $(SCRIPTS) && $(PYTHON) bench_l0_radix_micro.py

bench-cdf:
	cd $(SCRIPTS) && $(PYTHON) bench_cdf_pareto.py --force

bench-waterfall:
	cd $(SCRIPTS) && $(PYTHON) bench_waterfall_decomposition.py

bench-scalability:
	cd $(SCRIPTS) && $(PYTHON) bench_scalability.py

bench-ablation:
	cd $(SCRIPTS) && $(PYTHON) bench_ml_ablation.py

figures:
	cd $(SCRIPTS) && $(PYTHON) generate_plots.py

tables:
	cd $(SCRIPTS) && $(PYTHON) generate_latex_tables.py

metrics:
	cd $(SCRIPTS) && $(PYTHON) generate_master_metrics.py

pdf:
	pdflatex -interaction=nonstopmode main.tex || true
	pdflatex -interaction=nonstopmode main.tex || true
	@test -f main.pdf

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
