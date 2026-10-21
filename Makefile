#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = my_val_capstone_01
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) my_val_capstone_01/dataset.py


## Build the modelling tables of the regression (needs data/processed/model_state.csv from 02_fm_target_definition)
.PHONY: regression_dataset
regression_dataset:
	$(PYTHON_INTERPRETER) -m my_val_capstone_01.regression.dataset


## Train the regression models and learn the blend weight and the margins
.PHONY: regression_train
regression_train: regression_dataset
	$(PYTHON_INTERPRETER) -m my_val_capstone_01.regression.modeling.train


## Write data/processed/classification/model_state_with_reg.csv (model_state plus the five regression columns)
.PHONY: regression_features
regression_features: regression_train
	$(PYTHON_INTERPRETER) -m my_val_capstone_01.regression.features


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
