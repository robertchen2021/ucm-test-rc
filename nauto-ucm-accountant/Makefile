PIP_MODE := $(if $(VIRTUAL_ENV), , --user)
.PHONY: tests

local_install:
	pip install $(PIP_MODE) -e .[test] --progress-bar off
	pip list
ifneq (, $(shell which pipdeptree))
	pipdeptree
endif

tests:
	pytest ./tests

artifactory_pypi:
	@./artifactory_pypirc.sh

publish_library: artifactory_pypi
	python setup.py bdist_wheel upload -r local
