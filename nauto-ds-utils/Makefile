ifeq ($(VIRTUAL_ENV),)
	PIP_MODE := --user
endif

local_install:
	pip install $(PIP_MODE) -e .
	pip list
ifneq (, $(shell which pipdeptree))
	pipdeptree
endif

unittests:
	pytest ./tests

artifactory_pypi:
	@./artifactory_pypirc.sh

publish_library:
	python3 setup.py bdist_wheel upload -r local
