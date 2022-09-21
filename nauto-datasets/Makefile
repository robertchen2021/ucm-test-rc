ifeq ($(VIRTUAL_ENV),)
        PIP_MODE := --user
endif

local_install:
	pip install $(PIP_MODE) -e .
	pip list
ifneq (, $(shell which pipdeptree))
	pipdeptree
endif

local_install_cpu:
	pip install $(PIP_MODE) -e .[tf-cpu,spark,dev]
	pip list
ifneq (, $(shell which pipdeptree))
	pipdeptree
endif

local_install_gpu:
	pip install $(PIP_MODE) -e .[tf-gpu,spark,dev]
	pip list
ifneq (, $(shell which pipdeptree))
	pipdeptree
endif

unittests:
	pytest ./tests

artifactory_pypi:
	@./artifactory_pypirc.sh

publish_library: artifactory_pypi
	python setup.py bdist_wheel upload -r local

check_protos:
	./scripts/check_protos_up_to_date.sh

