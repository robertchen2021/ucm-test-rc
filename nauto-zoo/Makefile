PIP_MODE := $(if $(VIRTUAL_ENV), , --user)
.PHONY: tests

local_install:
	pip install $(PIP_MODE) -e .[dev,preprocess,test] --progress-bar off
	pip list
ifneq (, $(shell which pipdeptree))
	pipdeptree
endif

login-aws:
	@eval `aws ecr get-login --no-include-email --region=us-east-1`

get-aws-credentials:
	echo "get aws credentials..."
	./get_aws_credentials.sh $(ENV)

test_caffe:
	pytest ./tests -m caffe

test_no_caffe:
	pytest ./tests -m "not caffe"

artifactory_pypi:
	@./artifactory_pypirc.sh

publish_library: artifactory_pypi
	python setup.py bdist_wheel upload -r local
