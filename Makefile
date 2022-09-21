SHELL :=/bin/bash

TAG=$(shell git rev-parse --short=9 HEAD)
PWD=$(shell pwd)

genprotos:
	@git submodule update --remote --merge && cd $(PWD)/schema/protos && make genprotos
	@git submodule >> .submodule
	@echo " done. "
