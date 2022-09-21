# nauto-zoo

Term `UCM` in this documentation stands for `universal cloud model`.

This documentation tells how to add new model inference code to the repository structure, so that it can be embedded 
inside UCM image.

## Setting up environment

Execute:
```shell
make local_install
```
This will install project dependencies to Python user install directory for your platform (f.e. ~/.local)

## Adding new model

To add new model, create a submodule under `nauto_zoo.models` and put there a class that extends `nauto_zoo.Model`.

You have to implement the method `run(model_input: ModelInput): ModelResponse`.

You can also provide method `bootstrap()` - it will be called after instantiating your model. One common usecase is downloading model files.

It is recommended to add functional tests for your model (see folder `tests`). The library provides some exemplary inputs (in/out videos, snapshots, sensor files)

## Releasing new version

Update `nauto_zoo/__version__.py` by incrementating `__version__` attribute. You can reference your model later on inside universal cloud model image.
