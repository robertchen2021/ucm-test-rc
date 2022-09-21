This library contains common util tools and algorithm codes contributed mainly by the Nauto Data Science team members. 

# Installation

## For Development
```
pip install -e .
```

## Cloud Deployment
```
pip3 install --index-url=https://$JFROG_USERNAME:$JFROG_PASSWORD@nauto.jfrog.io/nauto/api/pypi/drt-virtual/simple/ nauto_ds_utils
```
* Credentials can be found in your 1Password Engineering Vault.

## Testing
Testing data and files are in `nauto-ds-utils/tests/`. I've created a file specifically for data (`nauto-ds-utils/tests/data/`)(we can use that as a data or model dump).
```
$ cd nauto-ai/nauto-ds-utils
$ pytest .
```


## Publishing Library
Stay within this directory (`nauto-ds-utils`) and run the following command:
```
$ cd nauto-ai/nauto-ds-utils
$ make publish_library
```

## Packaging
In order to build a wheel package run:
```
python setup.py bdist_wheel
```
Installing the wheel:
```
pip install <path-to-wheel>
```

If you're having trouble installing `nauto-datasets` and `nauto-ds-utils` on your local machine -- If you force download `nauto-datasets==0.8.0` locally and then `sudo nauto-ds-utils` it will work. 

## Versions
0.1.0 - Deserialization and Utility Functions

0.1.1 - Renaming package layout. Added more functionality.

0.1.3 -- MakeFile, added Atypical Detectors Preprocessor and Model, Crashnet Preprocess & Model.
         More unitttests, utility functions for tensorflow and sklearn.
