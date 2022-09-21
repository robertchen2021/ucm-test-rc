#!/bin/bash

cat <<EOF > ~/.pypirc
[distutils]
index-servers = local
[local]
repository: https://nauto.jfrog.io/nauto/api/pypi/ml-infra
username: $ARTIFACTORY_USER
password: $ARTIFACTORY_PASSWORD
EOF