#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT_DIR="$( cd $SCRIPT_DIR && cd .. && pwd )"
REPO_ROOT_DIR=$(cd $PROJECT_ROOT_DIR && cd .. && pwd)
SENSOR_PROTO_PATH=${REPO_ROOT_DIR}/"schema/protos/protobuf/sensors/sensor.proto"
SENSOR_PROTO_LOCAL_PATH=${PROJECT_ROOT_DIR}/sensor/sensor.proto

if [ "$OSTYPE" == "linux-gnu" ]; then
  MD5=md5sum
elif [[ "$OSTYPE" == "darwin"* ]]; then
  MD5=md5
fi

SENSOR_PROTO_HASH=`${MD5} ${SENSOR_PROTO_PATH} | awk '{print $4}'`
SENSOR_PROTO_LOCAL_HASH=`${MD5} ${SENSOR_PROTO_LOCAL_PATH} | awk '{print $4}'`

echo 'Running through protos update check'

if [ "${SENSOR_PROTO_HASH}" != "${SENSOR_PROTO_LOCAL_HASH}" ]; then
  echo 'sensor.proto hash has been changed, checking differences'
  DIFF_VALUE=`diff --ignore-all-space --ignore-blank-lines --unchanged-line-format="" --old-line-format="(protos        ):%dn: %L" --new-line-format="(nauto-datasets):%dn: %L" $SENSOR_PROTO_PATH $SENSOR_PROTO_LOCAL_PATH`
  if [ -n "$DIFF_VALUE" ]; then
    echo 'Found differences:'
    echo $DIFF_VALUE
    exit 1
  else
    echo 'Only whitespace differences, no real differences'
  fi
fi