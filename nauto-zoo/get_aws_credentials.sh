#!/bin/bash

ENV=$1
TAG=$(git rev-parse HEAD)
DATE=$(date -u +%Y%m%d%H%M)

if [ "$ENV" == "" ]
then
  echo "ENV variable not set"
  exit 1
fi

aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID_PROD_US
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY_PROD_US
aws configure set default.region us-east-1

ENV_UPPERCASE=$(echo $ENV | tr a-z A-Z)
ROLE_ARN=IAM_ROLE_CIRCLECI_$ENV_UPPERCASE
echo ${!ROLE_ARN}
ROLE=`aws sts assume-role --role-arn "${!ROLE_ARN}" --role-session-name "$IAM_ROLE_TEMP_SESSION" --output json`
# get the creds
TEMP_ACCESS_KEY_ID=`echo $ROLE | jq -r '.Credentials.AccessKeyId'`
TEMP_SECRET_ACCESS_KEY=`echo $ROLE | jq -r '.Credentials.SecretAccessKey'`
TEMP_SESSION_TOKEN=`echo $ROLE | jq -r '.Credentials.SessionToken'`
# override the aws creds
export AWS_ACCESS_KEY_ID=$TEMP_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$TEMP_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN=$TEMP_SESSION_TOKEN


aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID --profile $ENV
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY --profile $ENV
aws configure set aws_session_token $AWS_SESSION_TOKEN --profile $ENV
aws configure set region us-east-1 --profile $ENV