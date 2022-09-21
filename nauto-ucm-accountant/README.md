# Nauto UCM accountant

This library is used to generate accuracy monitoring of `universal cloud model` aka `UCM`. 

Metrics are calculated by comparing human judgements with model judgements, so if some model is not double-checked by a human - there is nothing to analyze.

## Usage

### Requirements

For the lib to work, some env vars should be provided.

"QUBOLE_API_TOKEN" should contain - surprise! - Qubole API token. This token should allow running Presto queries on prod-us.

There should also be env vars that enable reading and writing to the S3 bucket "nauto-cloud-models-test-us"

### Reading data

Results can be viewed in console by running `python3 -m nauto_ucm_accountant.read`

Results can be shown in Qubole, please find the example notebook [here](https://us.qubole.com/users/sign_in#home?id=62066).

### Writing data

To update data, run `python3 -m nauto_ucm_accountant.write`

There is an optional flag `--days` - how many past days should be updated (since not all judgements are produced immediately).
The default value is 3.
Pass value 0 to recalculate all data since UCM was launched.

## Updating

There are two important configs in `nauto_ucm_accountant/accountant/config.py`

Parameter `ALL_MODELS` lists all the models for which statistics should be gathered. After new model was deployed - add it to the list.

Parameter `S3_ROOT` controls where on S3 are stored the intermediate results. Notice it is used both for reading and writing.
