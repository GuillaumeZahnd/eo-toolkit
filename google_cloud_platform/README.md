# HOWTO: Google Cloud Platform

run_on_gcp.sh creates a temporary VM, copies a local folder to it, runs a Python script, copies the results back and deletes the VM.

## One-time setup

0. Prerequisites

The [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk) is installed and the project has billing enabled.

1. Authenticate

`gcloud auth login`

2. Set the project

```sh
PROJECT_ID=eo-toolkit-513110
gcloud config set project $PROJECT_ID
```

3. Enable the Compute Engine API

```sh
gcloud services enable compute.googleapis.com
```

## Run a script

```sh
PAYLOAD=./payload
SCRIPT_NAME=run_mock_up_embedding_layer.py
MACHINE_TYPE=e2-standard-4
ZONE=europe-west4-a
MAX_RUN=1h

./run_on_gcp.sh $PAYLOAD $SCRIPT_NAME $MACHINE_TYPE $ZONE $MAX_RUN
```

`PAYLOAD` is a folder that must contain the script and every module it imports. If it has a `requirements.txt`, it is installed too (PyTorch, CPU build, is always installed).

## Results

Results are downloaded in the `gcp_outputs` folder.

## Check for leftovers after a run 

All should return empty.

```sh
gcloud compute instances list
gcloud compute disks list
gcloud compute snapshots list
gcloud compute images list --no-standard-images
gcloud compute addresses list
gcloud storage buckets list
```
