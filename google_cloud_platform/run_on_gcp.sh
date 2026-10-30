#!/usr/bin/env bash
# Run a Python script on a temporary Compute Engine VM, fetch results, delete the VM.
# Usage: ./run_on_gcp.sh <local_dir> <script.py> [machine_type]
# Env:   ZONE (default europe-west4-a), MAX_RUN (default 2h; server-side auto-delete), KEEP_VM=1

set -euo pipefail

DIR="${1:?Missing local directory}"
SCRIPT="${2:?Missing script name}"
MACHINE="${3:-e2-standard-4}"
ZONE="${ZONE:-europe-west4-a}"
MAX_RUN="${MAX_RUN:-2h}"
VM="job-$(date +%Y%m%d-%H%M%S)"
DIR="${DIR%/}"
FETCHED=0
rc=0

[[ -f "$DIR/$SCRIPT" ]] || { echo "Error: $DIR/$SCRIPT not found" >&2; exit 1; }

cleanup() {
    trap - EXIT
    if [[ "${KEEP_VM:-0}" == "1" ]]; then
        echo "VM left running: $VM (Zone: $ZONE)"
        return
    fi
    # Nothing to clean up if the VM was never created.
    gcloud compute instances describe "$VM" --zone "$ZONE" &>/dev/null || return 0

    if [[ "$FETCHED" == "1" ]]; then
        echo "Deleting VM $VM..."
        gcloud compute instances delete "$VM" --zone "$ZONE" --quiet \
            || echo "!!! DELETE FAILED. Run: gcloud compute instances delete $VM --zone $ZONE" >&2
    else
        echo "!!! Results were NOT fetched. Stopping $VM instead of deleting it" >&2
        echo "    (compute billing stops, the disk is kept and still billed)." >&2
        gcloud compute instances stop "$VM" --zone "$ZONE" --quiet \
            || echo "!!! STOP FAILED. Run: gcloud compute instances stop $VM --zone $ZONE" >&2
        echo "    Recover: gcloud compute instances start $VM --zone $ZONE, copy ~/job/outputs," >&2
        echo "    then: gcloud compute instances delete $VM --zone $ZONE" >&2
    fi
}
trap cleanup EXIT

echo "Creating VM $VM ($MACHINE)..."
gcloud compute instances create "$VM" \
    --zone "$ZONE" \
    --machine-type "$MACHINE" \
    --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size 30GB \
    --max-run-duration "$MAX_RUN" --instance-termination-action DELETE \
    --quiet

echo "Waiting for SSH..."
for i in {1..60}; do
    gcloud compute ssh "$VM" --zone "$ZONE" --command "true" &>/dev/null && break
    [[ "$i" == 60 ]] && { echo "SSH never came up" >&2; exit 1; }
    sleep 5
done

echo "Syncing files..."
gcloud compute scp --recurse "$DIR" "$VM:~/job" --zone "$ZONE" --quiet

echo "Running execution..."
gcloud compute ssh "$VM" --zone "$ZONE" --command "
    set -e
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv
    python3 -m venv ~/venv
    ~/venv/bin/pip install -q --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
    cd ~/job
    if [ -f requirements.txt ]; then ~/venv/bin/pip install -q --no-cache-dir -r requirements.txt; fi
    ~/venv/bin/python '$SCRIPT'
" || rc=$?   # keep going on failure: partial outputs are still fetched below

echo "Fetching outputs..."
mkdir -p gcp_outputs
has_out=$(gcloud compute ssh "$VM" --zone "$ZONE" \
    --command 'test -d ~/job/outputs && echo yes || echo no' 2>/dev/null) || has_out=error
if [[ "$has_out" == "yes" ]]; then
    gcloud compute scp --recurse "$VM:~/job/outputs" gcp_outputs/ --zone "$ZONE" --quiet && FETCHED=1
elif [[ "$has_out" == "no" ]]; then
    echo "No outputs/ directory produced."
    FETCHED=1
fi

[[ "$rc" == 0 ]] || echo "!!! The script exited with status $rc" >&2
exit "$rc"
