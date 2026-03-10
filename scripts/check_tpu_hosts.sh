#!/bin/bash

# Source helper functions from get_tpu_bucket_name.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

get_zone() {
  curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F'/' '{print $NF}'
}

get_tpu_name() {
  local zone ip name
  zone=$(get_zone)
  ip=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip)

  name=$(gcloud compute tpus tpu-vm list --zone="$zone" \
    --format="value(name,networkEndpoints.ipAddress)" \
    | awk -v ip="$ip" 'index($2, ip) {print $1; exit}')

  if [[ ! "${name}" =~ yufeng ]]; then
    echo "❌ Error: TPU name '${name}' does not contain 'yufeng'. Exiting." >&2
    exit 1
  fi

  echo "$name"
}

SSH_TIMEOUT=${SSH_TIMEOUT:-5}

ZONE=$(get_zone)
TPU_NAME=$(get_tpu_name)
NUM_HOSTS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" \
  --format="value(networkEndpoints)" | grep -o "ipAddress" | wc -l)

echo "TPU: $TPU_NAME  Zone: $ZONE  Hosts: $NUM_HOSTS"
echo ""

reachable=0
unreachable=0
declare -a failed_workers=()

for ((worker=0; worker<NUM_HOSTS; worker++)); do
  result=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --worker="$worker" \
    --command="echo ok" \
    --ssh-flag="-o ConnectTimeout=${SSH_TIMEOUT}" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    2>/dev/null)

  if [[ "$result" == *"ok"* ]]; then
    echo "  [worker $worker] ✅ reachable"
    ((reachable++))
  else
    echo "  [worker $worker] ❌ unreachable"
    ((unreachable++))
    failed_workers+=("$worker")
  fi
done

echo ""
echo "========================================="
echo "Summary: $reachable/$NUM_HOSTS hosts reachable"
if [[ ${#failed_workers[@]} -gt 0 ]]; then
  echo "Unreachable workers: ${failed_workers[*]}"
fi
echo "========================================="
