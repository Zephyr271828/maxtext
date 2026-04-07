#!/bin/bash

# Older generated launcher scripts still call ensure_gcloud_ssh_key(), which in
# turn shells out to ssh-keygen/chmod for ~/.ssh/google_compute_engine. The
# current runner path uses the jobman-managed ~/.ssh/id_rsa instead, so short-
# circuit those helper calls when the requested output key is the legacy path.
_jobman_legacy_gcloud_key_path() {
  printf '%s' "$HOME/.ssh/google_compute_engine"
}

_jobman_has_id_rsa() {
  [[ -f "$HOME/.ssh/id_rsa" && -f "$HOME/.ssh/id_rsa.pub" ]]
}

ssh-keygen() {
  local out=""
  local prev=""
  for arg in "$@"; do
    if [[ "$prev" == "-f" ]]; then
      out="$arg"
      break
    fi
    prev="$arg"
  done

  if _jobman_has_id_rsa && [[ "$out" == "$(_jobman_legacy_gcloud_key_path)" ]]; then
    return 0
  fi

  command ssh-keygen "$@"
}

chmod() {
  local legacy_key="$(_jobman_legacy_gcloud_key_path)"
  local legacy_pub="${legacy_key}.pub"

  if _jobman_has_id_rsa && [[ "$#" -eq 3 && "$1" == "600" && "$2" == "$legacy_key" && "$3" == "$legacy_pub" ]]; then
    return 0
  fi

  command chmod "$@"
}

get_tpu_name() {
  local zone ip name
  zone=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F'/' '{print $NF}')
  ip=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip)

  # Match against any semicolon-separated IP list (exact match to avoid substring collisions)
  name=$(gcloud compute tpus tpu-vm list --zone="$zone" \
    --format="value(name,networkEndpoints.ipAddress)" \
    | awk -v ip="$ip" '{
        n = split($2, ips, ";")
        for (i = 1; i <= n; i++) {
          if (ips[i] == ip) { print $1; exit }
        }
      }')

  # Check that TPU name contains 'yufeng'
  if [[ ! "${name}" =~ yufeng ]]; then
    echo "❌ Error: TPU name '${name}' does not contain 'yufeng'. Exiting."
    exit 1
  fi

  echo "$name"
}

get_zone() {
  zone=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F'/' '{print $NF}')
  echo "$zone"
}

get_bucket_name() {
  MOUNT_DIR="$HOME/gcs-bucket" 
  BUCKET_NAME=$(mount | grep "on ${MOUNT_DIR}" | awk '{print $1}')
  echo "${BUCKET_NAME#gs://}"
}

get_data_bucket_name() {
  MOUNT_DIR="$HOME/gcs-data" 
  BUCKET_NAME=$(mount | grep "on ${MOUNT_DIR}" | awk '{print $1}')
  echo "${BUCKET_NAME#gs://}"
}

get_num_hosts() {
  ZONE=$(get_zone)
  TPU_NAME=$(get_tpu_name)
  HOSTS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format="value(networkEndpoints)" | grep -o "ipAddress" | wc -l) 
  echo "$HOSTS"
}

TPU_NAME=$(get_tpu_name)
echo "✅ Detected TPU name: ${TPU_NAME:-unknown}"

TPU_ZONE=$(get_zone)
echo "✅ Detected TPU zone: ${TPU_ZONE:-unknown}"

BUCKET_NAME=$(get_bucket_name)
echo "✅ Detected Bucket name: ${BUCKET_NAME:-unknown}"

if [[ -d $HOME/gcs-data ]]; then
  DATA_BUCKET_NAME=$(get_data_bucket_name)
  echo "✅ Detected Data Bucket name: ${DATA_BUCKET_NAME:-unknown}"
fi

NUM_HOSTS=$(get_num_hosts)
echo "✅ Detected number of hosts: ${NUM_HOSTS:-unknown}"
