#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Creating/updating ConfigMap 'gemma-abliterate-script' in namespace 'llm'..."
kubectl create configmap gemma-abliterate-script \
  --from-file=abliterate.py="${SCRIPT_DIR}/abliterate.py" \
  -n llm --dry-run=client -o yaml | kubectl apply -f -

echo "Cleaning up any previous job run..."
kubectl delete job gemma-abliterate -n llm --ignore-not-found=true

K8S_JOB="${REPO_ROOT}/k8s/job-abliterate.yaml"
if [ ! -f "$K8S_JOB" ]; then
  K8S_JOB="${SCRIPT_DIR}/job-abliterate.yaml"
fi

echo "Submitting abliteration job from ${K8S_JOB} to distiller node..."
kubectl apply -f "${K8S_JOB}"

echo ""
echo "Job submitted! Follow progress with:"
echo "  kubectl logs -n llm job/gemma-abliterate -f"
