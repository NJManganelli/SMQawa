#!/usr/bin/env bash
#
# Tear down the local HTCondor pool.
#   ./teardown-condor-docker.sh            # stop + remove containers/network
#   ./teardown-condor-docker.sh --volumes  # also drop the pool-password/secrets volume
#
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--volumes" ]]; then
  docker compose --profile histserv down --volumes
  echo ">>> Containers, network, and volumes (incl. pool password) removed."
else
  docker compose --profile histserv down
  echo ">>> Containers and network removed; 'secrets' volume kept (use --volumes to drop)."
fi
