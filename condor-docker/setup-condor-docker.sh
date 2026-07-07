#!/usr/bin/env bash
#
# Stand up a local HTCondor v25 pool in Docker: central manager (cm),
# access point (submit), and 3 worker nodes (execute), sharing a pool-password
# secret over a private bridge. Optionally also start the histserv service node.
#
#   ./setup-condor-docker.sh              # cm + submit + 3 workers
#   ./setup-condor-docker.sh --histserv   # + the long-running hist server node
#   CONDOR_VERSION=25.0-el9 ./setup-condor-docker.sh
#   WORKERS=5 ./setup-condor-docker.sh    # more than the default 3 workers
#
set -euo pipefail

cd "$(dirname "$0")"

export CONDOR_VERSION="${CONDOR_VERSION:-25.0-el9}"
WORKERS="${WORKERS:-3}"
POOL_PW="${POOL_PW:-$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24)}"
WITH_HISTSERV=0
[[ "${1:-}" == "--histserv" ]] && WITH_HISTSERV=1

# docker compose v2 (plugin) required.
DC="docker compose"
$DC version >/dev/null 2>&1 || { echo "ERROR: 'docker compose' (v2) is required." >&2; exit 1; }

echo ">>> HTCondor image tag: ${CONDOR_VERSION} (must be >= 25)"
case "${CONDOR_VERSION}" in
  25.*|2[6-9].*|[3-9][0-9].*) : ;;
  *) echo "!!! WARNING: CONDOR_VERSION=${CONDOR_VERSION} does not look like >= 25." >&2 ;;
esac

# 1) Create the shared pool-password blob in the `secrets` volume with a throwaway
#    container (no daemons running yet). condor_store_cred writes a root-owned 0600
#    file that every node reads via SEC_PASSWORD_FILE.
echo ">>> Creating shared pool password in the 'secrets' volume ..."
$DC run --rm --no-deps -e POOL_PW="${POOL_PW}" cm bash -lc '
  set -e
  if [ ! -s /etc/condor/secrets/pool_password ]; then
    condor_store_cred -c add -p "$POOL_PW" -f /etc/condor/secrets/pool_password
    chmod 600 /etc/condor/secrets/pool_password
  fi
  test -s /etc/condor/secrets/pool_password && echo "    pool password ready."
'

# 2) Bring up the pool.
echo ">>> Starting central manager, access point, and ${WORKERS} workers ..."
$DC up -d cm submit
$DC up -d --scale execute="${WORKERS}" execute
if [[ "${WITH_HISTSERV}" == "1" ]]; then
  echo ">>> Starting histserv service node (DAGMan SERVICE-node stand-in) ..."
  $DC --profile histserv up -d histserv
fi

# 3) Wait for the workers to register with the collector (via the AP's condor_status).
echo -n ">>> Waiting for ${WORKERS} STARTDs to report"
deadline=$(( $(date +%s) + 120 ))
while :; do
  n=$($DC exec -T submit bash -lc 'condor_status -startd -af Name 2>/dev/null | wc -l' 2>/dev/null | tr -d '[:space:]' || echo 0)
  [[ "${n:-0}" -ge "${WORKERS}" ]] && { echo " OK (${n})."; break; }
  [[ $(date +%s) -ge $deadline ]] && { echo; echo "!!! Timed out; only ${n:-0}/${WORKERS} reporting. See troubleshooting below."; break; }
  echo -n "."; sleep 3
done

echo
echo "=================== condor_status (from the access point) ==================="
$DC exec -T submit bash -lc 'condor_status; echo; condor_q -global 2>/dev/null || true' || true
echo "============================================================================="
cat <<EOF

Pool is up. Useful commands:
  docker compose exec submit condor_status                    # list worker slots
  docker compose exec -u submituser submit bash               # AP shell as the submitter
  docker compose exec -u submituser submit python3 /smqawa/condor-docker/probe_submit.py
  ./teardown-condor-docker.sh                                 # stop and remove everything

IMPORTANT: submit jobs as the 'submituser' account (-u submituser), NOT root.
Local client tools authenticate to the schedd via FS as the unix user; root maps
to the pool identity 'condor@submit', which the schedd rejects as a submitter
(HTCondor >= 23 user-record policy). submituser is the image's designated submitter.

Troubleshooting:
  * "Failed to create new User record for condor@..." -> you submitted as root;
    re-run as '-u submituser'.
  * "PERMISSION DENIED" between daemons -> pool password mismatch; run
    './teardown-condor-docker.sh --volumes' then re-run this script to regenerate it.
  * 0 workers reporting -> 'docker compose logs execute' (often the cm was not
    reachable yet; the images retry, give it a few more seconds).
EOF
