# Deployment

Unit file: [`deploy/pinance-ml-inference.service`](../deploy/pinance-ml-inference.service). This describes what the unit does; it does not claim that any particular host is running it now.

## Shape

One uvicorn process under systemd. It does **not** listen on `0.0.0.0`: the `ExecStart` line resolves the node's private-network (Tailscale) IPv4 address at start-up with `tailscale ip -4` and binds port 8001 to it, so the service is unreachable from the public internet and its address survives reboots. The backend calls it over that private network.

| Setting in the unit | Effect |
|---|---|
| `After=network-online.target tailscaled.service` | wait until the private-network address exists |
| `EnvironmentFile=` | secrets and endpoints come from an untracked `.env` next to the code |
| `Restart=on-failure`, `RestartSec=5` | restart after a crash |
| `NoNewPrivileges=true`, `ProtectSystem=strict` | no writable paths are needed: models live in memory, the database user is read-only |

## What the service needs at start-up

1. The environment variables from [`.env.example`](../.env.example) (see the table in the README).
2. A reachable MinIO with the models bucket. A wrong endpoint fails quickly (2 s connect, 5 s read, no retries) rather than hanging start-up.
3. A reachable database with a `candles` table, read once per poll for the symbol list. If the database (or MinIO) is unreachable at start-up, the initial poll logs the error and the app starts anyway; `/health` and `/models` answer, `/predict` returns 404 until the poll loop (which starts immediately and then repeats every interval) manages to load models (`test_app_starts_when_the_initial_poll_fails`).

The first poll is synchronous in the lifespan hook, so when everything is reachable the service has its models when it starts accepting requests.

## Rolling out a new feature schema

Because models trained on another feature list are rejected, a schema change is a coordinated release:

1. Re-vendor `vendor/features.py` from the new training commit and update `vendor/LOCK.md`.
2. Refresh `tests/fixtures/training_schema_vectors.json` from a freshly exported model's `metadata.json`; the parity tests must pass.
3. Deploy (restart) this service. Models are held in memory, so after the restart the old-schema models are polled again and rejected (`last_load_error` in `GET /models`); until new-schema models are pushed, the slot answers 404.
4. Push models on the new schema and let promotion see the new `expected_schema_version`.

Candidates on a new schema cannot be shadow-served by the old code, so promotion across a schema change is a manual step (also noted in the training repository).

## Trying it without a deployment

[docs/offline-demo.md](offline-demo.md) starts the same application with toy models and a local directory instead of MinIO. It does not exercise the unit file.

## Not verified

The unit was not started in the environment in which this documentation was written (no `systemd`/`tailscale` setup, no MinIO). The unit uses placeholder values: user and group `pinance`, code and virtualenv in `/opt/pinance_ml_inference`. Adapt them to the host.
