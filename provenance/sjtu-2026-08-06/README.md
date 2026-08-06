# sjtu provenance snapshot — 2026-08-06

Captured when the sjtu GPU node was decommissioned and wiped.

- `deploy/` — the three hand-written systemd units (`shapeflow-api-provider`,
  `shapeflow-p1-week1`, `shapeflow-vllm-causal`), the `sfsupervise` supervision script
  from `/usr/local/bin`, the repo's own unit templates, the service-account list
  (`sfinfer`, `sfprovider`, `sfrunner`, `sfsteward`, `sfevaluator`) and the
  `/run/shapeflow` runtime-state listing. This is the deployment topology the campaign
  actually ran under; none of it was in the repository.
- `deploy-backup-20260724/` — a pre-deploy snapshot (`HEAD`, `worktree.diff`,
  `untracked.tar.gz`) of an uncommitted working tree from 2026-07-24.

Provider credentials were staged at `/etc/shapeflow/{tavily,deepseek,exa}.key` on the
node. No key value is recorded here; the keys are being rotated.
