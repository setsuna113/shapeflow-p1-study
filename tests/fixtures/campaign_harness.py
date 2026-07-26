"""Provider + ledger + runner, wired the way the campaign wires them.

Shared so the crash tests and the happy-path tests exercise the same wiring. A harness that
differed between them would let a guard pass against a shape the real run never has.
"""

from __future__ import annotations

from pathlib import Path

from shapeflow_p1.campaign.runner import CampaignRunner, RunnerConfig
from shapeflow_p1.campaign.selector_client import SelectorModelCall
from shapeflow_p1.experiment.budget import Budget
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore
from shapeflow_p1.providers.provider_client import ProviderClient
from shapeflow_p1.runtime.provider_server import (
    ProviderConfig,
    ProviderService,
    RoleTokens,
    serve_forever,
)
from shapeflow_p1.secrets import SecretRedactor

TEST_EXECUTION_BINDING_SHA256 = "e" * 64
TEST_PROTOCOL_DOCUMENT_SHA256 = "d" * 64


class Harness:
    def __init__(self, settings, tmp_path: Path, engine, tokens: dict, repo: Path) -> None:
        self.settings = settings
        self.engine = engine
        self.repo = repo
        self.tokens = tokens
        self.provider_ledger = Ledger(str(tmp_path / "provider.sqlite"))
        budget = Budget(self.provider_ledger)
        for resource, cap in settings.budget_caps().items():
            budget.ensure_account(resource, cap)
        redactor = SecretRedactor()
        self.service = ProviderService(
            ProviderConfig(served_model="Qwen3-14B-AWQ", max_upstream_inflight=1),
            ledger=self.provider_ledger, budget=budget,
            store=ObjectStore(tmp_path / "provider-objects"), redactor=redactor,
            tokens=RoleTokens(tokens), upstream=engine, tavily_key=None, deepseek_key=None,
        )
        self.service.reconcile_on_start()
        self.tcp, _ = serve_forever(self.service, ProviderConfig(bind_port=0), redactor)
        self.base = f"http://127.0.0.1:{self.tcp.server_address[1]}"
        self.client = ProviderClient(base_url=self.base, token=tokens["runner"])
        self.run_ledger = Ledger(str(tmp_path / "run.sqlite"))
        self.store = ObjectStore(tmp_path / "run-objects")

    def runner(
        self,
        *,
        execution_binding_sha256: str = TEST_EXECUTION_BINDING_SHA256,
        protocol_document_sha256: str = TEST_PROTOCOL_DOCUMENT_SHA256,
        **kw,
    ) -> CampaignRunner:
        async def register(spec):
            await self.client.register_cell(
                cell_token=spec.cell_token, run_id=spec.run_id, task_id=spec.task_id,
                arm_id=spec.arm_id, variant_id=spec.variant_id,
                replicate_id=spec.replicate_id, work_key=spec.work_key)

        def model_call_factory(cell_token: str, *, seed: int) -> SelectorModelCall:
            return SelectorModelCall(
                self.client, cell_token=cell_token, repo=self.repo, temperature=0.0, top_p=1.0,
                max_completion_tokens=int(self.settings.get(
                    "week1", "measurement", "selector_max_completion_tokens")),
                seed=seed,
            )
        config = RunnerConfig(run_id="RUN-TEST", provider_base_url=self.base,
                              runner_token=self.tokens["runner"], **kw)
        async def fetch_work_summary(work_key: str):
            return await self.client.work_summary(work_key=work_key, require_isolated=True)

        return CampaignRunner(self.settings, ledger=self.run_ledger, store=self.store,
                              config=config,
                              execution_binding_sha256=execution_binding_sha256,
                              protocol_document_sha256=protocol_document_sha256,
                              model_call_factory=model_call_factory,
                              register_cell=register,
                              fetch_work_summary=fetch_work_summary,
                              engine_epoch="0" * 32)

    def close(self) -> None:
        self.tcp.shutdown()
        self.provider_ledger.close()
        self.run_ledger.close()
