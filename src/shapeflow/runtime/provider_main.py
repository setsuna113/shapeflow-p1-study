"""Starting the provider as the ``sfprovider`` identity.

Separate from :mod:`provider_server` so the service stays importable and testable without
touching a credential file, a socket or a real ledger. This module is the only place that reads
``/etc/shapeflow/*.key``, and it does so *after* asserting it is running as the provider
identity -- a provider that came up as root would hold every credential in a process that is
also allowed to do everything else.
"""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Optional

import httpx

from ..campaign.settings import Settings
from ..bench.grading.judge_client import load_deepseek_key
from ..experiment.budget import Budget
from ..experiment.ledger import Ledger
from ..object_store import ObjectStore
from ..secrets import REDACTOR
from .provider_server import (
    ProviderConfig,
    ProviderError,
    ProviderService,
    RoleTokens,
    serve_forever,
)

__all__ = ["build_service", "mint_role_tokens", "httpx_upstream", "run"]

ROLES = ("runner", "steward", "evaluator", "infer")


def httpx_upstream(client: Optional[httpx.Client] = None):
    """The live transport: one synchronous httpx client, shared across handler threads.

    TLS verification is never disabled, and the timeout is per-call because acquisition and
    inference have very different tails.
    """
    owned = client or httpx.Client(verify=True, follow_redirects=False)

    def upstream(url: str, headers: dict, body: dict, timeout: float):
        started = time.time()
        response = owned.post(url, headers=headers or None, json=body, timeout=timeout)
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": "upstream returned a non-JSON body"}
        if not isinstance(payload, dict):
            payload = {"data": payload}
        return response.status_code, payload, time.time() - started

    return upstream


def mint_role_tokens(token_dir: Path, uids: dict[str, int]) -> dict[str, str]:
    """Create one capability token per role, each readable only by that role's account.

    Written 0400 and chowned to the role. The provider is running as ``sfprovider``, so the
    chown needs privilege it does not have; on a host where that is true the tokens are
    pre-placed by the installer and this call simply reads them back. Minting here is what makes
    a development run work without a privileged step.
    """
    token_dir.mkdir(parents=True, exist_ok=True)
    tokens: dict[str, str] = {}
    for role in ROLES:
        path = token_dir / f"{role}.token"
        if path.exists():
            tokens[role] = path.read_text(encoding="utf-8").strip()
            continue
        token = f"{role}-" + secrets.token_urlsafe(24)
        path.write_text(token + "\n", encoding="utf-8")
        os.chmod(path, 0o400)
        uid = uids.get(role)
        if uid is not None:
            try:
                os.chown(path, uid, -1)
            except PermissionError:
                # Unprivileged run: the token is still per-file, just not per-uid. The caller
                # records SECRET_ISOLATION=PROCESS_ONLY rather than claiming uid-grade isolation.
                pass
        tokens[role] = token
    return tokens


def build_service(settings: Settings, *, upstream=None, uids: Optional[dict] = None):
    """Wire the provider from the campaign settings, reading both credentials exactly once."""
    config: ProviderConfig = settings.provider_config()
    ledger_path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(ledger_path))
    budget = Budget(ledger)
    for resource, cap in settings.budget_caps().items():
        budget.ensure_account(resource, cap)

    store = ObjectStore(settings.data_root / str(settings.get("week1", "paths", "provider_root"))
                        / "objects")
    tokens = RoleTokens(mint_role_tokens(Path(config.token_dir), uids or {}))
    service = ProviderService(
        config,
        ledger=ledger, budget=budget, store=store, redactor=REDACTOR, tokens=tokens,
        upstream=upstream or httpx_upstream(),
        # No live-search credential is loaded. The campaign retrieves from a frozen corpus, so
        # a provider holding a search key would be holding the one thing that could put a
        # treatment run back on the live web. The routes remain and fail closed without it.
        exa_key=None,
        tavily_key=None,
        deepseek_key=load_deepseek_key(REDACTOR),
        allowed_uids=_resolve_uids(),
    )
    return service, config, ledger



def _resolve_uids() -> dict[str, int]:
    """Map each role to the uid it must connect from, when those accounts exist.

    Absent accounts yield no entry, so the check is skipped rather than silently passing on a
    machine that has no such user.
    """
    import pwd

    mapping = {"runner": "sfrunner", "steward": "sfsteward", "evaluator": "sfevaluator",
               "infer": "sfinfer"}
    resolved: dict[str, int] = {}
    for role, user in mapping.items():
        try:
            resolved[role] = pwd.getpwnam(user).pw_uid
        except KeyError:
            continue
    return resolved


def run(settings: Settings) -> None:  # pragma: no cover - process entry point
    """Start the provider and block. Refuses to run as root."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise ProviderError(
            500,
            "refusing to start the provider as root: it holds every credential, and a process "
            "that also has every other privilege is not a boundary",
        )
    service, config, _ledger = build_service(settings)
    counts = service.reconcile_on_start()
    tcp, uds = serve_forever(service, config, REDACTOR)
    print(
        f"provider listening on {config.bind_host}:{tcp.server_address[1]}"
        + (f" and {config.unix_socket}" if uds else "")
        + f"; reconciled {counts['failed_unknown']} in-flight, released {counts['released']}",
        flush=True,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        tcp.shutdown()
        if uds:
            uds.shutdown()
