"""How everything else reaches the outside world: through the provider, never around it.

Steward, runner and evaluator processes get their outbound access from here. The methods
return transports shaped exactly like the ones
:class:`~shapeflow_p1.acquire.tavily_client.TavilyCaptureClient` and
:class:`~shapeflow_p1.evaluation.judge_client.DeepSeekJudge` already accept, so those clients
are reused unchanged rather than reimplemented against a second code path -- two
implementations of "call Tavily" would eventually disagree about what was frozen.

Every body leaves here with :data:`~shapeflow_p1.runtime.provider_server.PROVIDER_KEY_PLACEHOLDER`
where a credential would be. That is not a convenience: it means a real key was never in this
process's memory, so it cannot appear in its argv, its logs, its core dump or its exception text.

The inference path is a URL, not a transport. Vendor ODR builds its own OpenAI client, so the
only tagging channel that survives is the base URL: :meth:`ProviderClient.cell_base_url` returns
a per-cell path, and every model call made while that cell runs is attributed to it without any
mutable global that concurrent researchers could race on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx

from ..runtime.provider_server import PROVIDER_KEY_PLACEHOLDER

__all__ = [
    "ProviderClient",
    "ProviderCallError",
    "PROVIDER_KEY_PLACEHOLDER",
    "load_role_token",
]


class ProviderCallError(RuntimeError):
    """The provider refused or could not serve a call. Carries its status for classification."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"provider returned {status}: {message}")
        self.status = status


def load_role_token(token_dir: str, role: str) -> str:
    """Read this identity's role token. Absent means this role has no outbound access."""
    from pathlib import Path

    path = Path(token_dir) / f"{role}.token"
    if not path.exists():
        raise ProviderCallError(
            403,
            f"no {role} token at {path}; this identity has no route through the provider",
        )
    return path.read_text(encoding="utf-8").strip()


@dataclass
class ProviderClient:
    base_url: str
    token: str
    timeout_seconds: float = 900.0

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    async def _post(self, path: str, body: Mapping[str, Any]) -> tuple[int, dict]:
        url = self.base_url.rstrip("/") + path
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, headers=self._headers(), json=dict(body))
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"error": "provider returned a non-JSON body"}
        return response.status_code, payload if isinstance(payload, dict) else {"data": payload}

    # --- transports for the existing clients -------------------------------------------

    def tavily_transport(self, *, task_id: str = "", call_key: str = ""):
        """A transport for :class:`TavilyCaptureClient`: ``(url, body) -> (status, payload)``.

        The endpoint the caller passes is ignored on purpose -- the provider owns which host is
        contacted, so a client cannot redirect an acquisition query somewhere else and have it
        frozen as if it came from Tavily.
        """

        async def transport(_endpoint: str, body: dict) -> tuple[int, dict]:
            payload = dict(body)
            payload["api_key"] = PROVIDER_KEY_PLACEHOLDER
            payload["_task_id"] = task_id
            payload["_call_key"] = call_key or payload.get("query", "")
            return await self._post("/v1/tavily/search", payload)

        return transport

    def deepseek_transport(self, *, op_class: str, work_key: str = "", call_key: str = ""):
        """A transport for :class:`DeepSeekJudge`: ``(body) -> (status, payload)``."""

        async def transport(body: dict) -> tuple[int, dict]:
            payload = dict(body)
            payload["api_key"] = PROVIDER_KEY_PLACEHOLDER
            payload["_op_class"] = op_class
            payload["_work_key"] = work_key
            if call_key:
                payload["_call_key"] = call_key
            return await self._post("/v1/deepseek/chat", payload)

        return transport

    # --- inference ----------------------------------------------------------------------

    async def register_cell(
        self,
        *,
        cell_token: str,
        run_id: str,
        task_id: str,
        arm_id: str,
        variant_id: str,
        replicate_id: str,
        work_key: str,
        layer: str = "causal",
    ) -> None:
        status, payload = await self._post("/v1/cells", {
            "cell_token": cell_token, "run_id": run_id, "task_id": task_id, "arm_id": arm_id,
            "variant_id": variant_id, "replicate_id": replicate_id, "work_key": work_key,
            "layer": layer,
        })
        if status != 200:
            raise ProviderCallError(status, str(payload.get("error", payload)))

    def cell_base_url(self, cell_token: str) -> str:
        """The ``OPENAI_BASE_URL`` for one cell.

        Vendor ODR cannot be made to set a header, so the cell identity rides in the path. An
        OpenAI-compatible client appends ``/chat/completions`` to this, which is exactly the
        route the provider expects.
        """
        return f"{self.base_url.rstrip('/')}/v1/cell/{cell_token}"

    async def chat_completions(
        self, body: Mapping[str, Any], *, cell_token: Optional[str] = None
    ) -> dict:
        """Call the local engine directly (used by our own selectors, which can tag themselves)."""
        path = f"/v1/cell/{cell_token}/chat/completions" if cell_token else "/v1/chat/completions"
        status, payload = await self._post(path, body)
        if status != 200:
            raise ProviderCallError(status, str(payload.get("error", payload)))
        return payload

    async def healthy(self) -> bool:
        try:
            status, _ = await self._post("/healthz", {})
        except httpx.HTTPError:
            return False
        return status == 200

    async def ready(self) -> bool:
        try:
            status, _ = await self._post("/readyz", {})
        except httpx.HTTPError:
            return False
        return status == 200
