"""Amazon Bedrock Converse API tool-use loop with graceful degradation.

If Bedrock is unavailable (no creds, no access, disabled, or any error), this
module reports unavailability cleanly so the caller can fall back to a
deterministic answer. It NEVER pretends an unavailable model produced a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import config as C


@dataclass
class BedrockStatus:
    available: bool
    model_id: str | None = None
    reason: str = ""


@dataclass
class AgentTurn:
    text: str
    tool_trace: list[dict] = field(default_factory=list)
    model_id: str | None = None
    used_bedrock: bool = False


class BedrockClient:
    """Thin wrapper over boto3 bedrock-runtime Converse with tool use."""

    def __init__(self, region: str | None = None, model_id: str | None = None):
        self.region = region or C.AWS_REGION
        self.model_id = model_id or (C.BEDROCK_MODEL_ID or None)
        self._client = None
        self._status: BedrockStatus | None = None

    # --- availability ---
    def status(self) -> BedrockStatus:
        if self._status is not None:
            return self._status
        import os

        disabled = (
            C.BEDROCK_DISABLED
            or os.environ.get("POCKETSHIFT_DISABLE_BEDROCK", "0").strip() == "1"
        )
        if disabled:
            self._status = BedrockStatus(False, reason="disabled via env")
            return self._status
        try:
            import boto3  # noqa
            from botocore.config import Config

            cfg = Config(connect_timeout=4, read_timeout=40, retries={"max_attempts": 1})
            self._client = boto3.client(
                "bedrock-runtime", region_name=self.region, config=cfg
            )
            model = self._resolve_model()
            if model is None:
                self._status = BedrockStatus(
                    False, reason="no accessible Claude model found"
                )
            else:
                self.model_id = model
                self._status = BedrockStatus(True, model_id=model)
        except Exception as exc:  # noqa: BLE001 — degrade on any failure
            self._status = BedrockStatus(False, reason=f"{type(exc).__name__}: {exc}")
        return self._status

    def _resolve_model(self) -> str | None:
        """Pick a model: configured one if set, else first candidate that answers."""
        candidates = [self.model_id] if self.model_id else list(C.BEDROCK_MODEL_CANDIDATES)
        for mid in candidates:
            if not mid:
                continue
            try:
                self._client.converse(
                    modelId=mid,
                    messages=[{"role": "user", "content": [{"text": "ping"}]}],
                    inferenceConfig={"maxTokens": 5},
                )
                return mid
            except Exception:  # noqa: BLE001 — try next candidate
                continue
        return None

    # --- tool-use conversation ---
    def converse_with_tools(
        self,
        *,
        system: str,
        user_message: str,
        tool_specs: list[dict],
        executor: Callable[[str, dict], dict],
        max_iters: int = 6,
    ) -> AgentTurn:
        """Run the real tool loop. Returns final text + executed tool trace.

        tool_specs: Bedrock toolConfig 'tools' entries.
        executor(name, args) -> dict result (actually runs analytical functions).
        """
        st = self.status()
        if not st.available:
            raise RuntimeError(f"Bedrock unavailable: {st.reason}")

        messages: list[dict] = [{"role": "user", "content": [{"text": user_message}]}]
        trace: list[dict] = []
        tool_config = {"tools": tool_specs}

        for _ in range(max_iters):
            resp = self._client.converse(
                modelId=self.model_id,
                system=[{"text": system}],
                messages=messages,
                toolConfig=tool_config,
                inferenceConfig={"maxTokens": 1200, "temperature": 0.2},
            )
            out = resp["output"]["message"]
            messages.append(out)
            stop = resp.get("stopReason")

            if stop == "tool_use":
                tool_results = []
                for block in out["content"]:
                    if "toolUse" not in block:
                        continue
                    tu = block["toolUse"]
                    name = tu["name"]
                    args = tu.get("input", {}) or {}
                    try:
                        result = executor(name, args)
                        executed = True
                    except Exception as exc:  # noqa: BLE001
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                        executed = False
                    trace.append(
                        {"tool": name, "args": args, "executed": executed,
                         "result_keys": list(result.keys()) if isinstance(result, dict) else []}
                    )
                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tu["toolUseId"],
                                "content": [{"json": result}],
                            }
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                continue

            # end_turn / stop_sequence / max_tokens: collect text and finish.
            text = "".join(b.get("text", "") for b in out["content"])
            return AgentTurn(
                text=text, tool_trace=trace, model_id=self.model_id, used_bedrock=True
            )

        return AgentTurn(
            text="(Stopped after maximum tool iterations.)",
            tool_trace=trace,
            model_id=self.model_id,
            used_bedrock=True,
        )
