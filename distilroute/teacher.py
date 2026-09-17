"""The teacher: a hosted LLM asked, zero-shot, to route a batch of queries to one of N intents.

Two things matter here and both are about *trust in the labels*:

1. The prompt lists the intent **names only** — no example queries. Examples would be gold
   labels leaking into what we later call "LLM-labelled" data.
2. Parsing is strict. The model's answer must be one of the known names after a light
   normalisation; anything else is recorded as a failure (label ``None``), never guessed.
   The failure rate is itself a result that goes in the write-up.

Providers are reached over plain HTTP so the request is visible and there is no SDK to
install. ``transport`` is injectable so tests run without a key or a network.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

# transport(payload) -> (assistant_text, usage_dict)
Transport = Callable[[dict], tuple[str, dict]]


class TeacherError(Exception):
    """A 4xx the provider will keep returning (bad model, bad key, bad request) — do not retry."""


class RateLimited(Exception):
    """HTTP 429 — carries the provider's suggested wait (seconds), if it sent one."""

    def __init__(self, retry_after: float | None):
        super().__init__(f"rate limited (retry after {retry_after}s)")
        self.retry_after = retry_after


# "openai"-style providers all take the same chat/completions payload and bearer key.
# Free-tier catalogues rotate: GET <base>/models lists what a key can use today.
PROVIDERS: dict[str, dict] = {
    "groq": {
        "style": "openai",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "openai/gpt-oss-120b",
        "key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "style": "openai",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "key_env": "OPENROUTER_API_KEY",
    },
    "nvidia": {
        "style": "openai",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "default_model": "openai/gpt-oss-120b",
        "key_env": "NVIDIA_API_KEY",
    },
    "gemini": {
        "style": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "default_model": "gemini-2.0-flash",
        "key_env": "GEMINI_API_KEY",
    },
}


@dataclass
class LabelResult:
    text: str
    label: str | None  # None == the teacher's answer was not a known intent
    raw: str  # what the model actually said for this item, for the audit trail
    ranked: list[str] = field(default_factory=list)  # top-k valid intents, best first


@dataclass
class Teacher:
    labels: list[str]
    provider: str = "groq"
    model: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    descriptions: dict[str, str] | None = None  # intent -> one line; still no example queries
    reasoning_effort: str = "low"  # only sent to reasoning models (gpt-oss)
    top_k: int = 1  # >1 asks for a ranked list per query (soft labels for distillation)
    transport: Transport | None = None
    calls: int = field(default=0, init=False)
    tokens_in: int = field(default=0, init=False)
    tokens_out: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; choose from {list(PROVIDERS)}")
        spec = PROVIDERS[self.provider]
        self.model = self.model or spec["default_model"]
        self.api_key = self.api_key or os.environ.get(spec["key_env"])
        if self.transport is None:
            if not self.api_key:
                raise ValueError(f"set {spec['key_env']} in .env (see .env.example)")
            self.transport = self._http_transport
        # Match on the normalised form, answer with the canonical one (Banking77 has one
        # mixed-case name, `Refund_not_showing_up`).
        self._canonical = {self._normalise(name): name for name in self.labels}

    # ------------------------------------------------------------------ prompt

    def build_messages(self, queries: list[str]) -> tuple[str, str]:
        if self.descriptions:
            intents = "\n".join(
                f"- {n}: {self.descriptions[n]}" if n in self.descriptions else f"- {n}"
                for n in self.labels
            )
        else:
            intents = "\n".join(f"- {name}" for name in self.labels)
        system = (
            "You route customer messages sent to a banking app's support inbox. "
            "Assign each message exactly one intent from this list, using the name verbatim:\n"
            f"{intents}\n\n"
            + (
                "Reply with a single JSON object mapping each message number (as a string) to its "
                'intent name, e.g. {"1": "card_arrival", "2": "lost_or_stolen_card"}. '
                if self.top_k == 1
                else "Reply with a single JSON object mapping each message number (as a string) "
                f"to a list of the {self.top_k} most likely intent names, most likely first, "
                'e.g. {"1": ["card_arrival", "card_delivery_estimate", "get_physical_card"]}. '
            )
            + "No other text."
        )
        user = "\n".join(f"{i + 1}. {q.strip()}" for i, q in enumerate(queries))
        return system, user

    # ----------------------------------------------------------------- parsing

    @staticmethod
    def _normalise(name: str) -> str:
        # Banking77 has `reverted_card_payment?` (sic) and `Refund_not_showing_up`; models
        # return `reverted_card_payment` and `refund_not_showing_up`. Compare loosely.
        return re.sub(r"[\s\-]+", "_", name.strip().strip("\"'`?.!").lower())

    def _ranked(self, v: object) -> list[str]:
        """Valid canonical intents from a model value (one name or a list), best first, no dupes."""
        items = v if isinstance(v, list) else [v]
        seen: list[str] = []
        for item in items:
            name = self._canonical.get(self._normalise(str(item)))
            if name and name not in seen:
                seen.append(name)
        return seen[: self.top_k]

    def parse(self, text: str, n: int) -> dict[int, tuple[str | None, str, list[str]]]:
        """Map 1-based item number -> (label or None, raw answer, ranked valid intents)."""
        out: dict[int, tuple[str | None, str, list[str]]] = {
            i: (None, "", []) for i in range(1, n + 1)
        }
        # Models sometimes wrap JSON in a code fence or add a sentence; take the outermost {...}.
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return out
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return out
        if not isinstance(obj, dict):
            return out
        for k, v in obj.items():
            try:
                i = int(str(k).strip())
            except ValueError:
                continue
            if 1 <= i <= n:
                ranked = self._ranked(v)
                raw = json.dumps(v, ensure_ascii=False) if isinstance(v, list) else str(v)
                out[i] = (ranked[0] if ranked else None, raw, ranked)
        return out

    # ---------------------------------------------------------------- labelling

    def label_batch(self, queries: list[str]) -> list[LabelResult]:
        """Label one batch. Items the batch answer missed or mangled are retried one at a time.

        A wholly empty answer (reasoning model ran out of budget, transient garbage) is
        re-asked once as a batch first — far cheaper than N single-item retries.
        """
        parsed = self._ask(queries)
        if len(queries) > 1 and all(v[0] is None for v in parsed.values()):
            parsed = self._ask(queries)
        results: list[LabelResult] = []
        for i, q in enumerate(queries, start=1):
            label, raw, ranked = parsed[i]
            if label is None and len(queries) > 1:
                label, raw, ranked = self._ask([q])[1]
            results.append(LabelResult(text=q, label=label, raw=raw, ranked=ranked))
        return results

    def _ask(self, queries: list[str]) -> dict[int, tuple[str | None, str, list[str]]]:
        system, user = self.build_messages(queries)
        payload = self._payload(system, user)
        assert self.transport is not None
        text, usage = self.transport(payload)
        self.calls += 1
        self.tokens_in += int(usage.get("prompt_tokens", 0))
        self.tokens_out += int(usage.get("completion_tokens", 0))
        return self.parse(text, len(queries))

    # ---------------------------------------------------------------- providers

    def _payload(self, system: str, user: str) -> dict:
        if PROVIDERS[self.provider]["style"] == "openai":
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if "gpt-oss" in (self.model or ""):
                # Reasoning model: a routing decision does not need a long think.
                payload["reasoning_effort"] = self.reasoning_effort
            return payload
        # gemini
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }

    def _http_transport(self, payload: dict) -> tuple[str, dict]:
        spec = PROVIDERS[self.provider]
        if PROVIDERS[self.provider]["style"] == "openai":
            r = requests.post(
                spec["url"],
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120,
            )
        else:
            r = requests.post(
                spec["url"].format(model=self.model),
                json=payload,
                headers={"x-goog-api-key": self.api_key or ""},
                timeout=120,
            )
        if r.status_code == 429:
            ra = r.headers.get("retry-after")
            raise RateLimited(float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
        if 400 <= r.status_code < 500:
            raise TeacherError(f"HTTP {r.status_code} from {self.provider}: {r.text[:400]}")
        r.raise_for_status()  # 5xx -> requests.HTTPError, which the labeller retries
        body = r.json()
        if PROVIDERS[self.provider]["style"] == "openai":
            # Reasoning models can spend the whole budget thinking and return content: null;
            # an empty answer parses to "unparsed" and the per-item retry takes over.
            if "choices" not in body:
                # OpenRouter can answer 200 with {"error": {...}} when the upstream fails
                # or throttles; both are worth a retry.
                err = body.get("error", body)
                if isinstance(err, dict) and err.get("code") == 429:
                    raise RateLimited(None)
                raise requests.RequestException(f"no choices in response: {str(err)[:200]}")
            text = body["choices"][0]["message"].get("content") or ""
            usage = body.get("usage", {})
        else:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            um = body.get("usageMetadata", {})
            usage = {
                "prompt_tokens": um.get("promptTokenCount", 0),
                "completion_tokens": um.get("candidatesTokenCount", 0),
            }
        return text, usage
