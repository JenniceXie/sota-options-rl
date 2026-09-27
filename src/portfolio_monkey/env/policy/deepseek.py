"""DeepSeek over OpenRouter, as an append-only conversation.

The client is dependency-free ``urllib`` on purpose: this runs on PSC compute
nodes where adding a package to the environment is a scheduler round-trip, and
the request is one POST.

Four decisions specific to using an LLM as a *policy* rather than as a
summarizer:

**Append-only history (``R7``).**  The system block, the action grammar and the
episode header are sent once and never rewritten; each step appends one user
turn and one assistant turn.  DeepSeek's cache keys on the longest matching
prefix, so an edit anywhere in the history invalidates every token after it.
A run that rebuilt its prompt each step would pay full price on all ~21 steps
of an episode instead of on the first.

**Reasoning is captured, and replayed only within a bounded window.**
``message.reasoning`` is written to the ledger — that is the point of asking for
high effort.  Whether it also goes *back* into the conversation is
``reasoning_replay_turns``, and it is ``0`` by default, because replaying the
whole trace would spend the 32k budget on the model's own deliberation and would
make the prefix depend on a field the provider does not guarantee to be stable.
Measured on a 20-decision episode: reasoning runs ~1,522 tokens a step, so
replaying all of it adds 28,927 tokens to a 15,032-token prompt — 134% of the
32,768 budget, i.e. not an option.  A window of the last 3 turns adds 4,567.

The window exists because ``moonshotai/kimi-k3`` stops deliberating as the
conversation deepens: measured 2026-09-23 at depth 19 on Bedrock and on three
OpenRouter serves (DeepInfra bf16, Moonshot AI, Fireworks), it returned the
identical action with **zero** reasoning characters on all four, while the same
observation at depth 0-2 drew 3,368-9,477.  ``deepseek`` models do not do this —
491 of 492 steps carry a trace under the same history — which is why ``0``
remains the default and this is an arm-level opt-in rather than a global change.

Note the tension with the append-only prefix described below: a *sliding* window
mutates the tail of the history, so the cache matches only up to the oldest turn
whose attachment changed.  The re-processed suffix is bounded at roughly
``window x (observation + reasoning)`` tokens per step, ~6.4k at a window of 3.

**Temperature 0 with a fixed seed.**  ``docs/evaluation_protocol.md`` section 4
requires a rerun to reproduce; sampling would make the same market state
produce different orders and the arm's result would not be a property of the
policy.

**An empty completion is an abstain, not a crash.**  Providers return empty
content often enough that failing the run on one would mean losing an episode
to an outage.  It is retried, and if it stays empty the step is recorded as an
error and the book holds — which is a real, scoreable action, and the error rate
is in the ledger where it can be looked at.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..statespace import Observation
from ..tokens import HeuristicTokenCounter, TokenCounter, enforce_context_budget
from . import PolicyResponse

__all__ = ["DeepSeekPolicy", "OpenRouterError", "DEFAULT_MODEL", "DEFAULT_URL"]

DEFAULT_MODEL = "deepseek/deepseek-r1"
DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"
REASONING_EFFORTS = ("minimal", "low", "medium", "high")

#: Retried.  ``429`` and ``5xx`` are the provider's problem; ``408`` is a
#: timeout that may not recur.  ``400``/``401``/``404`` are ours and retrying
#: them just burns the rate limit.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class OpenRouterError(RuntimeError):
    """The request failed after every retry.

    ``reasoning`` carries whatever the model *did* think before the call went
    wrong.  A length-truncated completion is the case that matters: it has no
    content but often many thousands of characters of reasoning, and that trace
    is the artifact the run exists to collect.  Raising without it would drop
    the one thing the step produced.
    """

    def __init__(self, message: str, *, reasoning: str = "") -> None:
        super().__init__(message)
        self.reasoning = reasoning


@dataclass(frozen=True, slots=True)
class _Turn:
    role: str
    content: str
    #: The trace the model emitted while producing ``content``, kept on the turn
    #: so ``messages`` can decide per step whether to replay it.  Empty for user
    #: turns, and for assistant turns the model answered without deliberating.
    reasoning: str = ""


class DeepSeekPolicy:
    """An LLM policy holding one append-only conversation per episode."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        url: str | None = None,
        name: str = "deepseek",
        timeout_seconds: float = 300.0,
        max_retries: int = 5,
        empty_response_attempts: int = 6,
        empty_response_backoff_seconds: float = 2.0,
        reasoning_effort: str = "high",
        # Measured against ``deepseek/deepseek-r1`` at ``effort=high`` on a real
        # ten-name observation: the model spent 3,853 / 6,757 / 3,550 reasoning
        # tokens on three draws.  ``max_tokens`` is a *completion* budget that
        # reasoning is charged against, so the old 4,000 sat inside that spread
        # and the first smoke step truncated with zero content.  16,000 clears
        # the observed maximum by better than 2x; the order lines themselves are
        # ~150 characters, so nearly all of it is headroom for deliberation.
        max_completion_tokens: int = 16_000,
        max_context_tokens: int = 32_768,
        # ``None`` omits the field from the request body entirely, for endpoints
        # that do not accept it.  OpenRouter drops an unsupported parameter
        # silently rather than erroring, so sending ``temperature`` to a model
        # that ignores it produces a run whose recorded sampling setting was
        # never applied -- the same false disclosure the bedrock arm refuses,
        # arriving quietly instead of as a ``ValidationException``.
        temperature: float | None = 0.0,
        seed: int = 0,
        providers: Sequence[str] = (),
        allow_provider_fallbacks: bool = True,
        # How many of the most recent assistant turns carry their reasoning back
        # into the prompt.  ``0`` is the behaviour every arm before 2026-09-23
        # ran under, and is the default so that turning this on is a property of
        # the arm rather than a silent change to results already on disk.
        reasoning_replay_turns: int = 0,
        # How ``max_context_tokens`` is counted.  Defaults to the ``len//4``
        # estimate every run before 2026-09-24 used, so passing nothing changes
        # nothing; pass a ``FileTokenizerCounter`` to have the budget checked
        # against the tokenizer that will actually have to hold the trajectory.
        # Measured on this state space, the two disagree by 1.85x.
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        # Deliberately *not* read from ``OPENROUTER_MODEL``.  That variable is
        # set in ``configs/psc.env`` for the context-summarization jobs, and a
        # sourced env file silently replaced the arm under evaluation with an
        # unrelated model — the run completed, reconciled, and recorded 100%
        # provider errors against a name nobody chose.  The model is part of the
        # arm's identity, like ``arm`` and ``track``, so it comes from the
        # caller or from the default and from nowhere else.
        self.model = model or DEFAULT_MODEL
        self.url = url or os.getenv("OPENROUTER_CHAT_COMPLETIONS_URL", DEFAULT_URL)
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.empty_response_attempts = max(1, empty_response_attempts)
        self.empty_response_backoff_seconds = max(0.0, empty_response_backoff_seconds)
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.max_context_tokens = max_context_tokens
        self.token_counter: TokenCounter = token_counter or HeuristicTokenCounter()
        self.temperature = temperature
        self.seed = seed
        # OpenRouter's default routing picks a host per *request*, and the hosts
        # serving one model id are not one machine: they differ in quantization
        # and in kernel, so they differ in output.  Measured on a single recorded
        # observation at ``temperature=0``, six hosts returned four different
        # actions -- ``O PLTR ic n``, ``C p13\nO PLTR ic n``,
        # ``O PLTR ic n\nO TSLA ic n`` and ``H`` -- and their decode rates spread
        # 48 to 255 tok/s.  Unpinned, the arm is therefore a mixture over hosts
        # in unrecorded proportions, and its wall time is set by whichever host
        # answered.  ``providers`` is part of the arm's identity for the same
        # reason ``model`` is.
        self.providers = tuple(providers)
        self.allow_provider_fallbacks = allow_provider_fallbacks
        if reasoning_replay_turns < 0:
            raise ValueError("reasoning_replay_turns must be >= 0")
        self.reasoning_replay_turns = reasoning_replay_turns

        self._turns: list[_Turn] = []
        self._system = ""
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- Policy ----------------------------------------------------------

    def reset(
        self,
        *,
        system: str,
        grammar: str,
        episode_header: str,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Start the episode's conversation.

        The grammar goes in the system message with the schema rather than in
        the first user turn, because it never changes and everything in the
        system message is cached across *episodes*, not just across steps.

        A non-empty ``tools`` is **refused**, not dropped.  This class does not
        speak the tool channel yet, and under that channel the state space
        takes the ``Q`` lines out of the grammar -- so accepting and ignoring
        the schema would run a full month against a prompt with no way to ask
        for a price, and the run would look entirely healthy: valid orders,
        ``end_turn`` throughout, and a quote count of zero that reads as a
        policy that chose not to quote.
        """
        if tools:
            names = ", ".join(str(t.get("name") or "?") for t in tools)
            raise ValueError(
                f"--policy deepseek cannot carry tools ({names}): the OpenRouter "
                "arm does not speak the tool channel. Run it with "
                "--quote-channel text, which prints the Q verb in the grammar, "
                "or with --quote-channel off"
            )
        self._system = f"{system}\n\n{grammar}".strip()
        self._turns = [_Turn("user", episode_header)] if episode_header else []

    def act(self, observation: Observation) -> PolicyResponse:
        self._turns.append(_Turn("user", observation.text))
        # Before the call, not after: a request whose prompt already exceeds the
        # budget should not be paid for, and the failure belongs to the state
        # rather than to anything the model did with it.
        self._check_budget("observation")
        started = time.monotonic()
        try:
            response = self._complete()
        except OpenRouterError as exc:
            # Hold rather than propagate.  Losing a step to a provider outage
            # is a data-quality fact for the ledger; losing the episode is a
            # loss of every step before it.
            self._turns.append(_Turn("assistant", "H"))
            return PolicyResponse(
                text="H",
                reasoning=getattr(exc, "reasoning", ""),
                model=self.model,
                latency_seconds=time.monotonic() - started,
                error=str(exc),
            )

        content, reasoning, usage, finish_reason, provider = response
        self._turns.append(_Turn("assistant", content or "H", reasoning))
        # The completion is part of the trajectory a student has to hold, so the
        # budget is checked again with it in.  Raising *after* the turn is
        # appended is deliberate: the conversation object then reflects the
        # state that overflowed, which is what makes the number in the message
        # reproducible from the recording.
        self._check_budget("completion")
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        return PolicyResponse(
            text=content or "H",
            reasoning=reasoning,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_seconds=time.monotonic() - started,
            model=self.model,
            finish_reason=finish_reason,
            extra={
                "cached_tokens": _cached_tokens(usage),
                "context_estimate": self.context_estimate(),
                # Recorded per step, not per run: ``allow_fallbacks`` means a
                # pinned arm can still be served by a second host, and the only
                # place that is visible is the step it happened on.
                "provider": provider,
            },
        )

    # -- context ---------------------------------------------------------

    def _check_budget(self, where: str) -> None:
        enforce_context_budget(
            used=self.context_estimate(),
            budget=self.max_context_tokens,
            counter=self.token_counter,
            where=where,
        )

    def _replayed_turns(self) -> frozenset[int]:
        """Indices of the assistant turns whose reasoning goes back in the prompt.

        The most recent ``reasoning_replay_turns`` that actually have a trace,
        so a step the model answered tersely does not consume a slot and leave
        the window emptier than it was asked to be.
        """
        if not self.reasoning_replay_turns:
            return frozenset()
        carried = [
            index
            for index, turn in enumerate(self._turns)
            if turn.role == "assistant" and turn.reasoning
        ]
        return frozenset(carried[-self.reasoning_replay_turns :])

    def messages(self) -> list[dict[str, Any]]:
        replayed = self._replayed_turns()
        out: list[dict[str, Any]] = [{"role": "system", "content": self._system}]
        for index, turn in enumerate(self._turns):
            message: dict[str, Any] = {"role": turn.role, "content": turn.content}
            if index in replayed:
                # ``reasoning_details`` is OpenRouter's documented shape for
                # carrying a trace back across turns.  Measured 2026-09-23 on
                # ``moonshotai/kimi-k3`` via ``deepinfra/bf16``: sending it as
                # ``reasoning`` (a bare string), as ``reasoning_details``, or as
                # both produced an identical 7,368 prompt tokens against 3,457
                # with it stripped -- so the field is genuinely transmitted and
                # the three shapes cost the same.  The structured one is used
                # because it is the one the API documents as durable.
                message["reasoning_details"] = [
                    {"type": "reasoning.text", "text": turn.reasoning}
                ]
            out.append(message)
        return out

    def context_estimate(self) -> int:
        replayed = self._replayed_turns()
        count = self.token_counter.count
        return (
            count(self._system)
            + sum(count(t.content) for t in self._turns)
            # Counted, not ignored: replayed reasoning is ~1,522 tokens a step
            # against a ~625 token observation, so leaving it out would make
            # ``context_pressure`` under-report by more than the thing it
            # measures.
            + sum(count(self._turns[i].reasoning) for i in replayed)
        )

    def context_pressure(self) -> float:
        """Fraction of the budget used.

        Reported rather than acted on.  Trimming the history to fit would break
        the append-only prefix, so the correct response to pressure is a shorter
        episode or a smaller state space — both of which are configuration
        changes the run should make deliberately, not something the client
        should paper over mid-episode.
        """
        return self.context_estimate() / max(1, self.max_context_tokens)

    # -- the call --------------------------------------------------------

    def _complete(self) -> tuple[str, str, Mapping[str, Any], str, str]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages(),
            "seed": self.seed,
            "max_tokens": self.max_completion_tokens,
            # ``exclude`` is false: the reasoning trace is the artifact the run
            # is being asked to keep.
            "reasoning": {"effort": self.reasoning_effort, "exclude": False},
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.providers:
            body["provider"] = {
                "order": list(self.providers),
                "allow_fallbacks": self.allow_provider_fallbacks,
            }

        issue = "empty content"
        salvaged = ""
        for attempt in range(self.empty_response_attempts):
            payload = self._post(body)
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                issue = "response has no choices"
            else:
                message = choices[0].get("message")
                finish_reason = str(choices[0].get("finish_reason") or "")
                if isinstance(message, Mapping):
                    content = _text(message.get("content"))
                    reasoning = _text(message.get("reasoning")) or _reasoning_details(
                        message.get("reasoning_details")
                    )
                    if content.strip():
                        usage = payload.get("usage")
                        return (
                            content.strip(),
                            reasoning,
                            usage if isinstance(usage, Mapping) else {},
                            finish_reason,
                            str(payload.get("provider") or ""),
                        )
                    # Keep the longest trace seen across attempts.  The step is
                    # going to be recorded as an error either way; whether it is
                    # recorded with the model's thinking attached is the whole
                    # difference between a scoreable failure and a blank row.
                    if len(reasoning) > len(salvaged):
                        salvaged = reasoning
                    # A completion that spent its whole budget reasoning is a
                    # length problem, not an outage, and retrying it unchanged
                    # would produce the same truncation.
                    if finish_reason == "length":
                        issue = "completion truncated before any content"
                        break
                    issue = "empty content"
            if attempt < self.empty_response_attempts - 1:
                time.sleep(self.empty_response_backoff_seconds * (attempt + 1))
        raise OpenRouterError(
            f"OpenRouter returned no usable completion: {issue}", reasoning=salvaged
        )

    def _post(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        data = json.dumps(body).encode()
        for attempt in range(self.max_retries + 1):
            request = Request(
                self.url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "portfolio-monkey options policy evaluation",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, Mapping):
                    raise OpenRouterError("OpenRouter returned a non-object response")
                if "error" in decoded and "choices" not in decoded:
                    raise OpenRouterError(f"OpenRouter error: {decoded['error']}")
                return decoded
            except json.JSONDecodeError as exc:
                # A truncated body on an HTTP 200 is a provider transport
                # failure; treat it like a retryable 5xx.
                if attempt >= self.max_retries:
                    raise OpenRouterError("OpenRouter returned malformed JSON") from exc
                delay = 0.0
            except HTTPError as exc:
                if exc.code not in RETRYABLE_STATUS or attempt >= self.max_retries:
                    raise OpenRouterError(f"OpenRouter HTTP {exc.code}") from exc
                delay = _retry_after(exc)
            except (URLError, TimeoutError, HTTPException, ConnectionError) as exc:
                # ``HTTPException`` is here because of two runs killed by
                # ``http.client.IncompleteRead``: OpenRouter answered 200 with a
                # chunked body and dropped the connection mid-stream.  That is
                # raised out of ``response.read()``, so it never reaches the
                # ``JSONDecodeError`` branch below — which was written for
                # exactly this failure and only catches the half of it where
                # enough bytes arrive to attempt a parse.  ``IncompleteRead``
                # subclasses ``HTTPException``, not ``URLError``, so it matched
                # no clause at all and propagated out through ``act`` into the
                # runner, ending one run at step 6 and another after an hour.
                # ``ConnectionError`` covers the sibling case where the peer
                # resets before any body arrives.
                if attempt >= self.max_retries:
                    raise OpenRouterError(
                        f"OpenRouter network request failed: {type(exc).__name__}"
                    ) from exc
                delay = 0.0
            time.sleep(max(delay, min(60.0, 2**attempt + random.random())))
        raise AssertionError("unreachable")


def _retry_after(exc: HTTPError) -> float:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _text(value: Any) -> str:
    """OpenRouter returns content as a string or as a list of content parts."""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "".join(
            part.get("text", "") for part in value if isinstance(part, Mapping)
        )
    return ""


def _reasoning_details(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ""
    return "\n".join(
        str(part.get("text") or part.get("summary") or "")
        for part in value
        if isinstance(part, Mapping)
    ).strip()


def _cached_tokens(usage: Mapping[str, Any]) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        return int(details.get("cached_tokens") or 0)
    return 0
