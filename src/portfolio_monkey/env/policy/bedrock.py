"""Bedrock models, as the same append-only conversation as the DeepSeek arm.

This is a sibling of :mod:`portfolio_monkey.env.policy.deepseek`, not a
generalization of it.  The two are kept apart because the differences are not
configuration -- they are capability limits of the endpoint, and hiding them
behind a shared client would let an arm silently run under a setting it did not
get.

**No temperature, no top-p, no seed.**  Bedrock rejects all three for this model
with a ``ValidationException`` that names the field.  ``DeepSeekPolicy`` takes
``temperature`` and ``seed`` and ``docs/evaluation_protocol.md`` section 4 leans
on them for reproducibility; here they do not exist, so this class refuses them
at construction rather than accepting and dropping them.  A run whose manifest
records ``temperature=0`` while the endpoint sampled freely is a worse outcome
than a run that will not start.  Four identical calls returned four distinct
outputs, so this arm has real, uncontrolled sampling diversity: a lost response
cannot be regenerated, which is why every response is written to the ledger on
receipt rather than at the end of the episode.

**Whether reasoning is readable depends on the model, so it is read off the
response rather than assumed.**  ``us.openai.gpt-6-astra`` returns a
``reasoningContent.redactedContent`` blob of opaque bytes (``rsn_...``) that
exists for stateless round-tripping and is not decryptable; it is recorded as a
byte count, never as text, because writing ``rsn_...`` into the ``reasoning``
column would make the ledger look like it holds a trace when every downstream
reader -- SFT label extraction most of all -- would find ciphertext.
``us.moonshotai.kimi-k3`` instead returns ``reasoningContent.reasoningText``,
plain and readable, which is what makes it usable as an SFT teacher where
gpt-6-astra is an evaluation benchmark only.  ``_complete`` reads both shapes
and lets whichever is present win, so a third model needs no branch here.

**Caching is automatic, and asking for it fails.**  Sending an explicit
``cachePoint`` block raises ``AccessDeniedException``.  Not sending one costs
nothing: measured on a 17.7k-token prefix, turn 2 onward read the entire prefix
from cache and wrote only the ~28 tokens of the new turn.  The append-only
history that ``R7`` requires for DeepSeek's prefix cache is exactly what this
endpoint rewards too, and it still hits when the assistant turns are replayed as
text only, with the reasoning block dropped.

**Reasoning replays as a ``reasoningContent`` block, and needs no signature
here.**  ``reasoning_replay_turns`` mirrors the OpenRouter arm and defaults to
``0`` for the same reason: turning it on is a property of the arm, not a silent
change to results on disk.  It exists because ``moonshotai.kimi-k3`` stops
deliberating as the conversation deepens -- measured 2026-09-23 over the 12
Bedrock K3 arms, 142 of 756 steps carry any trace at all, and the loss is
positional rather than random (36/36 at decision 1, 23/36 at 3, ~0 from 7 on)
while ``stopReason`` stays ``end_turn`` and the order lines stay valid.  The
model keeps acting and stops thinking.

Probed against the live endpoint before being written
(``scripts/analysis/probe_bedrock_reasoning_replay.py``, 2026-09-23): kimi-k3
returns ``reasoningText`` with **keys ``['text']`` only and no ``signature``**,
an input assistant message carrying a ``reasoningContent`` block is *accepted*,
and it is genuinely transmitted -- a 747-character trace moved the prompt from
255 to 444 tokens, ``+189`` against the ~186 the trace estimates at.  Sending a
``signature`` key and omitting it produced the identical 444, so the field is
inert for this model; it is still captured and replayed when a response carries
one, because Anthropic models on Bedrock reject a thinking block whose signature
is missing, and this class decides by reading the response rather than by
knowing the id.  Acceptance alone would prove nothing --
``additionalModelRequestFields`` is unvalidated on this endpoint, so a field can
be taken and ignored -- which is why the token delta is the evidence.

The replayed trace also changed the *next* answer: on the same second turn the
model returned 120 characters of reasoning with the trace stripped and 1,696
with it replayed.  That is the effect the window is for, and it is also why a
*sliding* window mutates the tail of the history and costs a partial cache miss,
bounded at roughly ``window x (observation + trace)`` per step.

**``inputTokens`` is not the prompt size.**  It counts only what was *not*
served from cache, so a 17,700-token prompt reports as 2.  ``prompt_tokens``
here is the sum of the three input counters, so that a token comparison against
the OpenRouter arms is a comparison of prompts rather than of cache luck.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..actions import render_quote_call
from ..statespace import Observation
from ..tokens import HeuristicTokenCounter, TokenCounter, enforce_context_budget
from . import PolicyResponse

__all__ = [
    "BedrockPolicy",
    "BedrockError",
    "DEFAULT_MODEL",
    "DEFAULT_REGION",
    "KIMI_K3_MODEL",
]

#: The inference profile, not the bare model id.  Neither ``openai.gpt-6-astra``
#: nor ``moonshotai.kimi-k3`` is servable without the ``us.`` prefix: both list
#: ``inferenceTypesSupported: ['INFERENCE_PROFILE']`` and are refused as
#: on-demand-unsupported.  The profile spans us-east-1, us-east-2 and us-west-2,
#: so this arm is region-unpinned by construction and says so in its manifest
#: rather than inheriting the pinned-host convention of the OpenRouter arms.
DEFAULT_MODEL = "us.openai.gpt-6-astra"

#: The SFT teacher.  Same endpoint and same request shape as the default, and
#: it differs in exactly one way that matters: it returns a readable trace.
#: ``reasoning.effort`` is honoured -- measured 2026-09-21 over 5 calls each on
#: one prompt, ``low`` produced 897-3,908 characters of reasoning and ``high``
#: produced 17,017-19,178, two ranges that do not overlap.  Worth pinning down
#: because ``additionalModelRequestFields`` is *not* validated: a bogus key is
#: accepted silently, so acceptance alone never proves a field was read.
KIMI_K3_MODEL = "us.moonshotai.kimi-k3"

DEFAULT_REGION = "us-west-2"
REASONING_EFFORTS = ("low", "medium", "high")

#: Retried.  Everything else -- ``ValidationException``, ``AccessDeniedException``,
#: ``ResourceNotFoundException`` -- is a request we built wrong, and retrying it
#: only spends the throttle budget on the same rejection.
RETRYABLE_ERRORS = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
        "ModelNotReadyException",
    }
)


class BedrockError(RuntimeError):
    """The request failed after every retry.

    Carries no salvaged reasoning, unlike its OpenRouter counterpart: a
    truncated completion here returns ciphertext, so there is nothing to keep.
    """


@dataclass(frozen=True, slots=True)
class _Turn:
    role: str
    text: str
    #: The trace the model emitted while producing ``text``, kept on the turn so
    #: ``messages`` can decide per step whether to replay it.  Empty for user
    #: turns, for a model that encrypts its trace, and for an assistant turn the
    #: model answered without deliberating -- which on kimi-k3 is most of them.
    reasoning: str = ""
    #: Absent on kimi-k3 and on gpt-6-astra, present on Anthropic models, which
    #: reject a replayed thinking block that does not carry it back.
    signature: str = ""


class BedrockPolicy:
    """An LLM policy holding one append-only conversation per episode."""

    def __init__(
        self,
        *,
        model: str | None = None,
        region: str | None = None,
        name: str = "gpt6",
        client: Any = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 5,
        empty_response_attempts: int = 6,
        empty_response_backoff_seconds: float = 2.0,
        reasoning_effort: str = "high",
        max_completion_tokens: int = 16_000,
        max_context_tokens: int = 32_768,
        # How many of the most recent assistant turns carry their trace back
        # into the prompt.  ``0`` is what every Bedrock arm before 2026-09-23
        # ran under.
        reasoning_replay_turns: int = 0,
        # How ``max_context_tokens`` is counted.  Defaults to the ``len//4``
        # estimate every run before 2026-09-24 used, so passing nothing changes
        # nothing; pass a ``FileTokenizerCounter`` to have the budget checked
        # against the tokenizer that will actually have to hold the trajectory.
        # Measured on this state space, the two disagree by 1.85x.
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.region = region or os.getenv("AWS_REGION") or DEFAULT_REGION
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.empty_response_attempts = max(1, empty_response_attempts)
        self.empty_response_backoff_seconds = max(0.0, empty_response_backoff_seconds)
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}"
            )
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.max_context_tokens = max_context_tokens
        self.token_counter: TokenCounter = token_counter or HeuristicTokenCounter()
        if reasoning_replay_turns < 0:
            raise ValueError("reasoning_replay_turns must be >= 0")
        self.reasoning_replay_turns = reasoning_replay_turns
        self._client = (
            client if client is not None else _build_client(self.region, timeout_seconds)
        )

        self._turns: list[_Turn] = []
        self._system = ""
        self._tools: tuple[Mapping[str, Any], ...] = ()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.tool_calls = 0

    # -- Policy ----------------------------------------------------------

    def reset(
        self,
        *,
        system: str,
        grammar: str,
        episode_header: str,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if len(tools) > 1:
            # One declared tool, one renderer.  ``_complete`` normalizes every
            # ``toolUse`` it sees through ``render_quote_call``, which is sound
            # only while the quote tool is the only thing on offer; a second
            # tool would come back as a ``Q`` line with the wrong fields in it
            # and be refused by the parser as a grammar error, which reads as
            # the model failing rather than as this class being out of date.
            names = ", ".join(str(t.get("name") or "?") for t in tools)
            raise ValueError(
                f"BedrockPolicy renders one tool and was given {len(tools)} "
                f"({names}); add a renderer per tool before declaring a second"
            )
        self._system = f"{system}\n\n{grammar}".strip()
        self._tools = tuple(tools)
        self._turns = [_Turn("user", episode_header)] if episode_header else []

    def act(self, observation: Observation) -> PolicyResponse:
        self._turns.append(_Turn("user", observation.text))
        # Before the call, not after: a request whose prompt already exceeds the
        # budget should not be paid for, and the failure belongs to the state
        # rather than to anything the model did with it.
        self._check_budget("observation")
        started = time.monotonic()
        try:
            (
                content,
                readable,
                signature,
                redacted_bytes,
                usage,
                stop_reason,
                tool_uses,
            ) = self._complete()
        except BedrockError as exc:
            # Hold rather than propagate, for the same reason the DeepSeek arm
            # does: losing a step to an outage is a data-quality fact for the
            # ledger, losing the episode is a loss of every step before it.
            self._turns.append(_Turn("assistant", "H"))
            return PolicyResponse(
                text="H",
                model=self.model,
                latency_seconds=time.monotonic() - started,
                error=str(exc),
                extra=self._provenance(),
            )

        # ``content`` here already carries the tool calls as ``Q`` lines, so the
        # turn goes into the history as text and the conversation from turn 2 on
        # is byte-identical to the text channel's.  Two consequences worth
        # naming.  The ledger's ``completion`` column is the normalized text,
        # which is what the SFT corpus needs -- the student is trained to emit
        # the DSL, not tool calls -- and the raw call is kept in ``extra`` so
        # the substitution is auditable rather than merely asserted.
        self._turns.append(_Turn("assistant", content or "H", readable, signature))
        # The completion is part of the trajectory a student has to hold, so the
        # budget is checked again with it in.  Raising *after* the turn is
        # appended is deliberate: the conversation then reflects the state that
        # overflowed, which is what makes the number in the message
        # reproducible from the recording.
        self._check_budget("completion")
        self.tool_calls += len(tool_uses)
        cache_read = _count(usage, "cacheReadInputTokens")
        prompt = (
            _count(usage, "inputTokens")
            + cache_read
            + _count(usage, "cacheWriteInputTokens")
        )
        completion = _count(usage, "outputTokens")
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_tokens += cache_read
        return PolicyResponse(
            text=content or "H",
            # Empty for a model that encrypts its trace, and the real text for
            # one that does not -- decided by the response, not by the id.  For
            # the encrypted case the byte count in ``extra`` still shows the
            # model reasoned rather than answered blind, but nothing that would
            # read as a trace is written here: an SFT label extractor must find
            # either usable text or nothing, never ciphertext.
            reasoning=readable,
            prompt_tokens=prompt,
            completion_tokens=completion,
            latency_seconds=time.monotonic() - started,
            model=self.model,
            finish_reason=stop_reason,
            extra={
                "cached_tokens": cache_read,
                "uncached_prompt_tokens": _count(usage, "inputTokens"),
                "context_estimate": self.context_estimate(),
                "reasoning_redacted_bytes": redacted_bytes,
                # Stated so a corpus built from this ledger can be filtered on
                # "the teacher actually left a trace" without reparsing text.
                "reasoning_readable_chars": len(readable),
                # The wire form of what ``text`` now states in DSL.  Recorded
                # so "the quote round ran on the tool channel" is checkable
                # from the ledger rather than inferred from the manifest's
                # ``quotes.channel``: a schema that was declared and never
                # called and a schema that was never declared produce the same
                # zero here otherwise, and those are different failures.
                "tool_calls": [
                    {
                        "name": str(use.get("name") or ""),
                        "input": dict(use.get("input") or {}),
                    }
                    for use in tool_uses
                ],
                **self._provenance(),
            },
        )

    def _provenance(self) -> dict[str, Any]:
        return {
            "provider": "bedrock",
            "region_requested": self.region,
            # The ``us.`` inference profile may serve from any of three regions
            # and the response does not say which, so the arm cannot claim a
            # pinned host the way the OpenRouter arms do.
            "region_pinned": False,
        }

    # -- context ---------------------------------------------------------

    def _replayed_turns(self) -> frozenset[int]:
        """Indices of the assistant turns whose trace goes back in the prompt.

        The most recent ``reasoning_replay_turns`` that actually *have* a trace,
        so a step the model answered tersely does not consume a slot and leave
        the window emptier than it was asked to be.  On kimi-k3 that is the
        difference between a window of 3 and a window of nothing: from decision
        7 on, the three preceding turns usually carry no trace at all.
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
        out: list[dict[str, Any]] = []
        for index, turn in enumerate(self._turns):
            content: list[dict[str, Any]] = []
            if index in replayed:
                inner: dict[str, Any] = {"text": turn.reasoning}
                if turn.signature:
                    inner["signature"] = turn.signature
                # Reasoning block first, then the answer: that is the order the
                # model produced them in and the order Converse round-trips.
                content.append({"reasoningContent": {"reasoningText": inner}})
            content.append({"text": turn.text})
            out.append({"role": turn.role, "content": content})
        return out

    def _check_budget(self, where: str) -> None:
        enforce_context_budget(
            used=self.context_estimate(),
            budget=self.max_context_tokens,
            counter=self.token_counter,
            where=where,
        )

    def context_estimate(self) -> int:
        replayed = self._replayed_turns()
        count = self.token_counter.count
        return (
            count(self._system)
            + sum(count(t.text) for t in self._turns)
            # Counted, not ignored: a kimi-k3 trace at ``effort=high`` runs
            # 17,017-19,178 characters, so leaving replayed traces out would
            # make ``context_pressure`` under-report by several times the thing
            # it measures.
            + sum(count(self._turns[i].reasoning) for i in replayed)
        )

    def context_pressure(self) -> float:
        return self.context_estimate() / max(1, self.max_context_tokens)

    # -- the call --------------------------------------------------------

    def _complete(self) -> tuple[str, str, str, int, Mapping[str, Any], str, tuple[Mapping[str, Any], ...]]:
        issue = "empty content"
        for attempt in range(self.empty_response_attempts):
            payload = self._converse()
            message = payload.get("output", {}).get("message", {})
            blocks = message.get("content") or []
            # Normalized into the completion text, in the order the model sent
            # them, rather than answered as ``toolResult`` blocks.  Measured on
            # the live endpoint 2026-09-24: a turn whose ``toolUse`` is replayed
            # as plain assistant text and answered with a plain user turn is
            # *accepted* -- no ``toolResult`` is required -- and the reply came
            # back ``end_turn``.  That is what makes this design possible, and
            # it is the only one that works, because ``OptionsEnv.quote``
            # returns a list that is not one row per call: real-name refusals
            # come first, then parse errors, then refused order lines, then the
            # answers.  Pairing that by index is wrong and pairing it by the
            # rendered line is ambiguous the moment the policy asks for the
            # same package twice.  Keeping the history in text also means the
            # channel costs nothing per quote: the schema is ~468 tokens once
            # at the head of the prefix and cached from turn 2.
            tool_uses = tuple(
                b["toolUse"]
                for b in blocks
                if isinstance(b, Mapping) and isinstance(b.get("toolUse"), Mapping)
            )
            content = "\n".join(
                part
                for part in (
                    "".join(
                        b.get("text", "") for b in blocks if isinstance(b, Mapping)
                    ).strip(),
                    *(
                        render_quote_call(use.get("input") or {})
                        for use in tool_uses
                    ),
                )
                if part
            ).strip()
            reasoning_blocks = [
                b["reasoningContent"]
                for b in blocks
                if isinstance(b, Mapping) and isinstance(b.get("reasoningContent"), Mapping)
            ]
            redacted = sum(
                len(rc.get("redactedContent") or b"") for rc in reasoning_blocks
            )
            # Which of the two a model returns is a property of the *response*,
            # not of an id this class was told.  ``openai.gpt-6-astra`` returns
            # ``redactedContent`` and nothing else; ``moonshotai.kimi-k3``
            # returns ``reasoningText`` and nothing else.  Reading both and
            # letting whichever is present win keeps the third model working
            # without a branch that has to be remembered.
            readable = "\n".join(
                text
                for rc in reasoning_blocks
                if isinstance(rc.get("reasoningText"), Mapping)
                and (text := str(rc["reasoningText"].get("text") or "").strip())
            )
            # Absent on both models served today, and required by Anthropic's,
            # which refuse a replayed thinking block without it.  Taking the
            # first one is right because ``readable`` joins the blocks into one
            # trace, so there is one block to send back.
            signature = next(
                (
                    sig
                    for rc in reasoning_blocks
                    if isinstance(rc.get("reasoningText"), Mapping)
                    and (sig := str(rc["reasoningText"].get("signature") or ""))
                ),
                "",
            )
            stop_reason = str(payload.get("stopReason") or "")
            if content:
                usage = payload.get("usage")
                return (
                    content,
                    readable,
                    signature,
                    redacted,
                    usage if isinstance(usage, Mapping) else {},
                    stop_reason,
                    tool_uses,
                )
            # A completion that spent its whole budget reasoning is a length
            # problem, not an outage; retrying it unchanged reproduces it.
            if stop_reason == "max_tokens":
                issue = "completion truncated before any content"
                break
            issue = "empty content"
            if attempt < self.empty_response_attempts - 1:
                time.sleep(self.empty_response_backoff_seconds * (attempt + 1))
        raise BedrockError(f"Bedrock returned no usable completion: {issue}")

    def _converse(self) -> Mapping[str, Any]:
        request = {
            "modelId": self.model,
            "system": [{"text": self._system}],
            "messages": self.messages(),
            # No ``temperature``, no ``topP``, no ``seed``: the endpoint rejects
            # each by name.  No ``cachePoint`` either -- asking for caching is
            # an ``AccessDeniedException``, while not asking for it still reads
            # the whole prefix from cache from the second turn on.
            "inferenceConfig": {"maxTokens": self.max_completion_tokens},
            "additionalModelRequestFields": {
                "reasoning": {"effort": self.reasoning_effort}
            },
        }
        if self._tools:
            # Sent on *every* request of the episode, not only the ones where a
            # quote would be useful.  Converse validates the history against
            # the tools declared on the current call, and it sits at the head
            # of the prefix where it is read from cache from turn 2 on --
            # measured 468 tokens for the quote schema, paid once.  Declaring
            # it conditionally would mutate that head and cost the whole prefix.
            request["toolConfig"] = {
                "tools": [{"toolSpec": dict(spec)} for spec in self._tools]
            }
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.converse(**request)
            except Exception as exc:  # noqa: BLE001 - classified just below
                code = _error_code(exc)
                if code not in RETRYABLE_ERRORS or attempt >= self.max_retries:
                    raise BedrockError(f"Bedrock {code}: {exc}") from exc
            time.sleep(min(60.0, 2**attempt + random.random()))
        raise AssertionError("unreachable")


def _build_client(region: str, timeout_seconds: float) -> Any:
    """Imported lazily so a hold or rule run needs neither boto3 nor credentials."""
    import boto3
    from botocore.config import Config

    if not os.getenv("AWS_BEARER_TOKEN_BEDROCK") and not os.getenv("AWS_ACCESS_KEY_ID"):
        # ``configs/psc.env`` uses bare ``NAME="value"``, so it has to be sourced
        # with ``set -a`` or the variable never reaches a Python child and the
        # failure arrives much later as an opaque credentials error.
        raise RuntimeError(
            "AWS_BEARER_TOKEN_BEDROCK is required; source configs/psc.env with "
            "'set -a && . configs/psc.env && set +a'"
        )
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            read_timeout=timeout_seconds,
            connect_timeout=30,
            # botocore's own retries would sit inside ``_converse``'s retry loop
            # and multiply it; the classification above is the one that decides.
            retries={"max_attempts": 1, "mode": "standard"},
        ),
    )


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping) and error.get("Code"):
            return str(error["Code"])
    return type(exc).__name__


def _count(usage: Mapping[str, Any], key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0
