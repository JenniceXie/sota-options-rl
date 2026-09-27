"""One chat rendering for SFT, RL rollouts and evaluation.

WHY NOT THE STOCK QWEN3 TEMPLATE.  It emits ``<think>\\n\\n</think>\\n\\n`` on the
*last* assistant turn only.  Two consequences, both measured on this corpus:

* veRL 0.7.1's ``MultiTurnSFTDataset`` renders each message on its own, so every
  assistant turn is "last" and gets the block; its own sanity check then compares
  that against a whole-conversation render and raises ``AssertionError``.
* at generation time ``enable_thinking=False`` puts the block on *every* turn,
  while SFT (whole-conversation render) would have trained 60 of 61 turns
  without it -- a train/inference mismatch on the one token position that
  decides whether the model starts reasoning.

So: no think block anywhere.  The template below is Qwen3's ChatML framing with
the reasoning and tool branches removed.  It is concatenative -- rendering
messages one at a time and joining equals rendering them together -- which is
the property both veRL's SFT dataset and the rollout loop depend on.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

#: ``content`` may be a string or, when a trainer routes text through a VLM
#: processor (veRL 0.9.1 + Qwen3.8), a list of ``{"type": "text", ...}`` parts.
#: Both render to the same bytes; for strings this is identical to the template
#: every 4B run used.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['content'] is string -%}{{- message['content'] -}}"
    "{%- else -%}{%- for part in message['content'] -%}"
    "{%- if part['type'] == 'text' -%}{{- part['text'] -}}{%- endif -%}"
    "{%- endfor -%}{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)


def render_message(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


GENERATION_PROMPT = f"{IM_START}assistant\n"


def make_model_dir(src: str | Path, dst: str | Path) -> Path:
    """``dst`` = ``src`` with only ``chat_template`` replaced.

    Weights and every other file are symlinked, so the directory is the base
    model byte for byte except for the one field this module exists to pin.
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in ("tokenizer_config.json", "chat_template.jinja"):
            continue
        target = dst / item.name
        if not target.exists():
            os.symlink(item.resolve(), target)
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    cfg["chat_template"] = CHAT_TEMPLATE
    (dst / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))
    return dst


def install_template(model_dir: str | Path) -> None:
    """Overwrite ``chat_template`` in a checkpoint directory a trainer wrote.

    A trainer that saves the tokenizer it loaded will already carry it; this is
    for the ones that do not, and it is idempotent.
    """
    path = Path(model_dir) / "tokenizer_config.json"
    cfg = json.loads(path.read_text())
    if cfg.get("chat_template") != CHAT_TEMPLATE:
        cfg["chat_template"] = CHAT_TEMPLATE
        path.write_text(json.dumps(cfg, indent=2))
    # A VLM processor (Qwen3.8) reads chat_template.jinja rather than the
    # tokenizer config, so the pinned template is written there too.
    jinja = Path(model_dir) / "chat_template.jinja"
    if jinja.exists() and jinja.read_text() != CHAT_TEMPLATE and not Path(str(jinja) + ".stock").exists():
        shutil.copy(str(jinja), str(jinja) + ".stock")
    jinja.write_text(CHAT_TEMPLATE)


VLM_PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json")


def text_only_view(model_dir: str | Path) -> None:
    """Hide a VLM's processor configs so every framework treats it as text-only.

    veRL 0.9.1's ``hf_processor`` maps Qwen3.5/3.8 to ``Qwen3VLProcessor`` and
    binds Qwen3-VL's ``get_rope_index``, producing 4-row position ids that the
    Qwen3.5 text model rejects (``apply_rotary_pos_emb`` size mismatch).  With the
    processor configs moved aside ``hf_processor`` returns None and positions are
    1-D, as they are for any text-only model.  The data here is text-only, and
    vLLM is run with ``language_model_only=True``.  Idempotent; files are moved,
    not deleted, into ``_vlm_processor/``.
    """
    d = Path(model_dir)
    aside = d / "_vlm_processor"
    for name in VLM_PROCESSOR_FILES:
        f = d / name
        if f.exists() or f.is_symlink():
            aside.mkdir(exist_ok=True)
            shutil.move(str(f), str(aside / name))

