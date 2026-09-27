"""After a veRL job: publish the last HF export at ``<output_dir>/hf``.

``<output_dir>/hf`` is what ``PMVerlBackend`` returns as
``Invocation.expected_checkpoint``, so the next stage reads it from the plan
rather than re-deriving veRL's ``global_step_N/...`` layout.

Usage: python -m pm_train.finalize <output_dir> [sft|rl]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from .chat import install_template, text_only_view


def finalize(output_dir: str | Path, stage: str) -> Path:
    out = Path(output_dir)
    steps = sorted(
        (int(m.group(1)), p) for p in (out / "checkpoints").glob("global_step_*")
        if (m := re.fullmatch(r"global_step_(\d+)", p.name))
    )
    if not steps:
        raise SystemExit(f"{out}: no global_step_* checkpoint")
    step, last = steps[-1]
    hf = last / "huggingface" if stage == "sft" else last / "actor" / "huggingface"
    if not (hf / "config.json").exists():
        raise SystemExit(f"{hf}: no HF export (was hf_model in save_contents?)")
    if not any(hf.glob("*.safetensors")):
        raise SystemExit(f"{hf}: HF export has config but no weights")
    install_template(hf)
    text_only_view(hf)
    link = out / "hf"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(hf.resolve())
    (out / "hf_step.txt").write_text(f"{step}\n{hf.resolve()}\n")
    print(f"{link} -> {hf.resolve()} (global_step {step})")
    return link


if __name__ == "__main__":
    finalize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "sft")
