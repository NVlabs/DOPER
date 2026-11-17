# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard: the public agent guidance names every consent flag and resolves its skill.

The agent files live in ``release/public-root/`` on a development checkout and at the tree root
in an exported tree; the tests find them in whichever place exists.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "release" / "public-root"
AGENT_ROOT = STAGING if STAGING.is_dir() else ROOT
CONSENT_FLAGS = (
    "--accept-license",
    "--accept-new-layout",
    "--accept-non-paper-masks",
    "--accept-registration",
    "--accept-unreviewed",
    "--allow-test-inputs",
    "--allow-partial-bank",
    "--allow-unpinned-checkpoint",
    "--allow-nonstandard-backbone",
    "--force",
    "--force-from",
)
GUARDED_ENV_VARS = ("DOPER_ALLOW_BAD_DATA", "SYNTH_ALLOW_MISSING_ASSETS")
SKILL_NAME = "doper-cad-workflow"


def read_agents_md() -> str:
    return (AGENT_ROOT / "AGENTS.md").read_text(encoding="utf-8")


def test_agents_md_mentions_every_consent_flag_and_guarded_env_var():
    text = read_agents_md()
    missing = [name for name in (*CONSENT_FLAGS, *GUARDED_ENV_VARS) if name not in text]
    assert missing == []
    assert "DOPER_BLENDER" in text
    assert "python -m doper.train --config" in text


def test_agents_md_points_at_domain_randomization_notes():
    text = read_agents_md()
    assert "examples/minimal/README.md" in text
    assert "Domain randomization" in text
    assert re.search(r"(?i)opt-in", text)
    assert "robust_appearance" in text


def test_claude_md_points_at_agents_md():
    lines = [
        line
        for line in (AGENT_ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    assert "AGENTS.md" in lines[0]


def test_skill_has_front_matter_and_symlink_resolves():
    skill_dir = ROOT / "skills" / SKILL_NAME
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    front = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", text)
    assert front is not None
    assert front.group(1) == SKILL_NAME
    link = AGENT_ROOT / ".claude" / "skills" / SKILL_NAME
    assert link.is_symlink()
    assert link.resolve() == skill_dir.resolve()
    assert (link / "SKILL.md").is_file()


def read_skill_md() -> str:
    return (ROOT / "skills" / SKILL_NAME / "SKILL.md").read_text(encoding="utf-8")


def test_agents_md_tells_agents_to_run_uv_sync_first_without_asking():
    text = " ".join(read_agents_md().split())
    assert "first step in a fresh checkout is `uv sync`" in text
    assert "without asking" in text


def test_cad_example_dry_run_means_real_renders_and_a_training_dry_run():
    for text in (read_agents_md(), read_skill_md()):
        flat = " ".join(text.split())
        assert "CAD example dry run" in flat
        assert "steps 1-4" in flat
        assert "real renders" in flat
        assert "only training is a dry run" in flat.lower()
        assert "python -m doper.train" in flat
    skill = " ".join(read_skill_md().split())
    assert "optional preflight" in skill
    assert "`uv sync` first" in skill
