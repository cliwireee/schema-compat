"""Every ```python block in README.md must run."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def test_readme_has_examples() -> None:
    assert len(BLOCKS) >= 3


@pytest.mark.parametrize("code", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_readme_block_runs(code: str) -> None:
    exec(compile(code, str(README), "exec"), {})
