"""Markdown linting helpers for OKFgraph ingestion.

These functions wrap ``mordant.lint`` / ``mordant.fix`` with OKFgraph's
non-blocking policy: fixable whitespace/formatting issues (MD009, MD012,
MD047) are auto-fixed; structural errors (MD001, MD031, MD033) are reported
as warnings but never block ingestion.

Extracted from ``okfgraph.router.OKFRouter._lint_converted_md`` /
``_lint_converted_md_str`` (Gap #5c).
"""

from pathlib import Path
from typing import Any, Dict

import logging
import mordant

logger = logging.getLogger(__name__)

# Whitespace/formatting rules that mordant can auto-fix.
FIXABLE_RULES = {"MD009", "MD012", "MD047"}
# Structural rules that indicate malformed markdown — warn only.
ERROR_RULES = {"MD001", "MD031", "MD033"}


def lint_converted_md(
    md_path: Path,
    *,
    auto_fix: bool = True,
) -> Dict[str, Any]:
    """Lint a markdown file and optionally auto-fix fixable issues.

    Returns a dict with:
    - ``content``: the (possibly fixed) markdown content (str)
    - ``fixed``: whether content was modified (bool)
    - ``fixed_count``: number of auto-fixed issues (int)
    - ``unfixable``: list of unfixable diagnostics (list)
    - ``errors``: list of error-level diagnostics (list)
    """
    content = Path(md_path).read_text(encoding="utf-8")
    diagnostics = mordant.lint(content, gfm_opts=mordant.GfmOptions.all())

    if not diagnostics:
        return {
            "content": content,
            "fixed": False,
            "fixed_count": 0,
            "unfixable": [],
            "errors": [],
        }

    unfixable = [d for d in diagnostics if d.rule not in FIXABLE_RULES]
    errors = [d for d in diagnostics if d.rule in ERROR_RULES]

    fixed_content = content
    fixed_count = 0

    if auto_fix:
        fixable = [d for d in diagnostics if d.rule in FIXABLE_RULES]
        if fixable:
            result = mordant.fix(content, gfm_opts=mordant.GfmOptions.all())
            if result.fixed:
                fixed_content = result.output
                fixed_count = len(result.fixed)
                logger.info(
                    "auto-fixed %d issues in %s",
                    fixed_count,
                    Path(md_path).name,
                )

    if unfixable:
        logger.warning(
            "%d unfixable issues in %s: %s",
            len(unfixable),
            Path(md_path).name,
            ", ".join(f"{d.rule} (line {d.line})" for d in unfixable[:5]),
        )

    if errors:
        logger.warning(
            "%d structural errors in %s — import may produce unexpected results: %s",
            len(errors),
            Path(md_path).name,
            ", ".join(f"{d.rule} (line {d.line})" for d in errors[:3]),
        )

    return {
        "content": fixed_content,
        "fixed": fixed_count > 0,
        "fixed_count": fixed_count,
        "unfixable": unfixable,
        "errors": errors,
    }


def lint_converted_md_str(
    content: str,
    *,
    auto_fix: bool = True,
) -> Dict[str, Any]:
    """Lint markdown content in-memory (no file I/O).

    Returns a dict with the same keys as :func:`lint_converted_md`, but
    ``unfixable`` / ``errors`` contain rule-name strings (not diagnostic
    objects) for easier programmatic inspection.
    """
    diagnostics = mordant.lint(content, gfm_opts=mordant.GfmOptions.all())

    if not diagnostics:
        return {
            "content": content,
            "fixed": False,
            "fixed_count": 0,
            "unfixable": [],
            "errors": [],
        }

    unfixable = [d for d in diagnostics if d.rule not in FIXABLE_RULES]
    errors = [d for d in diagnostics if d.rule in ERROR_RULES]

    fixed_content = content
    fixed_count = 0

    if auto_fix:
        fixable = [d for d in diagnostics if d.rule in FIXABLE_RULES]
        if fixable:
            result = mordant.fix(content, gfm_opts=mordant.GfmOptions.all())
            if result.fixed:
                fixed_content = result.output
                fixed_count = len(result.fixed)

    return {
        "content": fixed_content,
        "fixed": fixed_count > 0,
        "fixed_count": fixed_count,
        "unfixable": [d.rule for d in unfixable],
        "errors": [d.rule for d in errors],
    }
