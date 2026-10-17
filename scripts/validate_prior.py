#!/usr/bin/env python3
"""Check a manual prior against the schema and the protocol's honesty rules.

Not a generic JSON-Schema runner: the rules that matter here are the ones
docs/zero_shot_protocol.md states — every manual fact carries a page, scan
realisations say how they were found, and concepts never silently become
type numbers.

Usage:
  uv run python scripts/validate_prior.py assets/priors/prior_ContraJ-Nes-v0.json
  uv run python scripts/validate_prior.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "assets" / "priors" / "schema" / "prior.schema.json"
PRIORS = ROOT / "assets" / "priors"

BUTTONS = {"A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"}
KINDS = {
    "goal", "target", "controls", "exit", "hazard", "player",
    "economy", "menu", "hud", "collectible", "ability", "other",
}


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def check_citation(cit, where: str, errors: list[str]) -> None:
    if not isinstance(cit, dict):
        fail(f"{where}: citation missing", errors)
        return
    page = cit.get("page")
    if not isinstance(page, int) or page < 1:
        fail(f"{where}: citation.page must be a positive integer", errors)
    if not (cit.get("quote") or cit.get("figure")):
        fail(f"{where}: citation needs quote or figure (page {page})", errors)


def check_templates(prior: dict, errors: list[str], warnings: list[str]) -> None:
    for i, t in enumerate(prior.get("templates") or []):
        where = f"templates[{i}] ({t.get('name', '?')})"
        if not t.get("name"):
            fail(f"{where}: name required", errors)
        if "phases" in t:
            # composed template: frames live on the phases
            for j, ph in enumerate(t["phases"] or []):
                n = ph.get("frames")
                if not isinstance(n, int) or n < 1:
                    fail(f"{where}.phases[{j}]: frames must be a positive integer", errors)
                for key in ("hold", "tap", "rest"):
                    for b in ph.get(key) or []:
                        if b not in BUTTONS:
                            fail(f"{where}.phases[{j}].{key}: unknown button {b!r}", errors)
        else:
            frames = t.get("frames")
            if not isinstance(frames, int) or frames < 1:
                fail(f"{where}: frames must be a positive integer", errors)
        for key in ("hold", "tap", "rest"):
            for b in t.get(key) or []:
                if b not in BUTTONS:
                    fail(f"{where}.{key}: unknown button {b!r}", errors)
        has_cit = isinstance(t.get("citation"), dict)
        has_scan = bool(t.get("found_by"))
        if not has_cit and not has_scan:
            warnings.append(
                f"{where}: neither citation nor found_by — is this from the "
                f"manual or from controllability.py?"
            )
        if has_cit:
            check_citation(t["citation"], f"{where}.citation", errors)


def check_v2_concept_layer(prior: dict, errors: list[str], warnings: list[str]) -> None:
    for i, fact in enumerate(prior.get("facts") or []):
        where = f"facts[{i}] ({fact.get('id', '?')})"
        if not fact.get("id"):
            fail(f"{where}: id required", errors)
        kind = fact.get("kind")
        if kind not in KINDS:
            fail(f"{where}: kind must be one of {sorted(KINDS)}", errors)
        if not fact.get("statement"):
            fail(f"{where}: statement required", errors)
        check_citation(fact.get("citation"), f"{where}.citation", errors)
        status = fact.get("status", "extracted")
        if status not in ("extracted", "human_confirmed", "rejected"):
            fail(f"{where}: bad status {status!r}", errors)

    for section in ("goal", "player", "controls", "exit"):
        block = prior.get(section) or {}
        for cit in block.get("citations") or []:
            check_citation(cit, f"{section}.citations", errors)

    targets = prior.get("targets") or {}
    for i, c in enumerate(targets.get("concepts") or []):
        check_citation(c.get("citation"), f"targets.concepts[{i}].citation", errors)
    real = targets.get("realisation") or {}
    if real and not real.get("found_by"):
        warnings.append(
            "targets.realisation is present without found_by — protocol "
            "requires the scan that produced the numbers to be named"
        )
    if real.get("types") is not None and not targets.get("concepts"):
        warnings.append(
            "targets.realisation.types set but targets.concepts is empty — "
            "a realisation without a concept is the mix-up the schema exists "
            "to prevent"
        )

    exit_real = (prior.get("exit") or {}).get("realisation") or {}
    if exit_real and not exit_real.get("found_by"):
        warnings.append("exit.realisation is present without found_by")

    for i, h in enumerate(prior.get("hazards") or []):
        check_citation(h.get("citation"), f"hazards[{i}].citation", errors)
    for i, e in enumerate(prior.get("economy") or []):
        check_citation(e.get("citation"), f"economy[{i}].citation", errors)

    tables = prior.get("tables") or {}
    if tables and not tables.get("found_by"):
        warnings.append("tables is present without found_by")


def check_legacy_mixups(prior: dict, errors: list[str], warnings: list[str]) -> None:
    """v1 files and the legacy flat fields of v2 files."""
    src = prior.get("source") or ""
    if "recollection" in src.lower() or "memory" in src.lower() and "scan" not in src.lower():
        warnings.append(
            "source sounds like recollection rather than a scan — "
            "docs/manuals.md requires a scan a person can open"
        )
    if prior.get("target_types") is not None and not (
        (prior.get("targets") or {}).get("concepts") or prior.get("notes")
    ):
        warnings.append(
            "target_types set with neither targets.concepts nor notes — "
            "name the manual concept it realises"
        )
    why = prior.get("target_px_why") or ""
    if prior.get("target_px") is not None and not why:
        warnings.append(
            "target_px set without target_px_why — the pricing that swamped "
            "progress once already needs its arithmetic on file"
        )


def validate(path: Path) -> int:
    try:
        prior = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{path}: cannot read: {exc}", file=sys.stderr)
        return 2

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(prior, dict):
        print(f"{path}: top level must be an object", file=sys.stderr)
        return 2

    version = prior.get("schema_version")
    if version not in (1, 2):
        fail(f"schema_version must be 1 or 2, got {version!r}", errors)
    for key in ("game", "source"):
        if not prior.get(key):
            fail(f"{key} required", errors)

    if version == 2:
        check_v2_concept_layer(prior, errors, warnings)
    check_templates(prior, errors, warnings)
    check_legacy_mixups(prior, errors, warnings)

    # Structural sanity against the schema file's required top-level set.
    if SCHEMA.exists():
        try:
            schema = json.loads(SCHEMA.read_text())
            allowed = set(schema.get("properties", {}))
            unknown = set(prior) - allowed
            if unknown:
                warnings.append(f"fields not in schema: {sorted(unknown)}")
        except (OSError, json.JSONDecodeError):
            warnings.append(f"could not read {SCHEMA.name} for the allow-list")

    label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"{label}: schema_version={version}, "
          f"{len(prior.get('facts') or [])} facts, "
          f"{len(prior.get('templates') or [])} templates")
    for w in warnings:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print(f"  FAIL ({len(errors)} error(s), {len(warnings)} warning(s))")
        return 1
    print(f"  ok ({len(warnings)} warning(s))")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path,
                    help="prior JSON files (default: --all)")
    ap.add_argument("--all", action="store_true",
                    help="check every assets/priors/prior_*.json")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.all or not paths:
        paths = sorted(PRIORS.glob("prior_*.json"))
    if not paths:
        print("no prior files found", file=sys.stderr)
        return 2

    rc = 0
    for p in paths:
        rc = max(rc, validate(p))
    return rc


if __name__ == "__main__":
    sys.exit(main())
