"""Load and validate the frozen Stage 3 timing parameters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.timing.stage1_fixtures import ENSEMBLE_REALIZATIONS, NOISE_ARMS, SEED_BLOCKS, stage1_fixtures

FROZEN_STAGE3_PARAMETERS = Path(__file__).with_name("frozen_stage3_parameters.json")


def _fingerprint(parameters):
    payload = {key: value for key, value in parameters.items() if key != "fingerprint"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def _expected_arm_labels():
    labels = set()
    for scope in ("fixed_aperture", "end_to_end"):
        for fixture, spec in stage1_fixtures().items():
            if scope == "fixed_aperture" and spec["scope"] == "end_to_end":
                continue
            for noise_arm in NOISE_ARMS:
                for harmonics_enabled in (False, True):
                    for tool in ("isoster", "photutils", "autoprof"):
                        labels.add("|".join((scope, tool, fixture, noise_arm, str(harmonics_enabled))))
    return labels


def load_stage3_parameters(path=FROZEN_STAGE3_PARAMETERS):
    parameters = json.loads(Path(path).read_text())
    if parameters.get("fingerprint") != _fingerprint(parameters):
        raise ValueError("Stage 3 parameter fingerprint does not match its contents")
    if parameters.get("sessions") != 3 or parameters.get("repetitions_per_session") != ENSEMBLE_REALIZATIONS:
        raise ValueError("Stage 3 session or repetition count differs from the frozen decision")
    if parameters.get("campaign_seed_block") != SEED_BLOCKS["campaign"]:
        raise ValueError("Stage 3 campaign seed block differs from the frozen Stage 1 seed")
    batches = parameters.get("calls_per_batch_by_arm", {})
    if set(batches) != _expected_arm_labels():
        raise ValueError("Stage 3 batch table does not contain exactly the 132 frozen arms")
    if any(not isinstance(value, int) or value < 1 for value in batches.values()):
        raise ValueError("Stage 3 batch counts must be positive integers")
    return parameters
