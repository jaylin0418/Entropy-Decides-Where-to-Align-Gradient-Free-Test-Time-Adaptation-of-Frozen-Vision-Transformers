"""Arms of the rule-chosen second-opinion stack are built from their name (scripts/episodic/run_forward_only.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.episodic.run_forward_only import ARMS, rule_arm  # noqa: E402
from scripts.streams.run_streams import ARMS as SARMS, rule_arm as stream_rule_arm  # noqa: E402
from scripts.cost.time_batches import ARMS as TARMS, rule_arm as timing_rule_arm  # noqa: E402


def test_rule_arm_names_the_blocks_one_based_and_matches_the_method():
    assert rule_arm('rule_t6_b5b9_gated') == ARMS['exit_b5b9_gated']
    assert rule_arm('rule_t6_b9_gated') == ARMS['exit_b9_gated']
    assert rule_arm('rule_t6_b5b9') == ARMS['exit_b5b9']
    a = rule_arm('rule_t7_b4b11_gated')
    assert a['place'] == (6,) and a['exit_sites'] == (3, 10) and a['exit_theta'] == 1.5 and a['gate'] == {'spec': 0.4, 'mean': 0.3}
    assert rule_arm('rule_t6_b11b4_gated')['exit_sites'] == (3, 10)


def test_rule_arm_rejects_other_names():
    with pytest.raises(KeyError):
        rule_arm('exit_b5b9_gated')


def test_random_control_and_the_stream_and_timing_arms():
    r = rule_arm('rule_t6_b4b11_random_gated')
    assert r['exit_random'] == 0.35 and r['exit_sites'] == (3, 10) and 'gate' in r
    assert rule_arm('rule_t6_b5b9_random_gated') == ARMS['exit_b5b9_random_gated']
    v = rule_arm('rule_t6_views_gated')                                 # the views control: no extra block, two views of the main path
    assert v['exit_sites'] == () and v['exit_views'] == 2 and v['place'] == (5,) and 'gate' in v
    assert stream_rule_arm('rule_t6_b5b9_gated') == SARMS['exit_b5b9_gated']
    assert timing_rule_arm('rule_t6_b5b9_k16_fp32_stats32') == TARMS['exit_b5b9_k16_fp32_stats32']
    with pytest.raises(KeyError):
        timing_rule_arm('rule_t6_b4b11_gated')
