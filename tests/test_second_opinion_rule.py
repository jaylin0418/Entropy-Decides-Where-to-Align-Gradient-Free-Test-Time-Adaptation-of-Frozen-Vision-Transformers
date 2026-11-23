"""The pair rule of the calibration: the two blocks (other than the main block) whose mixture entropy is lowest, in depth order."""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('site_choice', Path(__file__).resolve().parents[1] / 'scripts' / 'analysis' / 'site_choice.py')
site_choice = importlib.util.module_from_spec(spec)
sys.modules['site_choice'] = site_choice
spec.loader.exec_module(site_choice)
pick = site_choice.pick_second_opinions


def ent(singles, pairs):
    e = {f'px+b{k}': v for k, v in singles.items()}
    e.update({f'px+b{a}|px+b{b}': v for (a, b), v in pairs.items()})
    return e


def test_lowest_pair_in_depth_order():
    e = ent({4: 1.1, 11: 1.0, 5: 1.6}, {(4, 11): 0.9, (4, 5): 0.95, (5, 11): 1.2})
    assert pick(e, 6, 12) == ('px+b4', 'px+b11', 'px+b4|px+b11')
    e = ent({}, {(7, 8): 0.8, (5, 8): 0.9})                      # both branches may lie on one side of the main block
    assert pick(e, 6, 12) == ('px+b7', 'px+b8', 'px+b7|px+b8')


def test_singles_pixel_and_main_block_are_not_candidates():
    e = ent({4: 0.1}, {(4, 11): 0.9, (6, 11): 0.5})
    e['px'] = 0.0; e['px|px+b4'] = 0.0
    assert pick(e, 6, 12) == ('px+b4', 'px+b11', 'px+b4|px+b11')
