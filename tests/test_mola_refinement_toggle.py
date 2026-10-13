import pytest
from pathlib import Path
from raven_app.cli import parse

def test_cli_revisit_refinement_flag_default_false():
    args = parse([
        'workflow',
        '--bag', 'dummy.bag',
        '--insv', 'dummy.insv',
        '--output', 'dummy_out',
    ])
    assert getattr(args, 'revisit_refinement', None) is False

def test_cli_revisit_refinement_flag_enabled():
    args = parse([
        'workflow',
        '--bag', 'dummy.bag',
        '--insv', 'dummy.insv',
        '--output', 'dummy_out',
        '--revisit-refinement',
    ])
    assert getattr(args, 'revisit_refinement', None) is True

def test_cli_revisit_refinement_flag_disabled():
    args = parse([
        'workflow',
        '--bag', 'dummy.bag',
        '--insv', 'dummy.insv',
        '--output', 'dummy_out',
        '--no-revisit-refinement',
    ])
    assert getattr(args, 'revisit_refinement', None) is False
