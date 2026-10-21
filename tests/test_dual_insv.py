import pytest
from pathlib import Path
import tempfile
from raven_app.video import find_partner_insv, normalize_dual_insv_pair
from raven_app.cli import parse


def test_find_partner_insv_both_exist():
    with tempfile.TemporaryDirectory() as tmp:
        p0 = Path(tmp) / 'VID_20260629_151302_00_013.insv'
        p1 = Path(tmp) / 'VID_20260629_151302_10_013.insv'
        p0.write_bytes(b'0')
        p1.write_bytes(b'1')

        c0, c1 = find_partner_insv(p0)
        assert c0 == p0
        assert c1 == p1

        # Test detection when passing p1
        c0_rev, c1_rev = find_partner_insv(p1)
        assert c0_rev == p0
        assert c1_rev == p1


def test_find_partner_insv_missing():
    with tempfile.TemporaryDirectory() as tmp:
        p0 = Path(tmp) / 'VID_20260629_151302_00_013.insv'
        p0.write_bytes(b'0')

        c0, c1 = find_partner_insv(p0)
        assert c0 == p0
        assert c1 is None


def test_normalize_dual_insv_pair():
    with tempfile.TemporaryDirectory() as tmp:
        p0 = Path(tmp) / 'VID_20260629_151302_00_013.insv'
        p1 = Path(tmp) / 'VID_20260629_151302_10_013.insv'
        p0.write_bytes(b'0')
        p1.write_bytes(b'1')

        # Correct order
        norm0, norm1 = normalize_dual_insv_pair(p0, p1)
        assert norm0 == p0
        assert norm1 == p1

        # Reversed order
        norm0, norm1 = normalize_dual_insv_pair(p1, p0)
        assert norm0 == p0
        assert norm1 == p1

        # Single input with auto-discovery
        norm0, norm1 = normalize_dual_insv_pair(p0, None)
        assert norm0 == p0
        assert norm1 == p1


def test_cli_parsing_secondary_insv():
    args = parse([
        'workflow',
        '--bag', 'test.bag',
        '--insv', 'VID_00.insv',
        '--secondary-insv', 'VID_10.insv',
        '--output', 'out_dir'
    ])
    assert args.insv == Path('VID_00.insv')
    assert args.secondary_insv == Path('VID_10.insv')
