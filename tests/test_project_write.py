"""write_project: stable id, and nothing is written when the manifest would not validate."""
import json
import re

import pytest

from raven_app.project import ManifestError, write_project

UUID4 = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')


def _project(tmp_path):
    (tmp_path / 'deliverables').mkdir()
    (tmp_path / 'deliverables' / 'colored_cloud.ply').write_bytes(b'x')
    return tmp_path


def test_first_export_has_a_uuid4_and_second_keeps_it(tmp_path):
    root = _project(tmp_path)
    first = write_project(root)
    assert UUID4.match(first['id']) and first['updated_utc']
    second = write_project(root)
    assert second['id'] == first['id']
    assert json.loads((root / 'project.json').read_text())['id'] == first['id']


def test_a_bad_old_id_is_replaced(tmp_path):
    root = _project(tmp_path)
    (root / 'project.json').write_text(json.dumps({'id': 'nope', 'assets': {}}))
    assert UUID4.match(write_project(root)['id'])


def test_invalid_manifest_leaves_the_old_file_untouched(tmp_path):
    root = _project(tmp_path)
    write_project(root)
    old = json.loads((root / 'project.json').read_text())
    old['crs'] = {'transform': '../escape.json'}
    (root / 'project.json').write_text(json.dumps(old))
    before = (root / 'project.json').read_bytes()
    with pytest.raises(ManifestError) as err:
        write_project(root)
    assert {f.code for f in err.value.findings} == {'E_PATH_ESCAPE'}
    assert (root / 'project.json').read_bytes() == before
