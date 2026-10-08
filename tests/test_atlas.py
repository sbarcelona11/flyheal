"""Atlas integrity: every gray context soma is distinct from modeled cells."""
from pathlib import Path
import hashlib
import json
import math

DATA = Path(__file__).resolve().parents[1] / 'flyheal/data/malecns80'


def test_official_context_atlas_integrity_and_unmodeled_semantics():
    atlas_bytes = (DATA / 'atlas.json').read_bytes()
    atlas = json.loads(atlas_bytes)
    manifest = json.loads((DATA / 'atlas_manifest.json').read_text())
    graph = json.loads((DATA / 'graph.json').read_text())
    assert hashlib.sha256(atlas_bytes).hexdigest() == manifest['sha256']
    assert len(atlas_bytes) < 5_000_000
    assert atlas['count'] == manifest['count'] == len(atlas['points']) == 124209
    assert atlas['count'] + len(graph['nodes']) == 124289
    assert atlas['source']['sha256'] == manifest['source_sha256'] == '2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2'
    assert atlas['license'] == 'CC BY 4.0'
    assert atlas['modeled'] is False and atlas['includesNeurites'] is False
    assert atlas['pointLayout'] == ['bodyId', 'x', 'y', 'z']
    ids = [point[0] for point in atlas['points']]
    assert len(set(ids)) == len(ids) and ids == sorted(ids)
    assert set(ids).isdisjoint(node['id'] for node in graph['nodes'])
    assert all(len(point) == 4 and isinstance(point[0], int) and point[0] > 0 for point in atlas['points'])
    assert all(isinstance(value, (int, float)) and math.isfinite(value) for point in atlas['points'] for value in point[1:])
    assert atlas['selection']['status']=='Traced'
    assert 'vnc_intrinsic' not in atlas['selection']['superclasses']
