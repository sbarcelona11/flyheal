import json,hashlib
from pathlib import Path
import pyarrow.feather as feather
import argparse
parser=argparse.ArgumentParser(description='Export measured gray brain soma context, excluding modeled cells')
parser.add_argument('raw_dir',type=Path)
parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'flyheal/data/malecns80')
args=parser.parse_args();d=args.data_dir;p=next(args.raw_dir.glob('*annotations*'))
assert hashlib.sha256(p.read_bytes()).hexdigest()=='2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2', 'Unexpected official annotation hash'
g=json.loads((d/'graph.json').read_text());excluded={n['id'] for n in g['nodes']}
classes=['ol_intrinsic','ol_sensory','visual_projection','visual_centrifugal','visual_projection_tbc','cb_intrinsic','cb_sensory','cb_motor','cb_endocrine','cb_sensory_tbc','cb_efferent','descending_neuron','descending_neuron_tbc','sensory_descending','efferent_descending']
t=feather.read_table(p,columns=['bodyId','somaLocation','superclass','status']).to_pylist()
points=[[row['bodyId'],*row['somaLocation']] for row in t if row['status']=='Traced' and row['superclass'] in classes and row['somaLocation'] is not None and len(row['somaLocation'])==3 and row['bodyId'] not in excluded]
points.sort(key=lambda row:row[0])
a={'modeled':False,'includesNeurites':False,'pointLayout':['bodyId','x','y','z'],'count':len(points),'points':points,'coordinateUnits':'8 nm voxels, native MaleCNS EM space','selection':{'status':'Traced','superclasses':classes,'excludedModeledIds':sorted(excluded)},'meaning':'Unmodeled soma context only; no activity or neurites. Superclass selection is not a complete anatomical segmentation.','source':{'url':'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'+'body-annotations-male-cns-v1.0-minconf-0.5.feather','sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'license':'CC BY 4.0','attribution':'FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research','selectionReference':'cobanov/flyjump brain-atlas manifest, pinned e34c6614e7d13a1018585f705cdeb59b9f38291d; generated independently from official table'}
(d/'atlas.json').write_text(json.dumps(a,separators=(',',':')))
(d/'atlas_manifest.json').write_text(json.dumps({'sha256':hashlib.sha256((d/'atlas.json').read_bytes()).hexdigest(),'count':len(points),'source_sha256':a['source']['sha256']},indent=2))
print('atlas',len(points),(d/'atlas.json').stat().st_size)
