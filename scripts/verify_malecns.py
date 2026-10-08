from pathlib import Path
import hashlib,json,datetime,collections
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.compute as pc
import argparse
parser=argparse.ArgumentParser(description='Validate packaged MaleCNS80 against complete official feather tables.')
parser.add_argument('raw_dir',type=Path,help='Directory containing annotations.feather, edges.feather, neurotransmitters.feather')
parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'flyheal/data/malecns80')
parser.add_argument('--report',type=Path,help='Optional independent JSON report output (does not modify graph or manifest)')
args=parser.parse_args()
raw=args.raw_dir
source=args.data_dir
graph_bytes=(source/'graph.json').read_bytes()
g=json.loads(graph_bytes)
m=json.loads((source/'manifest.json').read_text())
ids=[n['id'] for n in g['nodes']]
checks={}; errors=[]; sources={}
base='https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
filenames={'annotations':'body-annotations-male-cns-v1.0-minconf-0.5.feather','edges':'connectome-weights-male-cns-v1.0-minconf-0.5.feather','neurotransmitters':'body-neurotransmitters-male-cns-v1.0.feather'}
for name,filename in filenames.items():
 p=raw/(name+'.feather')
 with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 sources[name]={'url':base+filename,'sha256':digest,'bytes':p.stat().st_size,'matchesPinnedManifest':digest==m['sources'][name]}
 if not sources[name]['matchesPinnedManifest']: errors.append('Source checksum: '+name)
 print(name,digest,sources[name]['bytes'],flush=True)
checks['rawSourceHashes']=not errors
checks['graphHash']=hashlib.sha256(graph_bytes).hexdigest()==m['graphSha256']
checks['uniqueBodyIds']=len(set(ids))==len(ids)==80
ann=feather.read_table(raw/'annotations.feather',columns=['bodyId','type','superclass','somaLocation'])
ann=ann.filter(pc.is_in(ann['bodyId'],value_set=pa.array(ids,type=ann['bodyId'].type))).to_pylist()
a={r['bodyId']:r for r in ann}
nt=feather.read_table(raw/'neurotransmitters.feather',columns=['body','consensus_nt'])
nt=nt.filter(pc.is_in(nt['body'],value_set=pa.array(ids,type=nt['body'].type))).to_pylist()
t={r['body']:r['consensus_nt'] for r in nt}
checks['all80BodiesPresent']=set(a)==set(ids)
checks['all80TypesExact']=all(n['id'] in a and n['type']==a[n['id']]['type'] for n in g['nodes'])
checks['all80SomaPositionsExact']=all(n['id'] in a and n['position']==a[n['id']]['somaLocation'] for n in g['nodes'])
checks['all80NeurotransmittersExact']=all(n['nt']==t.get(n['id']) for n in g['nodes'])
signs={'acetylcholine':1,'gaba':-1,'glutamate':-1}
checks['assumedSignsMatchDocumentedRule']=all(n['sign']==signs.get(n['nt'],0) for n in g['nodes'])
actual=[];total=0
with pa.memory_map(str(raw/'edges.feather'),'r') as mapped:
 reader=pa.ipc.open_file(mapped)
 for k in range(reader.num_record_batches):
  batch=reader.get_batch(k)
  total+=batch.num_rows
  pre=batch.column(batch.schema.get_field_index('body_pre'))
  post=batch.column(batch.schema.get_field_index('body_post'))
  keep=pc.and_(pc.is_in(pre,value_set=pa.array(ids,type=pre.type)),pc.is_in(post,value_set=pa.array(ids,type=post.type)))
  sub=batch.filter(keep)
  for r in sub.to_pylist(): actual.append((int(r['body_pre']),int(r['body_post']),int(r['weight'])))
  if k%100==0: print('edge batches',k,'rows',total,flush=True)
expected=[(ids[e[0]],ids[e[1]],e[2]) for e in g['edges']]
checks['allInternalDirectedEdgesAndCountsExact']=collections.Counter(actual)==collections.Counter(expected)
checks['1296Edges']=len(actual)==len(expected)==1296
checks['26029SynapticContacts']=sum(e[2] for e in actual)==sum(e[2] for e in expected)==26029
checks['inputAndReadoutCounts']=len(g['inputs'])==32 and len(g['outputs'])==16
checks['inputChannelsMatchOfficialTypes']=all(a[ids[i]]['type']==g['channels'][channel] and g['nodes'][i]['role']=='input' for i,channel in g['inputs'])
checks['inputChannelMultiplicity']=collections.Counter(channel for i,channel in g['inputs'])==collections.Counter({i:4 for i in range(8)})
checks['outputRolesExact']=all(g['nodes'][i]['role']=='output' for i in g['outputs'])
checks['outputsAreAnnotatedDescending']=all(a[ids[i]]['superclass']=='descending_neuron' for i in g['outputs'])
errors.extend(k for k,v in checks.items() if not v)
report={'schemaVersion':1,'status':'passed' if not errors else 'failed','verifiedAtUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Independent Python verifier; SHA-256 of all official files; exact body ID/type/soma/NT comparison; streaming scan of every official edge-table record batch and multiset equality of the complete induced subgraph. Upstream builder is never executed by this verifier.','graphSha256':hashlib.sha256(graph_bytes).hexdigest(),'sources':sources,'checks':checks,'errors':errors,'officialEdgeRowsScanned':total,'selectedNodes':len(ids),'selectedDirectedEdges':len(actual),'selectedSynapticContacts':sum(e[2] for e in actual),'excludedValidationClaims':['No independent reproduction of upstream node-selection ranking','No physiological validity of transmitter sign, rate dynamics, artificial inputs or action readout','No olfactory circuit or food-seeking circuit claim','No full-brain simulation claim']}
if args.report:
 args.report.parent.mkdir(parents=True,exist_ok=True)
 args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
if errors:raise SystemExit(1)
