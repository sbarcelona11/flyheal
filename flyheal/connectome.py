"""Measured MaleCNS subset and a stateless differentiable circuit approximation.

Sensor injection, transmitter signs, dynamics and learned gains are modeling
assumptions. Anatomical contacts and cell identities are validated data.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

GRAPH_PATH = Path(__file__).parent / 'data' / 'malecns80' / 'graph.json'
REAL_KINDS = ('male_cns', 'rewired_male_cns')


def load_graph(path=None):
    path = Path(path) if path else GRAPH_PATH
    validation = json.loads(path.with_name('validation.json').read_text())
    if validation.get('status') != 'passed' or not all(validation.get('checks', {}).values()):
        raise ValueError('MaleCNS asset validation has not passed')
    if hashlib.sha256(path.read_bytes()).hexdigest() != validation.get('graphSha256'):
        raise ValueError('MaleCNS graph hash differs from validated asset')
    graph = json.loads(path.read_text())
    if len(graph['nodes']) != 80 or len(graph['channels']) != 8:
        raise ValueError('Expected 80 cells and eight input channels')
    pairs = set()
    for source, target, contacts in graph['edges']:
        if not (0 <= source < 80 and 0 <= target < 80 and contacts > 0) or (source,target) in pairs:
            raise ValueError('Invalid or duplicate directed anatomical edge')
        pairs.add((source,target))
    if len({n['id'] for n in graph['nodes']}) != 80 or any(n['sign'] not in (-1,0,1) for n in graph['nodes']):
        raise ValueError('Invalid cell identities or modeled transmitter signs')
    if len(graph['inputs']) != 32 or len({i for i,c in graph['inputs']}) != 32:
        raise ValueError('Expected 32 distinct input cells')
    if any(not (0 <= i < 80 and 0 <= c < 8) for i,c in graph['inputs']):
        raise ValueError('Invalid sensor injection mapping')
    return graph


def matrices(graph):
    counts = np.zeros((80,80),dtype=np.float32)
    for source,target,contacts in graph['edges']:
        counts[target,source] = contacts
    injection = np.zeros((80,8),dtype=np.float32)
    for cell,channel in graph['inputs']:
        injection[cell,channel] = 1.
    return counts,injection


def rewire_edges(edges, width=80, seed=0):
    """Swap targets, preserving directed degrees and source contact multisets."""
    result=np.array(edges,dtype=np.int64).copy()
    occupied={(int(s),int(t)) for s,t,_ in result}
    rng=np.random.default_rng(seed)
    for _ in range(20*len(result)):
        i,j=rng.integers(len(result),size=2)
        a,b,_=result[i]; c,d,_=result[j]
        if a==c or b==d or a==d or c==b or (a,d) in occupied or (c,b) in occupied:
            continue
        occupied.remove((a,b)); occupied.remove((c,d))
        occupied.add((a,d)); occupied.add((c,b))
        result[i,1]=d; result[j,1]=b
    return result.tolist()


class ConnectomeExtractor(BaseFeaturesExtractor):
    """80 real cells, five reset-within-decision leaky tanh iterations.

    No state survives forward(), keeping PPO minibatch shuffling valid.
    PPO learns a shared actor/critic readout from all 80 cells; recorded output
    roles are metadata, not a claim that the learned readout is biological.
    """
    def __init__(self, observation_space, kind='male_cns', width=80, seed=0,
                 density=None, graph_path=None, iterations=5, leak=.5):
        if kind not in REAL_KINDS or width!=80 or observation_space.shape!=(8,):
            raise ValueError('MaleCNS requires kind male_cns/rewired_male_cns, width80, eight observations')
        if iterations < 1 or not 0 < leak <= 1:
            raise ValueError('Invalid within-decision circuit dynamics')
        super().__init__(observation_space,features_dim=80)
        self.kind=kind; self.iterations=iterations; self.leak=leak
        graph=load_graph(graph_path)
        self.cell_metadata=graph['nodes']
        self.channels=graph['channels']
        self.output_indices=graph['outputs']
        self.graph_sha256=hashlib.sha256((Path(graph_path) if graph_path else GRAPH_PATH).read_bytes()).hexdigest()
        self.rewired = kind=='rewired_male_cns'
        original_counts,injection=matrices(graph)
        if self.rewired:
            graph={**graph,'edges':rewire_edges(graph['edges'],80,seed)}
        counts,_=matrices(graph)
        signs=np.array([n['sign'] for n in graph['nodes']],dtype=np.float32)
        signed=counts*signs[None,:]/np.maximum(counts.sum(axis=1,keepdims=True),1.)
        targets,sources=np.nonzero(counts)
        self.register_buffer('original_counts',torch.tensor(original_counts))
        self.register_buffer('base_counts',torch.tensor(counts))
        self.register_buffer('base_signed',torch.tensor(signed))
        self.register_buffer('input_map',torch.tensor(injection))
        self.register_buffer('original_mask',torch.tensor((counts>0).astype(np.float32)))
        self.register_buffer('mask',self.original_mask.clone())
        self.register_buffer('alive',torch.ones(80))
        self.register_buffer('edge_indices',torch.tensor(targets*80+sources,dtype=torch.long))
        self.log_gain=nn.Parameter(torch.zeros(len(sources)))
        self.log_input_gain=nn.Parameter(torch.zeros(8))
        self.bias=nn.Parameter(torch.zeros(80))

    def edge_gains(self):
        return torch.exp(self.log_gain.clamp(-3.,3.))

    def weight_matrix(self):
        gains=torch.zeros(6400,device=self.log_gain.device,dtype=self.log_gain.dtype)
        gains=gains.scatter(0,self.edge_indices,self.edge_gains()).reshape(80,80)
        return self.base_signed*gains*self.mask*self.alive[:,None]*self.alive[None,:]

    def _drive(self,observations):
        return F.linear(observations*torch.exp(self.log_input_gain.clamp(-3.,3.)),self.input_map)*self.alive

    def iteration_activities(self,observations):
        drive=self._drive(observations)
        state=torch.zeros_like(drive)
        weights=self.weight_matrix()
        activities=[]
        for _ in range(self.iterations):
            proposed=torch.tanh(drive+F.linear(state,weights,self.bias))
            state=((1-self.leak)*state+self.leak*proposed)*self.alive
            activities.append(state)
        return activities

    def activations(self,observations):
        return self._drive(observations),self.iteration_activities(observations)[-1]

    def forward(self,observations):
        return self.iteration_activities(observations)[-1]

    def effective_connections(self):
        return int((self.mask*self.alive[:,None]*self.alive[None,:]).sum().item())

    def modeled_connections(self):
        return int(((self.base_signed!=0)*self.mask*self.alive[:,None]*self.alive[None,:]).sum().item())

    def lesion(self,mode,fraction,seed):
        if mode not in ('neurons','edges') or not 0 <= fraction <= 1:
            raise ValueError('mode neurons/edges and fraction in [0,1] required')
        rng=np.random.default_rng(seed)
        before=self.effective_connections()
        candidates=(np.flatnonzero(self.alive.detach().cpu().numpy()) if mode=='neurons' else
                    np.argwhere((self.mask*self.alive[:,None]*self.alive[None,:]).detach().cpu().numpy()!=0))
        count=int(round(fraction*len(candidates)))
        selected=candidates[rng.choice(len(candidates),count,replace=False)]
        with torch.no_grad():
            if mode=='neurons':
                self.alive[torch.as_tensor(selected,dtype=torch.long,device=self.alive.device)]=0
            elif count:
                self.mask[selected[:,0],selected[:,1]]=0
        return dict(mode=mode,fraction=fraction,seed=seed,removed=count,eligible=len(candidates),
                    connections_before=before,connections_after=self.effective_connections(),
                    modeled_connections_after=self.modeled_connections(),selected=selected.tolist())
