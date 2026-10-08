"""Synthetic graphs and permanent lesions. No biological data are used."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


SYNTHETIC_KINDS = ('dense', 'structured', 'random')
KINDS = SYNTHETIC_KINDS + ('male_cns', 'rewired_male_cns')


def connectivity(kind: str, width: int = 64, seed: int = 0, density: float | None = None) -> np.ndarray:
    """Rows are target neurons, columns source neurons in a feedforward layer.

    The random control uses directed bipartite double-edge swaps, preserving
    both degree sequences. Blocks are synthetic modules, not anatomical regions.
    """
    if kind not in SYNTHETIC_KINDS or width < 16 or width % 4:
        raise ValueError('kind must be dense/structured/random; width >=16, multiple of 4')
    if kind == 'dense':
        return np.ones((width, width), dtype=np.float32)
    block = width // 4
    density = .3125 if density is None else density
    if not np.isfinite(density) or not 0 < density <= 1:
        raise ValueError('density must be finite and in (0,1]')
    degree = max(1, int(round(width * density)))
    mask = np.zeros((width, width), dtype=np.float32)
    for row in range(width):
        module, offset = divmod(row, block)
        for connection in range(degree):
            jump, delta = divmod(connection, block)
            col = ((module + jump) % 4) * block + (offset + delta) % block
            mask[row, col] = 1
    if kind == 'random' and degree < width:
        rng = np.random.default_rng(seed)
        edges = np.argwhere(mask == 1)
        for _ in range(20 * len(edges)):
            i, j = rng.integers(len(edges), size=2)
            a, b = edges[i]
            c, d = edges[j]
            if a != c and b != d and mask[a, d] == 0 and mask[c, b] == 0:
                mask[a, b] = mask[c, d] = 0
                mask[a, d] = mask[c, b] = 1
                edges[i] = (a, d)
                edges[j] = (c, b)
    return mask


class FlyExtractor(BaseFeaturesExtractor):
    """Shared actor/critic observations→width→width extractor with persistent buffers.

    Storage/compute remains dense; masks enforce functional sparsity. This
    implementation does not claim sparse-kernel memory or speed advantages.
    """
    def __init__(self, observation_space, kind='dense', width=64, seed=0, density=None):
        super().__init__(observation_space, features_dim=width)
        self.observation_space = observation_space
        self.kind = kind
        self.layer1 = nn.Linear(int(np.prod(observation_space.shape)), width)
        self.layer2 = nn.Linear(width, width)
        self.register_buffer('mask', torch.tensor(connectivity(kind, width, seed, density=density)))
        self.register_buffer('alive', torch.ones(width))

    def activations(self, observations):
        hidden = torch.tanh(self.layer1(observations))
        output = F.linear(hidden, self.layer2.weight * self.mask, self.layer2.bias)
        return hidden, torch.tanh(output) * self.alive

    def forward(self, observations):
        return self.activations(observations)[1]

    def effective_connections(self) -> int:
        return int((self.mask * self.alive[:, None]).sum().item())

    def lesion(self, mode: str, fraction: float, seed: int) -> dict:
        """Remove a rounded fraction of currently active nodes or active edges.

        Lesion calls are cumulative. Recovery never calls this method to undo
        damage. Node silence occurs after bias and nonlinearity, preventing bias
        from reviving the output. Lesions affect the shared actor/critic extractor.
        """
        if mode not in ('neurons', 'edges') or not 0 <= fraction <= 1:
            raise ValueError('mode neurons/edges and fraction in [0,1] required')
        rng = np.random.default_rng(seed)
        before = self.effective_connections()
        if mode == 'neurons':
            candidates = np.flatnonzero(self.alive.detach().cpu().numpy())
        else:
            candidates = np.argwhere(
                (self.mask * self.alive[:, None]).detach().cpu().numpy() != 0)
        count = int(round(fraction * len(candidates)))
        selected = candidates[rng.choice(len(candidates), count, replace=False)]
        with torch.no_grad():
            if mode == 'neurons':
                self.alive[torch.as_tensor(selected, dtype=torch.long, device=self.alive.device)] = 0
            elif count:
                self.mask[selected[:, 0], selected[:, 1]] = 0
        return {'mode': mode, 'fraction': fraction, 'seed': seed, 'removed': count,
                'eligible': len(candidates), 'connections_before': before,
                'connections_after': self.effective_connections(),
                'selected': selected.tolist()}
