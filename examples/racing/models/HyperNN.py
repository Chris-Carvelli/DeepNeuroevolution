import random
import math
from functools import reduce
import torch
import torch.nn as nn


def random_z_v(z_dim, z_num):
    # ret = np.random.normal(0.01, 1.0, z_dim * z_num)
    return torch.distributions.normal.Normal(torch.zeros([z_num, z_dim]), 0.1).sample()


class HyperNN(nn.Module):
    def __init__(self, obs_space, action_space, pnn, tiling=64, shrink=1):
        super().__init__()

        self._tiling = tiling

        self.z_dim = int(32 * shrink)
        self.z_v_evolve_prob = 0.5

        self.pnn = pnn(obs_space, action_space)
        self.pnn_modules = list(dict(self.pnn.named_children()).keys())

        self.out_features = self._get_out_features()
        self.z_num, self.z_indexer = self._get_z_num()

        in_size = int(128 * shrink)
        self.hnn = nn.Sequential(
            nn.Linear(self.z_dim, in_size),
            nn.ReLU(),
            nn.Linear(in_size, in_size),
            nn.ReLU(),
            nn.Linear(in_size, self.out_features),
        )

        self.register_buffer('z_v', random_z_v(self.z_dim, self.z_num))
        self.add_tensors = {}

        self._init_nn()

    def forward(self, layer_index=None):
        if layer_index is None:
            return [self.hnn(x) for x in self.z_v]
        else:
            if isinstance(layer_index, int):
                module_name = self.pnn_modules[layer_index]
            else:
                module_name = layer_index
            z_shard = self.z_indexer[module_name]
            return [self.hnn(x) for x in self.z_v[z_shard]]

    def evolve(self, sigma):
        coin_toss = random.random()
        if coin_toss > self.z_v_evolve_prob:
            # evolve z vector
            module_idx = math.floor(random.random() * len(self.pnn_modules))
            module_name = self.pnn_modules[module_idx]

            for name in self.z_indexer:
                if module_name in name:
                    z_shard = self.z_indexer[name]
                    self.z_v[z_shard] += torch.distributions.normal.Normal(
                        torch.zeros([z_shard.stop - z_shard.start, self.z_dim]),
                        sigma
                    ).sample()
            self._update_pnn()
        else:
            # evolve weights
            params = self.named_parameters()
            for name, tensor in sorted(params):
                if 'z_v' not in name:
                    to_add = self.add_tensors[tensor.size()]
                    to_add.normal_(0.0, sigma)
                    tensor.data.add_(to_add)
            self._update_pnn()

    def evaluate(self, env, max_eval, render=False, fps=60):
        return self.pnn.evaluate(env, max_eval, render, fps)

    def _init_nn(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal_(tensor)
            elif 'z_v' not in name:
                tensor.data.zero_()

        self._update_pnn()

    # tiling not supported (but it should be a bit faster, performance gain unclear)
    def _update_pnn(self):
        weights = self()

        if self._tiling:
            for name, param in self.pnn.named_parameters():
                z_shard = self.z_indexer[name]
                param.data = self._shape_w(weights[z_shard], param.shape).data
        else:
            i = 0
            for name, param in self.pnn.named_parameters():
                param.data = self._shape_w(weights[i], param.shape).data
                i += 1

    def _shape_w(self, w, layer_shape):
        if isinstance(w, list):
            w = torch.cat(w)
        w = torch.Tensor(w)
        w = torch.narrow(w, 0, 0, reduce((lambda x, y: x * y), layer_shape))
        w = w.view(layer_shape)

        return w

    def _get_z_num(self):
        z_num = 0
        z_indexer = {}

        # tiling
        for name, param in self.pnn.named_parameters():
            if self._tiling is not None:
                layer_shape = param.shape
                layer_size = reduce((lambda x, y: x * y), layer_shape, 1)
                z_shard = math.ceil(layer_size / self.out_features)

                z_indexer[name] = slice(z_num, z_num + z_shard, 1)

                z_num += z_shard
            else:
                z_num += 1

        return z_num, z_indexer

    def _get_out_features(self):
        if self._tiling is not None:
            return self._tiling

        ret = 0
        for name, param in self.pnn.named_parameters():
            if 'weight' in name:
                layer_shape = param.shape
                layer_size = reduce((lambda x, y: x * y), layer_shape)
                if layer_size > ret:
                    ret = layer_size
        return ret

