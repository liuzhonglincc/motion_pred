import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
import heapq

# 定义MLP类
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MotionPredModule(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, k):
        super().__init__()
        self.args = args
        self._build_layers(dim_in, hidden_dim, k)
        self._reset_parameters()

    def _build_layers(self, dim_in, hidden_dim, k):
        self.mlp = MLP(4, hidden_dim, 4)

        # 将轨迹的过去k帧偏移 映射到1
        self.linear_k = nn.Linear(k, 1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    # -> 表示方法返回的类型
    def forward(self, data_x_1, data_x_2):
        data_x_2 = torch.transpose(data_x_2, 2, 3)   # 换成 1xNx4xk
        data_x_2_out = self.linear_k(data_x_2).squeeze(-1)  # 1xNx4x1  压缩成 1xNx4
 
        P_offs = self.mlp(data_x_2_out)

        return P_offs

def build(args, layer_name, dim_in, hidden_dim, k):
    interaction_layers = {
        'MP': MotionPredModule,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, k)