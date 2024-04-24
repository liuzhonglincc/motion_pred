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

# 定义GCN模型, 两层GCN
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class MotionPredModule(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, k):
        super().__init__()
        self.args = args
        self._build_layers(dim_in, hidden_dim, k)
        self._reset_parameters()

    def _build_layers(self, dim_in, hidden_dim, k):
        self.gcn = GCN(8, dim_in//2, dim_in)

        self.linear_key = nn.Linear(dim_in*3, dim_in)
        self.linear_query = nn.Linear(dim_in*3, dim_in)
        self.d = dim_in

        self.linear_offset = nn.Linear(4, dim_in)
        self.prelu = nn.PReLU()
        self.mlp = MLP(dim_in, hidden_dim, 4)

        # 将轨迹的过去k帧偏移 映射到1
        self.linear_k = nn.Linear(k, 1)

        self.conv_y_3 = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, (3-1)//2))
        self.conv_x_3 = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=((3-1)//2, 0))

        self.conv_y_5 = nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, (5-1)//2))
        self.conv_x_5 = nn.Conv2d(1, 1, kernel_size=(5, 1), padding=((5-1)//2, 0))

        self.conv_y_9 = nn.Conv2d(1, 1, kernel_size=(1, 9), padding=(0, (9-1)//2))
        self.conv_x_9 = nn.Conv2d(1, 1, kernel_size=(9, 1), padding=((9-1)//2, 0))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def find_k_nearest_neighbors(self, distance_matrix, k):
        n = distance_matrix.shape[0]
        res = [[], []]
        for i in range(n):
            max_heap = []
            for j in range(n):
                if i != j:
                    distance = distance_matrix[i, j]
                    if len(max_heap) < k:
                        heapq.heappush(max_heap, (-distance, j))
                    elif -max_heap[0][0] > distance:
                        heapq.heappop(max_heap)
                        heapq.heappush(max_heap, (-distance, j))
            for _, idx in max_heap:
                res[0].append(i)
                res[1].append(idx)
        return res

    # -> 表示方法返回的类型
    def forward(self, data_x_1, data_x_2):
        data_x_2 = torch.transpose(data_x_2, 2, 3)   # 换成 1xNx4xk
        data_x_2_out = self.linear_k(data_x_2).squeeze(-1)  # 1xNx4x1  压缩成 1xNx4
        data_x = torch.cat((data_x_1, data_x_2_out), dim=2)
        Ot = data_x[:, :, 4:]

        # 计算每个点到其他点的欧氏距离
        expanded_tensor = data_x[:, :, 0:2].unsqueeze(2)  # Shape: (1, K, 1, 2)
        expanded_tensor_transposed = expanded_tensor.transpose(1, 2)  # Shape: (1, 1, K, 2)
        diff = expanded_tensor - expanded_tensor_transposed  # (1, K, K, 2)
        distances = torch.norm(diff, dim=3)  # Shape: (1, K, K)
        max_values, _ = torch.max(distances, dim=2, keepdim=True)
        oushi_atte = 1 - distances / max_values

        # 使用GCN代替 linear
        edge_index_3 = self.find_k_nearest_neighbors(distances.detach().squeeze(0).cpu().numpy(), 3)
        embedding_3 = self.gcn(data_x.squeeze(0), torch.tensor(edge_index_3, dtype=torch.long).to(torch.device("cuda"))).unsqueeze(0)
        edge_index_5 = self.find_k_nearest_neighbors(distances.detach().squeeze(0).cpu().numpy(), 5)
        embedding_5 = self.gcn(data_x.squeeze(0), torch.tensor(edge_index_5, dtype=torch.long).to(torch.device("cuda"))).unsqueeze(0)
        edge_index_9 = self.find_k_nearest_neighbors(distances.detach().squeeze(0).cpu().numpy(), 9)
        embedding_9 = self.gcn(data_x.squeeze(0), torch.tensor(edge_index_9, dtype=torch.long).to(torch.device("cuda"))).unsqueeze(0)
        embedding = torch.cat((embedding_3, embedding_5, embedding_9), dim=2)

        query = self.linear_query(embedding).unsqueeze(1)  # batch_size, num_heads, seq_length, d_k
        key = self.linear_key(embedding).unsqueeze(1)  # batch_size, num_heads, seq_length, d_k

        # 计算注意力分数矩阵
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(self.d))
        scaled_attention_scores = scaled_attention_scores.squeeze(1)
        max_values, _ = torch.max(scaled_attention_scores, dim=2, keepdim=True)
        A_atte = scaled_attention_scores / max_values

        # 多尺度非对称卷积
        A_atte_y_3 = self.conv_y_3(A_atte.unsqueeze(0))
        A_atte_x_3 = self.conv_x_3(A_atte.unsqueeze(0))

        A_atte_y_5 = self.conv_y_5(A_atte.unsqueeze(0))
        A_atte_x_5 = self.conv_x_5(A_atte.unsqueeze(0))

        A_atte_y_9 = self.conv_y_9(A_atte.unsqueeze(0))
        A_atte_x_9 = self.conv_x_9(A_atte.unsqueeze(0))

        A_atte = torch.cat((A_atte, oushi_atte, (A_atte_y_3+A_atte_x_3).squeeze(0), (A_atte_y_5+A_atte_x_5).squeeze(0), (A_atte_y_9+A_atte_x_9).squeeze(0)), dim=2) # 1xNx5N

        # 送入mlp之前，对A_atte 归一化  normalize,  尽管 A_atte 和 oushi_atte 都各自做了归一化，这里还是要整体归一化，不然性能会降低。
        A_atte_min = A_atte.min()
        A_atte_max = A_atte.max()
        A_atte = (A_atte - A_atte_min) / (A_atte_max - A_atte_min)

        Ot = Ot.repeat(1, 5, 1)
        P_offs = self.mlp(self.prelu(self.linear_offset(torch.matmul(A_atte, Ot))))

        return P_offs

def build(args, layer_name, dim_in, hidden_dim, k):
    interaction_layers = {
        'MP': MotionPredModule,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, k)