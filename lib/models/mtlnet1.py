import torch
import math
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class PositionalEncoding(nn.Module):
    def __init__(self,
        emb_size: int,
        dropout = 0.3,
        maxlen: int = 5000,
        batch_first = False):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000)/emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) #(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2) #(maxlen, 1, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        self.batch_first = batch_first

    def forward(self, token_embedding:Tensor):
        if self.batch_first:
            return self.pos_embedding.transpose(0,1)[:,:token_embedding.size(1)]
        else:
            return self.pos_embedding[:token_embedding.size(0), :]


class AUClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, seq_input):
        bs, seq_len = seq_input.size(0), seq_input.size(1)
        weight = self.fc.weight
        bias = self.fc.bias
        seq_input = seq_input.reshape((bs * seq_len, 1, -1))  # bs*seq_len, 1, metric_dim
        weight = weight.unsqueeze(0).repeat((bs, 1, 1))  # bs,seq_len, metric_dim
        weight = weight.view((bs * seq_len, -1)).unsqueeze(-1)  # bs*seq_len, metric_dim, 1
        inner_product = torch.bmm(seq_input, weight).squeeze(-1).squeeze(-1)  # bs*seq_len
        inner_product = inner_product.view((bs, seq_len))
        return inner_product + bias

class FEATURE_REDUCE1(nn.Module):
    def __init__(self, ori_feature, reduce_feature):
        super(FEATURE_REDUCE1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=ori_feature, out_channels=reduce_feature,
                               kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=reduce_feature, out_channels=reduce_feature,
                               kernel_size=1, bias=True)
        self.mish = nn.ReLU()

    def forward(self, x):
        feat1 = self.conv1(x)
        feat1 = self.mish(feat1)
        feat2 = self.conv2(feat1)
        feat2 = self.mish(feat2)
        return feat2

class MTLNet1(nn.Module):
    def __init__(self, cfg):
        super(MTLNet1, self).__init__()
        self.n_embed = 1536
        self.hidden_dim = cfg.MODEL.HIDDEN_DIM

        if self.hidden_dim != 1024:
            self.input_proj = FEATURE_REDUCE1(self.n_embed, self.hidden_dim)
        self.query_embed = nn.Embedding(22, self.hidden_dim)
        self.positional_encoding = PositionalEncoding(self.hidden_dim, batch_first=True)

        # block1
        self.trans1 = CA_FFN(d_model=self.hidden_dim, nhead=8, activation='relu')
        # block2
        self.trans2 = SA_CA_FFN(d_model=self.hidden_dim, nhead=8, activation='relu')
        # block3
        self.trans3 = SA_CA_FFN(d_model=self.hidden_dim, nhead=8, activation='relu')
        # block4
        self.trans4 = SA_CA_FFN(d_model=self.hidden_dim, nhead=8, activation='relu')

        # task linear layers
        self.expr_classifier = nn.Linear(self.hidden_dim, 8)
        self.au_classifier = AUClassifier(self.hidden_dim, 12)
        self.va_classifier = nn.Linear(self.hidden_dim, 2)

        # expr proj
        self.transformation_matrices = []
        for i_au in range(10):
            matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.transformation_matrices.append(matrix)
        self.transformation_matrices = nn.ModuleList(self.transformation_matrices)

        # learnable mask vectors
        self.point_mask = nn.Parameter(torch.zeros(cfg.TRAIN.BATCH_SIZE, 12))
        nn.init.xavier_uniform_(self.point_mask)

        # gcn_embedding
        self.graph_embedding = torch.nn.Sequential(GCN(self.hidden_dim, self.hidden_dim))
        self.ten_graph_embedding = torch.nn.Sequential(GCN_TEN(self.hidden_dim, self.hidden_dim, self.hidden_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        bs, _, channel = x.shape    # b, t, c
        outputs = {}
        if self.hidden_dim != 1024:
            x = self.input_proj(x.permute(0, 2, 1)).permute(0, 2, 1)

        pos_x = self.positional_encoding(x).permute(1, 0, 2) # t, 1, c
        x = x.permute(1, 0, 2) # t, b, c
        query_input = self.query_embed.weight  # t, c
        query_input = query_input.unsqueeze(1).repeat(1, bs, 1) # t, b, c
        tgt = torch.zeros_like(query_input)

        # blocks
        block1_output = self.trans1(tgt, x, query_pos = query_input, pos = pos_x)
        block2_output = self.trans2(block1_output, x, query_pos = query_input, pos = pos_x)
        block3_output = self.trans3(block2_output, x, query_pos = query_input, pos = pos_x)
        block4_output = self.trans4(block3_output, x, query_pos = query_input, pos = pos_x)

        query_output = block4_output

        query_output = query_output.permute(1, 2, 0) # b, c, t

        au_output = query_output[:, :, :12]
        outputs['AU'] = self.au_classifier(au_output.permute(0, 2, 1))

        # AU-GCN
        au_output = self.graph_embedding(au_output.permute(0, 2, 1)).permute(0, 2, 1)
        point_mask = F.softmax(self.point_mask, dim=-1)

        mask = torch.ones_like(point_mask)
        for i in range(point_mask.shape[0]):
            _, indices = torch.topk(point_mask[i], k=2, largest=False)
            mask[i, indices] = 0
        new_au_output = []
        for i in range(bs):
            batch_mask = mask[i, :]
            batch_au = au_output[i, :, :]
            new_batch_au = []
            for i in range(len(batch_mask)):
                if batch_mask[i] == 1:
                    new_batch_au.append(batch_au[:, i])
            new_batch_au = torch.stack(new_batch_au, dim=-1)
            new_au_output.append(new_batch_au)
        mask_au_output = torch.stack(new_au_output, dim=0)  # b, c, t

        mask_au_output = self.ten_graph_embedding(mask_au_output.permute(0, 2, 1)).permute(0, 2, 1)
        mask_au_output = mask_au_output + query_output[:, :, 12:]
        expr_va_output = []
        for i in range(10):
            mask_au = mask_au_output[:, :, i]
            projected = self.transformation_matrices[i](mask_au)
            expr_va_output.append(projected)
        expr_va_output = torch.stack(expr_va_output, dim=1)  # bs, numeber of regions, dim

        outputs['EXPR'] = self.expr_classifier(expr_va_output.mean(1))
        outputs['VA'] = self.va_classifier(expr_va_output.mean(1))

        return outputs

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Param:
        in_features, out_features, bias
    Input:
        features: N x C (n = # nodes), C = in_features
        adj: adjacency matrix (N x N)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        b, n, c = input.shape
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.adj = Adj(nfeat, 12)

    def forward(self, x):
        adj1 = self.adj(x)
        x = self.gc1(x, adj1)
        x = x.transpose(1, 2).contiguous()
        x = self.bn1(x).transpose(1, 2).contiguous()
        x = F.relu(x)

        return x

class GCN_TEN(nn.Module):
    def __init__(self, nfeat, nhid, nout):
        super(GCN_TEN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.bn2 = nn.BatchNorm1d(nout)
        self.adj = Adj(nfeat, 10)

    def forward(self, x):
        adj1 = self.adj(x)
        x = self.gc1(x, adj1)
        x = x.transpose(1, 2).contiguous()
        x = self.bn1(x).transpose(1, 2).contiguous()
        x = F.relu(x)

        adj2 = self.adj(x)
        x = self.gc2(x, adj2)
        x = x.transpose(1, 2).contiguous()
        x = self.bn2(x).transpose(1, 2).contiguous()
        x = F.relu(x)

        return x


class SA_CA_FFN(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.dim_feedforward = d_model * 4
        self.linear1 = nn.Linear(d_model, self.dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_pos: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt1 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt1)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt3 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt3)
        tgt = self.norm3(tgt)
        return tgt

class CA_FFN(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.dim_feedforward = d_model * 4
        self.linear1 = nn.Linear(d_model, self.dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_pos: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        tgt1 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout1(tgt1)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Adj(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(Adj, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        si = x.detach()
        si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
        threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
        adj = (si >= threshold).float()

        # Adj process
        A = normalize_digraph(adj)
        return A

def normalize_digraph(A):
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim = -1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix,A),norm_degs_matrix)
    return norm_A