import torch.nn as nn
import torch.nn.functional as F
import torch
import math



class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, activation='softmax', top_k=0, transform=True, n_type=1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.n_type = n_type
        if n_type == 1:
            self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        else:
            self.query_linear = nn.Linear(d_model, d_model, bias=False)
            self.key_linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for i in range(self.n_type)])
            self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(n_heads=h, activation=activation, top_k=top_k)
        self.transform = transform

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        if self.transform:
            if self.n_type==1:
                query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                                     for l, x in zip(self.linear_layers, (query, key, value))]
            else:
                query = self.query_linear(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                key = [key_linear(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for key_linear in self.key_linears]
                value = self.value_linear(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn, score = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.dropout(self.output_linear(self.dropout(x)))

    def get_attention(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn, score = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        return attn

    def get_score(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn, score = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        return score

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, n_heads=1, activation='softmax', top_k=0):
        self.n_heads = n_heads
        super().__init__()
        self.activation_type = activation
        if activation=='softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation=='sigmoid':
            self.activation = nn.Sigmoid()
        elif activation=='tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None
        #self.top_k = top_k

    def forward(self, query, key, value, mask=None, dropout=None):
        if type(key)==list:
            scores = [torch.matmul(query, key_i.transpose(-2, -1)) \
                     / torch.sqrt(torch.tensor(query.size(-1)))
                      for key_i in key]
            if mask is not None:
                score_sum = []
                for score, mask_i in zip(scores, mask):
                    weight, mask_to_fill = mask_i
                    mask_to_fill = mask_to_fill == 0
                    mask_to_fill = mask_to_fill.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
                    if (self.activation_type=='softmax') or (self.activation_type=='sigmoid'):
                        score_sum.append(score.masked_fill(mask_to_fill, -1e9))
                    else:
                        score_sum.append(score.masked_fill(mask_to_fill, 0))
                scores = sum(score_sum)
            else:
                scores = sum(scores)
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) \
                     / torch.sqrt(torch.tensor(query.size(-1)))
            if mask is not None:
                mask = mask == 0
                mask = mask.unsqueeze(1).expand(-1, self.n_heads, - 1, -1)
                scores = scores.masked_fill(mask, -1e9)
            else:
                mask = torch.ones_like(scores, dtype=torch.float32)
                if dropout is not None:
                    mask = dropout(mask)
                mask = mask == 0
                scores = scores.masked_fill(mask, -1e9)
        #print(mask.shape)
        '''
        if self.top_k!=0:
            top_k_values, top_k_indices = torch.topk(scores, k=self.top_k, dim=-1)
            top_k_mask = torch.stack([torch.stack([score_ij.lt(top_k_ij[:, -1])
                                                   for score_ij, top_k_ij in zip(score_i, top_k_i)], dim=0)
                                      for score_i, top_k_i in zip(scores, top_k_values)], dim=0)

            scores = scores.masked_fill(top_k_mask, -1e9)
        '''
        if self.activation:
            p_attn = self.activation(scores)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn, scores


class FewShotAttention(Attention):

    def __init__(self, d_model, n_heads=1, activation='softmax', dropout=0.2):
        super(FewShotAttention, self).__init__(n_heads=n_heads, activation=activation)
        self.d_k = d_model // self.n_heads
        self.query_linears = nn.ModuleList([ReluLinear(self.d_k, int(self.d_k/4), bias=False) for _ in range(self.n_heads)]) #
        self.key_linears = nn.ModuleList([ReluLinear(self.d_k, int(self.d_k/4), bias=False) for _ in range(self.n_heads)]) # to reduce the number of params
        #self.query_norm = nn.LayerNorm(d_model)
        #self.key_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None, dropout=None, transform=True):
        batch_size = query.size(0)
        #query = self.query_norm(query)
        #key = self.key_norm(key)
        query = query.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        if transform:
            query = [linear(query[:, i, :, :]) for i, linear in enumerate(self.query_linears)]
            key = [linear(key[:, i, :, :]) for i, linear in enumerate(self.key_linears)]

            if self.n_heads!=1:
                query = torch.stack(query, dim=1)
                key = torch.stack(key, dim=1)
            else:
                query = query[0].unsqueeze(1)
                key = key[0].unsqueeze(1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / torch.sqrt(torch.tensor(query.size(-1)))
        mask = torch.ones_like(scores)
        mask = self.dropout(mask)
        scores = scores.masked_fill(mask==0, -1e9)
        p_attn = self.activation(scores)

        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return x, p_attn, scores

class ReluLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.relu = nn.ReLU()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        weight = self.relu(weight)
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
