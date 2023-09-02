import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from carbonmatrix.model.common_modules import Linear, LayerNorm

class Attention(nn.Module):
    def __init__(self, input_dim, key_dim, value_dim, output_dim, num_head,
            split_first=True, gating=True):
        super().__init__()
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0

        self.key_dim, self.value_dim = key_dim, value_dim

        self.num_head = num_head

        self.split_first = split_first

        if self.split_first:
            self.proj_q = Linear(input_dim, key_dim, init='attn', bias=False)
            self.proj_k = Linear(input_dim, key_dim, init='attn', bias=False)
            self.proj_v = Linear(input_dim, value_dim, init='attn', bias=False)
        else:
            assert (key_dim == value_dim)
            self.proj_in = Linear(input_dim, key_dim * 3, init='attn', bias=False)

        self.gating = gating
        if gating:
            self.gate= Linear(input_dim, value_dim, init='gate')

        self.proj_out = Linear(value_dim, output_dim, init='final')

    def forward(self, q_data, k_data=None, bias=None, k_mask=None):
        """
        Arguments:
            q_data: (batch_size, N_seqs, N_queries, q_channel)
            k_data: (batch_size, N_seqs, N_keys, k_channel)
            k_mask: (batch_size, N_seqs, N_keys)
            bias  : (batch_size, N_queries, N_keys). shared by all seqs
        Returns:
            (b s l c)
        """
        key_dim, value_dim = self.key_dim // self.num_head, self.value_dim // self.num_head

        if self.split_first:
            assert (k_data is not None)
            q = self.proj_q(q_data)
            k = self.proj_k(k_data)
            v = self.proj_v(k_data)
            q, k, v = map(lambda t: rearrange(t, 'b s l (h d) -> b s h l d', h = self.num_head), (q, k, v))
        else:
            assert (k_data is None)
            t = rearrange(self.proj_in(q_data), "... l (h d) -> ... h l d", h=self.num_head)
            q, k, v = torch.chunk(t, 3, dim=-1)

        q = q* key_dim**(-0.5)

        logits = torch.einsum('... h q d, ... h k d -> ... h q k', q, k)

        if bias is not None:
            logits = logits + rearrange(bias,  'b h q k -> b () h q k')

        if k_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            k_mask = rearrange(k_mask, 'b s k -> b s () () k')
            logits = logits.masked_fill(~k_mask.bool(), mask_value)

        weights = F.softmax(logits, dim = -1)
        weighted_avg = torch.einsum('b s h q k, b s h k d -> b s h q d', weights, v)
        weighted_avg = rearrange(weighted_avg, 'b s h q d -> b s q (h d)')

        if self.gating:
            gate_values = torch.sigmoid(self.gate(q_data))
            weighted_avg = weighted_avg * gate_values

        output = self.proj_out(weighted_avg)

        return output

class SeqAttentionWithPairBias(nn.Module):
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()
        c = config

        self.seq_norm = LayerNorm(num_in_seq_channel)
        self.pair_norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False)

        self.attn = Attention(
                input_dim=num_in_seq_channel,
                key_dim=num_in_seq_channel,
                value_dim=num_in_seq_channel,
                output_dim=num_in_seq_channel,
                num_head=c.num_head,
                split_first=False)
        self.config = config

    def forward(self, seq_act, pair_act, mask):
        """
        Arguments:
            seq_act: (b l c)
            pair_act: (b l l c)
            mask: (b l), padding mask
        Returns:
            (b l c)
        """
        mask = rearrange(mask, 'b l -> b () l')
        seq_act = self.seq_norm(seq_act)

        pair_act = self.pair_norm(pair_act)
        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')

        seq_act = rearrange(seq_act, 'b l c -> b () l c')
        seq_act = self.attn(q_data=seq_act, bias=bias, k_mask=mask)
        seq_act = rearrange(seq_act, 'b s l c -> (b s) l c')
        return seq_act

class Transition(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()

        c = config

        intermediate_channel = num_in_channel * c.num_intermediate_factor
        self.transition = nn.Sequential(
                LayerNorm(num_in_channel),
                Linear(num_in_channel, intermediate_channel, init='linear'),
                nn.ReLU(),
                Linear(intermediate_channel, num_in_channel, init='final'),
                )

    def forward(self, act, mask):
        return self.transition(act)

# AlphaFold and ESMFold have different implementations
# Here we follow ESMFold.
class OuterProductMean(nn.Module):
    def __init__(self, config, num_in_channel, num_out_channel):
        super().__init__()

        c = config
        self.norm = LayerNorm(num_in_channel)
        self.left_proj = Linear(num_in_channel, c.num_outer_channel, init='linear')
        self.right_proj = Linear(num_in_channel, c.num_outer_channel, init='linear')

        self.out_proj = Linear(2 * c.num_outer_channel, num_out_channel, init='final')

    def forward(self, act, mask):
        """
        act: (b l c)
        mask: (b l)
        """
        mask = rearrange(mask, 'b l -> b l ()')
        act = self.norm(act)
        left_act = mask * self.left_proj(act)
        right_act = mask * self.right_proj(act)

        #act = rearrange(left_act, 'b l c -> b l () c ()') * rearrange(right_act, 'b l c -> b  () l () c')
        #act = torch.einsum('b i c, b j d -> b i j c d', left_act, right_act)
        #act = rearrange(act, 'b i j c d -> b i j (c d)')

        prod = left_act[:, None, :, :] * right_act[:, :, None, :]
        diff = left_act[:, None, :, :] - right_act[:, :, None, :]

        act = torch.cat([prod, diff], dim=-1)
        act = self.out_proj(act)

        return act

class TriangleMultiplication(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        assert c.orientation in ['per_row', 'per_column']

        self.norm = LayerNorm(num_in_channel)

        self.left_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear')
        self.right_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear')

        self.final_norm = LayerNorm(c.num_intermediate_channel)

        if c.gating:
            self.left_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate')
            self.right_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate')
            self.final_gate = Linear(num_in_channel, num_in_channel, init='gate')

        self.proj_out = Linear(c.num_intermediate_channel, num_in_channel, init='final')


        self.config = c

    def forward(self, act, mask):
        """
        act: (b l l c)
        mask: (b l)
        """
        c = self.config

        #pair_mask = rearrange(mask, 'b l -> b l () ()') * rearrange(mask, 'b l -> b () l ()')
        pair_mask = mask[:,:,None,None] * mask[:,None,:,None]

        act = self.norm(act)

        input_act = act

        left_proj_act = self.left_proj(act)
        right_proj_act = self.right_proj(act)

        left_proj_act = pair_mask * left_proj_act
        right_proj_act = pair_mask * right_proj_act

        if c.gating:
            left_gate_values = torch.sigmoid(self.left_gate(act))
            right_gate_values = torch.sigmoid(self.right_gate(act))

            left_proj_act = left_proj_act * left_gate_values
            right_proj_act = right_proj_act * right_gate_values

        if c.orientation == 'per_row':
            act = torch.einsum('b i k c, b j k c -> b i j c', left_proj_act, right_proj_act)
        elif c.orientation == 'per_column':
            act = torch.einsum('b k i c, b k j c -> b i j c', left_proj_act, right_proj_act)
        else:
            raise NotImplementedError(f'{self.orientation} not Implemented')

        act = self.final_norm(act)
        act = self.proj_out(act)

        if c.gating:
            gate_values = torch.sigmoid(self.final_gate(input_act))
            act = act * gate_values

        return act

class TriangleAttention(nn.Module):
    def __init__(self, config, num_in_pair_channel):
        super().__init__()
        c = config

        assert c.orientation in ['per_row', 'per_column']

        self.norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False)
        self.attn = Attention(
                input_dim=num_in_pair_channel,
                key_dim=num_in_pair_channel,
                value_dim=num_in_pair_channel,
                output_dim=num_in_pair_channel,
                num_head=c.num_head,
                gating=c.gating)

        self.config = config

    def forward(self, pair_act, seq_mask):
        '''
        pair_act: (b l l c)
        seq_mask: (b l)
        '''
        c = self.config
        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        pair_act = self.norm(pair_act)
        seq_mask = rearrange(seq_mask, 'b l -> b () l')

        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')

        pair_act = self.attn(q_data=pair_act, k_data=pair_act, bias=bias, k_mask=seq_mask)

        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        return pair_act
