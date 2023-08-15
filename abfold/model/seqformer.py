import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from einops import rearrange

from abfold.model.common_modules import(
        Linear,
        LayerNorm,
        apply_dropout,)
from abfold.common import residue_constants

class EmbeddingAndSeqformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.num_token = residue_constants.restype_num + 3
        self.num_region = residue_constants.num_ab_regions + 1

        self.proj_aa_type = nn.Embedding(self.num_token, c.seq_channel, padding_idx=0)

        if c.abrep.enabled:
            abrep_embed_weights = torch.log(torch.tensor([(1-0.5)/c.abrep.num_layers] * c.abrep.num_layers + [0.5]))
            self.abrep_embed_weights = nn.Parameter(abrep_embed_weights)
            
            self.proj_abrep_embed = nn.Sequential(
                LayerNorm(c.abrep.embed_channel),
                Linear(c.abrep.embed_channel, c.seq_channel, init='linear', bias=True),
                nn.ReLU(),
                Linear(c.seq_channel, c.seq_channel, init='linear', bias=True),
                nn.Dropout(p=c.abrep.dropout_rate),
                )
            
            if c.abrep.pair_enabled:
                self.proj_abrep_embed_pair = Linear(c.abrep.embed_pair_channel, c.pair_channel, init='final', bias=False)

        if c.esm.enabled:
            esm_embed_weights = torch.zeros((c.esm.num_layers + 1,))
            #esm_embed_weights = torch.log(torch.tensor([(1-0.5)/c.esm.num_layers] * c.esm.num_layers + [0.5]))
            self.esm_embed_weights = nn.Parameter(esm_embed_weights)

            self.proj_esm_embed = nn.Sequential(
                LayerNorm(c.esm.embed_channel),
                Linear(c.esm.embed_channel, c.seq_channel, init='linear', bias=True),
                nn.ReLU(),
                Linear(c.seq_channel, c.seq_channel, init='linear', bias=True),
                #nn.Dropout(p=c.esm.dropout_rate),
                )

            if c.esm.pair_enabled:
                self.proj_esm_embed_pair = Linear(c.esm.embed_pair_channel, c.pair_channel, init='linear', bias=True)
        
        self.proj_rel_pos = torch.nn.Embedding(c.max_relative_feature * 2 + 2, c.pair_channel)

        if c.recycle_features:
            self.prev_seq_norm = LayerNorm(c.seq_channel)
            self.prev_pair_norm = LayerNorm(c.pair_channel)

        if c.recycle_pos:
            self.proj_prev_pos = nn.Embedding(c.prev_pos.num_bins, c.pair_channel)

        self.seqformer = Seqformer(c)

        self.config = config

    def forward(self, batch):
        c = self.config

        seq, mask= batch['seq'], batch['mask']

        batch_size, num_residue = seq.shape[:2]
       
        seq_act = self.proj_aa_type(seq)

        #if 'residx' in batch:
        seq_pos = batch['residx'][:,1:-1] - 1
        #else:
        #    seq_pos = torch.tile(torch.arange(num_residue, device=seq.device), [batch_size, 1])

        offset = rearrange(seq_pos, 'b l -> b () l') - rearrange(seq_pos, 'b l -> b l ()')
        rel_pos = torch.clip(offset + c.max_relative_feature, min=0, max=2*c.max_relative_feature) + 1
        pair_act = self.proj_rel_pos(rel_pos)

        if c.esm.enabled:
            layer_weights = F.softmax(self.esm_embed_weights, dim=-1)
            esm_embed = batch['esm_embed'].to(dtype=layer_weights.dtype)

            esm_embed = torch.einsum('b l c n, n -> b l c', esm_embed, layer_weights)
            
            esm_embed = self.proj_esm_embed(esm_embed)
            seq_act = seq_act + esm_embed

            if c.esm.pair_enabled and 'esm_embed_pair' in batch:
                pair_embed = self.proj_esm_embed_pair(batch['esm_embed_pair'])
                pair_act = pair_act + pair_embed
        
        if c.recycle_features:
            if 'prev_seq' in batch:
                seq_act = seq_act + self.prev_seq_norm(batch['prev_seq'])
            if 'prev_pair' in batch:
                pair_act = pair_act + self.prev_pair_norm(batch['prev_pair'])

        if c.recycle_pos and 'prev_pos' in batch:
            pair_act = pair_act + self.proj_prev_pos(batch['prev_pos'])

        seq_act, pair_act = self.seqformer(seq_act, pair_act, mask=mask, is_recycling=batch['is_recycling'])

        return seq_act, pair_act

class Attention(nn.Module):
    def __init__(self, input_dim, key_dim, value_dim, output_dim, num_head,
            split_first=True, 
            gating=True,
            inp_kernels=None):
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
         
        self.inp_kernels = inp_kernels
        if inp_kernels:
            self.inp_q = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_k = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_v = SpatialDepthWiseInception(value_dim // num_head, inp_kernels)

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
        
        if self.inp_kernels:
            q, k, v = map(lambda t: rearrange(t, 'b s h l d-> b (s h) l d'), (q, k, v))
            q = self.inp_q(q)
            k = self.inp_k(k)
            v = self.inp_v(v)
            q, k, v = map(lambda t: rearrange(t, 'b (s h) l d-> b s h l d', h = self.num_head), (q, k, v))
        
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
                split_first=False,
                inp_kernels=c.inp_kernels)

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

# AF2 and ESM-FOLD have different implementations
# Here we just follow ESMFOLD
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

        
        if c.inp_kernels:
            self.inp_left = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)
            self.inp_right = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)

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
        
        if c.inp_kernels:
            if c.orientation == 'per_row':
                equation = 'b i j (h d) -> b (i h) j d'
            else:
                equation = 'b i j (h d) -> b (j h) i d'

            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))

            left_proj_act = self.inp_left(left_proj_act)
            right_proj_act = self.inp_right(right_proj_act)
            
            if c.orientation == 'per_row':
                equation = 'b (i h) j d -> b i j (h d)'
            else:
                equation = 'b (j h) i d -> b i j (h d)'
            
            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))
        
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
                gating=c.gating,
                inp_kernels=c.inp_kernels)

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

class SeqformerIteration(nn.Module):
    def __init__(self, config, seq_channel, pair_channel):
        super().__init__()
        c = config

        self.seq_attn = SeqAttentionWithPairBias(c.seq_attention_with_pair_bias, seq_channel, pair_channel)
        self.seq_transition = Transition(c.seq_transition, seq_channel)
        self.outer_product_mean = OuterProductMean(c.outer_product_mean, seq_channel, pair_channel)
        
        self.triangle_multiplication_outgoing = TriangleMultiplication(c.triangle_multiplication_outgoing, pair_channel)
        self.triangle_multiplication_incoming = TriangleMultiplication(c.triangle_multiplication_incoming, pair_channel)
        self.triangle_attention_starting_node = TriangleAttention(c.triangle_attention_starting_node, pair_channel)
        self.triangle_attention_ending_node = TriangleAttention(c.triangle_attention_ending_node, pair_channel)
        self.pair_transition = Transition(c.pair_transition, pair_channel)

        self.config = config

    def forward(self, seq_act, pair_act, seq_mask):
        """
        seq_act: (b l c)
        pair_act: (b l l c)
        seq_mask: (b l)
        """
        c = self.config

        def dropout_fn(input_act, act, config):
            if self.training and config.dropout_rate > 0.:
                if config.shared_dropout:
                    if config.orientation == 'per_row':
                        broadcast_dim = 1
                    else:
                        broadcast_dim = 2
                else:
                    broadcast_dim = None
                act = apply_dropout(act, config.dropout_rate,
                        is_training=True, broadcast_dim=broadcast_dim)
            return input_act + act
        
        seq_act = dropout_fn(
                seq_act, self.seq_attn(seq_act, pair_act, seq_mask), c.seq_attention_with_pair_bias)
        seq_act = seq_act + self.seq_transition(seq_act, seq_mask)
        
        pair_act = pair_act + self.outer_product_mean(seq_act, seq_mask)
        
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_outgoing(pair_act, seq_mask), c.triangle_multiplication_outgoing)
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_incoming(pair_act, seq_mask), c.triangle_multiplication_incoming)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_starting_node(pair_act, seq_mask), c.triangle_attention_starting_node)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_ending_node(pair_act, seq_mask), c.triangle_attention_ending_node)
        pair_act = pair_act + self.pair_transition(pair_act, seq_mask)
        
        return seq_act, pair_act

class Seqformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.blocks = nn.ModuleList([SeqformerIteration(c.seqformer, c.seq_channel, c.pair_channel) for _ in range(c.seqformer_num_block)])

    def forward(self, seq_act, pair_act, mask, is_recycling=True):
        for it, block in enumerate(self.blocks):
            block_fn = functools.partial(block, seq_mask=mask)
            if self.training and not is_recycling and it > 0:
                seq_act, pair_act = checkpoint(block_fn, seq_act, pair_act)
                #seq_act, pair_act = block_fn(seq_act, pair_act)
            else:
                seq_act, pair_act = block_fn(seq_act, pair_act)

        return seq_act, pair_act

class SpatialDepthWiseConvolution(nn.Module):
    def __init__(self, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=head_dim, out_channels=head_dim,
                kernel_size=(kernel_size,),
                # padding=(kernel_size - 1,),
                padding=kernel_size//2,
                groups=head_dim)
    
    def forward(self, x: torch.Tensor):
        batch_size, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * heads, head_dim, seq_len)
        x = self.conv(x)
        #if self.kernel_size>1:
        #    x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, head_dim, seq_len)
        x = x.permute(0, 1, 3, 2)
        return x

class SpatialDepthWiseInception(nn.Module):
    def __init__(self, head_dim, kernels):
        super().__init__()
       
        assert len(kernels) > 1 and  kernels[0] == 1

        self.convs = torch.nn.ModuleList([SpatialDepthWiseConvolution(head_dim, kernel_size=k) for k in kernels[1:]])
        self.kernels = kernels
    def forward(self, x):
        # x: (batch, num_heads, len, head_dim)
        
        assert x.shape[1] % len(self.kernels) == 0
        group_num_head = x.shape[1] // len(self.kernels)
        
        outputs = [x[:,:group_num_head]]

        for i, conv in enumerate(self.convs):
            outputs.append(conv(x[:,group_num_head*(i+1):group_num_head*(i+2)]))

        outputs = torch.cat(outputs, dim=1)

        return outputs
