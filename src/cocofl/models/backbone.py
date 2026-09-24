"""The full TFN model, adapted to ETHOS token sequences.

TFN's multimodal architecture (StaticEncoder, TimeAwareLSTM,
TimeAwareAttentionEncoder, fusion layer) is kept as is. For token input it adds a
token embedding before the LSTM encoder and replaces the time-series reconstruction
head with a vocabulary LM head.

Source: TFN_Temporal-Fusion-Nexus/src/models.py (T-LSTM from Baytas et al.).
"""
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# TFN CONFIG values, copied from TFN_Temporal-Fusion-Nexus/src/config.py
TFN_HIDDEN = 512
TFN_LAYERS = 2
TFN_NUM_HEADS = 2
TFN_STATIC_EMBEDDING_DIM = 16
TFN_STATIC_OUTPUT_DIM = 256


class TLSTMCell(nn.Module):
    """Time-aware LSTM cell from TFN. Verbatim copy."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_decomp = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_decomp = nn.Parameter(torch.Tensor(hidden_size))
        self.Wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Ui = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))
        self.Wf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))
        self.Wo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uo = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))
        self.Wc = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bc = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for w in self.parameters():
            nn.init.uniform_(w, -stdv, stdv)

    def map_elapse_time(self, t):
        t_clamped = torch.clamp(t, min=0.0)
        T = 1.0 / torch.log(t_clamped + 2.7183)
        return T.repeat(1, self.hidden_size)

    def forward(self, x_t, states, elapsed_time):
        h_prev, c_prev = states
        if elapsed_time.dim() == 1:
            elapsed_time = elapsed_time.unsqueeze(1)
        T = self.map_elapse_time(elapsed_time)
        C_ST = torch.tanh(c_prev @ self.W_decomp + self.b_decomp)
        c_prev_new = c_prev - C_ST + T * C_ST
        i_t = torch.sigmoid(x_t @ self.Wi + h_prev @ self.Ui + self.bi)
        f_t = torch.sigmoid(x_t @ self.Wf + h_prev @ self.Uf + self.bf)
        o_t = torch.sigmoid(x_t @ self.Wo + h_prev @ self.Uo + self.bo)
        g_t = torch.tanh(x_t @ self.Wc + h_prev @ self.Uc + self.bc)
        c_t = f_t * c_prev_new + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class TimeAwareLSTM(nn.Module):
    """Multi-layer T-LSTM from TFN. Verbatim copy."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList(
            [TLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )

    def forward(self, input_seq, elapsed_times, initial_states=None):
        batch_size, seq_length, _ = input_seq.size()
        device = input_seq.device
        if initial_states is None:
            h_t = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        else:
            h_t, c_t = zip(*[(h, c) for h, c in initial_states])
        outputs = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            delta_t = elapsed_times[:, t]
            for layer in range(self.num_layers):
                h, c = self.lstm_cells[layer](x, (h_t[layer], c_t[layer]), delta_t)
                h_t[layer], c_t[layer] = h, c
                x = h
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h_t, c_t)


class TimeAwareAttentionEncoder(nn.Module):
    """TFN's encoder: T-LSTM plus temporal self-attention. Takes ``input_size`` instead of the hard-coded CONFIG['ts_features']."""

    def __init__(self, input_size, hidden_size=TFN_HIDDEN, num_layers=TFN_LAYERS, num_heads=TFN_NUM_HEADS, use_temporal_attention=True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.use_temporal_attention = use_temporal_attention
        self.lstm = TimeAwareLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, x, elapsed_times, mask=None):
        lstm_out, _ = self.lstm(x, elapsed_times)
        batch_size, seq_len, _ = lstm_out.size()
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
        attn_weights = None
        if self.use_temporal_attention:
            attn_output, attn_weights = self.attention(
                query=lstm_out, key=lstm_out, value=lstm_out,
                key_padding_mask=key_padding_mask,
                need_weights=True, average_attn_weights=True,
                attn_mask=causal_mask, is_causal=True,
            )
            lstm_out = self.layer_norm_1(lstm_out + attn_output)
            ff_out = self.ff(lstm_out)
            lstm_out = self.layer_norm_2(lstm_out + ff_out)
        return lstm_out, attn_weights


class StaticEncoder(nn.Module):
    """TFN's static feature encoder. Verbatim copy."""

    def __init__(self, categorical_cardinalities, numerical_dim,
                 embedding_dim=TFN_STATIC_EMBEDDING_DIM, output_dim=TFN_STATIC_OUTPUT_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.numerical_dim = numerical_dim
        self.embeddings = nn.ModuleList([nn.Embedding(n, embedding_dim) for n in categorical_cardinalities])
        self.numerical_processor = nn.Sequential(nn.Linear(numerical_dim, embedding_dim), nn.ReLU())
        total = embedding_dim * len(categorical_cardinalities) + embedding_dim
        self.fc = nn.Linear(total, output_dim)
        self.relu = nn.ReLU()

    def forward(self, categorical_features, numerical_features):
        embedded = [emb(categorical_features[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        numerical = self.numerical_processor(numerical_features)
        combined = torch.cat([embedded, numerical], dim=1)
        return self.relu(self.fc(combined))


class TFNFullModel(nn.Module):
    """The full TFN architecture for ETHOS token sequences.

    TFN's multimodal model (static encoder, T-LSTM, temporal attention) with a token
    embedding at the input and a vocabulary LM head at the output (next-token
    prediction).
    """

    def __init__(self, vocab_size, n_embd=TFN_HIDDEN, n_layer=TFN_LAYERS,
                 n_positions=2048, categorical_cardinalities=(2, 6, 5),
                 numerical_dim=2, use_static=True, num_heads=TFN_NUM_HEADS):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_positions = n_positions
        self.use_static = use_static

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.lstm_encoder = TimeAwareAttentionEncoder(
            input_size=n_embd, hidden_size=n_embd, num_layers=n_layer, num_heads=num_heads,
        )
        if use_static:
            self.static_encoder = StaticEncoder(
                categorical_cardinalities=categorical_cardinalities, numerical_dim=numerical_dim,
            )
            self.static_fusion_layer = nn.Linear(n_embd + TFN_STATIC_OUTPUT_DIM, n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying, as in GPT-2
        self.wte.weight = self.lm_head.weight

        # Two optional adapter slots for the hierarchy (WP4):
        #   cohort_adapter   per centre (medium pace)
        #   patient_adapter  per patient (fast)
        # They are applied in that order in forward, each with its own residual, and are
        # created by ``cocofl.adapters.attach_ffn_adapter``. The population model itself is
        # not adapted in place, so there are exactly two.
        self.cohort_adapter = None
        self.patient_adapter = None

        self.config = SimpleNamespace(
            vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_positions=n_positions,
            categorical_cardinalities=list(categorical_cardinalities), numerical_dim=numerical_dim,
            use_static=use_static, num_heads=num_heads,
        )

    def num_parameters(self, exclude_embeddings=True):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.wte.weight.numel()
        return n

    def forward(self, input_ids, elapsed_times: Optional[torch.Tensor] = None,
                static_categorical: Optional[torch.Tensor] = None,
                static_numerical: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        B, T = input_ids.size()
        device = input_ids.device

        if elapsed_times is None:
            elapsed_times = torch.ones(B, T, device=device, dtype=torch.float32)
        x = self.wte(input_ids)  # (B, T, n_embd)
        lstm_out, _ = self.lstm_encoder(x, elapsed_times)  # (B, T, n_embd)

        if self.use_static:
            assert static_categorical is not None and static_numerical is not None, \
                "use_static=True requires static_categorical and static_numerical"
            static_encoding = self.static_encoder(static_categorical, static_numerical)  # (B, static_out)
            expanded = static_encoding.unsqueeze(1).expand(-1, lstm_out.size(1), -1)  # (B, T, static_out)
            fused = self.static_fusion_layer(torch.cat([lstm_out, expanded], dim=-1))
            lstm_out = F.relu(fused + lstm_out)

        # Adapters, if attached: cohort first, then patient on top.
        if self.cohort_adapter is not None:
            lstm_out = self.cohort_adapter(lstm_out)
        if self.patient_adapter is not None:
            lstm_out = self.patient_adapter(lstm_out)

        logits = self.lm_head(lstm_out)  # (B, T, vocab)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=lstm_out)
