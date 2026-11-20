"""POD-Attention-LSTM (KAN) for the BS 8414 Part1 geometry-variant corpus.

Lineage
-------
Parent: `bs8414_KAN_surrogate/model_part1.py` (`Part1KANAttentionLSTM`).

Kept, verbatim: the parameter encoder (cladding/insulation/geometry embeddings
+ 13 material features through KANLinear), the sinusoidal time encoding,
input projection, MultiScaleConv (k=3/9/27), the 2-layer BiLSTM(96), 4-head
self-attention + LayerNorm, the KANSensorDecoder blocks, the param->output skip
projections, and the hard ambient / zero-HRR clamps.

The change: the output representation. The parent decodes the 16 thermocouples
directly, through two grouped decoders (External LV1, LV2), and the 5 HRR channels
through a third. Here the same backbone features decode to **POD coefficients at
every timestep**, and fixed sensor-wise POD modes fitted on the training frames
(`pod_rom_part1.SpatialPODReconstruction`) rebuild the channels:

    features (B, 181, 192) -> KANSensorDecoder -> a_tc(t)  (B, 181, 8) -> Phi_tc  (16 x 8) -> 16 TC
                           -> KANSensorDecoder -> a_hrr(t) (B, 181, 4) -> Phi_hrr ( 5 x 4) ->  5 HRR
    + param skips into the coefficients, then the parent's clamps

This is the POD-LSTM form of a reduced-order model: fixed spatial structures,
temporal coefficients from a recurrent network, the q ~ qbar + Phi a(t) of the
CYPHER POD-Galerkin submission with the Galerkin operator replaced by the
Attention-LSTM.

Disclosed consequences of that one change:
  * The two grouped TC decoders become one coefficient decoder, because the modes
    mix both levels; the parameter count falls slightly (see `__main__`).
  * Outputs are confined to the span of the r leading sensor modes. Measured
    projection floor: TC r=8 combined valid+test R2 0.981; HRR r=4 1.000 (4 d.p.).
    The HRR head therefore carries the budget's closure structure in its modes.

Twin: `bs8414_MLP_pod_surrogate/model_part1.py` is identical except that every
KANLinear block is an MLPLinear. That is the parents' ablation, carried over to
the POD head. Both twins load a byte-identical spatial basis.
`train_part1.py` and `evaluate_part1.py` are byte-identical to the parents'.

The earlier space-time POD-NN variant (no backbone) is archived as
`model_part1_podnn.py` with its runs in `models_part1_podnn_*`.
"""
import torch
import torch.nn as nn

from config_part1 import (
    N_CLADDING, N_INSULATION, N_GEOMETRY, N_MATERIAL_FEATURES,
    CLADDING_EMBED_DIM, INSULATION_EMBED_DIM, GEOMETRY_EMBED_DIM,
    N_SENSORS, N_HRR_CHANNELS, N_TIMESTEPS,
    EMBEDDING_DIM, LSTM_HIDDEN_SIZE, ATTENTION_HEADS, DROPOUT, NUM_KNOTS,
    COL_CLADDING, COL_INSULATION, COL_GEOM,
)
from layers_part1 import (KANLinear, TimeEncoding, MultiScaleConv, KANSensorDecoder,
                          kan_regularization)
from pod_rom_part1 import SpatialPODReconstruction, R_TC_SPATIAL, R_HRR_SPATIAL


class Part1ParameterEncoder(nn.Module):
    """[cladding_id, insulation_id, geom_id, 13 material] -> embedding. Verbatim from the parent."""

    def __init__(self, output_dim=EMBEDDING_DIM, num_knots=NUM_KNOTS):
        super().__init__()
        self.cladding_embedding = nn.Embedding(N_CLADDING, CLADDING_EMBED_DIM)
        self.insulation_embedding = nn.Embedding(N_INSULATION, INSULATION_EMBED_DIM)
        self.geometry_embedding = nn.Embedding(N_GEOMETRY, GEOMETRY_EMBED_DIM)

        input_dim = (CLADDING_EMBED_DIM + INSULATION_EMBED_DIM
                     + GEOMETRY_EMBED_DIM + N_MATERIAL_FEATURES)
        self.kan1 = KANLinear(input_dim, 48, num_knots=num_knots)
        self.kan2 = KANLinear(48, output_dim, num_knots=num_knots)
        self.output_dim = output_dim

    def forward(self, params):
        clad = self.cladding_embedding(params[:, COL_CLADDING].long())
        ins = self.insulation_embedding(params[:, COL_INSULATION].long())
        geom = self.geometry_embedding(params[:, COL_GEOM].long())
        material = params[:, 3:]
        x = torch.cat([clad, ins, geom, material], dim=-1)
        return self.kan2(self.kan1(x))


class Part1KANPODAttentionLSTM(nn.Module):
    """Parent Attention-LSTM backbone, POD-coefficient heads."""

    def __init__(self, r_tc=R_TC_SPATIAL, r_hrr=R_HRR_SPATIAL,
                 hidden_size=LSTM_HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM,
                 n_heads=ATTENTION_HEADS, dropout=DROPOUT, num_knots=NUM_KNOTS,
                 basis_path=None):
        super().__init__()
        self.n_sensors = N_SENSORS
        self.n_hrr_channels = N_HRR_CHANNELS

        # ── backbone, verbatim from the parent ────────────────────────────
        self.param_encoder = Part1ParameterEncoder(output_dim=embedding_dim,
                                                   num_knots=num_knots)
        self.time_encoding = TimeEncoding(d_model=embedding_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_size), nn.GELU(), nn.Dropout(dropout),
        )
        self.multi_scale = MultiScaleConv(hidden_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=2, batch_first=True, bidirectional=True, dropout=dropout,
        )
        lstm_out = hidden_size * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out)

        # ── POD head (the change) ─────────────────────────────────────────
        self.tc_coef_decoder = KANSensorDecoder(lstm_out, r_tc, num_knots=num_knots)
        self.hrr_coef_decoder = KANSensorDecoder(lstm_out, r_hrr, num_knots=num_knots)
        self.tc_coef_skip = nn.Linear(embedding_dim, r_tc)
        self.hrr_coef_skip = nn.Linear(embedding_dim, r_hrr)
        self.rom = SpatialPODReconstruction(r_tc, r_hrr, path=basis_path)

    def set_output_scaling(self, tc_scaler, hrr_scaler, hrr_nonnegative_idx=(0,), **kw):
        self.rom.set_output_scaling(tc_scaler, hrr_scaler, hrr_nonnegative_idx, **kw)

    def coefficients(self, params, time_array):
        """-> a_tc (B, T, r_tc), a_hrr (B, T, r_hrr), attention weights."""
        B, T = params.shape[0], len(time_array)
        param_embed = self.param_encoder(params)
        time_embed = self.time_encoding(time_array)
        combined = torch.cat([
            param_embed.unsqueeze(1).expand(-1, T, -1),
            time_embed.unsqueeze(0).expand(B, -1, -1),
        ], dim=-1)

        x = self.input_proj(combined)
        x = x + self.multi_scale(x)
        lstm_out, _ = self.lstm(x)
        attn_out, attn_w = self.attention(lstm_out, lstm_out, lstm_out)
        features = self.attn_norm(lstm_out + attn_out)

        a_tc = (self.tc_coef_decoder(features)
                + self.tc_coef_skip(param_embed).unsqueeze(1).expand(-1, T, -1))
        a_hrr = (self.hrr_coef_decoder(features)
                 + self.hrr_coef_skip(param_embed).unsqueeze(1).expand(-1, T, -1))
        return a_tc, a_hrr, attn_w

    def forward(self, params, time_array):
        a_tc, a_hrr, attn_w = self.coefficients(params, time_array)
        tc, hrr = self.rom(a_tc, a_hrr)
        return tc, hrr, attn_w


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── uniform interface (see bs8414_KAN_surrogate/model_part1.py) ───────────
MODEL_NAME = "KAN POD-Attention-LSTM (Part1)"
Part1Surrogate = Part1KANPODAttentionLSTM
LAMBDA_REG = 2e-3  # parent's spline L2 + smoothness weight, unchanged


def regularization(model):
    return kan_regularization(model)


if __name__ == "__main__":
    from config_part1 import N_INPUT_PARAMS
    from data_loader_part1 import ChannelScaler
    from pod_rom_part1 import load_spatial_basis

    model = Part1Surrogate()
    print(f"{MODEL_NAME}: {count_parameters(model):,} trainable parameters "
          f"(parent Part1 KAN-Attention-LSTM: 850,765)")
    print(f"  r_tc={model.rom.r_tc}  r_hrr={model.rom.r_hrr}  "
          f"basis corpus {model.rom.basis_digest}")

    blob = load_spatial_basis()
    model.set_output_scaling(ChannelScaler().load_state_dict(blob["tc_scaler"]),
                             ChannelScaler().load_state_dict(blob["hrr_scaler"]))
    p = torch.rand(4, N_INPUT_PARAMS)
    p[:, COL_CLADDING] = torch.randint(0, N_CLADDING, (4,)).float()
    p[:, COL_INSULATION] = torch.randint(0, N_INSULATION, (4,)).float()
    p[:, COL_GEOM] = torch.randint(0, N_GEOMETRY, (4,)).float()
    tc, hrr, attn = model(p, torch.linspace(0, 1, N_TIMESTEPS))
    a_tc, a_hrr, _ = model.coefficients(p, torch.linspace(0, 1, N_TIMESTEPS))
    print(f"  tc {tuple(tc.shape)}  hrr {tuple(hrr.shape)}  "
          f"a_tc {tuple(a_tc.shape)}  a_hrr {tuple(a_hrr.shape)}  attn {tuple(attn.shape)}")
    print(f"  ambient clamp holds: {bool((tc >= model.rom.ambient_scaled - 1e-6).all())}")
    m = model.rom.tc_modes.double()
    print(f"  sensor-mode orthonormality error: "
          f"{(m @ m.T - torch.eye(len(m), dtype=torch.float64)).abs().max():.2e}")
