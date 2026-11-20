"""ARCHIVED (2026-09-14): space-time POD-NN variant, superseded by the
POD-Attention-LSTM in `model_part1.py` at the user's direction (keep the
Attention-LSTM, add POD). Runs: `models_part1_podnn_hash_s{42,45,48}_r1`. To
re-evaluate them, copy this file over `model_part1.py` in a scratch checkout —
their checkpoints carry the space-time basis in their buffers.

KAN POD-ROM for the BS 8414 Part1 geometry-variant corpus.

Lineage
-------
Parent: `bs8414_KAN_surrogate/model_part1.py` (`Part1KANAttentionLSTM`, champion
`models_part1_r4_seed48`, combined TC R2 0.8526 on the 185-case corpus).

The change: the output representation. The parent generates 181 timesteps
through a time-encoded BiLSTM + self-attention backbone and per-group decoders.
This model is a non-intrusive POD reduced-order model — the trajectory lives in
a fixed r-dimensional POD space fitted on the training cases
(`pod_rom_part1.py`), and the network only has to map the build-up to r_tc = 32
TC and r_hrr = 16 HRR coefficients:

    params -> Part1ParameterEncoder (KANLinear x2, unchanged from parent)
           -> KANSensorDecoder -> c_tc  (+ linear skip)   -> V_tc  -> 16 TC x 181
           -> KANSensorDecoder -> c_hrr (+ linear skip)   -> V_hrr ->  5 HRR x 181
           -> hard ambient / zero-HRR clamps (unchanged from parent)

Disclosed consequence: removing the temporal backbone is part of the change, not
a separate one — in a POD ROM time is carried by the modes, so there is nothing
left for an LSTM to do. Parameter count drops accordingly (see `__main__`).

Why it was worth building: the projection floor of this basis is test R2 0.977,
combined valid+test 0.975, at r=32 (fit_pod_part1.py, scored with the frozen
evaluator's masked_r2) — well above every Part1 TC surrogate trained so far, so
truncation is not what would limit it.

Twin: `bs8414_MLP_pod_surrogate/model_part1.py` is identical except every
KANLinear block is an MLPLinear — the parents' KAN-vs-MLP ablation, carried into
the reduced space. Both twins load a byte-identical basis.

`train_part1.py` and `evaluate_part1.py` are byte-identical to the parents':
reconstruction happens inside `forward`, so the loss is still the frozen masked
MSE on the full standardised trajectories, not a coefficient-space loss.
"""
import torch
import torch.nn as nn

from config_part1 import (
    N_CLADDING, N_INSULATION, N_GEOMETRY, N_MATERIAL_FEATURES,
    CLADDING_EMBED_DIM, INSULATION_EMBED_DIM, GEOMETRY_EMBED_DIM,
    N_SENSORS, N_HRR_CHANNELS, N_TIMESTEPS, EMBEDDING_DIM, NUM_KNOTS,
    COL_CLADDING, COL_INSULATION, COL_GEOM,
)
from layers_part1 import KANLinear, KANSensorDecoder, kan_regularization
from pod_rom_part1 import PODReconstruction, R_TC, R_HRR


class Part1ParameterEncoder(nn.Module):
    """[cladding_id, insulation_id, geom_id, 13 material] -> embedding.

    Verbatim from the parent `bs8414_KAN_surrogate/model_part1.py`.
    """

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


class Part1KANPODROM(nn.Module):
    """Build-up -> POD coefficients -> TC and HRR trajectories."""

    def __init__(self, r_tc=R_TC, r_hrr=R_HRR, embedding_dim=EMBEDDING_DIM,
                 num_knots=NUM_KNOTS, basis_path=None):
        super().__init__()
        self.param_encoder = Part1ParameterEncoder(embedding_dim, num_knots)
        self.tc_coef_decoder = KANSensorDecoder(embedding_dim, r_tc, num_knots=num_knots)
        self.hrr_coef_decoder = KANSensorDecoder(embedding_dim, r_hrr, num_knots=num_knots)
        # Linear skips, the reduced-space counterpart of the parent's
        # param->output skip projections. They also restore a path to arbitrary
        # coefficient vectors, which the decoders' final LayerNorm constrains.
        self.tc_coef_skip = nn.Linear(embedding_dim, r_tc)
        self.hrr_coef_skip = nn.Linear(embedding_dim, r_hrr)
        self.rom = PODReconstruction(r_tc, r_hrr, path=basis_path)

    def set_output_scaling(self, tc_scaler, hrr_scaler, hrr_nonnegative_idx=(0,),
                           **kw):
        self.rom.set_output_scaling(tc_scaler, hrr_scaler, hrr_nonnegative_idx, **kw)

    def coefficients(self, params, time_array=None):
        e = self.param_encoder(params)
        return (self.tc_coef_decoder(e) + self.tc_coef_skip(e),
                self.hrr_coef_decoder(e) + self.hrr_coef_skip(e))

    def forward(self, params, time_array):
        if len(time_array) != N_TIMESTEPS:
            raise ValueError(f"a POD ROM reconstructs the fixed {N_TIMESTEPS}-step "
                             f"grid its modes were fitted on; got {len(time_array)}")
        c_tc, c_hrr = self.coefficients(params)
        tc, hrr = self.rom(c_tc, c_hrr)
        return tc, hrr, (c_tc, c_hrr)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── uniform interface (see bs8414_KAN_surrogate/model_part1.py) ───────────
MODEL_NAME = "KAN POD-ROM (Part1)"
Part1Surrogate = Part1KANPODROM
LAMBDA_REG = 2e-3  # parent's spline L2 + smoothness weight, unchanged


def regularization(model):
    return kan_regularization(model)


if __name__ == "__main__":
    import numpy as np
    from config_part1 import N_INPUT_PARAMS
    from data_loader_part1 import ChannelScaler
    from pod_rom_part1 import load_basis

    model = Part1Surrogate()
    print(f"{MODEL_NAME}: {count_parameters(model):,} trainable parameters "
          f"(parent Part1 KAN-Attention-LSTM: 850,765)")
    print(f"  r_tc={model.rom.r_tc}  r_hrr={model.rom.r_hrr}  "
          f"basis corpus {model.rom.basis_digest}")

    blob = load_basis()
    tc_scaler = ChannelScaler().load_state_dict(blob["tc_scaler"])
    hrr_scaler = ChannelScaler().load_state_dict(blob["hrr_scaler"])
    model.set_output_scaling(tc_scaler, hrr_scaler)

    p = torch.rand(4, N_INPUT_PARAMS)
    p[:, COL_CLADDING] = torch.randint(0, N_CLADDING, (4,)).float()
    p[:, COL_INSULATION] = torch.randint(0, N_INSULATION, (4,)).float()
    p[:, COL_GEOM] = torch.randint(0, N_GEOMETRY, (4,)).float()
    tc, hrr, (c_tc, c_hrr) = model(p, torch.linspace(0, 1, N_TIMESTEPS))
    print(f"  tc {tuple(tc.shape)}  hrr {tuple(hrr.shape)}  "
          f"coef {tuple(c_tc.shape)} {tuple(c_hrr.shape)}")
    print(f"  ambient clamp holds: {bool((tc >= model.rom.ambient_scaled - 1e-6).all())}")

    # The basis must reproduce a training case from its own coefficients.
    modes = model.rom.tc_modes.double()
    print(f"  basis orthonormality error: "
          f"{(modes @ modes.T - torch.eye(len(modes), dtype=torch.float64)).abs().max():.2e}")
