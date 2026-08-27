class FunctionTypes:
    LOSS = "loss"
    ESTIMATING = "estimating"


class SandwichFormationMethods:
    BREAD_T_QR = "bread_T_qr"
    MEAT_SVD_SOLVE = "meat_svd_solve"
    NAIVE = "naive"


# Auto-mode heuristic for post_deployment_analysis.resolve_jacobian_row_chunk_size
# (jacobian_row_chunk_size=None). HONESTY NOTE: out_dim (= num_updates *
# beta_dim + theta_dim) is a single-variable PROXY for the true backward-pass
# peak-memory driver -- the per-cotangent backward-graph footprint also grows
# with num_subjects x num_updates x per-subject history length -- and these
# numbers were calibrated on exactly ONE study shape (oralytics, beta_dim=135,
# 70 subjects, 31 updates) on ONE 24GB machine. The explicit
# jacobian_row_chunk_size parameter is the escape hatch in both directions
# (smaller to use less memory, larger or 0 to go faster when memory is
# plentiful); see resolve_jacobian_row_chunk_size's docstring.
#
# Empirical anchors (real oralytics study, beta_dim=135, 24GB Mac):
# - out_dim ~28-56 (this repo's own benchmark fixtures): unchunked is fine and
#   the jitted chunk path measured ~2.4x SLOWER (compile dominates) -- so auto
#   must stay unchunked at small out_dim or the benchmarks regress.
# - out_dim ~1500: UNCHUNKED crashed the machine; chunk 64 (budget 96k) worked.
# - out_dim ~2000: chunk 64 (budget 128k) worked.
# - out_dim ~3100: chunk 64 (budget ~198k) CRASHED; chunk 16 (budget ~50k)
#   worked, peak RSS ~2.1GB.
# - out_dim ~4185: chunk 16 (budget ~67k) worked, peak RSS ~3.1GB.
# So the crash boundary for budget = chunk_size * out_dim sits somewhere in
# (128k, 198k]; 65,536 is roughly one-third of the crash point and at-or-below
# every verified-safe budget in the dangerous out_dim >= 3100 regime --
# conservative on purpose: memory headroom matters more than the last 10% of
# speed. Nothing was measured between out_dim 56 and 1500 (unchunked fine vs.
# fatal), so the unchunked threshold takes the conservative end of that gap;
# the worst case just above it is one ~1.4-1.8s jit compile (plus one for a
# remainder chunk) on problems whose backward pass already costs seconds. The
# 64 cap pins the just-above-threshold range to the largest chunk ever
# verified safe (64, at out_dim ~1500).
JACOBIAN_AUTO_UNCHUNKED_MAX_OUT_DIM = 512
JACOBIAN_AUTO_ROW_BUDGET = 65536
JACOBIAN_AUTO_MAX_CHUNK = 64


class DiagnosticClassifications:
    SUPPORTED = "supported"
    LOCALLY_SUPPORTED = "locally_supported"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"


class CheckStatuses:
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
