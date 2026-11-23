# NordicDat external replication: protocol fixed before any results
- Data: NordicDat (Zenodo 10964584), 91,455 rows. Target: `delay` (network ping, ms).
- Exclusions: delay <= 0 or > 2000 ms (invalid or time-out measurements); service_status == -1.
- Sessions (runs): a new session starts after a timestamp gap > 60 s; keep sessions with >= 500 rows.
- Features (pre-outcome telemetry only): rsrq, rssi, rsrp, sinr, velocity_abs, acceleration_abs, velocity_long,
  acceleration_lat, acceleration_long, heading, elevation, band, ran, operator.
  Excluded: latitude, longitude, serving_cell_id, timestamp, throughput (concurrent outcomes).
- Folds: 5 outer folds over whole sessions (GroupKFold order fixed by seed 2027). Within the outer-training
  sessions, the original pipeline functions are used unchanged (train/calibration group split, nested EC gate,
  OOF scale, CQR, RF predictor). The RSRP tertile objective uses rsrp.
- Methods: Standard CP, OOF normalized CP, CQR, gated EC-NCS (explanation rank), and gated scale on the
  predicted-delay rank (same gate, bins and grid).
- Primary endpoint: mean per-session Winkler loss at alpha = 0.1.
  Primary test: EC-NCS vs Standard CP (paired Wilcoxon over test sessions).
  Secondary test: EC-NCS vs predicted-delay rank.
- No admission deadline (ping delay has a different meaning from CICV5G one-way delay).
- Pipeline compatibility: NetworkMode = ran (LTE / 5G-NSA); FrequencyHz = constant (single ping rate).
- Scenario = constant 'NordicDat' (needed by the split function).

## Protocol log (added after the run)
- Fixed before any result: data exclusions, the 60 s session rule, features, methods, primary and secondary endpoints and tests.
- Three compatibility settings were added after runs that crashed in the pipeline's split function, before any interval or metric had been computed:
  NetworkMode = ran, FrequencyHz = constant, Scenario = constant. They only label the sessions for the split's stratification; none uses the outcome.
- NordicDat was not used for any earlier development, tuning or gate choice.
- Tests: raw (unadjusted) two-sided paired Wilcoxon over sessions; scipy default drops zero differences (6 ties -> n = 24).
  Robustness (post hoc): zsplit ties p = 0.0005; sign test 19/5 p = 0.0066; day-clustered bootstrap of the relative change
  -12.6% [-17.0%, -5.5%]; leave-one-day-out -10.8% to -14.1%, all p <= 0.0045.
- Caveat: many consecutive sessions are separated by only 1-3 min and are likely parts of the same drive; the day-level analysis addresses this.

## Confirmatory test 2: few sessions for learning the error scale (fixed before running)
- Hypothesis carried over from CICV5G: with 2 or 4 sessions to learn the error scale, EC-NCS has a lower Winkler loss than OOF normalized CP and CQR.
- RF predictor fixed (all outer-training sessions); calibration and test sessions unchanged; 3 random draws of sessions per fold.
- EC-NCS: five bins, shrinkage 1, cap 2, no gate (same as CICV5G). Predicted-delay rank with the same bins is reported alongside.
- OOF: ExtraTrees(200, leaf 10) on |OOF residual|. CQR: GBR quantiles 0.05/0.95 (250 trees, depth 3, lr 0.04).
- Primary: mean per-session Winkler loss; paired Wilcoxon over (draw, session) pairs and over sessions.
