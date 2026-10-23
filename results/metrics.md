# results/metrics.md — drift benchmark, tracked per phase

## Days 2–5 — open-loop double integration (synthetic 120 s drive, `SYNTH-drive1.csv`)

| Phase / Day | Method | Final position error (m) | Drift rate (m/min) | RMSE (m) |
|---|---|---|---|---|
| Day 2 | Naive double-integration (zero correction) | 311.07 | 155.67 | 137.42 |
| Day 3 | Gravity-compensated + bias-corrected double integration | -233.25 | -116.72 | 106.24 |
| Day 4 | Mount-aligned (vehicle-frame) double integration | -233.79 | -116.99 | 106.49 |
| Day 5 | Mount-aligned + denoised (1.0 Hz low-pass) double integration | -233.59 | -116.89 | 106.38 |

Recomputed in the Day 8 audit with the current code. The values originally
logged (345.49 / -199.74 / -199.76 / -199.56 m) came from integrators that
set `dt[0] = dt[1]`, i.e. integrated a first step that never happened;
that was later fixed to `dt[0] = 0`, which alone moves the synthetic final
error by ~33 m. Day 4 now uses the GPS-interval heading estimator (1.2 deg
on this drive, whose true mount yaw is 0). On this drive calibration cuts
the final error by about a quarter (311 -> 233 m), and mount alignment and
denoising change almost nothing -- consistent with the Day 5 finding that
open-loop drift is driven by near-DC error, not high-frequency noise.

The earlier Day 8 rows (S-S1 whole-drive integration: 47,881 m hard ZUPT,
1,881,072 m no ZUPT) were removed in the Day 8 audit. They were produced
with GPS speed read as km/h when the file stores m/s, a broken heading
estimate and synthetic-tuned ZUPT thresholds, and their columns did not
match this table's header. See PROGRESS.md, "Day 8 audit".

The first outage-benchmark version (hold 46/123/283 m, INS+ZUPT 90/257/605 m
at 30/60/120 s) was also replaced: it calibrated each drive on its WHOLE
recording (including GPS from inside the simulated outage) and took the
entry speed from GPS interpolated towards the next fix. The tables below
use only information available before each outage, and wheel-speed truth
where a vehicle file exists. See PROGRESS.md, "Day 8 audit, part 3".

## Day 8 — simulated GPS-outage benchmark (real IO-VNBD drives)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 8 | Hold last GPS fix (baseline), 30s outage | 30 | 56.4 | 13.3 | 85.5 | 855 | 57 |
| Day 8 | Strapdown INS, no ZUPT, 30s outage | 30 | 104.3 | 25.3 | 136.6 | 855 | 57 |
| Day 8 | Strapdown INS + adaptive ZUPT, 30s outage | 30 | 106.5 | 26.6 | 143.8 | 855 | 57 |
| Day 8 | Hold last GPS fix (baseline), 60s outage | 60 | 151.1 | 17.7 | 206.6 | 855 | 57 |
| Day 8 | Strapdown INS, no ZUPT, 60s outage | 60 | 285.7 | 37.5 | 420.0 | 855 | 57 |
| Day 8 | Strapdown INS + adaptive ZUPT, 60s outage | 60 | 285.7 | 40.2 | 433.2 | 855 | 57 |
| Day 8 | Hold last GPS fix (baseline), 120s outage | 120 | 314.0 | 20.1 | 444.2 | 855 | 57 |
| Day 8 | Strapdown INS, no ZUPT, 120s outage | 120 | 767.1 | 48.4 | 1246.1 | 855 | 57 |
| Day 8 | Strapdown INS + adaptive ZUPT, 120s outage | 120 | 783.5 | 49.7 | 1257.6 | 855 | 57 |

## Day 9 — ML velocity during simulated GPS outages (TEST drives only)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 9 | Hold last GPS fix (baseline), 30s outage | 30 | 74.3 | 9.3 | 86.0 | 240 | 6 |
| Day 9 | Strapdown INS + adaptive ZUPT, 30s outage | 30 | 107.2 | 18.4 | 122.4 | 240 | 6 |
| Day 9 | ML velocity model, 30s outage | 30 | 86.7 | 19.6 | 167.4 | 240 | 6 |
| Day 9 | Anchored ML (entry correction, 20 s fade), 30s outage | 30 | 74.2 | 13.9 | 104.2 | 240 | 6 |
| Day 9 | Hold last GPS fix (baseline), 60s outage | 60 | 165.5 | 13.8 | 179.0 | 240 | 6 |
| Day 9 | Strapdown INS + adaptive ZUPT, 60s outage | 60 | 205.5 | 15.0 | 309.6 | 240 | 6 |
| Day 9 | ML velocity model, 60s outage | 60 | 135.5 | 15.1 | 273.9 | 240 | 6 |
| Day 9 | Anchored ML (entry correction, 20 s fade), 60s outage | 60 | 112.5 | 10.8 | 200.6 | 240 | 6 |
| Day 9 | Hold last GPS fix (baseline), 120s outage | 120 | 360.6 | 17.8 | 432.4 | 240 | 6 |
| Day 9 | Strapdown INS + adaptive ZUPT, 120s outage | 120 | 636.4 | 29.2 | 906.3 | 240 | 6 |
| Day 9 | ML velocity model, 120s outage | 120 | 221.4 | 12.3 | 429.1 | 240 | 6 |
| Day 9 | Anchored ML (entry correction, 20 s fade), 120s outage | 120 | 206.1 | 12.0 | 348.6 | 240 | 6 |
