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

## Day 10 — heading only: end-position error with TRUE distance (all drives)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 10 | True distance + last GPS course (straight line), 30s outage | 30 | 98.4 | 19.2 | 183.1 | 653 | 46 |
| Day 10 | True distance + calibrated gyro heading, 30s outage | 30 | 58.9 | 12.7 | 151.2 | 653 | 46 |
| Day 10 | True distance + last GPS course (straight line), 60s outage | 60 | 239.2 | 26.8 | 391.2 | 653 | 46 |
| Day 10 | True distance + calibrated gyro heading, 60s outage | 60 | 157.3 | 19.8 | 325.2 | 653 | 46 |
| Day 10 | True distance + last GPS course (straight line), 120s outage | 120 | 650.7 | 38.9 | 930.3 | 653 | 46 |
| Day 10 | True distance + calibrated gyro heading, 120s outage | 120 | 446.8 | 27.1 | 798.4 | 653 | 46 |

## Day 10 — full 2-D path during simulated GPS outages (TEST drives only)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 10 | Hold last fix + last course (baseline), 30s outage | 30 | 119.9 | 22.5 | 166.5 | 60 | 4 |
| Day 10 | Anchored ML + last course, 30s outage | 30 | 103.4 | 18.6 | 149.4 | 60 | 4 |
| Day 10 | Hold last fix + gyro heading, 30s outage | 30 | 123.1 | 20.7 | 159.7 | 60 | 4 |
| Day 10 | Anchored ML + gyro heading (full method), 30s outage | 30 | 93.7 | 17.5 | 138.2 | 60 | 4 |
| Day 10 | Hold last fix + last course (baseline), 60s outage | 60 | 331.6 | 28.7 | 368.9 | 60 | 4 |
| Day 10 | Anchored ML + last course, 60s outage | 60 | 231.2 | 23.8 | 304.0 | 60 | 4 |
| Day 10 | Hold last fix + gyro heading, 60s outage | 60 | 322.2 | 27.7 | 373.1 | 60 | 4 |
| Day 10 | Anchored ML + gyro heading (full method), 60s outage | 60 | 207.8 | 20.9 | 296.6 | 60 | 4 |
| Day 10 | Hold last fix + last course (baseline), 120s outage | 120 | 929.4 | 46.0 | 1002.1 | 60 | 4 |
| Day 10 | Anchored ML + last course, 120s outage | 120 | 747.2 | 37.1 | 898.2 | 60 | 4 |
| Day 10 | Hold last fix + gyro heading, 120s outage | 120 | 810.6 | 40.0 | 1031.0 | 60 | 4 |
| Day 10 | Anchored ML + gyro heading (full method), 120s outage | 120 | 746.8 | 35.1 | 926.7 | 60 | 4 |

## Day 11 — map matching, heading/path only: TRUE distance (all drives with a map)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 11 | True distance + last GPS course (straight line), 30s outage | 30 | 95.4 | 19.7 | 163.1 | 535 | 38 |
| Day 11 | True distance + gyro heading (Day 10), 30s outage | 30 | 50.9 | 11.5 | 125.8 | 535 | 38 |
| Day 11 | True distance + gyro + map (snap), 30s outage | 30 | 15.2 | 3.4 | 99.3 | 535 | 38 |
| Day 11 | True distance + gyro + map (route), 30s outage | 30 | 7.8 | 1.7 | 91.5 | 535 | 38 |
| Day 11 | True distance + last GPS course (straight line), 60s outage | 60 | 234.7 | 28.4 | 357.9 | 535 | 38 |
| Day 11 | True distance + gyro heading (Day 10), 60s outage | 60 | 140.5 | 18.0 | 272.4 | 535 | 38 |
| Day 11 | True distance + gyro + map (snap), 60s outage | 60 | 58.7 | 9.0 | 231.2 | 535 | 38 |
| Day 11 | True distance + gyro + map (route), 60s outage | 60 | 25.6 | 4.0 | 226.3 | 535 | 38 |
| Day 11 | True distance + last GPS course (straight line), 120s outage | 120 | 641.1 | 40.2 | 863.4 | 535 | 38 |
| Day 11 | True distance + gyro heading (Day 10), 120s outage | 120 | 381.6 | 24.9 | 655.0 | 535 | 38 |
| Day 11 | True distance + gyro + map (snap), 120s outage | 120 | 277.7 | 21.1 | 604.0 | 535 | 38 |
| Day 11 | True distance + gyro + map (route), 120s outage | 120 | 238.0 | 17.9 | 596.4 | 535 | 38 |

## Day 11 — map matching, full pipeline during simulated GPS outages (TEST drives)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 11 | Hold last fix + last course (baseline), 30s outage | 30 | 121.8 | 24.9 | 172.3 | 45 | 3 |
| Day 11 | Anchored ML + gyro heading (Day 10), 30s outage | 30 | 87.1 | 18.6 | 133.8 | 45 | 3 |
| Day 11 | Anchored ML + gyro + map (snap), 30s outage | 30 | 73.0 | 15.2 | 114.6 | 45 | 3 |
| Day 11 | Anchored ML + gyro + map (route, full method), 30s outage | 30 | 68.6 | 12.7 | 113.7 | 45 | 3 |
| Day 11 | Hold last fix + last course (baseline), 60s outage | 60 | 353.4 | 37.9 | 389.6 | 45 | 3 |
| Day 11 | Anchored ML + gyro heading (Day 10), 60s outage | 60 | 201.6 | 21.4 | 285.9 | 45 | 3 |
| Day 11 | Anchored ML + gyro + map (snap), 60s outage | 60 | 137.0 | 15.3 | 210.5 | 45 | 3 |
| Day 11 | Anchored ML + gyro + map (route, full method), 60s outage | 60 | 103.3 | 12.4 | 206.8 | 45 | 3 |
| Day 11 | Hold last fix + last course (baseline), 120s outage | 120 | 953.2 | 51.5 | 999.2 | 45 | 3 |
| Day 11 | Anchored ML + gyro heading (Day 10), 120s outage | 120 | 725.7 | 35.0 | 817.5 | 45 | 3 |
| Day 11 | Anchored ML + gyro + map (snap), 120s outage | 120 | 627.2 | 30.1 | 774.5 | 45 | 3 |
| Day 11 | Anchored ML + gyro + map (route, full method), 120s outage | 120 | 624.1 | 31.8 | 829.5 | 45 | 3 |

## Day 11 — map matching, full pipeline (TEST + VAL drives, more outages)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 11 | Hold last fix + last course (baseline) [test+val], 30s outage | 30 | 170.2 | 45.5 | 189.7 | 135 | 9 |
| Day 11 | Anchored ML + gyro heading (Day 10) [test+val], 30s outage | 30 | 101.5 | 24.5 | 145.1 | 135 | 9 |
| Day 11 | Anchored ML + gyro + map (snap) [test+val], 30s outage | 30 | 87.0 | 23.9 | 129.7 | 135 | 9 |
| Day 11 | Anchored ML + gyro + map (route, full method) [test+val], 30s outage | 30 | 77.7 | 20.2 | 127.3 | 135 | 9 |
| Day 11 | Hold last fix + last course (baseline) [test+val], 60s outage | 60 | 362.5 | 55.0 | 417.9 | 135 | 9 |
| Day 11 | Anchored ML + gyro heading (Day 10) [test+val], 60s outage | 60 | 209.2 | 30.0 | 307.1 | 135 | 9 |
| Day 11 | Anchored ML + gyro + map (snap) [test+val], 60s outage | 60 | 169.1 | 23.9 | 269.6 | 135 | 9 |
| Day 11 | Anchored ML + gyro + map (route, full method) [test+val], 60s outage | 60 | 155.6 | 22.6 | 262.6 | 135 | 9 |
| Day 11 | Hold last fix + last course (baseline) [test+val], 120s outage | 120 | 882.4 | 61.3 | 983.8 | 135 | 9 |
| Day 11 | Anchored ML + gyro heading (Day 10) [test+val], 120s outage | 120 | 561.2 | 37.1 | 724.4 | 135 | 9 |
| Day 11 | Anchored ML + gyro + map (snap) [test+val], 120s outage | 120 | 473.4 | 32.0 | 684.4 | 135 | 9 |
| Day 11 | Anchored ML + gyro + map (route, full method) [test+val], 120s outage | 120 | 458.6 | 31.2 | 690.6 | 135 | 9 |

## Day 12 — map matching + turn consistency, heading/path only: TRUE distance (all drives with a map)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 12 | True distance + last GPS course (straight line), 30s outage | 30 | 98.4 | 19.2 | 183.1 | 653 | 46 |
| Day 12 | True distance + gyro heading (Day 10), 30s outage | 30 | 58.9 | 12.7 | 151.2 | 653 | 46 |
| Day 12 | True distance + gyro + map (snap), 30s outage | 30 | 16.4 | 3.4 | 122.4 | 653 | 46 |
| Day 12 | True distance + gyro + map (route), 30s outage | 30 | 8.7 | 1.8 | 116.9 | 653 | 46 |
| Day 12 | True distance + last GPS course (straight line), 60s outage | 60 | 239.2 | 26.8 | 391.2 | 653 | 46 |
| Day 12 | True distance + gyro heading (Day 10), 60s outage | 60 | 157.3 | 19.8 | 325.2 | 653 | 46 |
| Day 12 | True distance + gyro + map (snap), 60s outage | 60 | 66.0 | 9.8 | 281.1 | 653 | 46 |
| Day 12 | True distance + gyro + map (route), 60s outage | 60 | 32.7 | 4.4 | 278.2 | 653 | 46 |
| Day 12 | True distance + last GPS course (straight line), 120s outage | 120 | 650.7 | 38.9 | 930.3 | 653 | 46 |
| Day 12 | True distance + gyro heading (Day 10), 120s outage | 120 | 446.8 | 27.1 | 798.4 | 653 | 46 |
| Day 12 | True distance + gyro + map (snap), 120s outage | 120 | 355.7 | 23.7 | 745.5 | 653 | 46 |
| Day 12 | True distance + gyro + map (route), 120s outage | 120 | 306.4 | 21.1 | 739.8 | 653 | 46 |

## Day 12 — map matching + turn consistency, full pipeline during simulated GPS outages (TEST drives)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 12 | Hold last fix + last course (baseline), 30s outage | 30 | 119.9 | 22.5 | 166.5 | 60 | 4 |
| Day 12 | Anchored ML + gyro heading (Day 10), 30s outage | 30 | 93.7 | 17.5 | 138.2 | 60 | 4 |
| Day 12 | Anchored ML + gyro + map (snap), 30s outage | 30 | 72.8 | 12.3 | 108.6 | 60 | 4 |
| Day 12 | Anchored ML + gyro + map (route, full method), 30s outage | 30 | 72.0 | 11.0 | 107.8 | 60 | 4 |
| Day 12 | Hold last fix + last course (baseline), 60s outage | 60 | 331.6 | 28.7 | 368.9 | 60 | 4 |
| Day 12 | Anchored ML + gyro heading (Day 10), 60s outage | 60 | 207.8 | 20.9 | 296.6 | 60 | 4 |
| Day 12 | Anchored ML + gyro + map (snap), 60s outage | 60 | 137.6 | 14.3 | 238.3 | 60 | 4 |
| Day 12 | Anchored ML + gyro + map (route, full method), 60s outage | 60 | 105.9 | 12.3 | 227.0 | 60 | 4 |
| Day 12 | Hold last fix + last course (baseline), 120s outage | 120 | 929.4 | 46.0 | 1002.1 | 60 | 4 |
| Day 12 | Anchored ML + gyro heading (Day 10), 120s outage | 120 | 746.8 | 35.1 | 926.7 | 60 | 4 |
| Day 12 | Anchored ML + gyro + map (snap), 120s outage | 120 | 696.4 | 31.1 | 878.9 | 60 | 4 |
| Day 12 | Anchored ML + gyro + map (route, full method), 120s outage | 120 | 711.8 | 35.6 | 932.6 | 60 | 4 |

## Day 12 — map matching + turn consistency, full pipeline (TEST + VAL drives, more outages)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 12 | Hold last fix + last course (baseline) [test+val], 30s outage | 30 | 160.1 | 38.3 | 185.7 | 150 | 10 |
| Day 12 | Anchored ML + gyro heading (Day 10) [test+val], 30s outage | 30 | 106.2 | 22.8 | 145.8 | 150 | 10 |
| Day 12 | Anchored ML + gyro + map (snap) [test+val], 30s outage | 30 | 79.7 | 17.3 | 124.7 | 150 | 10 |
| Day 12 | Anchored ML + gyro + map (route, full method) [test+val], 30s outage | 30 | 78.3 | 16.4 | 122.6 | 150 | 10 |
| Day 12 | Hold last fix + last course (baseline) [test+val], 60s outage | 60 | 354.3 | 48.2 | 406.8 | 150 | 10 |
| Day 12 | Anchored ML + gyro heading (Day 10) [test+val], 60s outage | 60 | 214.3 | 28.0 | 309.3 | 150 | 10 |
| Day 12 | Anchored ML + gyro + map (snap) [test+val], 60s outage | 60 | 169.9 | 23.9 | 275.2 | 150 | 10 |
| Day 12 | Anchored ML + gyro + map (route, full method) [test+val], 60s outage | 60 | 156.5 | 22.6 | 266.9 | 150 | 10 |
| Day 12 | Hold last fix + last course (baseline) [test+val], 120s outage | 120 | 873.7 | 57.1 | 986.4 | 150 | 10 |
| Day 12 | Anchored ML + gyro heading (Day 10) [test+val], 120s outage | 120 | 576.4 | 36.6 | 777.4 | 150 | 10 |
| Day 12 | Anchored ML + gyro + map (snap) [test+val], 120s outage | 120 | 522.9 | 31.5 | 733.6 | 150 | 10 |
| Day 12 | Anchored ML + gyro + map (route, full method) [test+val], 120s outage | 120 | 476.6 | 31.5 | 743.9 | 150 | 10 |

## Day 13 — EKF fusion, real-time end error (TEST drives)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 13 | Day 10: anchored ML + gyro, 30s outage | 30 | 97.2 | 18.4 | 139.2 | 60 | 4 |
| Day 13 | EKF fusion (real time), 30s outage | 30 | 113.0 | 19.9 | 128.6 | 60 | 4 |
| Day 13 | Day 12: anchored ML + gyro + map (route), 30s outage | 30 | 72.4 | 11.3 | 109.1 | 60 | 4 |
| Day 13 | EKF fusion + map (route), 30s outage | 30 | 66.1 | 11.0 | 95.1 | 60 | 4 |
| Day 13 | Day 10: anchored ML + gyro, 60s outage | 60 | 223.9 | 21.7 | 297.0 | 60 | 4 |
| Day 13 | EKF fusion (real time), 60s outage | 60 | 200.0 | 18.9 | 242.8 | 60 | 4 |
| Day 13 | Day 12: anchored ML + gyro + map (route), 60s outage | 60 | 109.7 | 12.3 | 233.1 | 60 | 4 |
| Day 13 | EKF fusion + map (route), 60s outage | 60 | 111.5 | 12.3 | 196.5 | 60 | 4 |
| Day 13 | Day 10: anchored ML + gyro, 120s outage | 120 | 750.6 | 36.1 | 928.6 | 60 | 4 |
| Day 13 | EKF fusion (real time), 120s outage | 120 | 564.7 | 26.0 | 680.1 | 60 | 4 |
| Day 13 | Day 12: anchored ML + gyro + map (route), 120s outage | 120 | 812.8 | 35.6 | 944.2 | 60 | 4 |
| Day 13 | EKF fusion + map (route), 120s outage | 120 | 484.3 | 22.9 | 660.8 | 60 | 4 |

## Day 13 — EKF fusion, offline tunnel track at hidden GPS fixes (TEST drives)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 13 | EKF real time (no exit fix) (track), 30s outage | 30 | 21.5 | nan | 37.3 | 578 | 4 |
| Day 13 | Straight line entry -> exit fix (track), 30s outage | 30 | 22.9 | nan | 29.4 | 578 | 4 |
| Day 13 | Day 10 DR, end error spread over time (track), 30s outage | 30 | 15.0 | nan | 22.3 | 578 | 4 |
| Day 13 | EKF + RTS smoother (uses exit fix) (track), 30s outage | 30 | 10.2 | nan | 16.0 | 578 | 4 |
| Day 13 | EKF real time (no exit fix) (track), 60s outage | 60 | 44.6 | nan | 76.4 | 1163 | 4 |
| Day 13 | Straight line entry -> exit fix (track), 60s outage | 60 | 52.3 | nan | 80.4 | 1163 | 4 |
| Day 13 | Day 10 DR, end error spread over time (track), 60s outage | 60 | 41.4 | nan | 52.0 | 1163 | 4 |
| Day 13 | EKF + RTS smoother (uses exit fix) (track), 60s outage | 60 | 20.5 | nan | 35.0 | 1163 | 4 |
| Day 13 | EKF real time (no exit fix) (track), 120s outage | 120 | 117.6 | nan | 186.7 | 2367 | 4 |
| Day 13 | Straight line entry -> exit fix (track), 120s outage | 120 | 134.3 | nan | 169.7 | 2367 | 4 |
| Day 13 | Day 10 DR, end error spread over time (track), 120s outage | 120 | 140.2 | nan | 153.6 | 2367 | 4 |
| Day 13 | EKF + RTS smoother (uses exit fix) (track), 120s outage | 120 | 44.6 | nan | 76.0 | 2367 | 4 |

## Day 13 — EKF fusion, real-time end error (TEST + VAL drives)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 13 | Day 10: anchored ML + gyro [test+val], 30s outage | 30 | 108.6 | 23.9 | 146.7 | 150 | 10 |
| Day 13 | EKF fusion (real time) [test+val], 30s outage | 30 | 111.0 | 25.9 | 158.9 | 150 | 10 |
| Day 13 | Day 12: anchored ML + gyro + map (route) [test+val], 30s outage | 30 | 78.9 | 17.0 | 124.2 | 150 | 10 |
| Day 13 | EKF fusion + map (route) [test+val], 30s outage | 30 | 80.1 | 16.1 | 134.1 | 150 | 10 |
| Day 13 | Day 10: anchored ML + gyro [test+val], 60s outage | 60 | 215.9 | 28.6 | 309.8 | 150 | 10 |
| Day 13 | EKF fusion (real time) [test+val], 60s outage | 60 | 203.7 | 28.2 | 314.3 | 150 | 10 |
| Day 13 | Day 12: anchored ML + gyro + map (route) [test+val], 60s outage | 60 | 167.9 | 23.6 | 271.5 | 150 | 10 |
| Day 13 | EKF fusion + map (route) [test+val], 60s outage | 60 | 154.4 | 20.5 | 277.3 | 150 | 10 |
| Day 13 | Day 10: anchored ML + gyro [test+val], 120s outage | 120 | 573.8 | 36.9 | 779.0 | 150 | 10 |
| Day 13 | EKF fusion (real time) [test+val], 120s outage | 120 | 521.6 | 30.8 | 711.0 | 150 | 10 |
| Day 13 | Day 12: anchored ML + gyro + map (route) [test+val], 120s outage | 120 | 505.0 | 31.3 | 746.3 | 150 | 10 |
| Day 13 | EKF fusion + map (route) [test+val], 120s outage | 120 | 412.3 | 29.9 | 679.1 | 150 | 10 |

## Day 13 — EKF fusion, offline tunnel track at hidden GPS fixes (TEST + VAL drives)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 13 | EKF real time (no exit fix) (track) [test+val], 30s outage | 30 | 27.1 | nan | 50.0 | 849 | 10 |
| Day 13 | Straight line entry -> exit fix (track) [test+val], 30s outage | 30 | 27.7 | nan | 34.3 | 849 | 10 |
| Day 13 | Day 10 DR, end error spread over time (track) [test+val], 30s outage | 30 | 17.6 | nan | 27.4 | 849 | 10 |
| Day 13 | EKF + RTS smoother (uses exit fix) (track) [test+val], 30s outage | 30 | 12.1 | nan | 20.7 | 849 | 10 |
| Day 13 | EKF real time (no exit fix) (track) [test+val], 60s outage | 60 | 54.5 | nan | 103.1 | 1700 | 10 |
| Day 13 | Straight line entry -> exit fix (track) [test+val], 60s outage | 60 | 59.8 | nan | 82.5 | 1700 | 10 |
| Day 13 | Day 10 DR, end error spread over time (track) [test+val], 60s outage | 60 | 42.8 | nan | 56.4 | 1700 | 10 |
| Day 13 | EKF + RTS smoother (uses exit fix) (track) [test+val], 60s outage | 60 | 25.2 | nan | 43.2 | 1700 | 10 |
| Day 13 | EKF real time (no exit fix) (track) [test+val], 120s outage | 120 | 137.3 | nan | 243.1 | 3505 | 10 |
| Day 13 | Straight line entry -> exit fix (track) [test+val], 120s outage | 120 | 136.3 | nan | 169.7 | 3505 | 10 |
| Day 13 | Day 10 DR, end error spread over time (track) [test+val], 120s outage | 120 | 129.3 | nan | 149.1 | 3505 | 10 |
| Day 13 | EKF + RTS smoother (uses exit fix) (track) [test+val], 120s outage | 120 | 61.1 | nan | 111.3 | 3505 | 10 |

## Day 13 — EKF fusion, robustness to GPS outliers (TEST + VAL drives)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 13 | Last fix + speed/course extrapolation (robustness) | 0 | 20.3 | nan | 51.5 | 150 | 10 |
| Day 13 | EKF without gate (robustness) | 0 | 26.0 | nan | 59.9 | 150 | 10 |
| Day 13 | EKF with chi-square gate (robustness) | 0 | 20.1 | nan | 37.4 | 150 | 10 |

## Day 15 — streaming edge engine replay (same outages)

| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) | Mean abs error (m) | Outages | Drives |
|---|---|---|---|---|---|---|---|
| Day 15 | Engine, fp32 .ptl, no map [test+val], 30s outage | 30 | 94.5 | nan | 128.2 | 60 | 10 |
| Day 15 | Engine, fp32 .ptl, no map [test+val], 60s outage | 60 | 181.5 | nan | 235.5 | 60 | 10 |
| Day 15 | Engine, fp32 .ptl, no map [test+val], 120s outage | 120 | 443.6 | nan | 635.7 | 60 | 10 |
| Day 15 | Engine, fp32 .ptl + map [test+val], 30s outage | 30 | 73.1 | nan | 111.4 | 60 | 10 |
| Day 15 | Engine, fp32 .ptl + map [test+val], 60s outage | 60 | 129.6 | nan | 210.5 | 60 | 10 |
| Day 15 | Engine, fp32 .ptl + map [test+val], 120s outage | 120 | 373.8 | nan | 588.3 | 60 | 10 |
| Day 15 | Engine, fp32 .ptl, no map, 30s outage | 30 | 94.5 | nan | 128.2 | 60 | 4 |
| Day 15 | Engine, fp32 .ptl, no map, 60s outage | 60 | 181.5 | nan | 235.5 | 60 | 4 |
| Day 15 | Engine, fp32 .ptl, no map, 120s outage | 120 | 443.6 | nan | 635.7 | 60 | 4 |
| Day 15 | Engine, fp32 .ptl + map, 30s outage | 30 | 73.1 | nan | 111.4 | 60 | 4 |
| Day 15 | Engine, fp32 .ptl + map, 60s outage | 60 | 129.6 | nan | 210.5 | 60 | 4 |
| Day 15 | Engine, fp32 .ptl + map, 120s outage | 120 | 373.8 | nan | 588.3 | 60 | 4 |
| Day 15 | Engine, int8 .ptl + map (deployed), 30s outage | 30 | 68.7 | nan | 110.4 | 60 | 4 |
| Day 15 | Engine, int8 .ptl + map (deployed), 60s outage | 60 | 129.8 | nan | 207.8 | 60 | 4 |
| Day 15 | Engine, int8 .ptl + map (deployed), 120s outage | 120 | 379.2 | nan | 581.4 | 60 | 4 |
| Day 15 | Engine, int8 .ptl + map (deployed) [test+val], 30s outage | 30 | 79.8 | nan | 149.0 | 150 | 10 |
| Day 15 | Engine, int8 .ptl + map (deployed) [test+val], 60s outage | 60 | 155.8 | nan | 282.0 | 150 | 10 |
| Day 15 | Engine, int8 .ptl + map (deployed) [test+val], 120s outage | 120 | 376.2 | nan | 593.4 | 150 | 10 |
