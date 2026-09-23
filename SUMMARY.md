# SUMMARY.md — Day-by-Day Concept + Coding Notes

> Companion to PROGRESS.md. This file explains **what each day's work
> means and why**, plus concrete coding tasks. Read the concept section
> before the coding section each day — the code will make more sense.

---

## Phase 1 — Setup & Dataset Understanding (Days 1–2)

### Day 1 — Environment & Repo Setup

**Concept:**
Before touching sensor fusion math, get the ground under your feet: a
reproducible environment and a mental model of the pipeline you're
building —
`raw IMU → calibrate/align → denoise → AI velocity estimate → strapdown
integration (dead reckoning) → map matching → GNSS fusion`.
Every later day builds one block of this chain.

**Coding tasks:**
- Create the folder structure from PROGRESS.md §8.
- Set up Python environment (conda/venv), core libs: `numpy`, `pandas`,
  `scipy`, `matplotlib`, `torch` (or `tensorflow`), `pymap3d` or `pyproj`
  (coordinate conversions), `filterpy` (Kalman filters as reference).
- Write a data-loading stub for IO-VNBD (`src/calibration/preprocessing.py`
  or a `data/loader.py`): load one drive's IMU stream (accel, gyro,
  timestamp) and GNSS ground truth into a pandas DataFrame.
- Sanity plot: raw accelerometer magnitude vs time for one drive.

### Day 2 — Exploratory Data Analysis (EDA)

**Concept:**
Understand your enemy: what does IMU noise actually look like on this
dataset? Look at sampling rate, units (m/s² vs g), sensor axes convention,
GNSS outage segments (if simulated in the dataset), and how much the
phone/sensor rig moves relative to the vehicle frame.

**Coding tasks:**
- Compute sampling rate consistency (timestamp diffs).
- Plot accel/gyro per-axis over a segment; identify stationary vs moving
  periods by eye.
- Plot the GNSS ground-truth trajectory (lat/lon → local ENU meters) for
  one drive — this is your "truth" for later drift evaluation.
- Compute a **naive double-integration baseline**: integrate raw
  accelerometer directly to position with no correction, and plot how
  fast it drifts. This number is your baseline to beat — write it into
  `results/metrics.md`.

---

## Phase 2 — Calibration & Preprocessing (Days 3–5)

### Day 3 — Coordinate Frames & Mount Alignment

**Concept:**
The IMU measures in its own **body frame**, but you need motion in the
**vehicle frame** (forward/lateral/vertical) to apply constraints like
"car doesn't slide sideways." The phone can be mounted at any pitch/roll/
yaw relative to the car. You must estimate this misalignment rotation
(a 3×3 rotation matrix or quaternion) so you can rotate every IMU sample
into the vehicle frame before doing anything else.

Standard trick: during straight-line driving, average forward
acceleration should align with the direction of travel and gravity should
be constant in the "down" axis — use gravity vector (from a stationary
period) plus the direction of dominant acceleration during acceleration/
braking events to estimate the alignment rotation.

**Coding tasks:**
- `src/calibration/mount_alignment.py`:
  - Detect a stationary window → estimate gravity vector → get roll/pitch
    of phone relative to level.
  - Detect acceleration/braking events → estimate yaw offset from vehicle
    forward axis.
  - Build a rotation matrix `R_body_to_vehicle` and apply it to a test
    segment.
- Unit test: after alignment, verify gravity is ~9.81 on the vertical
  vehicle axis and near-zero on lateral axis during straight driving.

### Day 4 — Bias Estimation & Denoising

**Concept:**
MEMS IMUs have **deterministic errors** — constant bias, scale factor
error, and slowly-varying bias drift — plus random noise. Bias alone,
uncorrected, causes error to grow *quadratically* with time when
integrated twice (accel → velocity → position). You need to:
1. Estimate a per-drive/per-session initial gyro & accel bias from
   stationary periods.
2. Apply a denoising filter (low-pass / complementary filter / wavelet)
   to remove high-frequency vibration (engine, road) without lagging the
   real vehicle motion too much.

**Coding tasks:**
- `src/calibration/preprocessing.py`:
  - `estimate_static_bias(imu_window)` — mean gyro/accel over a detected
    stationary window.
  - `lowpass_filter(signal, cutoff_hz, fs)` using `scipy.signal.butter` +
    `filtfilt`.
  - Apply and re-plot Day-2's raw signal vs filtered signal — visually
    confirm potholes/vibration are attenuated but braking/turning events
    are preserved.

### Day 5 — Motion Segmentation (Stationary / Moving Detector)

**Concept:**
This is the seed of your **Zero-Velocity Update (ZUPT)** — a classic INS
trick: whenever the vehicle is actually stationary (red light, traffic),
you *know* true velocity is 0, so you can reset the integrator and kill
accumulated velocity error. Detecting "is the car stopped right now?"
purely from IMU (variance of accel/gyro over a short window) is your
first small ML/statistical classifier, and it directly reduces drift.

**Coding tasks:**
- Build a simple feature set per window (e.g. 0.5s): accel variance, gyro
  variance, accel magnitude mean.
- Label windows using GNSS speed ≈ 0 as ground truth (from the dataset).
- Train a small classifier (logistic regression / small decision tree /
  1D-CNN) → `src/models/zupt_classifier.py`.
- Evaluate precision/recall on held-out drives — false positives (calling
  a slow-moving car "stopped") are worse than false negatives here, so
  tune threshold accordingly.

---

## Phase 3 — AI Velocity Estimation (Days 6–8)

### Day 6 — Problem Framing & Baseline Model

**Concept:**
Instead of only double-integrating noisy acceleration (which drifts
fast), train a model to **regress forward velocity directly from a
window of IMU samples** — this is the well-known "IMU-to-velocity"
learning approach (similar in spirit to RIDI / IONet-style inertial
odometry work, adapted here to vehicles instead of pedestrians). The
network learns an implicit calibration + integration + bias-correction
function from data, which classical integration can't do because it has
no way to "learn" systematic bias patterns.

**Coding tasks:**
- Define input: sliding window (e.g. 1–2s at IMU rate) of calibrated,
  filtered 6-axis IMU (accel + gyro) → output: forward velocity scalar
  (or vx, vy in vehicle frame) at the window's end/center.
- Ground truth velocity from GNSS Doppler/derivative of GNSS position in
  IO-VNBD.
- `src/models/velocity_net.py`: start with a simple 1D-CNN or small
  GRU/LSTM (few layers, small enough for mobile).
- Train/val split **by drive**, not by random windows (avoid leakage
  across time within the same drive).

### Day 7 — Training, Regularization, and Robustness

**Concept:**
Generalization matters more than fitting one drive perfectly — the model
must handle unseen roads/vibration profiles. Also handle the "noise
filter" requirement from the problem statement: the model (or a
preceding stage) must implicitly ignore potholes/idling vibration, which
you can reinforce by training on windows that include such events with
correct low-drift labels.

**Coding tasks:**
- Add data augmentation: inject synthetic noise bursts (simulated
  potholes) into training windows while keeping the velocity label
  correct, so the model learns to be robust to them.
- Track metrics: velocity RMSE (m/s) on held-out drives, and integrate
  predicted velocity to see resulting *position* drift (this is the
  metric that actually matters).
- Save checkpoints; log results to `results/metrics.md`.

### Day 8 — Model Compression for On-Device Inference

**Concept:**
The performance benchmark requires **10 Hz updates on a phone**. A
research-grade model that runs at 2 Hz on a laptop GPU is useless here.
Quantization/pruning/distillation trade a little accuracy for a large
speed/size win — necessary for the mobile deliverable and good practice
to establish early rather than at the last minute.

**Coding tasks:**
- Export trained model to ONNX (or TFLite) and benchmark inference
  latency on CPU (proxy for phone).
- If too slow/big: reduce layers/hidden size, or quantize
  (`torch.quantization` / TFLite post-training quantization) and re-check
  accuracy didn't collapse.
- Record final model size + latency in `results/metrics.md` — this is
  something judges will likely ask about directly.

---

## Phase 4 — Core Dead Reckoning / Strapdown INS (Days 9–10)

### Day 9 — Strapdown Integration

**Concept:**
"Strapdown INS" is the classical algorithm: integrate gyro to track
orientation (attitude), rotate the accel-derived (or your AI-predicted)
velocity into a fixed navigation frame, then integrate velocity to get
position. Your AI velocity model from Phase 3 *replaces or corrects* the
raw double-integration step, but you still need this orientation-tracking
skeleton around it, plus periodic ZUPT resets using Day 5's classifier.

**Coding tasks:**
- `src/dead_reckoning/strapdown_ins.py`:
  - Orientation integration from gyro (quaternion update, small-angle
    approx or full quaternion kinematics).
  - Combine with AI-predicted forward velocity (rotate into nav frame
    using current heading) to update position each step.
  - Apply ZUPT: when `zupt_classifier` says stationary, zero the velocity
    estimate before the next integration step.

### Day 10 — Drift Evaluation Against Benchmark

**Concept:**
This is where you check against the SIH performance benchmark directly:
drift <10% of distance traveled during a simulated GNSS blackout. Compare
three curves on the same plot — GNSS ground truth, naive double-
integration baseline (Day 2), and your AI+ZUPT dead-reckoning output —
so the improvement is visually obvious (good for the proposal too).

**Coding tasks:**
- `src/dead_reckoning/drift_eval.py`: given predicted trajectory + GNSS
  ground truth, compute drift % = (final position error) / (distance
  travelled) over simulated blackout windows of varying length (50 m,
  1 km equivalents).
- Generate the position-plot outputs required for the SIH screening
  submission → save to `results/plots/`.
- Tune ZUPT threshold / velocity model if drift target isn't met yet.

---

## Phase 5 — Map Matching (Days 11–12)

### Day 11 — OSM Road Graph + Non-Holonomic Constraints

**Concept:**
Even a good INS trajectory drifts laterally over time. Map matching uses
the fact that vehicles drive **on roads**: snap/constrain the noisy
trajectory onto the nearest plausible road segment. Non-Holonomic
Constraints (NHC) encode vehicle physics directly into the filter: a car
can't move sideways (no lateral slip) or vertically off the road plane,
which mathematically suppresses a large share of INS drift even before
map snapping.

**Coding tasks:**
- `src/map_matching/osm_loader.py`: load an offline OSM extract for the
  demo region (e.g. via `osmnx`) as a routable graph.
- `src/map_matching/nhc_constraints.py`: add a pseudo-measurement update
  to the filter — "lateral & vertical velocity in vehicle frame ≈ 0" —
  applied every step as a soft constraint.

### Day 12 — HMM / Particle-Filter Map Matcher

**Concept:**
A Hidden-Markov-Model map matcher treats "which road segment am I
actually on" as a hidden state, observations are your noisy INS
positions, and transition probabilities favor staying on/moving along
connected road segments (classic Newson & Krumm-style map matching). This
is what "snaps" your trajectory to the road grid, not just constrains its
physics.

**Coding tasks:**
- `src/map_matching/hmm_map_matcher.py`: candidate road segments near
  each estimated point (observation probability via distance), Viterbi
  decoding over a short sliding window for the most likely road path.
- Re-plot Day-10's trajectories with map-matching applied; recompute
  drift % — expect a further improvement, especially in the along-road
  direction.

---

## Phase 6 — GNSS+INS Fusion (Days 13–14)

### Day 13 — Fusion Filter (EKF/UKF) with AI-assisted Noise Modeling

**Concept:**
When GNSS *is* available, don't just trust it blindly (multipath in
cities) or ignore it — fuse it with INS using a Kalman-family filter. An
Unscented Kalman Filter (UKF) handles the nonlinear orientation/position
dynamics better than a plain EKF. The "AI" angle: instead of hand-tuned,
fixed process/measurement noise covariances, use a small learned model
(or heuristics from IMU variance) to adapt the filter's trust in
GNSS vs INS dynamically — e.g. down-weight GNSS when its reported HDOP is
poor or when it disagrees sharply with INS (likely multipath).

**Coding tasks:**
- `src/fusion/ekf_ukf_fusion.py`: state = [position, velocity,
  orientation, IMU biases]; process model = strapdown INS; measurement
  model = GNSS position/velocity when available.
- Add adaptive measurement noise: scale `R` (GNSS noise covariance) using
  a simple learned/heuristic confidence score.

### Day 14 — Seamless GNSS ↔ DR Handoff

**Concept:**
The spec explicitly calls for an **instant, seamless transition**
(milliseconds) both directions. Practically: the filter should already be
running continuously — GNSS outage just means "measurement update step is
skipped, prediction-only continues" (which is exactly what Day 9–10's DR
does), and GNSS return means "resume measurement updates," ideally with a
soft re-convergence (not a sudden position jump) by blending over a short
window rather than snapping instantly.

**Coding tasks:**
- `src/fusion/gnss_handoff.py`: outage detection (GNSS fix flag / HDOP
  threshold), automatic switch between "predict+update" and
  "predict-only" modes, and a short blending window on GNSS reacquisition
  to avoid visible jumps.
- End-to-end test: simulate a GNSS blackout window inside a full drive,
  run the complete fusion pipeline, plot the seam — should be smooth.

---

## Phase 7 — Integration, App, and Packaging (Day 15)

**Concept:**
Bring every component into two deliverables: (1) the mobile app — a thin
UI wrapper calling the exported on-device models + fusion engine in real
time, and (2) a generic edge engine — the same core algorithms exposed
via a clean IMU-in/position-out API, decoupled from "phone" so it can
take input from external/FOG IMUs at higher rates (~200 Hz) as required.

**Coding tasks:**
- `src/edge_engine/engine_api.py`: a clean function/class
  `process(imu_sample, gnss_sample=None) -> position`, hiding all
  internal modules (calibration → velocity model → DR → map match →
  fusion) behind one interface — this is what both the app and any
  external system call.
- `src/edge_engine/export/`: export final models (ONNX/TFLite) for
  mobile + a config for higher-rate FOG IMU operation.
- `mobile_app/`: minimal map UI showing a live vehicle icon, calling
  `engine_api` each tick — this satisfies the "Real-time Navigation
  Interface" requirement without needing to over-build the app itself.
- Update PROGRESS.md's Day 15 log + `results/metrics.md` with final
  drift %, update rate, and model size numbers for the proposal.
- Assemble the SIH screening submission: preliminary AI models + IO-VNBD
  subset position plots (already generated in Phase 4/5) + short write-up
  referencing this SUMMARY.md's phase descriptions.

---

## Quick Concept Glossary

- **Dead Reckoning (DR):** estimating current position from a known
  starting point using measured velocity/heading over time, without
  external positioning.
- **Strapdown INS:** IMU rigidly mounted (strapped down) to the vehicle;
  software (not gimbals) tracks orientation and integrates motion.
- **ZUPT (Zero-Velocity Update):** resetting velocity error to zero
  whenever the system is detected to be truly stationary.
- **NHC (Non-Holonomic Constraints):** physical motion constraints of a
  wheeled vehicle (no sideways slip, no vertical flight) used as
  pseudo-measurements to suppress filter drift.
- **EKF/UKF:** Extended/Unscented Kalman Filter — standard tools for
  fusing noisy, nonlinear sensor streams (GNSS + INS here).
- **Map Matching:** snapping a noisy estimated trajectory onto the most
  probable real road path using a known road network.
