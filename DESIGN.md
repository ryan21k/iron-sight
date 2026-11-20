# Squat analysis design

## Architecture and model choice

| Model | Accuracy and deployment tradeoff | Depth |
| --- | --- | --- |
| MediaPipe Pose Landmarker Full | MVP choice. Single-person tracking, 33 landmarks including heel and foot index, and a CPU-friendly Python API. Benchmark Lite for latency and Heavy for difficult frames with a rack or barbell. | Image landmarks plus learned hip-centered world landmarks. Treat world depth as a model estimate. |
| RTMPose / RTMW | RTMPose-m reports 75.8 COCO AP and over 90 FPS on an i7-11700 in its published setup. Choose a whole-body checkpoint for foot landmarks. Budget for person detection, tracking, and deployment complexity. | RTMPose/RTMW 2D checkpoints output image points. RTMW3D provides learned monocular 3D. |
| MoveNet Lightning / Thunder | Strong browser baseline. Lightning prioritizes latency and Thunder prediction quality. The 17-point topology includes ankles and omits separate heel and toe landmarks, limiting foot-relative measurements. | 2D. |

Published model scores and speeds describe different datasets, hardware, and pipelines. Compare models on the same squat clips using joint-angle error, near-side landmark coverage, and median and p95 capture-to-overlay latency. [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker), [RTMPose](https://arxiv.org/abs/2303.07399), [RTMW and RTMW3D](https://arxiv.org/abs/2407.08634), [MoveNet model card](https://storage.googleapis.com/movenet/MoveNet.SinglePose%20Model%20Card.pdf)

```text
File upload / webcam -> timestamped RGB frames -> PoseEstimator
 -> confidence + view gates -> coordinate transforms + smoothing
 -> MovementAnalyzer -> rep segmentation + view-specific features
 -> evidence events + optional symptom context -> safety policy
 -> curated cue / drill selection -> overlay, report, live feedback
```

Phase 1 uses Python 3.12, OpenCV and MediaPipe Tasks. Add NumPy/SciPy for batch analysis and FastAPI + React for the upload/live UI. Use background jobs for uploaded clips and browser inference or a bounded latest-frame worker for live cues. Decode presentation timestamps when adding variable-frame-rate support. Use monotonic capture time for webcam. Keep a shared pose record with timestamp, track ID, landmark identities, confidence, and coordinate space. A movement-specific analyzer receives that record and emits typed phases, features and events. Deadlift and press analyzers can supply their own phases and feature sets to the same renderer and safety policy.

### Capture, normalization and smoothing

- **Side profile** supports sagittal angle proxies and hip/shoulder timing. **Front** supports projected knee tracking and bilateral comparison when both sides are visible. **45°** supports overlay and review until its inferred 3D measurements pass validation. Use capture instructions, a level reference or calibrated extrinsics, and visibility/segment-length consistency checks to reject unsuitable clips.
- Convert normalized image landmarks to `p = (W*x, -H*y)` before geometry. If calibration is available, undistort and rotate using a known level reference. Normalize displacement by a stable standing segment length measured in the same coordinates. Monocular 2D projection loses out-of-plane information. For 45° measurements, validate learned 3D by view, or use synchronized calibrated cameras and triangulate. With calibrated 3D, align world vertical to gravity, estimate the athlete's horizontal right axis from standing bilateral hip/shoulder landmarks, and use its orthogonal horizontal forward axis with sign set from feet or capture direction. Project vectors into the forward/up or right/up plane as appropriate.
- Phase 1 filters image coordinates in frame-height units with One Euro. For time step dt, use `a(f)=2π*f*dt/(1+2π*f*dt)`. Low-pass the raw finite-difference velocity at 1 Hz and low-pass position with cutoff `1.5 + 2*abs(filtered_velocity)` Hz. Gate each point at 0.65 minimum presence and visibility and restart its filter on reacquisition. Offline, a centered Savitzky–Golay filter on a uniform time grid is another option. Split at gaps and verify peak attenuation and timing against hand-labeled clips. [One Euro reference implementation](https://github.com/casiez/OneEuroFilter)

## Geometry and phase specification

Use Cartesian coordinates with up positive. Write S, H, K, A, E, T for shoulder, hip, knee, ankle, heel, and foot index. For nonzero vectors define `ang(u,v)=atan2(||u×v||,u·v)*180/π`, using absolute scalar cross product in 2D. Evaluate each metric only with its required valid landmarks.

| Metric | Definition |
| --- | --- |
| Knee flexion | `180 - ang(H-K, A-K)` |
| Hip flexion proxy | `180 - ang(S-H, K-H)`. Trunk-relative thigh flexion includes changes in trunk orientation. |
| Ankle dorsiflexion proxy | `90 - ang(K-A, T-E)`. This uses the shoe/foot axis as a reference. Subtract the standing value for excursion comparisons. |
| Torso inclination | `ang(S-H, up)`. With a forward axis, report signed lean as `atan2((S-H)·forward,(S-H)·up)` in degrees. |
| Early ascent hip-to-shoulder rise | Over the first third of total hip rise from bottom, `R=(H_end-H_start)·up / ((S_end-S_start)·up)`. Require shoulder rise above 3% of standing leg length for the ratio. Also report absolute hip-minus-shoulder rise divided by standing leg length and change in lean in the same window. |
| Medial knee displacement | In front-plane right/up coordinates, interpolate the hip–ankle line at knee height with `λ=(K_y-A_y)/(H_y-A_y)`. For each side use `s*(K_x-[A_x+λ*(H_x-A_x)])/standing_ankle_width`, with `s=+1` for left and `-1` for right so positive is medial. Compare to setup. Require a nonzero vertical span, valid interpolation and stable stance. This is projected medial motion. |
| Pelvic tilt at depth | With validated sagittal ASIS and PSIS marker positions, let `v=ASIS-PSIS`. Compute `atan2(v·up,v·forward)` in degrees and subtract the setup value with angle unwrapping. Positive change indicates posterior rotation under this convention. Generic hip landmarks lack the anterior/posterior pelvic axis. Record this metric as unavailable in the starter. |
| Bilateral asymmetry | At the same timestamp, take `abs(θ_L-θ_R)` for comparable angles. For distinct left/right movement onsets report `abs(t_L-t_R)`, with onsets defined using the same calibrated criteria. Compare only with valid, comparable views for both sides. |

The implemented rep signal is projected knee flexion `q` on the selected side. Compute speed from consecutive accepted samples `(q_t-q_prev)/(t-t_prev)`. Thresholds below are engineering defaults for segmentation.

1. **Setup.** Establish baseline `b` as the mean of at least 0.4 s of samples with `q ≤ 35°` and `abs(speed) ≤ 12°/s`. Start descent when `q-b ≥ 15°` and speed is at least 12°/s.
2. **Descent.** Track the running peak. After peak excursion reaches 45°, show bottom when speed drops to at most 12°/s and the sample is within 7° of peak. Resume descent if speed rises above 12°/s in the bottom state.
3. **Bottom and ascent.** Enter ascent from descent or bottom on a drop of at least 7° from a qualifying peak and negative speed. Retrospectively label the contiguous interval around the first peak within 5° of that peak as bottom. Short transitions can have a zero-duration sampled bottom. The live phase has hysteresis and the exported timings use this retrospective interval.
4. **Completion.** Count once when ascent reaches `q-b ≤ 12°`. Discard shallow returns, attempts over 15 s, unfinished reps, and attempts with over 0.20 s between usable knee samples. After an interruption that discards a rep, require a return to the setup band before arming another rep. A shallow ROM exercise may need a separate threshold profile.

## Evidence, symptom context and recommendations

Record the optional symptom region, side, phase, quality, persistence, and optional rep/timestamp, plus lift variant, load and footwear. Summarize temporal overlap with valid events. Permit multiple hypotheses and missing measurements. Establish a reference within the same athlete, variant, stance, view, depth band and load band. Limb proportions and stance influence squat mechanics. [Stance and anthropometry study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6033510/)

| Observation and provisional rule | Coaching hypothesis and response when symptoms allow |
| --- | --- |
| Repeated heel rise (>3% of projected foot length relative to setup, stable toe), with a plateau in forward shank excursion | Check whether ankle motion, stance, footwear, or balance contributes. Trial a comfortable-depth bodyweight squat and compare. If independent ankle assessment supports a mobility limitation, offer a gentle knee-to-wall drill. Cue “Keep your whole foot heavy.” |
| Early ascent ratio >1.3 plus at least 8° increase in forward lean relative to bottom | Check load, bar position, fatigue, and habitual strategy. Try a lighter tempo squat or pause squat. Cue “Drive your hips and shoulders up together.” |
| Medial knee shift >0.08 standing ankle widths above setup | Check toe direction, stance, motion control, and view error. With a clean repeatable observation, trial a controlled supported split squat. Cue “Track your knees in the direction of your toes.” Video alone cannot establish hip abductor weakness. |
| Posterior pelvic rotation increases by >8° within the deepest 20% of descent, with a validated pelvis axis | Review depth, stance and load. If hip pinch coincides, pause the provoking set and consider a comfortable box height after safety triage. |
| Bilateral angle difference >10° at matched depth and time | Check camera placement, stance symmetry, and repeatability before offering a supported split squat or stance cue. |

These are initial review thresholds. Require at least 150 ms of coherent evidence for continuous events and recurrence in two of three comparable reps for automatic coaching. Flag single large, well-observed deviations for review, especially apparent loss of balance or repeated worsening under load. Never turn an angle cutoff into an injury-risk prediction. A useful later hybrid is phase-normalized trajectories and robust personal residuals `(x-median)/max(1.4826*MAD, noise_floor)` with confidence masking, followed by supervised event calibration on athlete-separated validation data. Track false coaching alerts, coverage and abstentions by angle, load, clothing and lift variant.

Safety precedes recommendation generation. Sharp, radiating, persistent, or worsening symptoms suppress exercise progressions and trigger guidance to stop the provoking movement and consult a physical therapist or physician. For severe low back pain radiating to the leg with new bladder or bowel disturbance or saddle numbness, route to emergency care. For co-occurring symptoms and uncertain pose evidence, prioritize the symptom report and withhold causal claims. [NICE referral guidance](https://www.nice.org.uk/guidance/ng127/chapter/Recommendations-for-adults-aged-over-16), [NHS back pain guidance](https://www.nhs.uk/conditions/back-pain/)

The recommendation engine selects approved drill and cue IDs after safety and quality checks. Prioritize one modifiable issue, then one cue and at most one drill or accessory with a symptom-free regression. Allow an LLM to explain selected items using structured evidence, with output validation and deterministic fallback copy. Keep triage, drill selection constraints, and uncertainty outside the LLM. Store symptom data only with consent and give users control over retention. Live feedback should speak at most one cue between reps and stay silent during tracking loss.

## Three milestones

1. **Pose overlay, reps, angles.** Run the supplied CLI on side-profile bodyweight, back and front squats. Hand-label onset, bottom and lockout on a diverse set, including partial reps and occlusions. Initial acceptance targets are at least 95% count agreement on usable clips, median bottom timing error under 150 ms and median sagittal knee error under 7° against annotated reference landmarks. Report unusable-clip rate separately.
2. **Compensation detection.** Add side and front capture checks, evidence event windows, personal baselines, and structured symptom logging. Start with heel-rise, early hip rise and front-plane knee/side differences. Review precision and abstentions against blinded coach labels. Keep pelvic tilt and 45° scoring gated on validated capture and landmarks.
3. **Corrective protocol and UI.** Add FastAPI upload jobs, a React scrubber with evidence markers, webcam preview and optional post-rep cues. Add the approved content library and constrained explanation generator. Verify every symptom-triage branch, enforce retention controls, and measure p95 capture-to-overlay latency on target hardware.
