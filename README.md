# Iron Sight

Analyze a single athlete's squat from side-profile video. Phase 1 produces a silent annotated MP4 with a skeleton, projected joint angles, live phase labels, and a rep count, plus a JSON summary with per-rep timing, angle statistics, and tracking coverage. See [Design and biomechanics](DESIGN.md) for the full architecture and later phases.

## Run on Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task" -OutFile "pose_landmarker_full.task"
.venv\Scripts\python.exe analyze.py squat.mp4 --model pose_landmarker_full.task --side right --variant bodyweight --output output/session-01
```

Use a constant-frame-rate MP4 or a MOV with a codec your OpenCV build can decode. Start with one second of still standing. Keep one athlete fully in frame, including the near heel and toes, with a level, fixed camera perpendicular to the direction of the squat. Use the athlete's camera-facing anatomical side. The output directory must be new. The variant field records the selected squat style and shares the same rep logic across styles.

Open `annotated.mp4` for frame-by-frame review. In `summary.json`, `tracked_fraction` is the fraction of frames with a usable knee angle. Each metric has its own sample count. Setup and incomplete reps are excluded from the per-rep statistics. A long occlusion invalidates an active rep. The timing uses frame index and nominal FPS, so transcode variable-frame-rate footage to constant frame rate before using the starter.

`pose.py` adapts MediaPipe output and smooths landmarks. `kinematics.py` computes angles. `squat.py` owns segmentation and aggregation. `analyze.py` handles decoding, the HUD, and output files.
