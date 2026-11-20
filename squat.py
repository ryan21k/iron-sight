"""Streaming squat segmentation and per-rep aggregation."""
from statistics import mean


class SquatTracker:
    START_DEG = 15
    MIN_EXCURSION_DEG = 45
    RETURN_DEG = 12
    TURN_DEG = 7
    MOTION_DEG_S = 12
    BOTTOM_WINDOW_DEG = 5
    MAX_GAP_S = 0.20
    MAX_REP_S = 15

    def __init__(self):
        self.baseline = None
        self.calibration = []
        self.previous = None
        self.frames = []
        self.peak = None
        self.phase = "setup"
        self.reps = []
        self.discarded = 0
        self.armed = True

    def reset(self):
        if self.frames:
            self.armed = False
        self.discarded += bool(self.frames)
        self.previous = None
        self.frames.clear()
        self.peak = None
        self.phase = "setup"
        self.calibration.clear()

    def update(self, time, metrics):
        knee = metrics.get("knee_flexion_deg")
        if self.previous and time - self.previous[0] > self.MAX_GAP_S:
            self.reset()
        if knee is None:
            return "untracked"
        speed = 0.0
        if self.previous:
            last_time, last_knee = self.previous
            speed = (knee - last_knee) / (time - last_time)
        self.previous = time, knee
        if self.baseline is None:
            if knee > 35 or abs(speed) > self.MOTION_DEG_S:
                self.calibration.clear()
            else:
                self.calibration.append((time, knee))
                if time - self.calibration[0][0] >= 0.4:
                    self.baseline = mean(value for _, value in self.calibration)
                    self.calibration.clear()
            return self.phase

        excursion = knee - self.baseline
        if not self.armed:
            self.armed = excursion <= self.RETURN_DEG
            return self.phase
        if self.phase == "setup":
            if excursion < self.START_DEG or speed < self.MOTION_DEG_S:
                return self.phase
            self.phase = "descent"
            self.peak = knee
        self.frames.append((time, metrics))
        self.peak = max(self.peak, knee)
        if time - self.frames[0][0] > self.MAX_REP_S:
            self.reset()
            return self.phase

        deep_enough = self.peak - self.baseline >= self.MIN_EXCURSION_DEG
        if self.phase in {"descent", "bottom"} and deep_enough:
            if self.peak - knee >= self.TURN_DEG and speed < 0:
                self.phase = "ascent"
            elif self.peak - knee <= self.TURN_DEG and speed <= self.MOTION_DEG_S:
                self.phase = "bottom"
            elif speed > self.MOTION_DEG_S:
                self.phase = "descent"
        if excursion <= self.RETURN_DEG:
            if self.phase == "ascent":
                self.reps.append(self.summarize())
                self.frames.clear()
            self.reset()
        return self.phase

    def summarize(self):
        start = self.frames[0][0]
        end = self.frames[-1][0]
        peak_index = max(range(len(self.frames)), key=lambda i: self.frames[i][1]["knee_flexion_deg"])
        first = last = peak_index
        while first > 0 and self.near_peak(first - 1):
            first -= 1
        while last + 1 < len(self.frames) and self.near_peak(last + 1):
            last += 1
        bottom_start, bottom_end = self.frames[first][0], self.frames[last][0]
        values_by_metric = {}
        for _, metrics in self.frames:
            for name, value in metrics.items():
                values_by_metric.setdefault(name, []).append(value)
        return {
            "rep": len(self.reps) + 1,
            "start_s": start,
            "end_s": end,
            "bottom_start_s": bottom_start,
            "bottom_end_s": bottom_end,
            "descent_s": bottom_start - start,
            "bottom_s": bottom_end - bottom_start,
            "ascent_s": end - bottom_end,
            "tracked_samples": len(self.frames),
            "angles": {
                name: {"min": min(values), "max": max(values), "mean": mean(values), "samples": len(values)}
                for name, values in values_by_metric.items()
            },
        }

    def near_peak(self, index):
        return self.peak - self.frames[index][1]["knee_flexion_deg"] <= self.BOTTOM_WINDOW_DEG
