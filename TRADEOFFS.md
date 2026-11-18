# Engineering trade-offs

Notes on the decisions behind hoopsvision, written to be adapted into a case study.
Each section covers the problem, the choice made, what it cost, and what the data
showed.

## 1. The homography is the product

Every tactical number (speed, spacing, space control, open space) is a function of
court coordinates. A detector that is 5% worse costs a few missed players; a
homography that is 5 ft off corrupts every metric on that frame, silently. So the
design rule for calibration was **never emit a number from a homography you can't
verify**:

- Each frame carries a confidence. Metrics are NaN below the threshold, and the
  dashboard greys out its tiles instead of showing stale or invented values.
- Confidence comes from two independent checks: agreement with detected court
  lines, and overlap between the projected lane and the observed paint. Line
  agreement alone isn't enough, because ICP can "earn" agreement by snapping to
  player edges. Adding the lane check removed confidently wrong frames that the
  line score had passed.
- A propagated (motion-carried) homography that no lane check has verified fades
  to zero over 2 s. The cost is gaps in the analytics during long pans. The benefit
  is that drift can't masquerade as data.

**Why classical, not a keypoint model?** A court-keypoint network is the stronger
long-term answer: it handles unpainted lanes and odd angles and gives many more
correspondences. Classical was chosen for this build because it needs no labelled
data or hosted weights, every stage can be inspected in the debug video, and the
verify step is reusable as-is. A keypoint model would plug in as one more
hypothesis generator next to the lane quads; the ICP refinement, confidence,
smoothing and propagation stay unchanged. On this clip the classical approach
reaches 86% of frames above the threshold. Its weak spot is a low-angle camera
whose near lane edge is hidden by the scorebug, which is exactly the case where
learned keypoints help.

## 2. Generate-and-verify beats clever single-shot detection

The first lane detector took "non-floor colour, largest blob, fit a quad". It
failed on most frames. The lane here is two colours (fill plus a dark border band),
players cut the blob into pieces, and the scorebug adds strong straight edges. Each
targeted fix (colour clustering, union of clusters, RANSAC lines on mask boundaries
ignoring occluder edges, circular angle grouping into two line families) made more
*candidates* right but none made candidates reliable. What made the system reliable
was cheap verification: generate several hypotheses, score them against
independent evidence, and keep the best.

Basket side (left or right lane) followed the same lesson. Letting the line score
choose between two side hypotheses was unreliable, because ICP can fit either one.
A geometric prior (end lines lean like "\\" on the right half of a sideline view)
made it deterministic.

## 3. Scene cuts: the obvious signal was wrong on real data

An HSV histogram cut detector flagged five cuts in the sample. All five were
*exposure flashes*: the broadcast's grading pulses the image, with identical camera
geometry. It also missed a true cut between two cameras with similar colours. The
replacement compares *structure* (correlation of gradient images after histogram
equalization) and treats a colour jump with intact structure as a flash. On the
clip the two real cuts score about 0.1 correlation and all other frames ≥ 0.58,
a wide margin. The same insight (equalize first) then fixed optical-flow camera
motion across flashes, which had been seeding calibration drift.

Cutting is still a reset, not a re-identification. IDs restart after each cut. That
keeps IDs honest at the cost of per-player continuity, and appearance re-ID across
cuts is the obvious next step.

## 4. Detection: recall over keypoints

The spec asks for pose skeletons. Pose models on 360p broadcast found 3–9 people
per frame; a same-family detection model found 15–17. Using pose as the detector
would have halved the tracked players. Instead detection drives tracking, and
pose keypoints are attached by IoU matching. Skeletons appear where pose finds a
person, and the analytics don't depend on pose. Upscaled inference (imgsz 1280) was
the other big recall lever, at about 300 ms/frame on CPU, the dominant cost of the
pipeline.

## 5. Team identification: what goes wrong is the role assignment, not the clustering

KMeans on torso colour separated white, blue and referee/other cleanly. The bugs
were in *naming* the clusters:

- "Referee = cluster with the fewest tracks" picked a team, because that team's
  tracks fragmented more under occlusion. Average simultaneous people per frame
  (refs ~1–3, teams ~2.5–5) is the robust signal.
- Plain majority voting lost players to the catch-all cluster (occluded or blurry
  crops land there). A track is now a referee only if ≥60% of its votes say so.
- Lightness is measured relative to the floor, so exposure changes don't move a
  white jersey across the boundary.

Team display colours (blue/amber) are deliberately *not* jersey colours. They come
from config and were checked for colour-vision-deficiency separation on the dark
background, so the dashboard reads the same for every matchup.

## 6. Kinematics: tests found real bugs

The synthetic test suite (a player at a known 5 m/s; glitches; gaps) caught three
issues that would have been invisible in the dashboard:

- A centred moving average shrank motion at track ends, so distance read about 6%
  low. Switched to per-run Savitzky–Golay, which preserves linear motion at the
  edges.
- `pandas.interpolate(limit=6)` *partially* fills long gaps (the first 6 frames of a
  20-frame hole). Replaced with run-length-aware filling.
- Frontcourt space control clamped players outside the half onto its boundary.
  Now it computes full-court Voronoi cells and clips them to the half.

Physically impossible steps (> 36 km/h) are *discarded*, not clipped: a 15 ft
projection glitch must add zero distance, not "36 km/h worth" of distance.

## 7. Caching keyed by code, not just config

Each stage's cache key hashes the video fingerprint, the config sections the stage
reads, the upstream keys, and **the stage's own source file**. Editing `teams.py`
re-runs teams and everything downstream, while detection (the 4-minute stage) stays
cached. This kept iteration on calibration and rendering in the seconds-to-minutes
range on CPU. `render` reads only cached tables, so visual changes never touch a
model.

## 8. What the numbers mean (and don't)

About 5–6 of ~10 players have valid court positions per analytics frame on this
clip. Team spacing (hull area) and distance covered are therefore **lower bounds**
and should be read comparatively (team A vs team B, possession vs possession).
Space control is more robust to missing players than hull area, because Voronoi
cells redistribute space rather than shrink. The possession summary reports a
`valid_frame_pct` so downstream users can weight or filter possessions by coverage.

## Next steps, in priority order

1. A court-keypoint model as an extra calibration hypothesis (biggest accuracy gain).
2. Camera-motion-compensated tracking (warp the Kalman state by the frame-to-frame
   homography already computed), to cut ID fragmentation during pans.
3. Appearance re-ID across cuts, for per-player continuity.
4. A ball tracker plus shot-clock OCR for possession boundaries, replacing the
   team/side heuristic.
5. A replay/logo-wipe classifier, to avoid double-counting same-camera replays.
