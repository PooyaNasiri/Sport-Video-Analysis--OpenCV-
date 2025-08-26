# Sport Video Analysis — OpenCV (C++17)

Lightweight football (soccer) video analysis in C++ with OpenCV. The pipeline detects players on the pitch using classical CV, clusters them into two teams by jersey color, and produces per-team heatmaps. An evaluation tool computes IoU-based metrics against a YOLO pseudo-ground-truth CSV.

> Course project — Computer Vision, University of Padova  
> Author: Pooya Nasiri (Student ID: 2071437)

---

## Highlights

- **Pure OpenCV + C++17**: no training required, runs on CPU.
- **Player detection**: field masking + background subtraction + contour filtering.
- **Team classification**: Lab-color features with k-means and temporal anchoring.
- **Heatmaps**: per-team spatial density and overlay images.
- **Metrics**: IoU matching vs. YOLO CSV to report Precision/Recall/F1/mIoU.

---

## Repo Structure

```
.
├─ CMakeLists.txt
├─ main.cpp                # entrypoint: runs detection, classification, heatmap
├─ detection.h/.cpp        # field mask, player mask, contouring, box merge
├─ classification.h/.cpp   # jersey-color features, k-means, temporal anchors
├─ heatmap.h/.cpp          # accumulation and visualization, PNG export
└─ eval.cpp                # IoU-based evaluation tool (ours.csv vs yolo.csv)
```

---

## Dependencies

- C++17 toolchain (g++/clang++)
- CMake ≥ 2.0
- OpenCV (tested with 4.x): core, imgproc, highgui, video, videoio

Ubuntu example:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev
```

---

## Build

### Recommended: CMake

```bash
mkdir build && cd build
cmake ..
make
```

This produces:

- `detect` — main detection pipeline (from `main.cpp`)
- Optionally build `eval` (see below)

> Note: `CMakeLists.txt` defines only `detect` by default. Build `eval` via one of the options in the Evaluation section.

### Alternative: Direct compile

```bash
# detection pipeline
g++ -std=c++17 main.cpp detection.cpp classification.cpp heatmap.cpp \
    `pkg-config --cflags --libs opencv4` -o detect

# evaluation tool
g++ -std=c++17 eval.cpp -o eval
```

---

## Usage

### 1) Run detection

```bash
./detect path/to/input_video.mp4
```

Windows close keys: press `q` or `Esc` in the video window.

**Outputs**

- `ours.csv` with header:
  ```
  frame,x1,y1,x2,y2,team
  ```
  where `team` is `0` = Team A (red overlay), `1` = Team B (blue overlay), `2` = Unknown (green overlay).
- Display windows:
  - `"Football Player Detection"` — annotated frames
  - `"Green Field Mask"` — binary pitch mask
  - `"Players"` — masked non-green regions
- Heatmap images on exit:
  - `combined_heatmap.png`
  - `heatmap_overlay.png`

### 2) Evaluate vs YOLO CSV

Prepare a YOLO detections CSV (`yolo.csv`) for the same video (person class only), then:

```bash
# build eval if not built yet
g++ -std=c++17 eval.cpp -o eval

# run
./eval ours.csv yolo.csv [iou_thr=0.5] [ours_offset=0] [yolo_offset=0]
```

**Example**

```bash
./eval ours.csv yolo.csv 0.5 0 -1
```

**Printed metrics**

```
TP=... FP=... FN=...
Precision=... Recall=... F1=... mIoU=...
```

Offsets help align frame indices if your CSVs start at different frames.

> Generating `yolo.csv`: run your preferred YOLO on the video, export per-frame bounding boxes, and convert to a 5-column CSV: `frame,x1,y1,x2,y2`. Ensure frames match the same resolution and indexing as `ours.csv`.

---

## How It Works

### Detection (`detection.cpp`)

1. **Pitch mask (HSV)**
   - Threshold green: `H≈40..90, S,V≥40`, then morphological clean-up.
   - Keep only large contours to isolate the field region.

2. **Foreground motion**
   - MOG2 background subtraction with a low learning rate.

3. **Player mask**
   - On the field-masked frame, suppress green and near-black to keep jersey regions, then dilate.

4. **Contours → boxes**
   - Filter by area and plausible sizes (`w∈[10,100], h∈[20,200]`), then merge overlapping boxes to avoid duplicates.

### Team Classification (`classification.cpp`)

- For each detected box:
  - Resize ROI to `32×64`.
  - Convert to Lab, remove green pixels (estimated via HSV), average the top-energy non-green Lab vectors to get a compact color descriptor.
- Run **k-means (k=2)** on descriptors per frame.
- **Temporal anchors** stabilize team labels across frames by slowly updating cluster centers over the first N frames.
- Simple spatial association with previous frame prevents flip-flops when objects are near.

### Heatmaps (`heatmap.cpp`)

- For each classified box, draw a small filled circle at the box center onto an RGB accumulator channel indexed by team.
- After the video:
  - Gaussian blur, normalize to 0..255.
  - Save combined heatmap and an overlay blended with the first frame.

---

## CSV Formats

- `ours.csv`
  ```
  frame,x1,y1,x2,y2,team
  0,  123,45,  170,160, 0
  0,  ...
  1,  ...
  ```
- `yolo.csv`
  ```
  frame,x1,y1,x2,y2
  0,  120,44,  172,161
  ...
  ```

All coordinates are pixel-space with the video’s original resolution. Frames are zero-based as produced by OpenCV’s `VideoCapture`.

---

## Tuning Tips

- **BackgroundSubtractorMOG2**: created with history `500`, varThreshold `16`, shadows disabled. Increase history for steadier backgrounds.
- **HSV thresholds**: adjust green ranges for different pitches/lighting.
- **Box filters**: widen `[w,h]` ranges for different camera zooms.
- **Team stability**: temporal anchors update for the first ~10 frames; increase if early frames are unstable.

---

## Known Limitations

- No multi-object tracking or ID persistence; metrics are per-frame.
- Color-based team clustering can struggle with green kits or harsh lighting.
- Evaluation uses YOLO pseudo-ground truth, not human labels.

---

## Example Workflow

```bash
# 1) Build
mkdir build && cd build && cmake .. && make

# 2) Detect
./detect ../sample_match.mp4   # produces ours.csv and heatmaps

# 3) Evaluate
# Assume you created yolo.csv from the same video
./eval ../ours.csv ../yolo.csv 0.5 0 -1
```

---

## License

Add your chosen license file (e.g., MIT, Apache-2.0).

---

## Acknowledgments

- OpenCV community and documentation  
- Inspiration from classical CV pipelines for sports footage analysis
