# AegisDrive-AI

> **🚧 Project Status: In Development**
>
> AegisDrive AI is an ongoing academic project currently under active development. Some features are experimental, while others are still being implemented and evaluated. The project's architecture, functionality, and documentation may change as development progresses.

- [Academic Supervision](#academic-supervision--acknowledgments)

An educational computer vision project focused on road-scene understanding and driver-assistance research. The project explores techniques for analyzing road environments and building a simplified prototype of an Advanced Driver Assistance System (ADAS).

The repository combines two complementary approaches:

- **YOLOP**, an open-source deep-learning model for unified perception in autonomous-driving research (experimental integration).
- **A classical computer-vision module (`lane_detector.py`)** that detects lane lines, estimates the ego-vehicle path, assigns detected vehicles to lanes, and displays approximate distances on a head-up display (HUD).

> **Safety Disclaimer:** This project is intended exclusively for educational and research purposes. It is not a certified ADAS or autonomous driving system. Its outputs may be inaccurate, incomplete, or unreliable. Never use this software as a substitute for driver attention or as a safety-critical system in a real vehicle.

---

## Table of Contents

- [Project Goals](#-project-goals)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Lane Detector Module](#-lane-detector-module)
- [Project Structure](#-project-structure)
- [YOLOP Overview](#-yolop-overview)
- [Limitations](#-limitations)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Author](#-author)

---

## 🎯 Project Goals

The primary goals of this project are to:

- Explore deep-learning-based road-scene perception.
- Experiment with vehicle and road-object detection.
- Investigate lane-line and drivable-area segmentation.
- Develop a modular foundation for a simplified ADAS prototype.
- Compare learned (YOLOP) and classical (OpenCV) approaches to lane perception.
- Evaluate computer vision models on road-scene videos.
- Establish a reproducible Python development environment.
- Gradually introduce visualization, performance measurement, and result export.

These goals describe the intended direction of the project, not a guarantee that every feature is already implemented.

## ✨ Features

| Feature                         | Description                                                            | Status                     |
| ------------------------------- | ---------------------------------------------------------------------- | -------------------------- |
| Lane-line detection (classical) | Detects and completes highway lane lines, including dashed markings    | Implemented (experimental) |
| Ego-vehicle path                | Highlights the driving lane of the camera vehicle with a blue gradient | Implemented (experimental) |
| Vehicle-to-lane assignment      | Determines which lane each detected vehicle occupies                   | Implemented (experimental) |
| Distance display (HUD)          | Shows approximate distance to the front, left, and right vehicles      | Implemented (experimental) |
| Traffic-sign candidates         | Heuristic detection based on color and shape                           | Experimental               |
| Installation script             | Automated environment setup and dependency verification on Windows     | Implemented                |
| YOLOP integration               | Experimentation with an existing road-perception model                 | In Progress                |
| Drivable-area segmentation      | Investigating road-area segmentation with YOLOP                        | Under Evaluation           |
| Video processing workflow       | End-to-end processing of road videos                                   | In Development             |
| Performance evaluation          | Measuring inference speed and prediction quality                       | Planned                    |
| Result export                   | Saving processed videos and evaluation reports                         | Planned                    |

Update these statuses as each feature is implemented and tested.

## 🧰 Technology Stack

- **Python 3.11** — primary programming language.
- **OpenCV** — image processing, line detection, and drawing.
- **NumPy** — numerical operations.
- **PyTorch** — deep-learning framework used by YOLOP.
- **YOLOP** — unified road-perception model (external repository).
- **Ultralytics YOLO** — optional; used by the demo in `lane_detector.py` to supply vehicle bounding boxes.
- **Git** — version control.

The exact dependency list is defined in `requirements.txt`.

## 📦 Requirements

Before installing the project, make sure you have:

- Python 3.11, available through the `py` launcher on Windows.
- Git.
- A working internet connection for downloading dependencies and model weights.
- Sufficient disk space for Python packages and model files.
- A compatible CPU or GPU.

An NVIDIA GPU with a compatible CUDA configuration can accelerate inference. CPU execution is also possible, although slower.

## 🔧 Installation

### 1. Clone the repository

Replace `<your-username>` with your GitHub username.

```bash
git clone https://github.com/<your-username>/AegisDrive AI.git
cd AegisDrive AI
```

### 2. Create a virtual environment

**Windows:**

```powershell
py -3.11 -m venv .venv
```

**Linux / macOS:**

```bash
python3.11 -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows CMD:**

```bat
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### Automated installation on Windows

Alternatively, run the installation script from the project root:

```bat
install.bat
```

The script performs the following steps:

1. Verifies that the Python launcher (`py`) and `requirements.txt` exist.
2. Creates a Python 3.11 virtual environment in `.venv` if one does not exist.
3. Upgrades `pip`.
4. Installs the dependencies, trying the official PyPI index first and then falling back to the Tsinghua and Aliyun mirrors if the previous attempt fails.
5. Imports the main libraries and prints their versions, including whether CUDA is available.

### 5. Verify the installation

Check the Python version:

```bash
python --version
```

Check the core computer vision dependencies:

```bash
python -c "import cv2, numpy, torch; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If a dependency is missing, review the installation output and the contents of `requirements.txt`.

## 🚀 Usage

The execution command depends on the entry point you want to use.

### Lane detector demo

The demo reads a video, obtains vehicle boxes from Ultralytics YOLO, and writes an annotated video to `lane_demo.mp4`:

```bash
python lane_detector.py path/to/video.mp4
```

Press `q` in the preview window to stop. The demo requires the `ultralytics` package and downloads `yolov8n.pt` on first use.

### Using the module in your own code

```python
import cv2
from lane_detector import LaneDetector

detector = LaneDetector()

cap = cv2.VideoCapture("path/to/video.mp4")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Vehicle boxes come from any detector (YOLO, YOLOP, etc.)
    vehicles = [
        {"box": (x1, y1, x2, y2), "cls": "car", "id": 7},
        # ...
    ]

    annotated, info = detector.process(frame, vehicles)
    cv2.imshow("Result", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Each vehicle may be a dictionary with `box`, `cls`, and optionally `id`, or simply a `(x1, y1, x2, y2)` tuple. Pedestrian classes (`person`, `pedestrian`) are ignored by this module.

### Streamlit interface

If the repository contains a Streamlit application named `app.py`, launch it with:

```bash
streamlit run app.py
```

Otherwise, use the documented entry point for the current implementation.

### YOLOP experiments

For YOLOP experiments, the workflow generally involves:

1. Preparing a compatible road-scene image or video.
2. Installing the model's required dependencies.
3. Obtaining the appropriate pretrained model weights.
4. Running the model's supported inference procedure.
5. Reviewing the generated predictions.
6. Recording performance and evaluation results.

Consult the model's official repository for its supported setup and inference instructions.

## 🚗 Lane Detector Module

`lane_detector.py` implements a classical (non-learned) pipeline for lane perception and simple scene analysis.

### What it produces

- **All detected lane lines**, drawn in red. Dashed markings are completed, and a missing line between two detected lines may be inferred and drawn as a dashed line.
- **The ego path**, drawn as a blue gradient between the two lane lines surrounding the camera vehicle.
- **Per-vehicle lane information**: each vehicle receives a lane label (for example `L3`), and the lines on its left and right are emphasized.
- **A head-up display** showing the approximate distance to the nearest vehicle in the ego lane (front) and in the adjacent lanes (left and right). Distances are color-coded by the `warn_dist` and `danger_dist` thresholds.
- **Traffic-sign candidates**, detected heuristically from color and shape.

### Processing pipeline

1. **Marking mask:** white and yellow markings are extracted in HSV space. When markings are faint, a top-hat filter supplies additional candidates. Wide bright regions and vehicle boxes are removed.
2. **Line segments:** edges are extracted with Canny and passed to a probabilistic Hough transform.
3. **Line extraction:** segments are converted to points and grouped by voting on line slope and position, so that fragments of a dashed line are merged into one line.
4. **Temporal tracking:** lines are smoothed across frames and retained briefly when they disappear.
5. **Gap filling:** missing lines between widely spaced neighbors are inferred.
6. **Vanishing point:** estimated from the intersection of the tracked lines.
7. **Lane assignment:** each vehicle is located between two lines at its ground-contact point.
8. **Distance estimation:** approximated from the bounding-box width, an assumed real-world vehicle width, and the camera's field of view.

### Main parameters

| Parameter      | Default | Description                                                            |
| -------------- | ------- | ---------------------------------------------------------------------- |
| `roi_top`      | `0.45`  | Top boundary of the road region, as a fraction of image height         |
| `hood_ratio`   | `0.0`   | Fraction of the image bottom occupied by the vehicle hood              |
| `ego_x_ratio`  | `0.5`   | Horizontal position of the camera vehicle in the image                 |
| `hfov_deg`     | `70.0`  | Horizontal field of view, used for distance estimation                 |
| `focal_px`     | `None`  | Focal length in pixels; overrides `hfov_deg` when provided             |
| `lane_width_m` | `3.5`   | Assumed lane width in meters                                           |
| `smooth`       | `0.3`   | Temporal smoothing factor for lane lines                               |
| `min_hits`     | `2`     | Frames required to confirm a line (use `1` for single images)          |
| `max_missed`   | `20`    | Frames a lost line is retained                                         |
| `fill_gaps`    | `True`  | Infer lane lines missing between detected lines                        |
| `enable_signs` | `True`  | Enable heuristic traffic-sign detection                                |
| `enable_hud`   | `True`  | Draw the head-up display                                               |
| `warn_dist`    | `20.0`  | Distance in meters below which the HUD turns orange                    |
| `danger_dist`  | `10.0`  | Distance in meters below which the HUD turns red                       |
| `occlusion`    | `body`  | `none`, `body`, or `full`: how much of each vehicle hides the overlays |
| `debug`        | `False` | Draw raw Hough segments in yellow                                      |

### Returned information

`process()` returns the annotated frame and a dictionary with the following keys: `lane_count`, `ego_lane`, `vanishing_point`, `lane_lines`, `vehicles`, `front`, `left`, `right`, and `signs`. This dictionary can be used for logging and event reports.

## 📁 Project Structure

The directory structure depends on the current implementation. A possible organization is:

```text
AegisDrive AI/
├── README.md
├── requirements.txt
├── .gitignore
├── install.bat                # Windows installation script
├── lane_detector.py           # Classical lane-perception module and demo
├── external/
│   └── YOLOP/                 # External model repository, if cloned
├── src/                       # Additional project-specific code
├── assets/
│   └── sample_videos/         # Optional sample videos
├── outputs/                   # Generated results
└── .venv/                     # Local Python environment; not committed
```

This is an illustrative structure, not a guarantee that every directory already exists.

The `.venv` directory, generated outputs, and large local assets should generally remain outside version control.

## 🧠 YOLOP Overview

YOLOP is an open-source perception model developed for autonomous-driving research.

Its original architecture focuses on three perception tasks:

- **Traffic-object detection:** identifying relevant objects in road scenes.
- **Lane-line detection:** identifying lane markings.
- **Drivable-area segmentation:** estimating which parts of the scene are suitable for driving.

These capabilities are model-specific. They should not be confused with independently implemented object tracking, metric distance estimation, collision prediction, or vehicle-control functionality.

The original project and its implementation details are available in the [official YOLOP repository](https://github.com/hustvl/YOLOP).

### YOLOP and the classical module

| Aspect             | YOLOP (learned)                           | `lane_detector.py` (classical)                          |
| ------------------ | ----------------------------------------- | ------------------------------------------------------- |
| Approach           | Neural network trained on driving data    | Color, edge, and geometric processing                   |
| Requires weights   | Yes                                       | No                                                      |
| Hardware demands   | Higher; benefits from a GPU               | Lower; runs on a CPU                                    |
| Strengths          | Can generalize across varied appearances  | Transparent, tunable, and easy to inspect               |
| Typical weaknesses | Depends on training data and domain match | Sensitive to lighting, markings quality, and thresholds |

Comparing the two on the same videos is one of the project's evaluation goals.

## 🚧 Limitations

- Predictions depend on the model, its training data, and the quality of the input.
- Poor lighting, rain, glare, worn markings, unusual camera angles, and unfamiliar road environments may reduce accuracy.
- The classical lane module draws straight lines; sharp curves are approximated less accurately.
- Distances are approximate. They depend on the assumed vehicle widths, the configured field of view, and the camera mounting, and they are less reliable when a vehicle is partially outside the frame.
- Traffic-sign detection is heuristic and may produce false positives or miss signs. A trained sign-detection model is recommended for reliable results.
- Pedestrian detection and pedestrian-crossing recognition are not part of the lane module.
- GPU acceleration depends on compatible hardware, drivers, and PyTorch configuration.
- Lane perception is not equivalent to a complete lane-keeping system.
- Collision-risk assessment requires additional validated logic and appropriate sensor or camera calibration.
- No safety certification or real-world driving validation is implied.

## 🩺 Troubleshooting

**Lane lines are not drawn.**
Run `LaneDetector(debug=True)` to display the raw Hough segments. If none appear, adjust `roi_top` and `hood_ratio` so that the region of interest covers the road and excludes the vehicle hood. When processing a single image rather than a video, set `min_hits=1`.

**Overlays disappear behind vehicles.**
Set `occlusion="none"` to draw the path and lines over vehicles, or keep the default `"body"` to hide them only behind the vehicle body.

**Distances look incorrect.**
Set `hfov_deg` (or `focal_px`) to match your camera. If the camera is not centered on the vehicle, adjust `ego_x_ratio`.

**Dependency installation fails.**
Check your internet connection, firewall, and proxy settings. On Windows, `install.bat` retries with alternative package mirrors; if all fail, keep the full error message for diagnosis.

**CUDA is reported as unavailable.**
Confirm that you have a compatible NVIDIA GPU, an up-to-date driver, and a PyTorch build that supports your CUDA version.

## 🧭 Roadmap

- [x] Provide an automated Windows installation script.
- [x] Implement a classical lane-perception module with a HUD.
- [ ] Complete and document the YOLOP setup.
- [ ] Verify model inference on sample road-scene videos.
- [ ] Organize project-specific code into reusable modules.
- [ ] Add clear visualizations of model predictions.
- [ ] Measure inference speed and resource usage.
- [ ] Compare YOLOP and the classical module on the same videos.
- [ ] Document supported input formats and execution commands.
- [ ] Add automated tests for project-specific functionality.
- [ ] Establish a reproducible evaluation workflow.
- [ ] Improve distance estimation through camera calibration.
- [ ] Investigate collision-risk indicators as a research feature.
- [ ] Export processed videos and event reports.

## 🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Commit your changes with a descriptive message.
5. Push your branch.
6. Open a pull request.

Example:

```bash
git checkout -b feature/road-scene-visualization
git add .
git commit -m "feat: add road-scene visualization"
git push origin feature/road-scene-visualization
```

When reporting a bug, please include the Python version, operating system, the command you ran, the full error message, and, where possible, a sample frame that reproduces the problem.

## 📄 License

The license for this project should be specified in the repository's `LICENSE` file.

If you distribute or modify YOLOP or other third-party components, comply with their respective licenses and attribution requirements.

## 🙏 Acknowledgments

- The authors of [YOLOP](https://github.com/hustvl/YOLOP). If you use YOLOP in your work, please cite the original paper as indicated in the official repository.
- The OpenCV, NumPy, PyTorch, and Ultralytics communities for their open-source tools.

## Academic Supervision & Acknowledgments

**Project Developer:** Mohammad Taha Rezazadeh
**Academic Supervisor:** Ms. Farahani

AegisDrive AI is developed by Mohammad Taha Rezazadeh as an academic computer vision project. The project has been carried out with the guidance and academic support of Ms. Farahani, whose supervision has contributed to the development and progress of this work.

I would like to express my sincere appreciation to Ms. Farahani for her guidance, support, and valuable contributions throughout the project.

This project reflects my ongoing efforts to explore computer vision, deep learning, and intelligent driver-assistance technologies in an academic and research-oriented environment.

## 👤 Author

**Project Developer:** Mohammad Taha Rezazadeh
**Academic Supervisor:** Ms. Farahani
Computer Vision and Deep Learning Enthusiast

---

If you find this project useful, consider giving the repository a star.
