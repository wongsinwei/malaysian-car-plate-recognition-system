# Malaysian Car Plate Recognition System

A desktop application for real-time vehicle detection and Malaysian license
plate recognition, built as a Final Year Project. It combines YOLO-based
object detection with EasyOCR text recognition, layered with custom logic
to correct common OCR misreads against real Malaysian plate formats.

## Features

- **Real-time detection** from a live camera feed, using a custom-trained
  YOLO segmentation model to locate vehicles and license plates.
- **License plate OCR** via EasyOCR, with a post-processing correction layer
  that:
  - Enforces valid plate structure (leading letters, max consecutive
    digits, optional trailing letters).
  - Applies context-aware character correction (e.g. `0↔O`, `1↔I/L`,
    `8↔B`, `5↔S`, `2↔Z`) based on whether a character falls in the
    prefix, numeric, or suffix zone of the plate.
  - Detects and preserves vanity plates rather than "correcting" them
    incorrectly.
  - Rejects partial/cropped reads that are missing expected letters.
- **Vehicle color classification** using a secondary YOLO model.
- **Distance and size estimation** for detected vehicles, with an
  adjustable focal length calibration setting.
- **Buffered majority-vote plate confirmation** — plate reads are
  buffered across frames and only committed once a stable, confident
  result emerges, reducing false positives from OCR flicker.
- **User accounts and history** — login/registration, per-user detection
  logs, manual correction of saved plates, and record editing, backed by
  Firebase.
- **Desktop GUI** built with `customtkinter`, including camera selection,
  live dashboard, and admin/history views.

## Tech Stack

- **UI:** customtkinter, Pillow
- **Computer Vision:** OpenCV, Ultralytics YOLO
- **OCR:** EasyOCR
- **Backend/Storage:** Firebase Realtime Database
- **Language:** Python 3

## Project Structure

```
├── LRP_system.py                  # Main application (GUI, camera, detection loop)
├── final_system_segmentation.py   # Firebase config and auth/db manager
├── best.pt                        # YOLO model — vehicle & plate detection/segmentation
├── color.pt                       # YOLO model — vehicle color classification
├── requirements.txt
└── README.md
```

> The packaged Windows build is distributed as an installer
> (`Setup.exe`) on the [Releases](../../releases) page rather than
> committed to the repo directly, due to its size.

## Getting Started

### Download & Run (recommended)

No Python installation required.

1. Go to the [**Releases**](../../releases/latest) page.
2. Download `Setup.exe` from the latest release (**v1.1.0**).
3. Run `Setup.exe` and follow the installation wizard.
4. Once installed, launch the application from the Start Menu or
   desktop shortcut created by the installer.

   > ⚠️ **Do not move `LRP_system.exe` out of its installed folder on
   > its own.** It depends on the accompanying `_internal` folder
   > sitting next to it — moving or copying only the `.exe` will
   > prevent the app from launching. Use the shortcut created by the
   > installer, or run it directly from the installed folder.
5. Log in or register, select a camera source, then launch the
   detection dashboard.

> A Firebase project is required for login/registration and cloud sync
> to work. If you're running this outside of the original developer
> setup, see [Running from Source](#running-from-source) for
> configuration details.

### Running from Source

For development, or to modify the detection/OCR logic.

#### Prerequisites

- Python 3.9+
- A webcam or IP camera feed
- A Firebase project (Realtime Database) — see
  `final_system_segmentation.py` for the expected configuration.

#### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wongsinwei/malaysian-car-plate-recognition-system.git
   cd malaysian-car-plate-recognition-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download `best.pt` and `color.pt` from the
   [latest Release](../../releases/latest) and place them in the project
   root (same folder as `LRP_system.py`).
4. Run the app:
   ```bash
   python LRP_system.py
   ```

## Known Limitations / Future Work

- The packaged application is built for **Windows only** (camera
  connection uses `cv2.CAP_DSHOW`, and the release is a Windows
  installer).
- Distance estimation assumes an average car width for all vehicle
  types; this could be made more accurate by branching the calibration
  by detected vehicle class (car / motorcycle / truck).
- Firebase security rules and password handling should be reviewed
  before any production deployment.

## License

Specify your license here (e.g. MIT).
