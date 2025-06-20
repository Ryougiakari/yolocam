# YOLOv8 Grad-CAM Heatmap Generation

This project provides the code structure and scripts to generate heatmaps on images using a pre-trained YOLOv8 object detection model and the Grad-CAM technique. The heatmaps visualize the regions of an image that the YOLOv8 model focuses on for its detections.

## Prerequisites

*   Python 3.8+
*   Sufficient disk space for PyTorch and other large libraries.

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    # If this were a git repo:
    # git clone <repository_url>
    # cd yolov8_gradcam
    ```
    For now, ensure you have all the files (`main.py`, `requirements.txt`, etc.) in a local directory.

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Important Note on Dependencies:** The `torch` (PyTorch) library, a core dependency for `ultralytics` (YOLOv8), is quite large. Installation might fail in environments with limited disk space (e.g., some cloud sandboxes). Ensure you are running this in an environment with several gigabytes of free space. If you encounter space issues, you might need to manage your environment's disk usage or use a machine with more resources.

## How it Works (Conceptual)

1.  **Object Detection:** YOLOv8 processes an input image to detect objects and their bounding boxes.
2.  **Target Layer Selection:** For Grad-CAM, a target convolutional layer within the YOLOv8 model is selected. This layer is typically one of the last convolutional layers, as it contains rich spatial and semantic information.
3.  **Grad-CAM Calculation:** Grad-CAM uses the gradients of the class score (for a detected object) with respect to the feature maps of the target layer. This produces a coarse localization map highlighting important regions for that specific class.
4.  **Heatmap Generation & Overlay:** The localization map is upscaled to the input image size and overlaid as a heatmap, providing a visual explanation for the detection.

## Usage

1. Download a pre-trained YOLOv8 model and place it inside the `models` folder.
2. Run `python main.py PATH_TO_IMAGE` to generate a heatmap.
   Use `--output` to change the output path and `--device cuda` for GPU acceleration.

## Downloading a Pre-trained YOLOv8 Model

You'll need a pre-trained YOLOv8 model file (e.g., a `.pt` file).

1.  **Choose a model:** Visit the official YOLOv8 repository or Ultralytics documentation to see available models (e.g., `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`). Smaller models like `yolov8n.pt` are faster and require less disk space, while larger models offer higher accuracy.
2.  **Download the model:**
    *   You can often find direct download links. For example, for `yolov8n.pt`:
        ```
        wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
        ```
    *   Alternatively, if you have the `ultralytics` package installed in an environment, you can download it via Python:
        ```python
        from ultralytics import YOLO
        # This will download yolov8n.pt if not present locally
        model = YOLO('yolov8n.pt')
        ```
3.  **Place the model:** Save the downloaded `.pt` file (e.g., `yolov8n.pt`) into the `models` directory within this project structure:
    ```
    yolov8_gradcam/
    ├── models/
    │   └── yolov8n.pt  <-- Place your downloaded model here
    ├── examples/
    ├── main.py
    ├── README.md
    └── requirements.txt
    ```
