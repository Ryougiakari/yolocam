import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return rgb, tensor.unsqueeze(0)


def generate_heatmap(model_path: Path, image_path: Path, output_path: Path, device: str = "cpu"):
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # Load YOLOv8 model
    yolo = YOLO(str(model_path))
    yolo.model.to(device).eval()

    rgb, tensor = load_image(image_path)
    tensor = tensor.to(device)

    # Forward pass through the model backbone
    with torch.no_grad():
        _ = yolo.model(tensor)

    # Run detection for class predictions
    results = yolo.predict(source=rgb, device=device, verbose=False)
    preds = results[0]
    if len(preds.boxes) == 0:
        raise RuntimeError("No objects detected in the image")
    class_id = int(preds.boxes.cls[0].item())

    # Set up Grad-CAM on the last convolutional block
    target_layer = list(yolo.model.model.named_children())[-2][0]
    cam_extractor = SmoothGradCAMpp(yolo.model, target_layer=target_layer)

    scores = yolo.model(tensor)
    loss = scores[0, class_id]
    yolo.model.zero_grad()
    loss.backward()

    activation_map = cam_extractor(tensor)[0].cpu()
    heatmap = to_pil_image(activation_map, mode="F")
    result = overlay_mask(to_pil_image(rgb), heatmap, alpha=0.5)
    result.save(output_path)

    print(f"Heatmap saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLOv8 Grad-CAM heatmap")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--model", dest="model", type=Path, default=Path("models/yolov8n.pt"), help="Path to YOLOv8 model")
    parser.add_argument("--output", dest="output", type=Path, default=Path("heatmap.jpg"), help="Output image path")
    parser.add_argument("--device", dest="device", choices=["cpu", "cuda"], default="cpu", help="Computation device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_heatmap(args.model, args.image, args.output, args.device)
