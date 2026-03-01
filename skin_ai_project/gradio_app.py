"""
Skin Disease Detection - Gradio Local Deployment
Supports: image upload, webcam capture, interactive crop via ImageEditor.
"""

import json
from pathlib import Path

import albumentations as A
import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ── paths & constants ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_skin_model.pth"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"

IMAGE_SIZE = 224
DROPOUT = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── preprocessing (must match train.py) ────────────────────────
def get_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── model loading (runs once at import) ────────────────────────
def _load_model():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = timm.create_model(
        "tf_efficientnetv2_b2", pretrained=False, drop_rate=DROPOUT
    )
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT), nn.Linear(in_features, len(class_names))
    )
    model = model.to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, get_val_transform()


MODEL, CLASS_NAMES, TRANSFORM = _load_model()
print(f"Model loaded on {DEVICE}  |  Classes: {CLASS_NAMES}")


# ── inference function ─────────────────────────────────────────
def predict(image):
    """
    Accept an image (PIL or numpy from gr.Image / gr.ImageEditor),
    run inference, and return results.
    """
    if image is None:
        return "No image provided", 0.0, {}

    # gr.ImageEditor returns a dict with composite/background/layers;
    # gr.Image returns PIL or ndarray directly.
    if isinstance(image, dict):
        composite = image.get("composite")
        image = composite if composite is not None else image.get("background")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image is None:
        return "No image provided", 0.0, {}

    image = image.convert("RGB")
    tensor = TRANSFORM(image=np.array(image))["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(MODEL(tensor), dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    prob_dict = {CLASS_NAMES[i]: float(round(p, 4)) for i, p in enumerate(probs)}
    return label, round(confidence, 4), prob_dict


# ── Gradio UI ──────────────────────────────────────────────────
def build_app():
    with gr.Blocks(
        title="Skin Disease AI Detection",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            # Skin Disease Detection - Local Test
            **Upload** an image, use your **Webcam**, or **Crop** interactively,
            then click **Predict**.
            """
        )

        with gr.Tabs():

            # ── Tab 1: Upload ──────────────────────────────────
            with gr.TabItem("Upload"):
                upload_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                upload_btn = gr.Button("Predict", variant="primary", size="lg")
                with gr.Row():
                    upload_label = gr.Textbox(label="Predicted Disease")
                    upload_conf = gr.Number(label="Confidence", precision=4)
                upload_probs = gr.Label(label="Class Probabilities")

                upload_btn.click(
                    fn=predict,
                    inputs=[upload_input],
                    outputs=[upload_label, upload_conf, upload_probs],
                )

            # ── Tab 2: Webcam ──────────────────────────────────
            with gr.TabItem("Live Camera"):
                cam_input = gr.Image(
                    label="Webcam Capture",
                    type="pil",
                    sources=["webcam"],
                )
                cam_btn = gr.Button("Predict", variant="primary", size="lg")
                with gr.Row():
                    cam_label = gr.Textbox(label="Predicted Disease")
                    cam_conf = gr.Number(label="Confidence", precision=4)
                cam_probs = gr.Label(label="Class Probabilities")

                cam_btn.click(
                    fn=predict,
                    inputs=[cam_input],
                    outputs=[cam_label, cam_conf, cam_probs],
                )

            # ── Tab 3: Crop Editor ─────────────────────────────
            with gr.TabItem("Crop Editor"):
                gr.Markdown(
                    "Upload an image, then **use the crop / draw tools** "
                    "in the editor before predicting."
                )
                editor_input = gr.ImageEditor(
                    label="Crop & Edit",
                    type="numpy",
                    sources=["upload", "webcam", "clipboard"],
                    crop_size=None,
                )
                editor_btn = gr.Button("Predict", variant="primary", size="lg")
                with gr.Row():
                    editor_label = gr.Textbox(label="Predicted Disease")
                    editor_conf = gr.Number(label="Confidence", precision=4)
                editor_probs = gr.Label(label="Class Probabilities")

                editor_btn.click(
                    fn=predict,
                    inputs=[editor_input],
                    outputs=[editor_label, editor_conf, editor_probs],
                )

    return demo


# ── launch ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=port)
