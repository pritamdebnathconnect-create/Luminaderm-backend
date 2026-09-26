from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import json
import io
import os
import urllib.request
import ast
import gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Existing image-only model (preserved) ----------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

model = None

def ensure_image_model_loaded():
    global model

    if model is not None:
        return

    print("Loading image-only model...")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    state_dict = torch.load(
        "skin_model.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    del state_dict

    model.eval()
    print("Image-only model loaded successfully.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

@app.get("/")
def home():
    return {"message": "Skin AI Backend is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Existing image-only endpoint."""
    ensure_image_model_loaded()
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

    confidence, predicted_index = torch.max(probabilities, 1)
    prediction = class_names[predicted_index.item()]
    return {
        "prediction": prediction,
        "confidence": round(confidence.item() * 100, 2),
    }


# ---------- New multimodal model ----------
MULTIMODAL_MODEL_PATH = os.getenv(
    "LUMINADERM_MULTIMODAL_PATH", "luminaderm_multimodal.pt"
)

# These are the raw SCIN fields used to create the 82 one-hot features.
SYMPTOM_FIELDS = [
    "textures_raised_or_bumpy",
    "textures_flat",
    "textures_rough_or_flaky",
    "textures_fluid_filled",
    "condition_symptoms_bothersome_appearance",
    "condition_symptoms_bleeding",
    "condition_symptoms_increasing_size",
    "condition_symptoms_darkening",
    "condition_symptoms_itching",
    "condition_symptoms_burning",
    "condition_symptoms_pain",
    "condition_symptoms_no_relevant_experience",
    "other_symptoms_fever",
    "other_symptoms_chills",
    "other_symptoms_fatigue",
    "other_symptoms_joint_pain",
    "other_symptoms_mouth_sores",
    "other_symptoms_shortness_of_breath",
    "other_symptoms_no_relevant_symptoms",
    "body_parts_head_or_neck",
    "body_parts_arm",
    "body_parts_palm",
    "body_parts_back_of_hand",
    "body_parts_torso_front",
    "body_parts_torso_back",
    "body_parts_genitalia_or_groin",
    "body_parts_buttocks",
    "body_parts_leg",
    "body_parts_foot_top_or_side",
    "body_parts_foot_sole",
    "body_parts_other",
    "condition_duration",
    "related_category",
]

class LuminaDermMultimodal(nn.Module):
    def __init__(self, symptom_dim, num_classes):
        super().__init__()
        self.image_encoder = models.resnet18(weights=None)
        image_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()
        self.symptom_encoder = nn.Sequential(
            nn.Linear(symptom_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, symptoms):
        image_features = self.image_encoder(image)
        symptom_features = self.symptom_encoder(symptoms)
        combined = torch.cat([image_features, symptom_features], dim=1)
        return self.classifier(combined)

multimodal_model = None
symptom_feature_columns = []
condition_names = []


def ensure_multimodal_model_loaded():
    global multimodal_model
    global symptom_feature_columns
    global condition_names

    if multimodal_model is not None:
        return

    print("Downloading/loading LuminaDerm multimodal model...")

    if not os.path.exists(MULTIMODAL_MODEL_PATH):
        MODEL_URL = (
            "https://github.com/pritam-debnath-dev/"
            "Luminaderm-backend/releases/download/"
            "v1.0.0/luminaderm_multimodal.pt"
        )

        urllib.request.urlretrieve(
            MODEL_URL,
            MULTIMODAL_MODEL_PATH
        )

    checkpoint = torch.load(
        MULTIMODAL_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    symptom_feature_columns = checkpoint[
        "symptom_feature_columns"
    ]
    condition_names = checkpoint["condition_names"]

    multimodal_model = LuminaDermMultimodal(
        symptom_dim=checkpoint["symptom_dim"],
        num_classes=len(condition_names),
    )

    multimodal_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    multimodal_model.eval()

    del checkpoint
    gc.collect()

    print("Multimodal model loaded successfully.")


def encode_symptoms(symptoms):
    """
    Convert raw SCIN-style symptom fields to the exact one-hot column order
    saved with the trained model. Unprovided fields are encoded as MISSING,
    matching the training preprocessing.
    """
    values = {}
    for field in SYMPTOM_FIELDS:
        value = symptoms.get(field)
        if value is None or str(value).strip() == "":
            values[field] = "MISSING"
        else:
            values[field] = str(value)

    encoded = []
    for column in symptom_feature_columns:
        matched_field = next(
            (field for field in SYMPTOM_FIELDS
             if column.startswith(field + "_")),
            None,
        )
        if matched_field is None:
            encoded.append(0.0)
            continue

        category = column[len(matched_field) + 1:]
        encoded.append(
            1.0 if values[matched_field] == category else 0.0
        )

    return torch.tensor([encoded], dtype=torch.float32)


@app.post("/predict-multimodal")
async def predict_multimodal(
    file: UploadFile = File(...),
    symptoms: str = Form(...),
):
    """
    Multipart form:
      - file: image
      - symptoms: JSON object of raw symptom/history fields
    """
    ensure_multimodal_model_loaded()
    
    if multimodal_model is None:
        raise HTTPException(
            status_code=503,
            detail="Multimodal model file is not installed on the server",
        )

    try:
        symptom_data = json.loads(symptoms)
        if not isinstance(symptom_data, dict):
            raise ValueError("symptoms must be a JSON object")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail='symptoms must be a JSON object encoded as a string',
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    image_tensor = transform(image).unsqueeze(0)
    symptom_tensor = encode_symptoms(symptom_data)

    with torch.no_grad():
        logits = multimodal_model(image_tensor, symptom_tensor)
        scores = torch.sigmoid(logits)[0].cpu().tolist()

    ranked = sorted(
        zip(condition_names, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    return {
        "model": "LuminaDerm multimodal (experimental)",
        "note": (
            "Scores are model outputs, not calibrated probabilities "
            "or medical diagnoses."
        ),
        "top_conditions": [
            {"condition": name, "score": round(float(score), 4)}
            for name, score in ranked
        ],
    }
