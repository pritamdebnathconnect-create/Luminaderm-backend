from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import json
import io

# Create FastAPI app
app = FastAPI()

# Allow the Lovable website to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Create ResNet18 model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

# Load our trained model
model.load_state_dict(torch.load("skin_model.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/")
def home():
    return {"message": "Skin AI Backend is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Prepare image
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

    # Find highest probability
    confidence, predicted_index = torch.max(probabilities, 1)

    prediction = class_names[predicted_index.item()]
    confidence_percentage = confidence.item() * 100

    return {
        "prediction": prediction,
        "confidence": round(confidence_percentage, 2)
    }