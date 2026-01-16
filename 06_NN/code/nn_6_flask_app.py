import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import os
import json
import urllib.request


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(device)

model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet Normalisierung
        std=[0.229, 0.224, 0.225]
    )
])

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Erwartet eine Bilddatei ('image') im POST-Request.
    Beispiel (Postman): POST -> http://127.0.0.1:5665/predict
    Body -> form-data -> key='image', value=<Bilddatei>
    """
    if "image" not in request.files:
        return jsonify({"error": "Kein Bild hochgeladen!"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Fehler beim Öffnen des Bildes: {str(e)}"}), 400

    # Preprocessing
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inferenz
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)

    # Ergebnis zurückgeben
    return jsonify({
        "predicted_class_index": predicted.item()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5665, debug=True)






