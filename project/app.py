from flask import Flask, request, render_template, redirect, url_for
import os
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# ----------------------------
# Flask App Setup
# ----------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load trained model
model = load_model("vgg16_butterfly75.h5")

# Butterfly class names
butterfly_names = {
    0: 'ADONIS', 1: 'AFRICAN GIANT SWALLOWTAIL', 2: 'AMERICAN SNOOT', 3: 'AN 88',
    4: 'APOLLO', 5: 'ATALA', 6: 'BANDED ORANGE HELICONIAN', 7: 'BANDED PEACOCK',
    8: 'BECKERS WHITE', 9: 'BLACK HAIRSTREAK', 10: 'BLUE MORPHO', 11: 'BLUE SPOTTED CROW',
    12: 'BROWN SIPROETA', 13: 'CABBAGE WHITE', 14: 'CAIRNS BIRDWING', 15: 'CHECKERED SKIPPER',
    16: 'CHESTNUT', 17: 'CLEOPATRA', 18: 'CLODIUS PARNASSIAN', 19: 'CLOUDED SULPHUR',
    20: 'COMMON BANDED AWL', 21: 'COMMON WOOD-NYMPH', 22: 'COPPER TAIL', 23: 'CRESCENT',
    24: 'CRIMSON PATCH', 25: 'DANAID EGGFLY', 26: 'EASTERN COMA', 27: 'EASTERN DAPPLE WHITE',
    28: 'EASTERN PINE ELFIN', 29: 'ELBOWED PIERROT', 30: 'GOLD BANDED', 31: 'GREAT EGGFLY',
    32: 'GREAT JAY', 33: 'GREEN CELLED CATTLEHEART', 34: 'GREY HAIRSTREAK', 35: 'INDRA SWALLOW',
    36: 'IPHICLUS SISTER', 37: 'JULIA', 38: 'LARGE MARBLE', 39: 'MALACHITE',
    40: 'MANGROVE SKIPPER', 41: 'MESTRA', 42: 'METALMARK', 43: 'MILTBERGS TORTOISESHELL',
    44: 'MONARCH', 45: 'MOURNING CLOAK', 46: 'ORANGE OAKLEAF', 47: 'ORANGE TIP',
    48: 'ORCHARD SWALLOW', 49: 'PAINTED LADY', 50: 'PAPER KITE', 51: 'PEACOCK',
    52: 'PINE WHITE', 53: 'PIPEVINE SWALLOW', 54: 'POPINJAY', 55: 'PURPLE HAIRSTREAK',
    56: 'PURPLISH COPPER', 57: 'QUESTION MARK', 58: 'RED ADMIRAL', 59: 'RED CRACKER',
    60: 'RED POSTMAN', 61: 'RED SPOTTED PURPLE', 62: 'SCARCE SWALLOW', 63: 'SILVER SPOT SKIPPER',
    64: 'SLEEPY ORANGE', 65: 'SOOTYWING', 66: 'SOUTHERN DOGFACE', 67: 'STRAITED QUEEN',
    68: 'TROPICAL LEAFWING', 69: 'TWO BARRED FLASHER', 70: 'ULYSES', 71: 'VICEROY',
    72: 'WOOD SATYR', 73: 'YELLOW SWALLOW TAIL', 74: 'ZEBRA LONG WING'
}

# Upload folder for images
UPLOAD_FOLDER = os.path.join("static", "images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# Routes
# ----------------------------

@app.route("/")
def main_index():
    return render_template("index.html")

@app.route("/input")
def input_page():
    return render_template("input.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess image
        image = load_img(file_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        # Prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        # Confidence check
        if confidence < 0.4:
            label = "Not a butterfly image"
        else:
            label = butterfly_names.get(predicted_class, "Unknown")

        return render_template("output.html",
                               label=label,
                               user_image=file.filename,
                               confidence=round(confidence * 100, 2))

    return redirect(url_for("input_page"))

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
