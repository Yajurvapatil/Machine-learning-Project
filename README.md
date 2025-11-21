# Machine-learning-Project

web application that detect the disease of plant based on leaf image.
flask
tensorflow>=2.10
tensorflow-hub
numpy
pillow
opencv-python
scikit-learn
matplotlib
gunicorn  # optional for deployment



# train.py
import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder

# Config
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_gen.num_classes
print(f"Detected {num_classes} classes: {train_gen.class_indices}")

# Build model (transfer learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base initially
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, 'mobilenetv2_finetune.h5'),
                             monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)

# Train head
history = model.fit(
    train_gen,
    epochs=6,
    validation_data=val_gen,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=LR/10), loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Save final model (best is saved by checkpoint)
model.save(os.path.join(MODEL_DIR, 'mobilenetv2_finetune_final.h5'))

# Save label map (class->index)
label_map = {v: k for k, v in train_gen.class_indices.items()}  # index->label
with open(os.path.join(MODEL_DIR, 'label_map.json'), 'w') as f:
    json.dump(label_map, f)
print("Saved model and label map.")


# model_utils.py
import json
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os

MODEL_DIR = "models"
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

def load_label_map(path=LABEL_MAP_PATH):
    import json
    with open(path, 'r') as f:
        label_map = json.load(f)
    # json gives keys as strings, convert to int keyed dict
    return {int(k): v for k, v in label_map.items()}

def prepare_image(image, target_size=(224, 224)):
    """
    image: PIL Image or path. Returns preprocessed array ready for model.predict.
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    else:
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = img_to_array(image)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


# infer.py
import argparse
from tensorflow.keras.models import load_model
from model_utils import prepare_image, load_label_map
import numpy as np
import os

MODEL_PATH = os.path.join("models", "mobilenetv2_finetune.h5")

def predict(image_path, model_path=MODEL_PATH):
    model = load_model(model_path)
    label_map = load_label_map()
    x = prepare_image(image_path)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = label_map[idx]
    return label, confidence, preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()
    label, conf, raw = predict(args.image, args.model)
    print(f"Predicted: {label} ({conf*100:.2f}%)")



# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from model_utils import prepare_image, load_label_map
import numpy as np

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = os.path.join('models', 'mobilenetv2_finetune.h5')
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # change this
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and labels once
model = load_model(MODEL_PATH)
label_map = load_label_map()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Predict
            x = prepare_image(save_path)
            preds = model.predict(x)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            label = label_map[idx]

            return render_template('index.html', filename=filename, label=label, confidence=f"{conf*100:.2f}")
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)



<!-- templates/index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Plant Disease Detector</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <div class="container">
    <h1>Plant Disease Detection</h1>

    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <button type="submit">Upload & Predict</button>
    </form>

    {% if filename %}
      <div class="result">
        <h2>Result</h2>
        <img src="{{ url_for('static', filename='uploads/' + filename) }}" alt="Uploaded image" width="300">
        <p><strong>Predicted:</strong> {{ label }}</p>
        <p><strong>Confidence:</strong> {{ confidence }}%</p>
      </div>
    {% endif %}
  </div>
</body>
</html>


/* static/style.css */
body { font-family: Arial, sans-serif; background:#f7f7f7; color:#222; }
.container { max-width:700px; margin:40px auto; padding:20px; background:#fff; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }
h1 { margin-top:0; }
form { margin-bottom:20px; }
button { padding:8px 14px; border:none; background:#2d6cdf; color:#fff; cursor:pointer; border-radius:4px; }
button:hover { opacity:0.95; }
.result img { display:block; margin:10px 0; border:1px solid #ddd; padding:4px; background:#fff; }
