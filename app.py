import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------------------------------
# Config (taken from your training notebook)
# ----------------------------------------------------------------------------
MODEL_PATH = "ResNet50V2-Coffee-Beans.h5"
IMG_SIZE = (256, 256)                          # target_size used in flow_from_directory
CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]  # sorted(os.listdir(train_dir))

CLASS_INFO = {
    "Dark": "Fully roasted, dark brown to almost black beans. Bold, smoky, low acidity.",
    "Green": "Unroasted raw coffee beans. Grassy, with no roasted flavour yet.",
    "Light": "Light brown beans with a short roast. Bright, acidic, fruity notes.",
    "Medium": "Medium brown beans with a balanced roast. Rounded body and sweetness.",
}

st.set_page_config(page_title="Coffee Bean Classifier", page_icon="☕", layout="centered")


# ----------------------------------------------------------------------------
# Model + inference helpers
# ----------------------------------------------------------------------------
def build_model():
    """Rebuild the exact architecture from the training notebook."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.applications import ResNet50V2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

    base_model = ResNet50V2(include_top=False, weights=None,
                            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    return Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(len(CLASS_NAMES), activation="softmax"),
    ])


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    errors = []

    # 1) Rebuild architecture + load only the trained weights (most compatible)
    try:
        model = build_model()
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        errors.append(f"build + load_weights error: {e}")

    # 2) Standard full-model load
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        errors.append(f"load_model error: {e}")

    raise RuntimeError("\n\n".join(errors))


def preprocess(image: Image.Image) -> np.ndarray:
    """Same preprocessing as training: resize to 256x256 and rescale by 1/255."""
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image) -> np.ndarray:
    return model.predict(preprocess(image), verbose=0)[0]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("☕ Coffee Bean Classifier")
st.write(
    "Upload a photo of a coffee bean to classify its roast level "
    "(**Dark, Green, Light, Medium**) using a fine-tuned **ResNet50V2** model."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`. Make sure it is in the same folder as app.py.\n\n{e}")
    st.stop()

source = st.radio("Input source", ["Upload image", "Use camera"], horizontal=True)
if source == "Upload image":
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
else:
    file = st.camera_input("Take a picture")

if file is not None:
    image = Image.open(file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input image", use_container_width=True)

    with st.spinner("Classifying..."):
        probs = predict(model, image)

    top = int(np.argmax(probs))
    label = CLASS_NAMES[top]

    with col2:
        st.subheader(f"Prediction: {label}")
        st.metric("Confidence", f"{probs[top] * 100:.2f}%")
        st.caption(CLASS_INFO[label])

    st.markdown("#### Class probabilities")
    for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        st.write(f"**{name}**")
        st.progress(float(p), text=f"{p * 100:.2f}%")
else:
    st.info("Upload or capture an image to get a prediction.")

st.markdown("---")
st.caption("Model: ResNet50V2 (transfer learning) · Input 256×256 · Classes: Dark, Green, Light, Medium")
