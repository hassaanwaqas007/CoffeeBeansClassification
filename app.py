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


def _dec(x):
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def _leaf(name):
    """'conv1_conv/kernel:0' -> 'kernel'"""
    return name.split("/")[-1].split(":")[0]


def load_weights_by_name(model, path):
    """Read a Keras-2 style .h5 directly and assign weights by layer/weight name.
    This avoids the weight-ordering differences between Keras 2 and Keras 3."""
    import h5py
    from tensorflow.keras.layers import Dense

    base_arrays = {}   # (layer_name, weight_name) -> array, for the ResNet50V2 base
    dense_groups = []  # [{kernel: arr, bias: arr}, ...] in saved order

    with h5py.File(path, "r") as f:
        mw = f["model_weights"] if "model_weights" in f else f
        for lname in [_dec(n) for n in mw.attrs["layer_names"]]:
            datasets = []

            def visit(name, obj, _d=datasets):
                if isinstance(obj, h5py.Dataset):
                    _d.append((name, obj[()]))

            mw[lname].visititems(visit)
            if not datasets:
                continue
            if len(datasets) > 10:  # the nested ResNet50V2 base
                for name, arr in datasets:
                    parts = name.split("/")
                    base_arrays[(parts[-2], _leaf(parts[-1]))] = arr
            elif lname.startswith("dense"):
                dense_groups.append({_leaf(n): a for n, a in datasets})

    # ResNet50V2 base
    base = model.layers[0]
    for layer in base.layers:
        for w in layer.weights:
            key = (layer.name, _leaf(w.name))
            if key not in base_arrays:
                raise KeyError(f"Weight {key} not found in h5 file")
            arr = base_arrays[key]
            if tuple(w.shape) != tuple(arr.shape):
                raise ValueError(f"Shape mismatch for {key}: {tuple(w.shape)} vs {arr.shape}")
            w.assign(arr)

    # Dense head: match by kernel shape
    for layer in [l for l in model.layers if isinstance(l, Dense)]:
        kernel_shape = tuple(layer.kernel.shape)
        match = next((g for g in dense_groups if tuple(g["kernel"].shape) == kernel_shape), None)
        if match is None:
            raise ValueError(f"No saved Dense weights with kernel shape {kernel_shape}")
        layer.kernel.assign(match["kernel"])
        layer.bias.assign(match["bias"])


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    errors = []

    # 1) Rebuild architecture + assign weights by name (most compatible)
    try:
        model = build_model()
        load_weights_by_name(model, MODEL_PATH)
        return model
    except Exception as e:
        errors.append(f"load_weights_by_name error: {e}")

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
