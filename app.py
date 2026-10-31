import base64
import io

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

st.set_page_config(page_title="Coffee Bean Roast Classifier", page_icon="☕", layout="wide")

ROAST_ORDER = ["Green", "Light", "Medium", "Dark"]
CLASS_COLORS = {"Green": "#7d9a4b", "Light": "#d2a169", "Medium": "#9a6236", "Dark": "#44261a"}

LINKS = [
    ("Live demo", "https://coffeebeansclassification.streamlit.app/"),
    ("GitHub", "https://github.com/hassaanwaqas007/CoffeeBeansClassification"),
    ("Notebook", "https://drive.google.com/file/d/1Am7X1RS2QEDPRJp0NrrxWt4A2uTlPvrj/view?usp=sharing"),
    ("Video", "https://drive.google.com/file/d/1dGrmhKRqH100f7WoK1S5VKxCQ9YXGDge/view?usp=sharing"),
    ("Dataset", "https://www.kaggle.com/datasets/utkarshsaxenadn/coffee-bean-classifier-resnet50v2"),
    ("Slides", "https://docs.google.com/presentation/d/1QNahkYoLdI5a9ToS-y4AdYXnBdVfPrFa/edit?usp=sharing&ouid=110537875862365994928&rtpof=true&sd=true"),
]


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
# Styling (uses inherited text color + translucent greys so it works in light and dark themes)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&display=swap');

.block-container { max-width: 1040px; padding-top: 2.5rem; padding-bottom: 3rem; }
footer, #MainMenu { visibility: hidden; }

.cb-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
            font-size: clamp(2rem, 5vw, 3.1rem); line-height: 1.1; margin: 0 0 .6rem; letter-spacing: -0.02em; }
.cb-sub { font-size: 1.08rem; line-height: 1.55; max-width: 60ch; opacity: .78; margin: 0 0 1.2rem; }

.cb-links { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 2rem; }
.cb-links a { text-decoration: none !important; color: inherit !important; font-size: .9rem; font-weight: 500;
              padding: .38rem .95rem; border-radius: 999px; border: 1px solid rgba(128,128,128,.35);
              background: rgba(128,128,128,.07); transition: background .15s, border-color .15s; }
.cb-links a:hover { background: rgba(125,154,75,.22); border-color: #7d9a4b; }
.cb-links a:focus-visible { outline: 2px solid #7d9a4b; outline-offset: 2px; }

.cb-h { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 500; font-size: 1.15rem; margin: 0 0 .7rem; }

.cb-card { border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.06);
           border-radius: 14px; padding: 1.4rem 1.5rem; }
.cb-frame { border-radius: 10px; overflow: hidden; border: 1px solid rgba(128,128,128,.28); }
.cb-frame img { display: block; width: 100%; height: auto; max-height: 420px; object-fit: contain;
                background: rgba(128,128,128,.12); }

.cb-result { display: flex; align-items: center; gap: 1rem; margin-bottom: .9rem; }
.cb-swatch { width: 56px; height: 56px; border-radius: 50%; flex: none;
             border: 2px solid rgba(128,128,128,.45); box-shadow: inset 0 -6px 12px rgba(0,0,0,.25); }
.cb-label { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2rem; line-height: 1.05; }
.cb-conf { opacity: .75; font-size: .98rem; margin-top: .15rem; }
.cb-desc { line-height: 1.55; opacity: .85; margin: 0 0 1.2rem; }
.cb-warn { border-left: 3px solid #d2a169; padding: .5rem .8rem; border-radius: 4px;
           background: rgba(210,161,105,.15); font-size: .92rem; margin-bottom: 1.2rem; }

.cb-spectrum { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin: .2rem 0 1.3rem; }
.cb-seg { text-align: center; }
.cb-bar { height: 16px; background: var(--c); border: 1px solid rgba(128,128,128,.35); }
.cb-seg:first-child .cb-bar { border-radius: 8px 0 0 8px; }
.cb-seg:last-child .cb-bar { border-radius: 0 8px 8px 0; }
.cb-seg .nm { font-size: .85rem; margin-top: .35rem; opacity: .6; }
.cb-seg.on .nm { opacity: 1; font-weight: 700; }
.cb-seg.on .bar-wrap::after { content: ""; display: block; width: 0; height: 0; margin: 4px auto 0;
                              border-left: 7px solid transparent; border-right: 7px solid transparent;
                              border-bottom: 8px solid currentColor; }
.cb-seg.dim .cb-bar { opacity: .35; }

.cb-prob { display: grid; grid-template-columns: 70px 1fr 58px; align-items: center; gap: .7rem; margin: .45rem 0; font-size: .95rem; }
.cb-track { height: 10px; border-radius: 6px; background: rgba(128,128,128,.2); overflow: hidden; }
.cb-fill { height: 100%; border-radius: 6px; background: var(--c); transform-origin: left;
           animation: cbgrow .7s ease-out; }
.cb-pct { text-align: right; font-variant-numeric: tabular-nums; opacity: .85; }
@keyframes cbgrow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (prefers-reduced-motion: reduce) { .cb-fill { animation: none; } }

.cb-empty p { margin: 0 0 1rem; line-height: 1.55; opacity: .85; }
.cb-legend { display: grid; gap: .6rem; }
.cb-legend div { display: flex; align-items: center; gap: .7rem; font-size: .93rem; line-height: 1.4; }
.cb-legend i { width: 14px; height: 14px; border-radius: 50%; flex: none; border: 1px solid rgba(128,128,128,.5); }

.cb-foot { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(128,128,128,.25);
           font-size: .88rem; opacity: .7; line-height: 1.6; }
</style>
"""


def html(markup: str):
    """Render HTML without markdown treating indented lines as code blocks."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def image_to_data_uri(image: Image.Image) -> str:
    img = image.convert("RGB")
    img.thumbnail((900, 900))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def spectrum_html(active=None) -> str:
    segs = []
    for name in ROAST_ORDER:
        state = "" if active is None else ("on" if name == active else "dim")
        segs.append(
            f'<div class="cb-seg {state}" style="--c:{CLASS_COLORS[name]}">'
            f'<div class="bar-wrap"><div class="cb-bar"></div></div>'
            f'<div class="nm">{name}</div></div>'
        )
    return f'<div class="cb-spectrum">{"".join(segs)}</div>'


def probs_html(probs: np.ndarray) -> str:
    rows = []
    for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        rows.append(
            f'<div class="cb-prob"><span>{name}</span>'
            f'<div class="cb-track"><div class="cb-fill" style="width:{p * 100:.2f}%;--c:{CLASS_COLORS[name]}"></div></div>'
            f'<span class="cb-pct">{p * 100:.1f}%</span></div>'
        )
    return "".join(rows)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

html("""
<h1 class="cb-title">Coffee bean roast classifier</h1>
<p class="cb-sub">Upload a photo of a coffee bean and the model tells you whether it is green, light, medium or dark roasted.</p>
""")
links = "".join(f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>' for label, url in LINKS)
html(f'<nav class="cb-links" aria-label="Project links">{links}</nav>')

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`. Make sure it is in the same folder as app.py.\n\n{e}")
    st.stop()

left, right = st.columns([1, 1], gap="large")

with left:
    html('<div class="cb-h">1. Add a bean photo</div>')
    tab_upload, tab_camera = st.tabs(["Upload image", "Use camera"])
    with tab_upload:
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"],
                                    label_visibility="collapsed")
    with tab_camera:
        captured = st.camera_input("Take a picture", label_visibility="collapsed")
    file = uploaded or captured

    image = None
    if file is not None:
        image = Image.open(file)
        html(f'<div class="cb-frame"><img src="{image_to_data_uri(image)}" alt="Uploaded coffee bean photo"></div>')

with right:
    html('<div class="cb-h">2. Roast level</div>')
    if image is None:
        legend = "".join(
            f'<div><i style="background:{CLASS_COLORS[n]}"></i><span><b>{n}.</b> {CLASS_INFO[n]}</span></div>'
            for n in ROAST_ORDER
        )
        html(f"""
        <div class="cb-card cb-empty">
        <p>Add a close, well-lit photo of a single bean or a small group. The result appears here.</p>
        {spectrum_html(None)}
        <div class="cb-legend">{legend}</div>
        </div>
        """)
    else:
        with st.spinner("Classifying..."):
            probs = predict(model, image)
        top = int(np.argmax(probs))
        label = CLASS_NAMES[top]
        warn = ""
        if probs[top] < 0.60:
            warn = ('<div class="cb-warn">The model is not very sure about this one. '
                    'Try a closer, brighter photo with a plain background.</div>')
        html(f"""
        <div class="cb-card">
        <div class="cb-result">
        <div class="cb-swatch" style="background:{CLASS_COLORS[label]}"></div>
        <div><div class="cb-label">{label} roast</div>
        <div class="cb-conf">{probs[top] * 100:.1f}% confidence</div></div>
        </div>
        <p class="cb-desc">{CLASS_INFO[label]}</p>
        {warn}
        {spectrum_html(label)}
        <div class="cb-h" style="font-size:1rem">Class probabilities</div>
        {probs_html(probs)}
        </div>
        """)

html("""
<div class="cb-foot">
Fine-tuned ResNet50V2 trained on the Kaggle coffee bean dataset. Images are resized to 256 x 256 before prediction.
The model only knows four classes, so photos of anything other than coffee beans will still get one of them.
</div>
""")
