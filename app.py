
import os
import json
import hashlib

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Flora AI | Flower Scanner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

.stApp {
    background: #f7faf7;
    color: #1e3328;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1050px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: #245a3c;
    letter-spacing: -1.2px;
}

.brand span {
    color: #86b798;
}

.tag {
    color: #64816e;
    font-size: 11px;
    letter-spacing: 1.5px;
    font-weight: 700;
    text-align: right;
    padding-top: 9px;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(32px, 5vw, 48px);
    font-weight: 800;
    line-height: 1.18;
    letter-spacing: -2px;
    text-align: center;
    color: #1c4931;
    margin-top: 35px;
    margin-bottom: 8px;
}

.hero-sub {
    color: #7b8980;
    text-align: center;
    font-size: 14px;
    margin-bottom: 32px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #e0e9e1;
    border-radius: 18px;
    background: #ffffff;
    box-shadow: 0 8px 26px rgba(32, 75, 48, 0.045);
}

.upload-heading {
    font-family: 'Manrope', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #234a32;
}

.muted {
    color: #829087;
    font-size: 13px;
}

[data-testid="stFileUploader"] {
    background: #f8fbf8;
    border: 1px dashed #a6c1ad;
    border-radius: 13px;
    padding: 10px;
}

div.stButton > button {
    width: 100%;
    background: #256344;
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #194c33;
    color: white;
    border: 0;
    transform: translateY(-1px);
}

.result-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #8a988e;
}

.result-name {
    font-family: 'Manrope', sans-serif;
    font-size: 31px;
    font-weight: 800;
    color: #286044;
    letter-spacing: -1px;
    margin: 8px 0;
}

.empty-state {
    background: #f5f8f5;
    border: 1px dashed #d4e1d6;
    border-radius: 14px;
    padding: 50px 15px;
    text-align: center;
    color: #809087;
}

div[data-testid="stProgress"] > div > div {
    background: #378357;
    border-radius: 20px;
}

hr {
    border: none;
    border-top: 1px solid #e5ebe6;
    margin: 30px 0 16px;
}

@media (max-width: 700px) {
    .hero-title {
        margin-top: 20px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL AND CLASS NAMES
# ==========================================

MODEL_PATH = "flower_cnn_model.keras"
CLASS_PATH = "class_names.json"
THRESHOLD_PATH = "rejection_threshold.json"


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not os.path.isfile(CLASS_PATH):
        return None

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_rejection_threshold():
    if not os.path.isfile(THRESHOLD_PATH):
        return None

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    threshold = float(data["flower_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The saved flower threshold must be between 0 and 1.")

    return threshold


model = load_model()
class_names = load_class_names()

try:
    rejection_threshold = load_rejection_threshold()
except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
    rejection_threshold = None
    st.error(f"Invalid rejection_threshold.json: {error}")


# ==========================================
# 4. HEADER
# ==========================================

left_header, right_header = st.columns([3, 1])

with left_header:
    st.markdown(
        '<div class="brand">flora<span>.</span>ai</div>',
        unsafe_allow_html=True
    )

with right_header:
    st.markdown(
        '<div class="tag">AI FLOWER SCANNER</div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="hero-title">Discover the flower<br>'
    'in your image.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-sub">'
    'Upload a photo and let AI identify the flower.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# 5. MAIN INTERFACE
# ==========================================

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            '<div class="upload-heading">Upload an image</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<p class="muted">Choose a clear flower photo '
            '(JPG, JPEG or PNG).</p>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        image = None
        image_key = None

        if uploaded_file is not None:
            try:
                image_bytes = uploaded_file.getvalue()
                image_key = hashlib.sha256(
                    image_bytes
                ).hexdigest()

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    use_container_width=True
                )

            except Exception:
                st.error("Unable to open this image.")

        # Reset the previous prediction when the image changes.
        if st.session_state.get("image_key") != image_key:
            st.session_state.image_key = image_key
            st.session_state.prediction = None

        st.write("")

        if model is None:
            st.error(
                "Trained model not found. Run train.py first."
            )
        elif class_names is None:
            st.error(
                "class_names.json not found. "
                "Run train.py to create it."
            )
        elif "Other" not in class_names:
            st.error(
                "The class list must contain an 'Other' class. "
                "Check your dataset folders and retrain."
            )
        elif rejection_threshold is None:
            st.warning(
                "rejection_threshold.json was not found or could not be "
                "loaded. Run train.py again before using image rejection."
            )

        classify = st.button(
            "✦  Identify Flower",
            disabled=(
                image is None
                or model is None
                or class_names is None
                or "Other" not in class_names
                or rejection_threshold is None
            )
        )


# ==========================================
# 6. CLASSIFICATION
# ==========================================

if classify and image is not None:
    try:
        input_shape = model.input_shape
        height = input_shape[1] or 160
        width = input_shape[2] or 160
        resized = image.resize((width, height))

        # The model performs MobileNetV2 preprocessing internally.
        image_array = np.asarray(resized, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Analysing your image..."):
            scores = model.predict(image_array, verbose=0)[0]

        if len(scores) != len(class_names):
            raise ValueError(
                "The model output count doesn't match class_names.json. "
                "Retrain the model."
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("Invalid model predictions.")

        other_index = class_names.index("Other")
        flower_indices = [
            i for i, name in enumerate(class_names)
            if name != "Other"
        ]
        if not flower_indices:
            raise ValueError("No flower classes were found.")

        best_flower_index = max(
            flower_indices, key=lambda i: float(scores[i])
        )
        best_flower_score = float(scores[best_flower_index])
        other_score = float(scores[other_index])
        top_index = int(np.argmax(scores))

        # Reject if Other is the top class or the best flower score
        # is below the threshold learned from the validation dataset.
        if top_index == other_index:
            predicted_name = "Other"
            result_reason = "The model's highest-scoring class is Other."
            result_score = other_score
            score_label = "Other class score"
        elif best_flower_score < rejection_threshold:
            predicted_name = "Other"
            result_reason = (
                "The image did not reach the minimum flower "
                "recognition threshold."
            )
            result_score = best_flower_score
            score_label = "Highest flower score"
        else:
            predicted_name = class_names[best_flower_index]
            result_reason = "A flower class passed the recognition threshold."
            result_score = best_flower_score
            score_label = "Flower prediction score"

        st.session_state.prediction = {
            "name": predicted_name,
            "score": result_score,
            "score_label": score_label,
            "reason": result_reason,
            "threshold": rejection_threshold,
            "classes": class_names,
            "scores": scores.tolist()
        }

    except Exception as error:
        st.session_state.prediction = None
        st.error(f"Could not classify image: {error}")


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

with right:
    with st.container(border=True):
        st.markdown(
            '<div class="result-label">ANALYSIS RESULT</div>',
            unsafe_allow_html=True
        )

        result = st.session_state.get("prediction")

        if result is None:
            st.markdown(
                '<div class="empty-state">'
                '<div style="font-size:34px;">✿</div>'
                '<b>Your result will appear here</b>'
                '<p style="font-size:13px;">'
                'Upload an image to begin.'
                '</p></div>',
                unsafe_allow_html=True
            )
        else:
            name = result["name"]
            score = result["score"]

            if name == "Other":
                st.markdown(
                    '<div class="result-name" '
                    'style="color:#a16c2b;">Other</div>',
                    unsafe_allow_html=True
                )
                st.info(
                    "This image was classified as Other. It may be a "
                    "non-flower image or an image the model could not "
                    "confidently recognise as a flower."
                )
            else:
                st.markdown(
                    f'<div class="result-name">{name}</div>',
                    unsafe_allow_html=True
                )
                st.success(
                    "A flower class passed the recognition threshold."
                )

            st.markdown(
                f"**{result['score_label']}: {score * 100:.2f}%**"
            )
            st.progress(max(0.0, min(1.0, score)))
            st.caption(result["reason"])
            st.caption(
                f"Flower rejection threshold: "
                f"{result['threshold'] * 100:.2f}%"
            )

            with st.expander("View all predictions"):
                for label, value in zip(
                    result["classes"], result["scores"]
                ):
                    st.markdown(f"**{label}** — {value * 100:.2f}%")
                    st.progress(
                        max(0.0, min(1.0, float(value)))
                    )

            st.caption(
                "Scores are model estimates, not guarantees. "
                "Unfamiliar images may still be misclassified."
            )


# ==========================================
# 8. FOOTER
# ==========================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    '<div class="muted" style="text-align:center;">'
    'flora.ai · AI-powered flower identification'
    '</div>',
    unsafe_allow_html=True
)