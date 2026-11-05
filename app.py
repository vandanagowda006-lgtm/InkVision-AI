# ============================================================
# INKVISION AI — THE DIGIT INTELLIGENCE LAB
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import html

from PIL import Image
from streamlit_drawable_canvas import st_canvas

from modules.authentication import register_user, authenticate_user
from utils.prediction import load_model, predict_digit, generate_gradcam
from utils.preprocessing import preprocess_image
from utils.multidigit import segment_digits

from database.db_manager import (
    save_prediction,
    get_user_history,
    get_user_statistics
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InkVision AI | Digit Intelligence Lab",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "logged_in": False,
    "user": None,
    "last_prediction": None,
    "last_multi_prediction": None,
    "processed_image": None,
    "original_image": None,
    "last_input_type": None,
    "gradcam_heatmap": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --background: #10130F;
    --surface: #191E18;
    --surface2: #222920;
    --lime: #D4FF62;
    --blue: #6C9EFF;
    --white: #F4F5EF;
    --muted: #A1AA9C;
}

.stApp {
    background: var(--background);
    color: var(--white);
    font-family: 'DM Sans', sans-serif;
}

/* Keep Streamlit's header available so the sidebar can always be reopened. */
header, [data-testid="stHeader"] {
    visibility: visible !important;
    display: block !important;
    background: transparent !important;
}

/* Hide only the footer and menu; do not hide the header/sidebar controls. */
footer, #MainMenu {
    visibility: hidden !important;
}

/* Keep the sidebar collapse/expand arrows visible and easy to click. */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarExpandButton"] {
    visibility: visible !important;
    display: inline-flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    color: var(--lime) !important;
    background: var(--surface2) !important;
    border: 1px solid rgba(212, 255, 98, 0.35) !important;
    border-radius: 8px !important;
    z-index: 999999 !important;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--white) !important;
}

p, label, span { color: inherit; }

.lab-eyebrow {
    color: var(--lime);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.lab-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(38px, 5vw, 65px);
    font-weight: 700;
    line-height: 1.05;
    color: var(--white);
}

.lab-title span { color: var(--lime); }

.lab-subtitle {
    color: var(--muted);
    font-size: 15px;
    line-height: 1.8;
    max-width: 650px;
}

.lab-card {
    background: var(--surface);
    border: 1px solid #30392D;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 16px;
}

.lab-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--lime);
    letter-spacing: 2px;
}

.lab-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 55px;
    font-weight: 700;
    color: var(--lime);
}

.lab-muted {
    color: var(--muted);
    font-size: 13px;
}

.stTextInput input, .stTextArea textarea {
    background: #20271E !important;
    color: white !important;
    border: 1px solid #46513F !important;
    border-radius: 10px !important;
}

.stTextInput label { color: var(--white) !important; }

.stButton > button, .stFormSubmitButton > button {
    background: var(--lime);
    color: #11170E !important;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 800;
    width: 100%;
}

.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #E5FF9B;
    color: #11170E !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #384233;
}

.stTabs [data-baseweb="tab"] { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--lime) !important; }

[data-testid="stSidebar"] {
    background: #171C16;
    border-right: 1px solid #30392D;
}

[data-testid="stSidebar"] * { color: var(--white); }

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid #30392D;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricValue"] { color: var(--lime); }

hr { border-color: #30392D; }

/* Dedicated logout styling; scoped to its unique Streamlit key. */
[data-testid="stSidebar"] div.st-key-logout_button button {
    background: #242B21 !important;
    color: #D4FF62 !important;
    border: 1px solid #D4FF62 !important;
    border-radius: 10px !important;
    min-height: 46px !important;
    width: 100% !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
}

[data-testid="stSidebar"] div.st-key-logout_button button:hover {
    background: #D4FF62 !important;
    color: #10130F !important;
    border: 1px solid #D4FF62 !important;
}

[data-testid="stSidebar"] div.st-key-logout_button {
    margin-top: 12px;
}

/* Model X-Ray architecture cards */
.architecture-step {
    background: #191E18;
    border: 1px solid #384233;
    border-left: 4px solid #D4FF62;
    border-radius: 13px;
    padding: 17px 22px;
    margin-bottom: 8px;
}

.architecture-number {
    color: #D4FF62;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
}

.architecture-name {
    color: #F4F5EF;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 19px;
    font-weight: 700;
    margin-top: 5px;
}

.architecture-config {
    color: #6C9EFF;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    margin-top: 5px;
}

.architecture-description {
    color: #A1AA9C;
    font-size: 13px;
    margin-top: 8px;
}

.multi-result-card {
    background: #191E18;
    border: 1px solid #384233;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def get_model():
    return load_model()


# ============================================================
# HEADER
# ============================================================

def show_brand():
    st.markdown("""
    <div class="lab-eyebrow">✳ INKVISION AI / RESEARCH DIVISION</div>
    <div style="font-family:Space Grotesk;font-size:27px;font-weight:700;margin-top:8px;">
        InkVision <span style="color:#D4FF62;">AI.</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# AUTHENTICATION
# ============================================================

def authentication_page():
    show_brand()
    st.write("")
    st.write("")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("""
        <div class="lab-eyebrow">A NEW PERSPECTIVE ON HANDWRITING</div>
        <br>
        <div class="lab-title">
            Beyond<br><span>recognition.</span><br>Into intelligence.
        </div>
        <br>
        <div class="lab-subtitle">
            Explore the science behind handwritten digits.
            Discover how computer vision processes strokes,
            extracts features and understands numbers.
        </div>
        <br>
        <div class="lab-card">
            <div class="lab-label">01 / DIGIT AUTOPSY</div>
            <h3>See every processing stage.</h3>
            <p class="lab-muted">Explore image enhancement and digit extraction.</p>
        </div>
        <div class="lab-card">
            <div class="lab-label">02 / SHAPE DNA</div>
            <h3>Explore the geometry.</h3>
            <p class="lab-muted">Discover measurable characteristics of handwriting.</p>
        </div>
        <div class="lab-card">
            <div class="lab-label">03 / MODEL X-RAY</div>
            <h3>Understand the AI.</h3>
            <p class="lab-muted">Explore the CNN architecture and evaluation.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.write("")
        st.write("")
        st.markdown("""
        <div class="lab-card">
            <div class="lab-eyebrow">YOUR PRIVATE WORKSPACE</div>
            <h2>Enter the lab.</h2>
            <p class="lab-muted">Sign in or create your InkVision account.</p>
        </div>
        """, unsafe_allow_html=True)

        login_tab, signup_tab = st.tabs(["SIGN IN", "CREATE ACCOUNT"])

        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("ENTER THE LAB →")

                if submitted:
                    if not username.strip() or not password:
                        st.error("Enter your username and password.")
                    else:
                        user = authenticate_user(username.strip(), password)
                        if user:
                            st.session_state.logged_in = True
                            st.session_state.user = user
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

        with signup_tab:
            with st.form("signup_form"):
                new_username = st.text_input("Choose username")
                email = st.text_input("Email address")
                new_password = st.text_input("Create password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("CREATE MY ACCOUNT →")

                if submitted:
                    if not all([new_username.strip(), email.strip(), new_password, confirm_password]):
                        st.error("Please complete all fields.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        success, message = register_user(
                            new_username.strip(), email.strip(), new_password
                        )
                        if success:
                            st.success(message)
                            st.info("You can now sign in.")
                        else:
                            st.error(message)


# ============================================================
# AI VISION IMAGE HELPERS
# ============================================================

def create_gradcam_visuals(processed_image, heatmap):
    """
    Resize Grad-CAM to the 28x28 CNN input and create a colour overlay.
    Returns (heatmap_rgb, overlay_rgb).
    """
    base = np.asarray(processed_image, dtype=np.float32)
    base = np.clip(base, 0.0, 1.0)
    base_u8 = (base * 255).astype(np.uint8)

    resized_heatmap = cv2.resize(
        np.asarray(heatmap, dtype=np.float32),
        (28, 28),
        interpolation=cv2.INTER_CUBIC
    )
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    heatmap_u8 = (resized_heatmap * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)
    overlay_rgb = cv2.addWeighted(
        base_rgb,
        0.58,
        heatmap_rgb,
        0.42,
        0
    )

    return heatmap_rgb, overlay_rgb


def show_ai_vision(results, processed_image, heatmap, heading=True):
    if heading:
        st.divider()
        st.markdown(
            '<div class="lab-eyebrow">01A / EXPLAINABLE AI</div>',
            unsafe_allow_html=True
        )
        st.markdown("### AI Vision — How does the model see your digit?")
        st.markdown(
            "Grad-CAM uses gradients from the CNN to create an approximate "
            "map of image regions that positively influence the selected class."
        )

    if heatmap is None:
        st.warning(
            "The digit was recognised, but an explanation heatmap could "
            "not be generated for this input."
        )
        return

    heatmap_rgb, overlay_rgb = create_gradcam_visuals(
        processed_image,
        heatmap
    )

    image_col, heat_col, overlay_col = st.columns(3)

    with image_col:
        st.markdown("**CNN input**")
        st.image(
            processed_image,
            clamp=True,
            use_container_width=True
        )
        st.caption("The normalized 28 × 28 image sent to the CNN.")

    with heat_col:
        st.markdown("**Grad-CAM heatmap**")
        st.image(
            heatmap_rgb,
            use_container_width=True
        )
        st.caption("Warmer colours indicate stronger positive contribution.")

    with overlay_col:
        st.markdown("**Vision overlay**")
        st.image(
            overlay_rgb,
            use_container_width=True
        )
        st.caption("The heatmap blended with the CNN input.")

    best = results[0]
    st.markdown(
        f"**Explaining predicted class: {best['digit']}** "
        f"· Model confidence: **{best['confidence']}%**"
    )
    st.info(
        "How to read this: highlighted regions are areas whose learned "
        "features contributed positively to the selected class score. "
        "Grad-CAM is a coarse approximation; it does not prove that the "
        "model relied only on those pixels or explain its decision perfectly."
    )


# ============================================================
# SINGLE-DIGIT PROCESSING
# ============================================================

def process_digit(image, input_type):
    try:
        original = np.asarray(image).copy()

        processed = preprocess_image(image)
        model = get_model()
        results = predict_digit(model, processed)
        best = results[0]

        # Keep the normal prediction successful even if explanation fails.
        heatmap = None
        try:
            heatmap = generate_gradcam(
                model,
                processed,
                class_index=best["digit"]
            )
        except Exception as gradcam_error:
            print(f"Grad-CAM generation failed: {gradcam_error}")

        user_id = st.session_state.user["id"]
        save_prediction(
            user_id,
            best["digit"],
            best["confidence"],
            input_type
        )

        st.session_state.last_prediction = results
        st.session_state.last_multi_prediction = None
        st.session_state.processed_image = processed[0, :, :, 0]
        st.session_state.original_image = original
        st.session_state.last_input_type = input_type
        st.session_state.gradcam_heatmap = heatmap

        st.success("Digit analysed successfully!")

    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ============================================================
# MULTI-DIGIT PROCESSING
# ============================================================

def process_multiple_digits(image, input_type):
    """
    Split a number image into individual digit regions, process each
    region with the existing single-digit CNN, and join predictions
    from left to right.

    This does not retrain or replace the CNN. It adds a segmentation
    step before the existing model.
    """
    try:
        original = np.asarray(image).copy()

        segments = segment_digits(image)

        if not segments:
            st.warning(
                "No clear digits were detected. Try using darker, thicker "
                "strokes and leave a small gap between neighbouring digits."
            )
            return

        if len(segments) == 1:
            st.info(
                "Only one separate digit was detected. "
                "The image will be analysed as a single digit."
            )
            process_digit(image, input_type)
            return

        model = get_model()
        user_id = st.session_state.user["id"]
        digit_results = []

        for position, segment in enumerate(segments, start=1):
            crop = segment["crop"]

            # Reuse the same MNIST-compatible preprocessing for each digit.
            processed = preprocess_image(crop)
            results = predict_digit(model, processed)
            best = results[0]

            # Generate an explanation independently for each digit.
            heatmap = None
            try:
                heatmap = generate_gradcam(
                    model,
                    processed,
                    class_index=best["digit"]
                )
            except Exception as gradcam_error:
                print(
                    f"Grad-CAM failed for digit {position}: "
                    f"{gradcam_error}"
                )

            # Store each digit as an individual prediction in the user's
            # collection because the existing database stores one digit
            # per prediction row.
            save_prediction(
                user_id,
                best["digit"],
                best["confidence"],
                f"{input_type} · Multi-digit"
            )

            digit_results.append({
                "position": position,
                "crop": crop,
                "results": results,
                "processed_image": processed[0, :, :, 0],
                "heatmap": heatmap,
                "predicted_digit": best["digit"],
                "confidence": best["confidence"]
            })

        recognised_number = "".join(
            str(item["predicted_digit"]) for item in digit_results
        )

        # Keep the first digit available to the existing Autopsy and Shape
        # DNA pages, while retaining all digit results separately.
        first_digit = digit_results[0]

        st.session_state.last_prediction = first_digit["results"]
        st.session_state.last_multi_prediction = {
            "number": recognised_number,
            "digits": digit_results,
            "count": len(digit_results)
        }
        st.session_state.processed_image = first_digit["processed_image"]
        st.session_state.original_image = original
        st.session_state.last_input_type = input_type
        st.session_state.gradcam_heatmap = first_digit["heatmap"]

        st.success(
            f"Multi-digit analysis completed: {len(digit_results)} "
            f"digit regions detected."
        )

    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Multi-digit analysis failed: {error}")


# ============================================================
# MULTI-DIGIT RESULTS DISPLAY
# ============================================================

def show_multiple_digit_results(multi_result):
    st.divider()
    st.markdown(
        '<div class="lab-eyebrow">AI INTERPRETATION / SEQUENCE MODE</div>',
        unsafe_allow_html=True
    )
    st.markdown("### Recognised number")

    st.markdown(
        f"""
        <div class="lab-card">
            <div class="lab-label">PREDICTED SEQUENCE</div>
            <div class="lab-number" style="font-size: clamp(48px, 8vw, 82px);">
                {html.escape(str(multi_result['number']))}
            </div>
            <div class="lab-muted">
                {multi_result['count']} individual digit regions analysed ·
                Read from left to right
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Individual digit analysis")

    columns = st.columns(min(multi_result["count"], 5))

    for index, item in enumerate(multi_result["digits"]):
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="multi-result-card">
                    <div class="lab-label">POSITION {item['position']:02d}</div>
                    <div class="lab-number" style="font-size:44px;">
                        {item['predicted_digit']}
                    </div>
                    <div class="lab-muted">{item['confidence']}% confidence</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(
                max(0.0, min(1.0, float(item["confidence"]) / 100))
            )

    st.divider()
    st.markdown("### Segmentation & CNN inputs")
    st.caption(
        "Each detected region is processed separately and resized to "
        "28 × 28 before being passed to the same CNN."
    )

    for item in multi_result["digits"]:
        with st.expander(
            f"Digit {item['position']} · Predicted {item['predicted_digit']} "
            f"· {item['confidence']}% confidence",
            expanded=(item["position"] == 1)
        ):
            crop_col, input_col, top_col = st.columns([1, 1, 1.4])

            with crop_col:
                st.markdown("**Segmented region**")
                st.image(item["crop"], clamp=True, use_container_width=True)

            with input_col:
                st.markdown("**CNN input · 28 × 28**")
                st.image(
                    item["processed_image"],
                    clamp=True,
                    use_container_width=True
                )

            with top_col:
                st.markdown("**Top 3 possibilities**")
                for possibility in item["results"]:
                    st.write(
                        f"Digit {possibility['digit']} — "
                        f"{possibility['confidence']}%"
                    )
                    st.progress(
                        max(
                            0.0,
                            min(
                                1.0,
                                float(possibility["confidence"]) / 100
                            )
                        )
                    )

            st.markdown("**AI Vision — individual digit explanation**")
            show_ai_vision(
                item["results"],
                item["processed_image"],
                item["heatmap"],
                heading=False
            )

    st.info(
        "Multi-digit recognition is performed by segmenting the image "
        "into separate regions and classifying each region with the "
        "existing MNIST CNN. It works best when digits do not touch or "
        "overlap. The model recognises individual digits; it does not "
        "understand the mathematical meaning of the complete number."
    )


# ============================================================
# THE LAB
# ============================================================

def show_lab():
    st.markdown("""
    <div class="lab-eyebrow">01 / INTERACTIVE WORKSPACE</div>
    <div class="lab-title">The <span>Digit Lab.</span></div>
    <p class="lab-subtitle">
        Draw a digit or upload handwriting.
        Let the CNN interpret your strokes.
    </p>
    """, unsafe_allow_html=True)

    recognition_mode = st.radio(
        "RECOGNITION MODE",
        ["Single digit", "Multi-digit number"],
        horizontal=True,
        key="recognition_mode"
    )

    if recognition_mode == "Single digit":
        st.caption(
            "Recognise one handwritten digit from 0 to 9."
        )
    else:
        st.caption(
            "Recognise a sequence such as 2026. Leave visible gaps "
            "between neighbouring digits for more reliable segmentation."
        )

    draw_tab, upload_tab = st.tabs(["✏️ DRAW", "📁 UPLOAD IMAGE"])

    with draw_tab:
        st.markdown(
            '<div class="lab-label">DRAW YOUR HANDWRITING BELOW</div>',
            unsafe_allow_html=True
        )

        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=16,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=400,
            drawing_mode="freedraw",
            key="digit_canvas",
            return_image_data=True
        )

        button_label = (
            "RECOGNIZE NUMBER →"
            if recognition_mode == "Multi-digit number"
            else "RECOGNIZE MY DIGIT →"
        )

        if st.button(button_label, key="recognize_drawing"):
            if canvas.image_data is not None:
                image = canvas.image_data.astype(np.uint8)

                if recognition_mode == "Multi-digit number":
                    process_multiple_digits(image, "Drawing")
                else:
                    process_digit(image, "Drawing")
            else:
                st.warning("Draw your handwriting first.")

    with upload_tab:
        upload_label = (
            "Upload a handwritten number"
            if recognition_mode == "Multi-digit number"
            else "Upload a handwritten digit"
        )

        uploaded_file = st.file_uploader(
            upload_label,
            type=["png", "jpg", "jpeg"],
            key="digit_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded handwriting", width=300)

            button_label = (
                "ANALYSE NUMBER →"
                if recognition_mode == "Multi-digit number"
                else "ANALYSE UPLOADED IMAGE →"
            )

            if st.button(button_label, key="analyse_upload"):
                image_array = np.array(image)

                if recognition_mode == "Multi-digit number":
                    process_multiple_digits(image_array, "Upload")
                else:
                    process_digit(image_array, "Upload")

    # Display the multi-digit result when sequence mode has been used.
    multi_result = st.session_state.last_multi_prediction

    if recognition_mode == "Multi-digit number" and multi_result:
        show_multiple_digit_results(multi_result)

    # Preserve the original single-digit result and AI Vision experience.
    elif st.session_state.last_prediction:
        results = st.session_state.last_prediction

        st.divider()
        st.markdown(
            '<div class="lab-eyebrow">AI INTERPRETATION</div>',
            unsafe_allow_html=True
        )

        best = results[0]
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"""
            <div class="lab-card">
                <div class="lab-label">PREDICTED DIGIT</div>
                <div class="lab-number">{best['digit']}</div>
                <div class="lab-muted">{best['confidence']}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Top 3 possibilities")
            for result in results:
                st.write(
                    f"Digit {result['digit']} — {result['confidence']}%"
                )
                st.progress(
                    max(
                        0.0,
                        min(1.0, float(result["confidence"]) / 100)
                    )
                )

        show_ai_vision(
            results,
            st.session_state.processed_image,
            st.session_state.gradcam_heatmap
        )


# ============================================================
# DIGIT AUTOPSY
# ============================================================

def show_autopsy():
    st.markdown("""
    <div class="lab-eyebrow">02 / IMAGE PROCESSING</div>
    <div class="lab-title">Digit <span>Autopsy.</span></div>
    <p class="lab-subtitle">
        Discover how your drawn or uploaded digit is transformed
        before reaching the CNN.
    </p>
    """, unsafe_allow_html=True)

    processed = st.session_state.processed_image
    original = st.session_state.original_image

    if processed is None:
        st.info("Recognize a digit in The Lab to inspect its processing.")
        return

    if st.session_state.last_multi_prediction:
        st.caption(
            "This page currently displays the first digit from your most "
            "recent multi-digit recognition. Open The Lab to inspect all digits."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original input")
        if original is not None:
            display_original = original
            if display_original.ndim == 3 and display_original.shape[-1] == 4:
                display_original = display_original[:, :, :3]
            st.image(display_original, use_container_width=True)
        else:
            st.info("Original image is not available for this prediction.")

    with col2:
        st.markdown("### CNN-ready image · 28 × 28")
        st.image(processed, clamp=True, width=220)

    st.divider()
    st.markdown("### Processing information")

    stages = [
        "Grayscale conversion",
        "Background normalization",
        "Gaussian noise reduction",
        "Otsu thresholding",
        "Digit boundary detection",
        "Aspect-ratio-preserving resize",
        "28 × 28 normalization",
    ]

    stage_cols = st.columns(2)
    for index, stage in enumerate(stages):
        with stage_cols[index % 2]:
            st.success(f"✓ {stage}")

    st.caption(
        "These stages reflect the preprocessing implementation "
        "in utils/preprocessing.py."
    )


# ============================================================
# SHAPE DNA
# ============================================================

def show_shape_dna():
    st.markdown("""
    <div class="lab-eyebrow">03 / GEOMETRIC ANALYSIS</div>
    <div class="lab-title">Shape <span>DNA.</span></div>
    <p class="lab-subtitle">
        Explore the measurable geometry of your handwritten digit.
    </p>
    """, unsafe_allow_html=True)

    processed = st.session_state.processed_image
    original = st.session_state.original_image

    if processed is None:
        st.info("Recognize a digit in The Lab to generate its Shape DNA.")
        return

    if st.session_state.last_multi_prediction:
        st.caption(
            "Geometric measurements below describe the first digit "
            "in your most recent multi-digit recognition."
        )

    gray = np.asarray(processed, dtype=np.float32)
    gray = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = np.clip(gray, 0.0, 1.0)
    binary = (gray > 0.5).astype(np.uint8)

    points = cv2.findNonZero(binary)
    if points is None:
        st.warning("No digit shape detected. Try drawing with thicker strokes.")
        return

    x, y, width, height = cv2.boundingRect(points)
    ys, xs = np.where(binary > 0)

    total_pixels = int(len(xs))
    density = total_pixels / binary.size * 100
    aspect_ratio = width / height if height else 0.0
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))

    binary_visual = (binary * 255).astype(np.uint8)

    geometry_visual = cv2.cvtColor(binary_visual, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(
        geometry_visual,
        (x, y),
        (x + width - 1, y + height - 1),
        (212, 255, 98),
        1
    )
    cv2.circle(
        geometry_visual,
        (int(round(center_x)), int(round(center_y))),
        2,
        (108, 158, 255),
        -1
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("WIDTH", f"{width} px")
    col2.metric("HEIGHT", f"{height} px")
    col3.metric("ASPECT RATIO", f"{aspect_ratio:.2f}")
    col4.metric("PIXEL DENSITY", f"{density:.1f}%")

    st.divider()

    visual1, visual2, visual3 = st.columns(3)

    with visual1:
        st.markdown("### Original input")
        if original is not None:
            original_display = original
            if original_display.ndim == 3 and original_display.shape[-1] == 4:
                original_display = original_display[:, :, :3]
            st.image(original_display, use_container_width=True)
        else:
            st.caption("Original input unavailable.")

    with visual2:
        st.markdown("### Binary shape")
        st.image(binary_visual, clamp=True, use_container_width=True)
        st.caption("White pixels represent the detected digit.")

    with visual3:
        st.markdown("### Geometry map")
        st.image(geometry_visual, clamp=True, use_container_width=True)
        st.caption("Lime = bounding box · Blue = geometric centre")

    st.divider()
    st.markdown("### Geometric fingerprint")

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("FOREGROUND PIXELS", total_pixels)
    metric2.metric("HORIZONTAL CENTRE", f"{center_x:.2f}")
    metric3.metric("VERTICAL CENTRE", f"{center_y:.2f}")
    metric4.metric("DIGIT DIMENSIONS", f"{width} × {height}")

    st.caption(
        "These are image measurements, not handwriting personality "
        "or identity predictions."
    )


# ============================================================
# MODEL X-RAY
# ============================================================

def show_model_xray():
    st.markdown("""
    <div class="lab-eyebrow">04 / MODEL EXPLORATION</div>
    <div class="lab-title">Model <span>X-Ray.</span></div>
    <p class="lab-subtitle">
        Explore the training and evaluation results
        of your convolutional neural network.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("### CNN Architecture")
    st.caption(
        "Follow the path from the 28 × 28 handwritten image "
        "to the model's final prediction."
    )

    layers = [
        {
            "number": "01",
            "name": "INPUT IMAGE",
            "config": "28 × 28 × 1",
            "description": "Receives the handwritten digit as a normalized grayscale image."
        },
        {
            "number": "02",
            "name": "CONVOLUTION 1",
            "config": "32 filters · 3 × 3 · ReLU",
            "description": "Detects basic visual patterns such as edges, curves and strokes."
        },
        {
            "number": "03",
            "name": "MAX POOLING 1",
            "config": "2 × 2",
            "description": "Reduces the feature-map size while retaining important visual information."
        },
        {
            "number": "04",
            "name": "CONVOLUTION 2",
            "config": "64 filters · 3 × 3 · ReLU",
            "description": "Learns more detailed patterns by combining the features from the earlier layer."
        },
        {
            "number": "05",
            "name": "MAX POOLING 2",
            "config": "2 × 2",
            "description": "Further reduces spatial dimensions and keeps stronger features."
        },
        {
            "number": "06",
            "name": "FLATTEN",
            "config": "Feature vector",
            "description": "Converts the two-dimensional feature maps into a one-dimensional vector."
        },
        {
            "number": "07",
            "name": "DENSE LAYER",
            "config": "128 neurons · ReLU",
            "description": "Combines the extracted features to learn patterns useful for digit classification."
        },
        {
            "number": "08",
            "name": "DROPOUT",
            "config": "Rate · 0.3",
            "description": "Randomly disables a portion of neurons during training to help reduce overfitting."
        },
        {
            "number": "09",
            "name": "OUTPUT LAYER",
            "config": "10 neurons · Softmax",
            "description": "Returns a probability score for each possible digit from 0 to 9."
        }
    ]

    for index, layer in enumerate(layers):
        st.markdown(
            f"""
            <div class="architecture-step">
                <div class="architecture-number">LAYER {layer['number']}</div>
                <div class="architecture-name">{layer['name']}</div>
                <div class="architecture-config">{layer['config']}</div>
                <div class="architecture-description">{layer['description']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if index < len(layers) - 1:
            st.markdown(
                """
                <div style="
                    text-align:center;
                    color:#D4FF62;
                    font-size:22px;
                    line-height:1.2;
                    margin:1px 0 5px 0;
                ">↓</div>
                """,
                unsafe_allow_html=True
            )

    st.info(
        "Architecture shown above follows the CNN definition in "
        "train_model.py. It is a visual explanation of the model, "
        "not a separate model or a live layer-inspection tool."
    )

    st.divider()
    st.markdown("### Training & Evaluation Reports")

    reports = [
        ("Training Accuracy", "reports/accuracy_graph.png"),
        ("Training Loss", "reports/loss_graph.png"),
        ("Confusion Matrix", "reports/confusion_matrix.png")
    ]

    for title, path in reports:
        st.markdown(f"### {title}")

        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.warning(
                f"{title} report is not available. "
                "Run train_model.py to generate it."
            )


# ============================================================
# MY COLLECTION
# ============================================================

def show_collection():
    st.markdown("""
    <div class="lab-eyebrow">05 / PERSONAL RECORDS</div>
    <div class="lab-title">My <span>Collection.</span></div>
    """, unsafe_allow_html=True)

    user_id = st.session_state.user["id"]
    history = get_user_history(user_id)

    if not history:
        st.info("Your collection is empty. Recognize your first digit!")
        return

    df = pd.DataFrame(
        history,
        columns=["ID", "Predicted Digit", "Confidence", "Input Type", "Date"]
    )

    st.dataframe(
        df.drop(columns=["ID"]),
        use_container_width=True,
        hide_index=True
    )

    stats = get_user_statistics(user_id)
    col1, col2, col3 = st.columns(3)

    col1.metric("TOTAL PREDICTIONS", stats["total_predictions"])
    col2.metric("AVERAGE CONFIDENCE", f"{stats['average_confidence']}%")
    col3.metric("UNIQUE DIGITS", stats["unique_digits"])

    st.caption(
        "For a multi-digit number, each individual digit is saved as "
        "a separate prediction record in your collection."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if not st.session_state.logged_in:
    authentication_page()

else:
    user = st.session_state.user

    with st.sidebar:
        show_brand()
        st.divider()

        st.markdown(f"### Hello, {html.escape(user['username'])}!")
        st.caption("YOUR INTELLIGENCE WORKSPACE")

        page = st.radio(
            "LAB NAVIGATION",
            [
                "The Lab",
                "Digit Autopsy",
                "Shape DNA",
                "Model X-Ray",
                "My Collection"
            ],
            key="lab_navigation"
        )

        st.divider()
        st.markdown(
            '<div class="lab-eyebrow">INKVISION AI / V1.0</div>',
            unsafe_allow_html=True
        )

        if st.button("LOG OUT ↗", key="logout_button"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.last_prediction = None
            st.session_state.last_multi_prediction = None
            st.session_state.processed_image = None
            st.session_state.original_image = None
            st.session_state.last_input_type = None
            st.session_state.gradcam_heatmap = None
            st.rerun()

    if page == "The Lab":
        show_lab()
    elif page == "Digit Autopsy":
        show_autopsy()
    elif page == "Shape DNA":
        show_shape_dna()
    elif page == "Model X-Ray":
        show_model_xray()
    elif page == "My Collection":
        show_collection()


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<br><br>
<hr>
<div style="
    text-align:center;
    color:#A1AA9C;
    font-family:DM Mono;
    font-size:11px;
    letter-spacing:2px;
">
    INKVISION AI · BEYOND RECOGNITION
    <br><br>
    EVERY STROKE TELLS A STORY.
</div>
""", unsafe_allow_html=True)
