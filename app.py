import streamlit as st
import tf_keras
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# SMART RECYCLE AI
# AI Waste Classification & Smart Recycling Assistant
# ---------------------------------------------------------

# Page configuration
st.set_page_config(
    page_title="Smart Recycle AI",
    page_icon="♻️",
    layout="centered"
)

# ---------------------------------------------------------
# Waste information
# ---------------------------------------------------------

WASTE_INFO = {
    "PLASTIC": {
        "category": "Recyclable",
        "icon": "🧴",
        "recommendation": (
            "Separate plastic items from wet waste. "
            "Clean the item when possible and place it "
            "in the appropriate dry/recyclable waste stream."
        )
    },

    "PAPER": {
        "category": "Recyclable",
        "icon": "📄",
        "recommendation": (
            "Keep paper clean and dry and place it "
            "with paper or dry recyclable waste."
        )
    },

    "METAL": {
        "category": "Recyclable",
        "icon": "🥫",
        "recommendation": (
            "Separate metal items from wet waste and "
            "place them in the appropriate recyclable "
            "or metal-waste collection stream."
        )
    },

    "GLASS": {
        "category": "Recyclable",
        "icon": "🍾",
        "recommendation": (
            "Handle glass carefully. Keep it separate "
            "from other waste and use an appropriate "
            "glass-recycling collection facility where available."
        )
    },

    "ORGANIC": {
        "category": "Compostable",
        "icon": "🍎",
        "recommendation": (
            "Place suitable organic waste in the wet/organic "
            "waste stream or use it for composting where available."
        )
    }
}

# ---------------------------------------------------------
# Load AI model
# ---------------------------------------------------------

MODEL_PATH = "converted_keras/keras_model.h5"
LABELS_PATH = "converted_keras/labels.txt"

@st.cache_resource
def load_model():
    model = tf_keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r") as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

model = load_model()
labels = load_labels()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("♻️ Smart Recycle AI")
st.subheader("AI Waste Classification & Smart Recycling Assistant")

st.write(
    "Upload an image of a waste item and the AI model will "
    "classify it into one of five waste categories."
)

st.info(
    "Supported categories: Plastic • Paper • Metal • Glass • Organic"
)

# ---------------------------------------------------------
# Image upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def predict_image(image):
    # Resize image to the size expected by the model
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.asarray(image)

    # Make sure image has RGB channels
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Normalize pixel values
    image_array = image_array.astype(np.float32) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array, verbose=0)[0]

    # Find class with highest probability
    predicted_index = int(np.argmax(prediction))

    confidence = float(prediction[predicted_index]) * 100

    predicted_label = labels[predicted_index]

    # Remove class number from label
    predicted_label = predicted_label.split(" ", 1)[-1].strip().upper()

    return predicted_label, confidence

# ---------------------------------------------------------
# Display and analyze image
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    st.write("")

    if st.button("🔍 ANALYZE WASTE", use_container_width=True):

        with st.spinner("AI is analyzing the image..."):

            predicted_label, confidence = predict_image(image)

        # -------------------------------------------------
        # Confidence check
        # -------------------------------------------------

        if confidence < 50:

            st.warning(
                f"⚠️ The AI is not very confident about this prediction "
                f"({confidence:.1f}%). Please try uploading a clearer "
                f"image with the waste item clearly visible."
            )

        else:

            info = WASTE_INFO.get(
                predicted_label,
                {
                    "category": "Unknown",
                    "icon": "♻️",
                    "recommendation": "Please check local waste-management guidelines."
                }
            )

            st.success("✅ Analysis Complete")

            st.markdown("---")

            st.header(
                f"{info['icon']} {predicted_label}"
            )

            st.metric(
                "AI Confidence",
                f"{confidence:.1f}%"
            )

            st.subheader("♻️ Waste Category")

            if info["category"] == "Recyclable":
                st.success(info["category"])
            elif info["category"] == "Compostable":
                st.success(info["category"])
            else:
                st.warning(info["category"])

            st.subheader("💡 Recommended Action")

            st.info(info["recommendation"])

            st.markdown("---")

            st.caption(
                "Note: This application is an educational AI prototype. "
                "Actual waste-disposal rules may vary by location."
            )

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("ℹ️ About the Project")

    st.write(
        "Smart Recycle AI is an educational AI application "
        "that uses image classification to identify common "
        "types of waste."
    )

    st.write("### 🤖 AI Technology")

    st.write(
        "Computer Vision\n\n"
        "Image Classification\n\n"
        "Machine Learning"
    )

    st.write("### 🌍 SDG Alignment")

    st.write(
        "**SDG 12:** Responsible Consumption and Production"
    )

    st.write(
        "**SDG 11:** Sustainable Cities and Communities"
    )

    st.write("### 📦 Classes")

    st.write(
        "• Plastic\n"
        "• Paper\n"
        "• Metal\n"
        "• Glass\n"
        "• Organic"
    )