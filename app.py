import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.title("🌱 Plant Disease Detector")

st.write("For best results please upload a clear image of the plant leaf.")

# --------------------------------------------------
# 1. Plant selection
# --------------------------------------------------

plant_classes = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato"
]

selected_plant = st.selectbox(
    "Choose the plant:",
    plant_classes,
    index=None,
    placeholder="Select a plant..."
)

# --------------------------------------------------
# 2. Load model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="tiganihatim/plant-disease-model",
        filename="best_plant_disease_model.keras"
    )

    return tf.keras.models.load_model(model_path)


model = load_model()

# These must match the class order used when training
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Background_without_leaves",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# --------------------------------------------------
# 3. Image input
# --------------------------------------------------

uploaded_file = None

if selected_plant:

    st.success(f"Selected plant: {selected_plant}")

    option = st.radio(
        "Upload image:",
        ["upload from files", "use camera"]
    )

    if option == "upload from files":

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"]
        )

    else:

        uploaded_file = st.camera_input(
            "Use camera"
        )

else:

    st.info("Please select a plant type to continue.")


# --------------------------------------------------
# 4. Make prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Selected image",
        use_container_width=True
    )

    # Resize to the same size used during training
    image = image.resize((180, 180))

    # Convert image to numpy array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # --------------------------------------------------
    # 5. Find overall prediction
    # --------------------------------------------------

    overall_index = np.argmax(predictions)

    overall_prediction = class_names[overall_index]

    overall_confidence = predictions[overall_index] * 100

    # Determine which plant the model thinks it sees
    if overall_prediction.startswith("Pepper,_bell___"):
        predicted_plant = "Pepper"

    elif "___" in overall_prediction:
        predicted_plant = overall_prediction.split("___")[0]

    else:
        predicted_plant = "Unknown"


    # --------------------------------------------------
    # 6. Check whether plant matches
    # --------------------------------------------------

    if predicted_plant != selected_plant:

        st.error(
            "❌ This image does not match the chosen plant."
        )

        st.write(
            f"You selected **{selected_plant}**, "
            f"but the model detected **{predicted_plant}**."
        )

        st.write(
            f"Model confidence: **{overall_confidence:.2f}%**"
        )


    else:

        # --------------------------------------------------
        # 7. Find disease for selected plant
        # --------------------------------------------------

        matching_indices = [
            i for i, name in enumerate(class_names)
            if name.startswith(selected_plant + "___")
            or (
                selected_plant == "Pepper"
                and name.startswith("Pepper,_bell___")
            )
        ]

        best_index = max(
            matching_indices,
            key=lambda i: predictions[i]
        )

        confidence = predictions[best_index] * 100

        predicted_class = class_names[best_index]

        # Remove plant name from result
        if "___" in predicted_class:
            disease = predicted_class.split("___", 1)[1]
        else:
            disease = predicted_class


        # --------------------------------------------------
        # 8. Display health status
        # --------------------------------------------------

        st.subheader("Plant Health Status")

        if disease == "healthy":

            st.success("🌿 Healthy")

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

        else:

            st.error("⚠️ Diseased")

            st.write(
                f"Possible disease: **{disease}**"
            )

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )