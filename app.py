import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import yaml

# Load model once
@st.cache_resource
def load_unet_model(model_path="models/final_model.h5"):
    return load_model(model_path, compile=False)

# Main app
def main():
    st.title("Satellite Segmentation with U-Net")
    model = load_unet_model()
    config = yaml.safe_load(open("config.yaml"))
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png"])

    if uploaded_file is not None:
        img_name = uploaded_file.name.split(".")[0]
        img_size = tuple(config["training"]["image_size"])

        image = Image.open(uploaded_file).convert("RGB")
        if "aug" in img_name:
            mask = Image.open(config["data"]["augmented_mask_dir"] + "/" + img_name + ".png")
        else:
            mask = Image.open(config["data"]["mask_dir"] + "/" + img_name + ".png")

        st.image(image, caption="Original Image", use_container_width=True)

        # Preprocess
        img = np.array(image.resize(img_size)) / 255.0
        img = np.expand_dims(img, axis=0)
        mask = np.array(mask.resize(img_size))

        # Predict
        prediction = model.predict(img)[0, :, :, 0]
        pred_mask = (prediction > 0.25).astype(np.uint8) * 255

        col1, col2 = st.columns(2)

        col1.image(pred_mask, caption="Predicted Mask", use_container_width=True)
        col2.image(mask, caption="Original Mask", use_container_width=True)

if __name__ == "__main__":
    main()
