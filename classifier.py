import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import nibabel as nib
import tempfile

# Load your Keras model once globally
model_path = r"D:\Projects\Brain_Tumour\Brain_tumor_Final\brain_tumor_classifier_new.keras"
model = load_model(model_path)

def preprocess_nii_image(nii_array):
    # Assuming the model expects 2D images with 3 channels (RGB)
    image_array = np.array(nii_array, dtype=np.float32)
    
    if image_array.ndim == 3:
        middle_index = image_array.shape[-1] // 2
        image_array = image_array[:, :, middle_index]

    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.ndim == 3 and image_array.shape[-1] == 1:
        image_array = np.concatenate([image_array] * 3, axis=-1)
    
    if image_array.shape[:2] != (200, 200):
        image_array = cv2.resize(image_array, (200, 200))
    
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  
    
    return image_array

def preprocess_jpg_image(image):
    image_array = np.array(image.resize((200, 200)))
    image_array = np.expand_dims(image_array, axis=0) / 255.0
    return image_array

def predict_image(image_array):
    # Ensure consistent input shape
    image_array = np.asarray(image_array, dtype=np.float32)
    
    # Perform prediction
    prediction = model.predict(image_array)
    
    return prediction

def predict_nii(nii_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(nii_file.read())
        temp_file_path = temp_file.name

    nii_image = nib.load(temp_file_path)
    nii_array = nii_image.get_fdata()
    
    processed_image = preprocess_nii_image(nii_array)
    
    prediction = predict_image(processed_image)
    return prediction

def app():
    st.title("Brain Tumor Classifier")
    st.write("This model was trained on .nii files. Upload a .jpg or .nii image for classification.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "nii"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]

        if file_extension == "jpg":
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("Classifying...")

            image_array = preprocess_jpg_image(image)
            prediction = predict_image(image_array)
        
        elif file_extension == "nii":
            st.write("Classifying...")
            prediction = predict_nii(uploaded_file)
        
        predicted_class_index = np.argmax(prediction)
        tumor_classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        predicted_tumor_name = tumor_classes[predicted_class_index]
        st.write(f"The predicted tumor is: {predicted_tumor_name}")

    # Clear session after each prediction to prevent errors when switching files
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    app()
