import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the pre-trained model
model = load_model('inceptionv3.h5')

# Custom class names (modify these based on your model's classes)
class_names = ['Cyst', 'Normal', 'Tumor', 'Stone']

# Streamlit application
st.title("Healthcare classification")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to a PIL image
    img = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Add a button to classify the image
    if st.button("Classify"):
        st.write("Classifying...")

        # Preprocess the image
        img_resized = img.resize((299, 299))  # VGG16 expects input images of size (224, 224)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict the class of the image
        predictions = model.predict(img_array)
        top_indices = predictions[0].argsort()[-3:][::-1]  # Get indices of top 3 predictions
        top_predictions = [(class_names[i], predictions[0][i]) for i in top_indices]
        
        # Display the predictions
        st.write("Top 3 Predictions:")
        for i, (label, score) in enumerate(top_predictions):
            st.write(f"{i+1}. {label}: {score*100:.2f}%")
        
        # # Draw the predictions on the image
        # draw = ImageDraw.Draw(img)
        # font = ImageFont.load_default()
        # text = f"{top_predictions[0][0]}: {top_predictions[0][1]*100:.2f}%"
        # draw.text((10, 10), text, font=font, fill=(255, 0, 0))
        
        # # Save and display the annotated image
        # annotated_image_path = "annotated_image.jpg"
        # st.image(annotated_image_path, caption='Annotated Image with Prediction.', use_column_width=True)
