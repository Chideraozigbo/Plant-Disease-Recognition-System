import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from keras.models import load_model

# Load the pre-trained model
load_models = tf.keras.models.load_model('already_trained_model.keras')

# Class names for prediction
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Recommendations for each class
recommendations = {
    'Apple___Apple_scab': 'Remove and destroy infected leaves. Apply fungicide as needed.',
    'Apple___Black_rot': 'Prune and destroy infected branches. Apply fungicide as needed.',
    'Apple___Cedar_apple_rust': 'Prune and destroy infected branches. Apply fungicide as needed.',
    'Apple___healthy': 'Ensure proper care and maintenance.',
    'Blueberry___healthy': 'Ensure proper care and maintenance.',
    'Cherry_(including_sour)___Powdery_mildew': 'Prune and destroy infected branches. Apply fungicide as needed.',
    'Cherry_(including_sour)___healthy': 'Ensure proper care and maintenance.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use disease-resistant varieties. Apply fungicide as needed.',
    'Corn_(maize)___Common_rust_': 'Plant resistant varieties. Use fungicides if necessary.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant varieties. Rotate crops. Apply fungicides as needed.',
    'Corn_(maize)___healthy': 'Ensure proper care and maintenance.',
    'Grape___Black_rot': 'Prune and destroy infected branches. Apply fungicide as needed.',
    'Grape___Esca_(Black_Measles)': 'Prune and destroy infected branches. Apply fungicide as needed.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Prune and destroy infected branches. Apply fungicide as needed.',
    'Grape___healthy': 'Ensure proper care and maintenance.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Remove and destroy infected trees. Use disease-resistant varieties.',
    'Peach___Bacterial_spot': 'Prune and destroy infected branches. Apply copper-based fungicides.',
    'Peach___healthy': 'Ensure proper care and maintenance.',
    'Pepper,_bell___Bacterial_spot': 'Prune and destroy infected branches. Apply copper-based fungicides.',
    'Pepper,_bell___healthy': 'Ensure proper care and maintenance.',
    'Potato___Early_blight': 'Practice crop rotation. Apply fungicides as needed.',
    'Potato___Late_blight': 'Practice crop rotation. Apply fungicides as needed.',
    'Potato___healthy': 'Ensure proper care and maintenance.',
    'Raspberry___healthy': 'Ensure proper care and maintenance.',
    'Soybean___healthy': 'Ensure proper care and maintenance.',
    'Squash___Powdery_mildew': 'Remove and destroy infected leaves. Apply fungicide as needed.',
    'Strawberry___Leaf_scorch': 'Prune and destroy infected branches. Apply fungicide as needed.',
    'Strawberry___healthy': 'Ensure proper care and maintenance.',
    'Tomato___Bacterial_spot': 'Prune and destroy infected branches. Apply copper-based fungicides.',
    'Tomato___Early_blight': 'Prune and destroy infected branches. Apply copper-based fungicides.',
    'Tomato___Late_blight': 'Prune and destroy infected branches. Apply copper-based fungicides.',
    'Tomato___Leaf_Mold': 'Prune and destroy infected branches. Apply copper-based fungicides.',
    'Tomato___Septoria_leaf_spot': 'Prune and destroy infected branches. Apply copper-based fungicides.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use insecticidal soap or neem oil. Ensure proper watering and ventilation.',
    'Tomato___Target_Spot': 'Prune and destroy infected branches. Apply copper-based fungicides.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Remove and destroy infected plants. Use disease-resistant varieties.',
    'Tomato___Tomato_mosaic_virus': 'Remove and destroy infected plants. Use disease-resistant varieties.',
    'Tomato___healthy': 'Ensure proper care and maintenance.'
}

# App Title
st.title('Plant Disease Diagnosis System')
st.markdown('Upload the picture of the plant')

# Upload button
plant_image = st.file_uploader('Choose an image ..', type=['jpg', 'png'])
submit = st.button('Predict Disease')

if submit:
    if plant_image is not None:

        # Convert to opencv
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying image with animation
        with st.spinner('Analyzing...'):
            st.image(opencv_image, channels="BGR", caption="Uploaded Image", use_column_width=True)
            # Resizing the image
            resized_image = cv2.resize(opencv_image, (128, 128))

            # Convert the image to 4 dimension
            input_array = np.expand_dims(resized_image, axis=0)

            # Make predictions
            prediction = load_models.predict(input_array)
            result_index = np.argmax(prediction)
            prediction_percentage = round(np.max(prediction) * 100, 2)

            # Display prediction
            predicted_class = class_names[result_index]
            st.write(f"Predicted Disease: {predicted_class}")
            st.write(f"Confidence: {prediction_percentage}%")

            # Display recommendations if available
            if predicted_class in recommendations:
                st.write("Recommendations:")
                st.write(recommendations[predicted_class])
            else:
                st.write("No recommendations available for this disease.")
            
    else:
        st.write("Please upload an image before predicting.")
else:
    st.write("Click the 'Predict Disease' button to make a prediction.")
