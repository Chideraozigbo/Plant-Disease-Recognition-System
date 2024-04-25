import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO

# Tensorflow Model Prediction
def model_prediction(test_image):
    model_url = "https://github.com/Chideraozigbo/largefiles/raw/main/already_trained_model.keras"
    try:
        # Download the model file from GitHub
        model_response = requests.get(model_url)
        model_response.raise_for_status()  # Raise an error for HTTP errors
        model_file = BytesIO(model_response.content)

        # Load the model from the downloaded file
        model = tf.keras.models.load_model(model_file)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, None

    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    predicted_class_index = np.argmax(predictions)
    prediction_percentage = predictions[0][predicted_class_index] * 100
    return predicted_class_index, prediction_percentage

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM PROJECT")
    image_path = "https://github.com/Chideraozigbo/Plant-Disease-Recognition-System/blob/main/streamlit-app/figure1.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### Project Description
                The "Plant Disease Recognition System" is an innovative project developed as a final year endeavor, aiming to revolutionize the agricultural sector by leveraging cutting-edge technology to identify and diagnose diseases in crops accurately and efficiently.
                
                **Key Features:**
                - **Image Recognition:** The system utilizes convolutional neural networks (CNNs) to analyze images of plant leaves and classify them into different disease categories.
                - **User-Friendly Interface:** Implemented using Streamlit, the interface offers seamless navigation and an intuitive user experience, allowing users to upload images and receive instant diagnoses.
                - **Recommendation Engine:** Upon identification of a disease, the system provides tailored recommendations and best practices for disease management and prevention, empowering farmers to take proactive measures.
                
                #### Implementation Details
                - **Data Collection and Preprocessing:** A comprehensive dataset comprising thousands of labeled images of healthy and diseased plants was collected and preprocessed. Offline augmentation techniques were applied to enhance the diversity and robustness of the dataset.
                - **Model Training:** Transfer learning techniques were employed to fine-tune pre-trained CNN architectures such as VGG16 and ResNet50 on the augmented dataset. Hyperparameter tuning and cross-validation were conducted to optimize model performance.
                - **Deployment:** The trained model was integrated into a Streamlit web application hosted on a cloud server, ensuring accessibility and scalability. The application allows users to upload images of plant leaves, triggering real-time prediction and recommendation generation.
                
                #### Impact and Future Directions
                The Plant Disease Recognition System holds immense potential to revolutionize crop management practices and contribute to global food security. Future enhancements may include expanding the dataset to encompass a broader range of crops and diseases, integrating real-time disease monitoring using IoT devices, and incorporating feedback mechanisms to continuously improve model accuracy.
                
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves which are categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    # Predict button
    if st.button("Predict"):
        st.write("Our Prediction")
        st.balloons()  # Add snow effect
        predicted_class_index, prediction_percentage = model_prediction(test_image)
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        predicted_class_name = class_name[predicted_class_index]
        st.success("Model predicts it's a {} with {:.2f}% confidence.".format(predicted_class_name, prediction_percentage))
