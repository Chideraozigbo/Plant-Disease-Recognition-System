import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model("already_trained_model.keras")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    predicted_class_index = np.argmax(predictions)
    prediction_percentage = predictions[0][predicted_class_index] * 100
    return predicted_class_index, prediction_percentage

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM PROJECT")
    image_path = "figure1.jpg"
    st.image(image_path,use_column_width=True)
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

#About Project
elif(app_mode=="About"):
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
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        st.balloons()  # Add snow effect
        predicted_class_index, prediction_percentage = model_prediction(test_image)
        #Reading Labels
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
        # Display recommendation
        if predicted_class_name in recommendations:
            recommendation = recommendations[predicted_class_name]
            st.write("Recommendation:", recommendation)
        else:
            st.write("No recommendation available for this class.")
