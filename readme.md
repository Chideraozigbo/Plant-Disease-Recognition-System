# Plant Disease Diagnosis System

This project is a plant disease diagnosis system that helps identify diseases in plants based on images uploaded by users. It utilizes a pre-trained deep learning model to make predictions about the type of disease affecting the plant. This project was done as my final year degree exam for the award of Bachelor of Science in Computer Science.

## Features

- Allows users to upload images of plants.
- Predicts the type of disease affecting the plant.
- Displays the prediction along with confidence percentage.
- Give a little recommendation

## Installation

1. Clone the repository: 
    ```bash
    git clone https://github.com/Chideraozigbo/Plant-Disease-Recognition-System.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained model [here](https://github.com/Chideraozigbo/largefiles/blob/main/already_trained_model.keras) and place it in the appropriate directory.


## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to the provided URL.

3. Upload an image of a plant.

4. Click the "Predict Disease" button to see the diagnosis.

## Dataset
The dataset used for training the model can be found [here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
