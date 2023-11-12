import streamlit as st
from pathlib import Path
from PIL import Image
import pickle
import tempfile
from roboflow import Roboflow
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import streamlit_authenticator as stauth

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Function to create a folder if it doesn't exist
def safe_create_path(path: str) -> None:
    path_parent = os.path.dirname(path)
    os.makedirs(path_parent, exist_ok=True)

# safe_create_path('predict')

# Function to dump content to a JSON file
def dump_to_json(target_path: str, content: dict) -> None:
    with open(target_path, 'w') as outfile:
        json.dump(content, outfile, indent=4)

# Set your Roboflow API key
roboflow_api_key = "dcZ99wzOfjJAOBZBqzQx"

# Initialize Roboflow
rf = Roboflow(api_key=roboflow_api_key)

# Hardcoded username and password (for demonstration purposes)
names = ["user", "user2"]
usernames = ["user", "user2"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "solar_panel", "abcdef", cookie_expiry_days=30)

names, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Incorrect username or password")

if authentication_status == None:
    st.error("Please login")

if authentication_status == True:
    # Sidebar for user input
    authenticator.logout("Logout", "sidebar")
    st.sidebar.header("Configure Inference")
    st.sidebar.write("Confidence Threshold: The minimum confidence required for a prediction to be considered a positive detection.")
    confidence = st.sidebar.slider("Confidence Threshold", 0, 100, 40)
    overlap = st.sidebar.slider("Overlap Threshold", 0, 100, 30)

    # Upload an image for inference
    st.sidebar.header("Upload an Image for Inference")
    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    col1, col2 = st.columns(2)
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        col1.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform inference on the uploaded image
        if st.sidebar.button("Run Inference"):
            st.write("Running Inference...")
            
            # Save the uploaded image as a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", dir='predict', delete=False) as temp_file:
                image.save(temp_file.name)
            
            # Perform inference with the temporary file path - Result in JSON format
            prediction = rf.project("pv-temperature-detection").version(2).model.predict(
                temp_file.name, confidence=confidence, overlap=overlap
            )
            # Perform inference with the temporary file path - RESULT IN IMAGE FORMAT
            prediction2 = rf.project("pv-temperature-detection").version(2).model.predict(
                temp_file.name, confidence=confidence, overlap=overlap
            ).save("results.jpg")
            
            # Remove the temporary file
            os.remove(temp_file.name)

            # Display the prediction
            st.header("Inference Results")
            image = Image.open("results.jpg")
            col2.image(image, caption="Result Image", use_column_width=True)
            
            # Count the number of predictions
            result = prediction.json()['predictions']
            num_predictions = len(result)
            st.write("Number of Predictions: ", num_predictions)
            
            json_result = prediction.json()
            dump_to_json('predict/result.json', json_result)

            f = open('predict/result.json',)

            data = json.load(f)
            col1, col2 = st.columns(2)
            total_confidence = 0
            for i in data["predictions"]:
                total_confidence += i['confidence']

            average_confidence = total_confidence / len(data["predictions"])

            col1.metric(label="Class", value=data["predictions"][0]["class"])   
            col2.metric(label="Avg Confidence", value=average_confidence)   

            col1, col2 = st.columns(2) 

            # Display the prediction in a table
            df = pd.DataFrame(data["predictions"], columns=['class', 'confidence'])
            col1.write("Confidence Analysis Summary:")
            col1.write(df['confidence'].describe())
            
            # Display the prediction in a table
            col2.bar_chart(pd.DataFrame(
                data["predictions"], 
                columns=['class','confidence']
                ))
            st.write("Detailed JSON Output:")
            st.write(prediction.json())