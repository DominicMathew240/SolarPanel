import streamlit as st
from pathlib import Path
from PIL import Image
import yaml 
from yaml.loader import SafeLoader
import tempfile
from roboflow import Roboflow
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import streamlit_authenticator as stauth
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


node_comp = Image.open("logo/node-comp.png")
stratetics = Image.open("logo/stratetics.png")
dronez = Image.open("logo/dronez.png")

# st.image([node_comp, stratetics, dronez], width=200)


from io import BytesIO
import base64

def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

node_comp_base64 = get_image_base64(node_comp)
stratetics_base64 = get_image_base64(stratetics)
dronez_base64 = get_image_base64(dronez)

# Display images in the footer
footer = f"""
<style>
.footer {{
    background-color: #ffffff;
    position: fixed;
    right: 0;
    float: right;
    top: 0;
    width: 26%;
    border-radius: 0px 0px 0px 12px;
    text-align: center;
    z-index: 1000;
}}

.footer-image {{
    margin: 6px;
}}

</style>
<div class="footer">
<img src="data:image/png;base64,{node_comp_base64}" width="32%" class="footer-image"/>
<img src="data:image/png;base64,{stratetics_base64}" width="32%" class="footer-image"/>
<img src="data:image/png;base64,{dronez_base64}" width="18%" class="footer-image"/>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            header {visibility: hidden;}
            .container {display:none;}
            [href*="https://streamlit.io/cloud"] {display: none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Function to create a folder if it doesn't exist
def safe_create_path(path: str) -> None:
    path_parent = os.path.dirname(path)
    os.makedirs(path_parent, exist_ok=True)

# Function to display distribution of classes
def display_class_distribution(predictions):
    st.subheader('Class Distribution')
    class_counts = pd.Series([prediction['class'] for prediction in predictions]).value_counts()
    st.bar_chart(class_counts)

# Function to display confidence distribution
def display_confidence_distribution(predictions):
    st.subheader('Confidence Distribution')
    confidence_values = pd.Series([prediction['confidence'] for prediction in predictions])
    
    # Create the histogram using Matplotlib
    fig, ax = plt.subplots()
    ax.hist(confidence_values, bins=20)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

# Function to display a heatmap of class vs confidence
def display_confidence_heatmap(predictions):
    st.subheader('Class vs Confidence Heatmap')
    df_confidence_heatmap = pd.DataFrame(predictions, columns=['class', 'confidence'])
    df_confidence_heatmap = df_confidence_heatmap.pivot_table(index='class', values='confidence', aggfunc='mean')
    fig, ax = plt.subplots()
    sns.heatmap(df_confidence_heatmap, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Mean Confidence'})
    st.pyplot(fig)

# Function to dump content to a JSON file
def dump_to_json(target_path: str, content: dict) -> None:
    with open(target_path, 'w') as outfile:
        json.dump(content, outfile, indent=4)

# Set your Roboflow API key
roboflow_api_key = "dcZ99wzOfjJAOBZBqzQx"

# Initialize Roboflow
rf = Roboflow(api_key=roboflow_api_key)

# Hardcoded username and password (for demonstration purposes)
names = ["Peter Parker", "Rebecca Miller"]
usernames = ["pparker", "rmiller"]

# load hashed passwords
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

names, authentication_status, username = authenticator.login("Login", "main")

if st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
elif st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar', key='unique_key')
    # Sidebar for user input
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
                temp_file.name, confidence=confidence, overlap=overlap, show=True
            )
            # Perform inference with the temporary file path - RESULT IN IMAGE FORMAT
            prediction2 = rf.project("pv-temperature-detection").version(2).model.predict(
                temp_file.name, confidence=confidence, overlap=overlap, show=True
            ).save("results.jpg")
            
            # Remove the temporary file
            os.remove(temp_file.name)

            # Display additional insights based on predictions
            result = prediction.json()['predictions']
            display_class_distribution(result)

            # Create two columns
            col3, col4 = st.columns(2)

            # Display the outputs in the columns
            with col3:
                display_confidence_distribution(result)
            with col4:
                display_confidence_heatmap(result)

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