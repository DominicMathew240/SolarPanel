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
import seaborn as sns
import streamlit_authenticator as stauth
from ultralytics import YOLO
from data import class_names



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


# Load a pretrained YOLOv8n model
model = YOLO('best1.pt')

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

    cls_list = predictions.tolist()
    
    # Map class indices to class names
    class_labels = [class_names[i] for i in cls_list]

    # Create a pandas Series for class counts
    class_counts = pd.Series(class_labels).value_counts()

    st.bar_chart(class_counts)

# Function to display confidence distribution
def display_confidence_distribution(predictions):
    st.subheader('Confidence Distribution')
    conf_list = predictions.tolist()
    confidence_values = conf_list
    
    # Create the histogram using Matplotlib
    fig, ax = plt.subplots()
    ax.hist(confidence_values, bins=20)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

# Function to display a heatmap of class vs confidence
def display_confidence_heatmap(conf, number):
    st.subheader('Class vs Confidence Heatmap')

    confidences = conf.tolist()
    cls = number.tolist() 
    
    # Map class indices to class names
    class_labels = [class_names[i] for i in cls]

    # Create a DataFrame
    df_confidence_heatmap = pd.DataFrame({'class': class_labels, 'confidence': confidences})

    # Pivot the DataFrame
    df_confidence_heatmap = df_confidence_heatmap.pivot_table(index='class', values='confidence', aggfunc='mean')
    
    # Create heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df_confidence_heatmap, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Mean Confidence'}, ax=ax)
    
    # Display the plot
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
            prediction = model.predict(
                temp_file.name
            )
            # Remove the temporary file
            os.remove(temp_file.name)

            # Display the prediction
            st.header("Inference Results")
            image = Image.open("results.jpg")
            col2.image(image, caption="Result Image", use_column_width=True)

            # Access the confidence values
            for r in prediction:
                number = r.boxes.cls
                confidence = r.boxes.conf
         

            # Display the class distribution
            display_class_distribution(number)
            # Create two columns
            col3, col4 = st.columns(2)

            # Display the outputs in the columns
            with col3:
                display_confidence_distribution(confidence)
            with col4:
                display_confidence_heatmap(confidence, number)

            # Count the number of predictions
            num_predictions = len(number)
            st.write("Number of Predictions: ", num_predictions)
            
            col1, col2 = st.columns(2)
            confidences = confidence.tolist()

            # Calculate total confidence
            total_confidence = sum(confidences)

            # Calculate average confidence
            if len(confidences) > 0:
                average_confidence = total_confidence / len(confidences)
            else:
                average_confidence = 0
            # Display rhe class name and average confidence
            class_name = class_names[int(number[0])]

            # Use st.metric to display the class name
            col1.metric(label="Class", value=class_name)  
            col2.metric(label="Avg Confidence", value=average_confidence)   

            col1, col2 = st.columns(2) 

            # Display the prediction in a table
            df = pd.DataFrame({'class': class_name, 'confidence': confidences})
            col1.write("Confidence Analysis Summary:")
            col1.write(df['confidence'].describe())  # Display summary statistics for confidenc
            
            # Display the prediction in a bar chart
            col2.bar_chart(pd.DataFrame({'class': class_name, 'confidence': confidences}))
            
            # st.write("Detailed JSON Output:")
            # st.write(prediction.json())