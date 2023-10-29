import streamlit as st
import json

f = open('predict/result.json',)

data = json.load(f)

# Create a Streamlit app
st.title("Confidence Analysis")

# Sidebar for user input
st.sidebar.header("Configure Analysis")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Filter data based on confidence threshold
filtered_data = [item for item in data if item['confidence'] >= confidence_threshold]

# Display descriptive analysis
st.header("Confidence Analysis")

# Display the total number of data points
st.write("Total Data Points:", len(data))

# Display the number of data points above the confidence threshold
st.write("Data Points Above Confidence Threshold:", len(filtered_data))

# Display a histogram of confidence values
confidence_values = [item['confidence'] for item in data]
st.subheader("Confidence Value Distribution")
st.hist(confidence_values, bins=20, color='blue', alpha=0.7, label='All Data')
st.hist([item['confidence'] for item in filtered_data], bins=20, color='red', alpha=0.7, label='Above Threshold')
st.legend(["All Data", "Above Threshold"])
st.pyplot()

# Display summary statistics
st.subheader("Summary Statistics")
st.write("Minimum Confidence:", min(confidence_values))
st.write("Maximum Confidence:", max(confidence_values))
st.write("Mean Confidence:", sum(confidence_values) / len(confidence_values))
st.write("Median Confidence:", sorted(confidence_values)[len(confidence_values) // 2])

# Display a table with the data
st.subheader("Data Table")
st.write(filtered_data)

# You can add more analysis or visualizations based on your specific data and requirements.
