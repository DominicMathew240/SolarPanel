## Login.txt v1.0

Description is the first version of the AI POC model, created by Dominic Mathew David

# Features
- Detect hotspot with thermal inspection, Model Architecture: YOLOv8, Mean Average Precision (mAP): 63.7%, Precision: 77.9%, Recall: 66.2%
- Confidence Threshold: The system allows users to set a confidence threshold, ensuring that only predictions with confidence above the specified level are considered.
- Overlap Threshold: Fine-tune the model's performance by adjusting the overlap threshold for more precise object detection.
- Predictions are presented visually on the uploaded image, Inference results are saved in a JSON file for further analysis.

## RoadMap & Future implementation
- [ ] Semantic Segmentation Analysis: Utilize SAM's capabilities to perform detailed semantic segmentation analysis, enabling the system to understand the context and relationships between different objects within an image. [planning]
- [x] User Annotation Interface: Introduce an annotation interface that allows users to manually label objects within images. This annotated data can be used for future training, enhancing the model's accuracy and adaptability. [planning]
- [ ] Data Analytics Dashboard: Develop a user-friendly dashboard that provides statistical insights into the dataset, including object distribution, common patterns, and potential areas for model improvement. [planning]
