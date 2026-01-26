

**Deep Learning for Automated Medical Image Diagnosis 🫁**


Group Name: The Deep Learning Friends

Project Date: April 27, 2025

Target Venue: IEEE Journal Submission



📋 Overview

This project presents an automated pneumonia diagnosis system using a deep learning architecture that integrates VGG16 with custom CNN layers. The goal is to provide rapid, accurate, and scalable medical diagnosis to assist healthcare professionals, particularly in resource-limited settings where expert radiologists may be unavailable.
The system is deployed as a real-time web application using Streamlit and Ngrok, allowing users to upload chest X-ray images and receive immediate predictions.


✨ Key Features

Deep Learning Model: Fine-tuned VGG16 architecture with custom Dense, Dropout, and Sigmoid layers.

Real-time Inference: Interactive web interface for instant image classification.

Robust Preprocessing: Includes resizing, pixel normalization, and data augmentation (flipping, rotation, zoom).

Public Access: Deployed via Ngrok for external accessibility.


🏗️ Architecture

The model utilizes a transfer learning approach:

Base Model: VGG16 (pretrained on ImageNet) with the top classification layer removed.

Custom Layers: * Flatten layer

Dense (128 units) with ReLU activation

Dropout (0.5) for overfitting prevention

Dense (1 unit) with Sigmoid activation for binary classification


📊 Dataset & Performance

Dataset: NIH Chest X-ray dataset obtained via Kaggle API.

Training: 8-10 epochs using Adam optimizer and Binary Cross-Entropy loss.

Results:

Validation Accuracy: 59% (weighted average F1-score: 0.60).

Clinical Generalization: Successfully distinguishes between Normal and Pneumonia cases in real-time testing.

Metric	Normal	Pneumonia
Precision	0.24	0.73
Recall	0.27	0.70
F1-Score	0.25	0.72


🚀 Installation & Usage

Prerequisites

Python 3.11+

TensorFlow, Streamlit, OpenCV, Matplotlib, Pyngrok

Local Setup

Clone the repository:

git clone https://github.com/your-username/Pneumonia-Detection.git

cd Pneumonia-Detection

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py


👥 Team Members & Contributions

Sai Sravani Madabhushi: Literature review, evaluation, Streamlit deployment, and documentation.

Vishanth Raj Chowhan Lavudya: Model training, hyperparameter tuning, and data cleaning.

Surya Shashank Pappu: Data collection, preprocessing, and initial model implementation.


🛠️ Future Enhancements

Implementation of ResNet50 architecture for improved feature extraction.

Threshold optimization to improve sensitivity vs. specificity ratios.

Expansion to larger, more diverse clinical datasets.


📜 License

Distributed under the GPL-3.0 License. See LICENSE for more information.

Instructor: Dr. Ghoraani Behnaaz

Institution: Florida Atlantic University
