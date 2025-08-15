# Facial Emotion Detection using CNN

## Project Overview
This project (July 2024) develops a deep Convolutional Neural Network (CNN) to detect human emotions from facial images. The model classifies facial expressions into seven categories (**angry, disgust, fear, happy, sad, surprise, neutral**) and includes a real-time emotion detection pipeline, with applications in human-computer interaction, mental health monitoring, and customer experience analysis.

## Objectives
- Preprocess facial images to enhance model performance.  
- Build and train a deep CNN with robust architecture to classify emotions accurately.  
- Deploy a real-time pipeline for emotion detection with visualization of results.  

## Dataset
The dataset used is **FER-2013** ([Kaggle link](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)) containing:
- **Images**: 35,887 grayscale images (48x48 pixels) split into training (28,709) and test (7,178) sets.  
- **Labels**: Seven emotion classes – angry, disgust, fear, happy, sad, surprise, neutral.  
- **Challenges**: Misaligned images, incorrect labels, and non-face images, addressed via preprocessing.  

## Methodology

### Data Preprocessing
- **CLAHE**: Enhanced image contrast for better feature visibility.  
- **Grayscale Conversion**: Standardized images to grayscale.  
- **Normalization**: Scaled pixel values to [0,1].  
- **Data Augmentation**: Applied rotation, flipping, and zoom to reduce overfitting.  
- **Reshaping**: Resized images to 48x48 pixels for CNN input.

### Model Architecture
- **Conv2D** layers for feature extraction.  
- **MaxPooling** to reduce spatial dimensions.  
- **Batch Normalization** to stabilize and accelerate training.  
- **Dropout** to prevent overfitting.  
- **Training**: Used categorical cross-entropy loss with Adam optimizer.  
- **Libraries**: TensorFlow/Keras.

### Real-Time Pipeline
- **Preprocessing**: CLAHE, grayscale, normalization.  
- **Reshaping**: Input images resized to 48x48.  
- **Visualization**: Displayed predicted probabilities for all seven emotion classes in real-time (bar chart/heatmap).  

### Model Evaluation
- **Metrics**: Accuracy, precision, recall, and F1-score/confusion matrix.  
- **Performance Boost**: Preprocessing improved CNN accuracy by 6%.  

## Results
- **Preprocessing**: CLAHE, grayscale, and normalization enhanced image quality, boosting accuracy by 6%.  
- **Model Performance**: Achieved high accuracy on FER-2013 test set with robust classification across all classes.  
- **Real-Time Pipeline**: Successfully deployed with 7-class probability visualization.  
- **Impact**: Enables real-time emotion recognition for applications in mental health, education, and customer experience analysis.  

## Requirements
Install dependencies:
```bash
pip install pandas numpy opencv-python tensorflow scikit-learn matplotlib seaborn

