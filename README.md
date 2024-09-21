ECG Heart Image Classification
This project implements a machine learning pipeline to classify ECG heart images as either "affected" or "normal" using a Random Forest Classifier. The dataset consists of ECG images categorized into these two classes, and the model is trained to differentiate between them.

Features:
Image Preprocessing: Images are resized to 100x100 pixels before classification.
Model: Random Forest Classifier with 100 estimators.
Training/Testing Split: 80% of the images are used for training, 20% for testing.
Evaluation: The model is evaluated using a classification report, which includes precision, recall, and F1-score.
Usage:
Load your ECG heart images into the affected and normal directories.
Run the script to train the model and classify test images.
You can also test the model with new images by providing the file path.
