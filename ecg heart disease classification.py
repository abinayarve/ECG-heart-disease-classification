#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[2]:


repo_path = 'C:/Users/rveab/Downloads/ecgheart'

# Defining the classes
classes = ['affected','normal']


# In[3]:


# Load images and labels
images = []
labels = []

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(repo_path, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(class_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # Resize images to a fixed size if necessary
                image = cv2.resize(image, (100, 100))  # Adjust dimensions as needed
                images.append(image)
                labels.append(class_idx)


# In[4]:


# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)


# In[5]:


# Split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# In[6]:


# Flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# In[7]:


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_flat, y_train)


# In[8]:


# Make predictions on the testing data
predictions = clf.predict(X_test_flat)


# In[9]:


# Evaluate the classifier
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=classes))


# In[10]:


# testing the model
input_image_path = "C:/Users/rveab/Downloads/ecg data/Normal/Normal (132).jpg"  # Replace with the path to your input image
input_image = cv2.imread(input_image_path)

# Preprocessiing the input image
input_image_resized = cv2.resize(input_image, (100, 100))  # Resize the image to match the training image size
input_image_flat = input_image_resized.reshape(1, -1)  # Flatten the image

# Ensure the input image has the correct number of features (30,000)
if input_image_flat.shape[1] != 30000:
    print("Input image does not have the correct number of features (30,000). Please resize the image to match the expected dimensions.")
else:
    predicted_class = clf.predict(input_image_flat)[0]  # Assuming 'clf' is your trained Random Forest Classifier
    predicted_class_name = classes[predicted_class]
    print("Predicted class:", predicted_class_name)


# In[ ]:




