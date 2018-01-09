# -*- coding: utf-8 -*-

'''
Provide helper functions and define classes that use pre-trained models to classify images
'''

import numpy as np
import cv2 # library to read images
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions

def path_to_tensor(img_path):
    '''
    Takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN
    Args: 
        Path to image
    Returns: 
        Tensor with dimensions (nb_samples, rows, columns, channels)
    '''

    # Load RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size = (224, 224))

    # Convert PIL.Image.Image type to 3D tensor with square shape with 3 filters for colors (224, 224, 3) 
    x = image.img_to_array(img)

    # Convert 3D tensor to 4D tensor and return
    return np.expand_dims(x, axis = 0)

def paths_to_tensor(img_paths):
    '''
    Takes a numpy array of string-valued image paths as input and returns a 4D tensor
    Args: 
        Vector of image paths
    Returns: 
        Tensor with dimensions (nb_samples, rows, columns, channels)
    '''

    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

class HumanDetector(object):
    '''
    Use OpenCV's implementation of Haar feature-based cascade classifier to detect human faces in images
    These models are stored in .xml files

    Attributes:
        face_classifier: The OpenCV classifier 
    '''

    def __init__(self):
        '''
        Inits class with the face classifier model
        '''
        self.model = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    
    def face_detector(self, img_path):
        '''
        Read in image and convert to grayscale to avoid redundant data processing/reduce processing time
        We don't need color information for a human facial recognition task
        Args:
            Image path
        Returns:
            Image, Detected faces
        '''
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(gray)
        
        return img, faces
    
    def display_face(self, faces):
        '''
        Read in image and plot with a box capturing a human face
        Args: 
            Faces is a numpy array of detected faces, where each row corresponds to a detected face. 
        '''
        
        print('Number of faces detected:', len(faces))

        img, faces = self.face_detector(faces)
        
        # Get bounding box for each detected face
        # X, Y = horizontal/vertical positions of bounding box
        # W, H = width/height of the box
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            
        # Convert BGR image to RGB for plotting
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(cv_rgb)
        plt.show()

    def count_images(self, imgs):
        '''
        Aggregates count of classification type and returns a percentage
        Args:
            Vector of images and detector function
        Returns:
            Percent of positive classifications
        '''

        count = 0
        for img in imgs:
            if(len(self.face_detector(img)[1]) > 0):
                count = count + 1
                
        percent_count = 100. * float(count) / len(imgs)
                
        return percent_count

class DogDetector(object):
    '''
    Use pre-trained ResNet-50 model on ImageNet data to process dog images
    
    Attributes:
        model: The ResNet50 model
    '''

    def __init__(self):
        '''
        Inits class with model trained on thousands of different images
        '''
        self.model = ResNet50(weights = 'imagenet')
    
    def ResNet50_predict_labels(self, img_path):
        '''
        Returns prediction vector for image located at img_path
        Convert RGB image to BGR by reordering the channels, normalize pixels prior to processing
        Args: 
            Image path
        Returns:
            Dictionary key value
        '''
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(self.model.predict(img))

    def dog_detector(self, img_path):
        '''
        Determines if a dog is detected in the image stored at img_path
        Dogs correspond with dictionary keys 151-268 (inclusive)
        Args:
            Image path
        Returns:
            T/F
        '''
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151)) 

    def count_images(self, imgs):
        '''
        Aggregates count of classification type and returns a percentage
        Args:
            Vector of images and detector function
        Returns:
            Percent of positive classifications
        '''

        count = 0
        for img in imgs:
            if(self.dog_detector(img)):
                count = count + 1
                
        percent_count = 100. * float(count) / len(imgs)
                
        return percent_count
