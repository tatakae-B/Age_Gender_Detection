import cv2
import numpy as np

class AgeGenderDetector:
    def __init__(self, 
                 age_prototxt_path='models/age_deploy.prototxt', 
                 age_model_path='models/age_net.caffemodel',
                 gender_prototxt_path='models/gender_deploy.prototxt',
                 gender_model_path='models/gender_net.caffemodel'):
        """
        Initialize the age and gender detection models
        
        Args:
            age_prototxt_path (str): Path to the age model architecture file
            age_model_path (str): Path to the age model weights
            gender_prototxt_path (str): Path to the gender model architecture file
            gender_model_path (str): Path to the gender model weights
        """
        # Age groups and configurations
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # Load pre-trained age detection model
        self.age_net = cv2.dnn.readNet(age_model_path, age_prototxt_path)
        
        # Load pre-trained gender detection model
        self.gender_net = cv2.dnn.readNet(gender_model_path, gender_prototxt_path)
        
        # Gender list
        self.gender_list = ['Male', 'Female']
    
    def detect_age(self, face_image):
        """
        Detect age for a given face image
        
        Args:
            face_image (numpy.ndarray): Cropped face image
        
        Returns:
            str: Predicted age group
        """
        # Preprocess the image for age detection
        blob = cv2.dnn.blobFromImage(
            face_image, 
            1.0, 
            (227, 227), 
            (78.4263377603, 87.7689143744, 114.895847746), 
            swapRB=False
        )
        
        # Set the blob as input and perform forward pass
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        
        # Get the age group with highest probability
        age_index = age_preds[0].argmax()
        age = self.age_list[age_index]
        
        return age
    
    def detect_gender(self, face_image):
        """
        Detect gender for a given face image
        
        Args:
            face_image (numpy.ndarray): Cropped face image
        
        Returns:
            str: Predicted gender
        """
        # Preprocess the image for gender detection
        blob = cv2.dnn.blobFromImage(
            face_image, 
            1.0, 
            (227, 227), 
            (78.4263377603, 87.7689143744, 114.895847746), 
            swapRB=False
        )
        
        # Set the blob as input and perform forward pass
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        
        # Get the gender with highest probability
        gender_index = gender_preds[0].argmax()
        gender = self.gender_list[gender_index]
        
        return gender