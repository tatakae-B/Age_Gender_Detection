import cv2
import numpy as np

class AgeDetector:
    def __init__(self, 
                 prototxt_path='models/age_deploy.prototxt', 
                 model_path='models/age_net.caffemodel'):
        """
        Initialize the age detection model
        
        Args:
            prototxt_path (str): Path to the model architecture file
            model_path (str): Path to the pre-trained model weights
        """
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # Load pre-trained age detection model
        self.age_net = cv2.dnn.readNet(model_path, prototxt_path)
    
    def detect_age(self, face_image):
        """
        Detect age for a given face image
        
        Args:
            face_image (numpy.ndarray): Cropped face image
        
        Returns:
            str: Predicted age group
        """
        # Preprocess the image
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
