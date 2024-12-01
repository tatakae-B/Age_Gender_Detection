import cv2
import numpy as np
from age_gender_detector import AgeGenderDetector

def main():
    # Initialize face and age/gender detectors
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    age_gender_detector = AgeGenderDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Detect age and gender
                age = age_gender_detector.detect_age(face_roi)
                gender = age_gender_detector.detect_gender(face_roi)
                
                # Display age and gender on the frame
                display_text = f'Age: {age}, Gender: {gender}'
                cv2.putText(
                    frame, 
                    display_text, 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (0, 255, 0), 
                    2
                )
            except Exception as e:
                print(f"Error detecting age/gender: {e}")
        
        # Display the frame
        cv2.imshow('Age and Gender Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
