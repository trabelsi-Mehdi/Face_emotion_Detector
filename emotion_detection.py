"""
Real-time Facial Emotion Recognition System
Uses pre-trained model for emotion detection from webcam feed
"""

import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time

class EmotionDetector:
    def __init__(self, model_path='models/emotion_model.h5'):
        """
        Initialize the emotion detector
        
        Args:
            model_path: Path to the trained emotion recognition model
        """
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Load the pre-trained model
        try:
            self.model = load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except:
            print(f"⚠ Could not load model from {model_path}")
            print("  Will create a placeholder model. Train a real model first!")
            self.model = None
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Color mapping for emotions
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 255, 0),    # Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 255),    # Yellow
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (255, 165, 0), # Orange
            'Neutral': (200, 200, 200) # Gray
        }
        
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model input
        
        Args:
            face_img: Cropped face region from frame
            
        Returns:
            Preprocessed image ready for model
        """
        # Resize to model input size (48x48 for FER2013 standard)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype('float32') / 255.0
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        return face_img
    
    def detect_emotion(self, face_img):
        """
        Detect emotion from face image
        
        Args:
            face_img: Cropped face region
            
        Returns:
            tuple: (emotion_label, confidence, all_predictions)
        """
        if self.model is None:
            return "No Model", 0.0, None
        
        processed_face = self.preprocess_face(face_img)
        predictions = self.model.predict(processed_face, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx]
        
        return self.emotion_labels[emotion_idx], confidence, predictions
    
    def draw_emotion_bar(self, frame, predictions, x, y, w):
        """
        Draw emotion probability bars on frame
        
        Args:
            frame: Video frame
            predictions: Array of emotion probabilities
            x, y, w: Face bounding box coordinates
        """
        if predictions is None:
            return
        
        bar_height = 15
        bar_spacing = 5
        start_y = y + 10
        
        for i, (emotion, prob) in enumerate(zip(self.emotion_labels, predictions)):
            bar_y = start_y + i * (bar_height + bar_spacing)
            bar_width = int(prob * 150)
            
            # Draw background
            cv2.rectangle(frame, (x + w + 10, bar_y), 
                         (x + w + 160, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Draw probability bar
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x + w + 10, bar_y), 
                         (x + w + 10 + bar_width, bar_y + bar_height), 
                         color, -1)
            
            # Draw text
            text = f"{emotion}: {prob:.2f}"
            cv2.putText(frame, text, (x + w + 165, bar_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_webcam(self, show_probabilities=True):
        """
        Run real-time emotion detection from webcam
        
        Args:
            show_probabilities: Whether to show emotion probability bars
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n🎥 Starting webcam emotion detection...")
        print("Press 'q' to quit, 'p' to toggle probability bars")
        
        fps_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Detect emotion
                emotion, confidence, predictions = self.detect_emotion(face_roi)
                
                # Draw bounding box
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label with confidence
                label = f"{emotion} ({confidence:.2f})"
                label_y = y - 10 if y - 10 > 10 else y + h + 20
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(frame, (x, label_y - text_h - 5), 
                            (x + text_w, label_y + 5), color, -1)
                cv2.putText(frame, label, (x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Draw probability bars
                if show_probabilities and predictions is not None:
                    self.draw_emotion_bar(frame, predictions, x, y, w)
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                show_probabilities = not show_probabilities
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run emotion detection"""
    detector = EmotionDetector()
    detector.run_webcam(show_probabilities=True)

if __name__ == "__main__":
    main()