from flask import Flask, render_template
import cv2
import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import easyocr

app = Flask(__name__)

# Load pre-trained ResNet50 model for helmet detection
helmet_model = ResNet50(weights='imagenet')

# Create an EasyOCR reader for license plate detection
license_plate_reader = easyocr.Reader(['en'])

# Function to preprocess the image for ResNet50
def preprocess_image(img):
    img_array = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to predict whether a helmet is present
def predict_helmet(img):
    processed_image = preprocess_image(img)
    predictions = helmet_model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions)
    
    # Get the top prediction
    top_prediction = decoded_predictions[0][0]
    
    # Extract label and probability
    label, _, confidence = top_prediction
    
    return label.lower(), confidence

# Function to detect faces in the image
def detect_face(img):
    # Load the Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Return True if a face is detected, otherwise False
    return len(faces) > 0

# Function to detect characters on the number plate
def detect_characters(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use EasyOCR to detect characters
    results = license_plate_reader.readtext(gray)

    # Extract characters from the results
    characters = [result[1] for result in results]

    return characters

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_helmet', methods=['POST'])
def detect_helmet():
    cap = cv2.VideoCapture(0)  # Use the default camera (0)

    while True:
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Check if a face is present
        has_head = detect_face(frame)

        if not has_head:
            cv2.putText(frame, 'No head detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Check if a helmet is present
            label, confidence = predict_helmet(frame)
            label_text = f"{label.capitalize()}"

            if confidence < 0.5:
                status_text = f'Not wearing helmet ({confidence:.2%})'
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Detect characters on the license plate
                characters = detect_characters(frame)

                # Display the detected characters
                if characters:
                    detected_text = ' '.join(characters)
                    cv2.putText(frame, f'Characters: {detected_text}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No characters detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            else:
                status_text = f'Wearing helmet ({confidence:.2%})'
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Helmet Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return render_template('index.html')

# ... (other routes and functions)

if __name__ == '__main__':
    app.run(debug=True)
