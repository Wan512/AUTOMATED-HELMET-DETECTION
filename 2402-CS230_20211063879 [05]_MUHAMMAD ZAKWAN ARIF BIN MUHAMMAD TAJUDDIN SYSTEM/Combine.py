import cv2
import easyocr
import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Load pre-trained ResNet50 model for helmet detection
helmet_model = ResNet50(weights='imagenet')

# Load Haarcascades-based face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create an EasyOCR reader for license plate detection
reader = easyocr.Reader(['en'])

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

# Function to detect characters on the number plate
def detect_characters(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use EasyOCR to detect characters
    results = reader.readtext(gray)

    # Extract characters from the results
    characters = [result[1] for result in results]

    return characters

# Function to detect the presence of a head in the image
def detect_head(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    return len(faces) > 0

# Function to save captured images into a PDF file
def save_images_to_pdf(images, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Add title to the PDF
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(letter[0] / 2, letter[1] - 30, "Captured Images Report")

    # Add images to the PDF
    for img_path in images:
        c.drawInlineImage(img_path, letter[0] / 4, letter[1] / 4, width=letter[0] / 2, height=letter[1] / 2)
        c.showPage()  # Add a new page for each image

    c.save()

# Function to detect helmet and characters on the number plate in webcam feed
def detect_helmet_and_characters_webcam():
    cap = cv2.VideoCapture(0)  # Use the default camera (0)

    # Initialize a list to store captured image paths
    captured_images = []

    while True:
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Detect head
        has_head = detect_head(frame)

        if has_head:
            # Detect helmet
            helmet_label, helmet_confidence = predict_helmet(frame)

            # Detect characters on the number plate
            characters = detect_characters(frame)

            # Display the results
            if helmet_confidence < 0.5:
                status_text = f'Not wearing helmet ({helmet_confidence:.2%})'
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Capture and save the image for not wearing a helmet
                save_path = os.path.join(os.path.expanduser("~"), "Desktop", "not_wearing_helmet.jpg")
                cv2.imwrite(save_path, frame)
                print(f"Image captured and saved: {save_path}")

                # Add the image path to the list
                captured_images.append(save_path)

                # If characters are detected, capture and save the image
                if characters:
                    detected_text = ' '.join(characters)
                    character_image_path = os.path.join(os.path.expanduser("~"), "Desktop", "characters_not_wearing_helmet.jpg")
                    cv2.imwrite(character_image_path, frame)
                    print(f"Character image captured and saved: {character_image_path}")
                    # Add the image path to the list
                    captured_images.append(character_image_path)

            else:
                status_text = f'Wearing helmet ({helmet_confidence:.2%})'
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Automatically save the frame as PNG when wearing a helmet
                png_save_path = os.path.join(os.path.expanduser("~"), "Desktop", "wearing_helmet.png")
                cv2.imwrite(png_save_path, frame)
                print(f"Frame captured and saved as PNG: {png_save_path}")

            if characters:
                detected_text = ' '.join(characters)
                cv2.putText(frame, f'Characters: {detected_text}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, 'No head detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Combined Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Input the image of not wearing a helmet
    not_wearing_helmet_img_path = input("Enter the path to the image of not wearing a helmet: ")
    if os.path.exists(not_wearing_helmet_img_path):
        captured_images.append(not_wearing_helmet_img_path)
    else:
        print("Invalid path to not wearing helmet image.")

    # Write number plate recognition
    number_plate_text = input("Enter the recognized number plate text: ")
    if number_plate_text:
        captured_images.append(number_plate_text)
    else:
        print("No number plate text entered.")

    # Save the captured images into a PDF
    pdf_path = os.path.join(os.path.expanduser("~"), "Desktop", "Captured_Images_Report.pdf")
    save_images_to_pdf(captured_images, pdf_path)
    print(f"Report saved as: {pdf_path}")

if __name__ == '__main__':
    detect_helmet_and_characters_webcam()
