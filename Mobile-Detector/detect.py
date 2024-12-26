import cv2
import torch
import pandas as pd
import os
import openpyxl
import smtplib
from email.mime.text import MIMEText
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor

# Load Faster R-CNN model for object detection
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to send an email
def send_email(email, name):
    sender_email = "k63887378@gmail.com"
    sender_password = "ANU@123_007"
    subject = "Mobile Phone Detected Alert"
    body = f"Dear {name},\n\nYour mobile phone usage was detected in a restricted zone.\n\nThank you."
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = email
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
            print(f"Email sent to {email}.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Load Excel file
excel_path = "Book1.xlsx"
data = pd.read_excel(excel_path)

# Ensure evidence folders exist
evidence_folder = "evidence"
face_folder = os.path.join(evidence_folder, "faces")
if not os.path.exists(evidence_folder):
    os.makedirs(evidence_folder)
if not os.path.exists(face_folder):
    os.makedirs(face_folder)

# Video capture
cap = cv2.VideoCapture(0)
frame_count = 0  # To give unique names to saved images

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    frame_tensor = ToTensor()(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(frame_tensor)

    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if label == 77 and score > 0.7:  # Mobile phone class
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            phone_image = frame[y1:y2, x1:x2]

            # Save the mobile phone image in the evidence folder
            phone_filename = os.path.join(evidence_folder, f"phone_{frame_count}.jpg")
            cv2.imwrite(phone_filename, phone_image)
            print(f"Saved phone image: {phone_filename}")

            # Detect face in the frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (fx, fy, fw, fh) in faces:
                face_image = frame[fy:fy+fh, fx:fx+fw]
                face_filename = os.path.join(face_folder, f"face_{frame_count}.jpg")
                cv2.imwrite(face_filename, face_image)
                print(f"Saved face image: {face_filename}")

                # Compare with Excel data (mock example)
                for _, row in data.iterrows():
                    # Add actual face matching logic here (placeholder condition for now)
                    if "match_condition":
                        send_email(row['Email'], row['Name'])
                        break

            frame_count += 1

    cv2.imshow("Mobile and Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
