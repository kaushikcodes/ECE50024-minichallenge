import pandas as pd
import os
import shutil
import cv2

def extract_features(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None  # Early return if image is not loaded

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(64, 64),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        resized_img = cv2.resize(img, (64, 64))
        return resized_img

    for (x, y, w, h) in faces[:1]:
        face_img = img[y:y+h, x:x+w]
        resized_face_img = cv2.resize(face_img, (64, 64))  
        return resized_face_img

testing_images = 'test'
target = 'test_cropped'

if not os.path.exists(target):
    os.makedirs(target)

for image in os.listdir(testing_images):
    img_path = os.path.join(testing_images, image)
    
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue
    
    processed_image = extract_features(img_path)
    if processed_image is not None:
        target_path = os.path.join(target, image)
        cv2.imwrite(target_path, processed_image)
        print(f"Processed and saved {image} to {target}")
    else:
        print(f"No face detected in {image}, skipped.")

