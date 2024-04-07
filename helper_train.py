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
        minNeighbors=15,
        minSize=(64, 64),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        resized_img = cv2.resize(img, (64,64))
        return resized_img

    for (x, y, w, h) in faces[:1]:
        face_img = img[y:y+h, x:x+w]
        resized_face_img = cv2.resize(face_img, (64, 64))  
        return resized_face_img

df = pd.read_csv('train.csv')
training_images = 'celebrities_cropped'
images = 'train'

if not os.path.exists(training_images):
    os.makedirs(training_images)

for index, row in df.iterrows():
    img_path = os.path.join(images, row['File Name'])
    celebrity_name = row['Category']
    target = os.path.join(training_images, celebrity_name)

    if not os.path.exists(target):
        os.makedirs(target)

    target_path = os.path.join(target, row['File Name'])
    processed_img = extract_features(img_path)
    if processed_img is not None:
        cv2.imwrite(target_path, processed_img)
        print(f"Processed and saved {img_path} to {target_path}")
    else:
        print(f"No face detected in {img_path}, skipped.")
