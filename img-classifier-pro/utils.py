import os

import cv2
import numpy as np
import pywt

face_cascade = cv2.CascadeClassifier('./opencv/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascade_eye.xml')

def get_faces_with_detected_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

def normalize_name(name):
    return name.lower().replace("_"," ")

def begin_cropping(img_dirs, path_to_cropped_images):
    cropped_img_dir = []
    celebrity_filename_list = {}

    for img_dir in img_dirs:
        count = 1
        celebrity_name = img_dir.split('/')[-1]
        celebrity_name = normalize_name(celebrity_name)
        celebrity_filename_list[celebrity_name] = []

        for entry in os.scandir(img_dir):
                roi_color = get_faces_with_detected_eyes(entry.path)
                if roi_color is not None:
                    cropped_folder = path_to_cropped_images + celebrity_name
                    if not os.path.exists(cropped_folder):
                        os.makedirs(cropped_folder)
                        cropped_img_dir.append(cropped_folder)
                        print('Generating cropped images in folder ', cropped_folder)
                    cropped_file_name = celebrity_name+str(count)+'.png'
                    cropped_file_path = cropped_folder+"/"+cropped_file_name

                    cv2.imwrite(cropped_file_path, roi_color)
                    celebrity_filename_list[celebrity_name].append(cropped_file_path)
                    count += 1

    return celebrity_filename_list

def create_filename_list(img_dirs):
    celebrity_filename_list = {}

    for img_dir in img_dirs:
        celebrity_name = img_dir.split('/')[-1]
        celebrity_name = normalize_name(celebrity_name)
        celebrity_filename_list.setdefault(celebrity_name, [])

        for entry in os.scandir(img_dir):
            if entry.is_file():
                celebrity_filename_list[celebrity_name].append(entry.path)

    return celebrity_filename_list

def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype Conversion
    # Convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H