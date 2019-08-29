import cv2
import openface
import dlib
import os
import face_recognition
import pandas as pd
import sys


def align_face(image_path):
    image = cv2.imread(image_path)
    try:
        detected_face = dlib.rectangles.pop(face_detector(image, 1))
        # detected_face = face_detector(image, 1)

        # align face
        aligned_face = face_aligner.align(534, image, detected_face,
                                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        cv2.imwrite('align_{}'.format(image_path), aligned_face)
    except:
        pass


def get_embedding(image_path):
    img = face_recognition.load_image_file(image_path)
    img_loc = [(0, len(img), len(img), 0)]
    img_enc = face_recognition.face_encodings(img, img_loc)[0]
    return img_enc


image_1_path = sys.argv[1]
image_2_path = sys.argv[2]

predictor_model = "lib/shape_predictor_68_face_landmarks.dat"

# create a HOG face detector using the built-it dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

# save align faces
align_face(image_1_path)
align_face(image_2_path)

# get embeddings
image_1_embed = get_embedding('align_{}'.format(image_1_path))
image_2_embed = get_embedding('align_{}'.format(image_2_path))

d = [image_1_embed - image_2_embed]

import pickle
from sklearn import svm
model = pickle.load(open('../face_comparison_svm.sav', 'rb'))
pred = model.predict(d)
print(pred[0])

