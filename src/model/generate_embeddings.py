import os
import face_recognition
import pandas as pd

PROJECT_PATH = os.path.abspath('../..')
os.chdir(PROJECT_PATH)

faces = os.listdir(PROJECT_PATH + '/align_face_dataset')

for face in faces:
    images = os.listdir('align_face_dataset/{}'.format(face))
    df = pd.DataFrame()
    for i in images:
        img_path = "align_face_dataset/{}/{}".format(face, i)
        img = face_recognition.load_image_file(img_path)

        img_loc = [(0, len(img), len(img), 0)]

        img_enc = face_recognition.face_encodings(img, img_loc)[0]

        df[i] = img_enc
    df.to_csv('embedded_face_dataset/{}.csv'.format(face), index=False)