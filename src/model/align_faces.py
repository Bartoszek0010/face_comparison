import cv2
import openface
import dlib
import os


PROJECT_PATH = os.path.abspath('../..')


def save_img(directory, file_name, img, source):
    # create firectory and enter it
    os.chdir(PROJECT_PATH + '/align_face_dataset')
    try:
        os.mkdir(directory)
    except:
        pass
    os.chdir(directory)
    # save aligned image
    cv2.imwrite(file_name, img)

    # back to the starting path
    os.chdir(source)


faces = os.listdir(PROJECT_PATH + '/face_dataset')
predictor_model = "../lib/shape_predictor_68_face_landmarks.dat"

# create a HOG face detector using the built-it dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

for face in faces:
    # exception when this will be a secret folder etc.
    try:
        os.chdir(PROJECT_PATH + '/face_dataset/' + face)
    except:
        continue

    images_name = os.listdir()
    for i in images_name:
        image = cv2.imread(i)
        try:
            detected_face = dlib.rectangles.pop(face_detector(image, 1))

            # get face's landmarks
            pose_landmarks = face_pose_predictor(image, detected_face)

            # align face
            aligned_face = face_aligner.align(534, image, detected_face, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            # save aligned face to align_face_dataset
            save_img(face, i, aligned_face, os.getcwd())

        except:
            continue
    os.chdir('..')

