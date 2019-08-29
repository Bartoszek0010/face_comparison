# face_comparison
<h2>ML project -> comparison of two faces (images) - Face Verification</h2>
-----------------------------------------------------------------------------

Target of this project is to create Face Verification with input of two face images. Machine Learning predictive model is 
a Support Vector Machine model. 

<h3>Training dataset (face dataset)</h3>
A training dataset of images of 100 perdon is the kaggle dataset:
https://www.kaggle.com/frules11/pins-face-recognition

<h3>Openface library</h3>
Openface is library to detect faces at the image, create locatization of faces and get landmarks.
Openface library : https://cmusatyalab.github.io/openface/

-----------------------------------------------------------------------------

<h2> Files Description</h2>

<h3>model/align_faces.py</h3>
Script which create a second faces dataset where all of images from face_dataset are aligned.<br>
Aligned face:<br>
Using face landmark estimation invented in 2014 by Vahid Kazemi and Josephine Sullivan,
warp each picture so that the eyes and lips are always in the sample place in the image.
This algorithm spot 68 points on the each face and pose and center this landmark.


<h3>model/generate_embedings.py</h3>
Script which create a dataset of .csv files of each person with dataframe of all embedded images (col = image, values = embedded image)<br>
Embeddings:
Respresent image as a vector of 128 numbers


<h3>model/svm_model.ipynb</h3>
Jupyter notebook script to prepare training and testing dataset and create SVM classifier.<br>
Model Accuracy : <b> 95,6% </b><br>
Script also save classifier to the .sav file (using pickle)


<h3>compare_faces.py</h3>
Script which use svm_model to predict input images (the same person - 1; different people - 0).<br>
Process of preprocessing is : <b>images -> align images -> images embedding -> model_prediction</b>
