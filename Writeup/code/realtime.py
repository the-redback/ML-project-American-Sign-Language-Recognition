import joblib
import numpy as np
import cv2
import tensorflow as tf
import argparse
import time
from tensorflow.keras.models import load_model

model = load_model('model-weight-v2.h5')
label_lines = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}



def predict(image_data):
    image_data = cv2.resize(image_data, (224, 224))
    image_data = tf.keras.applications.mobilenet.preprocess_input(image_data)
    image_data = image_data.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.

    predictions = model.predict(image_data)
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        # Just to get rid of the Z error for demo
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string

    return res, max_score

print('Model loaded')


def hand_area(img):
    hand = img[100:512, 100:512]
    hand = cv2.resize(hand, (224, 224))
    return hand


cap = cv2.VideoCapture(0)

time_counter = 0

captureFlag = False

realTime = True

spell_check = False

if (cap.isOpened() == False):
    print('Error opening camera')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while (cap.isOpened()):
    ret, frame = cap.read()

    cv2.rectangle(frame, (100, 100), (512, 512), (20, 34, 255), 2)
    hand = hand_area(frame)
    # hand = cv2.flip(hand, 1)

    image = hand


    outputs,score = predict(image)
    if score < .5:
        continue
    print(outputs,score)

    cv2.putText(frame, outputs, (90, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('image', frame)

    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
