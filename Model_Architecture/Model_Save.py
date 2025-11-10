from Custom_Model_Main import model_training
from tensorflow.keras.models import model_from_json
from Common_Imports import *
def model_saving():
    model=model_training()
    model_json = model.to_json()
    with open("Facial_Emotion/Model_Parameters/model_part2.json", "w") as json_file:
        json_file.write(model_json)
        

    model = model_from_json(open("Facial_Emotion/Model_Parameters/model_part2.json", "r").read())
    model.load_weights("Facial_Emotion/model_second_try.weights.h5")
    return model