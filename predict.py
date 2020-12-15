"""
Author: @Jopapy19  https://github.com/Jopapy19/CNN-App
Date: 2020/12-11 Umeå

"""
import os
import utils.config as config
import tensorflow as tf
import utils.data_preprocessing as dp
import matplotlib.pyplot as plt
#from utils import model


class Predict:
    def __init__(self, latest=True, model_index=None):
        if latest:
            self.get_latest_model_path()
        elif (model_index is not None) and (not latest):
            self.model_index = model_index
            self.get_other_models()
        self.ny_modellen = tf.keras.models.load_model(self.latest_model_path)

    def get_latest_model_path(self):
        current_models = os.listdir(config.TRAINED_MODEL_DIR) 
        latest_model = sorted(current_models)[-1]
        self.latest_model_path = os.path.join(
            config.TRAINED_MODEL_DIR, latest_model)
    
    def get_other_models(self):
        current_models = os.listdir(config.TRAINED_MODEL_DIR)
        latest_model = sorted(current_models)[self.model_index]
        self.latest_model_path = os.path.join(
            config.TRAINED_MODEL_DIR, latest_model)

    def predict(self, input_img_path=None):
        img = plt.imread(input_img_path)
        fit_img = dp.hantera_input_data(img)
        result = self.ny_modellen.predict(fit_img)
        print("### RESULT:", result)


if __name__ == "__main__":
    obj = Predict()
    obj.predict(input_img_path='nedladdad-Volvo.jpg')
