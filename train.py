"""
Author: @Jopapy19  https://github.com/Jopapy19/CNN-App
Date: 2020/12-11 Umeå

"""
import os
import utils.config as config
import tensorflow as tf
import utils.data_preprocessing as dp
import time
from utils import model

#from PIL import Image
#from sklearn.preprocessing import LabelEncoder


def get_unique_model_name(specific_name="VGG19_modell"):
    model_fileName = time.strftime(f"{specific_name}_at_%Y%m%d_%H%M%S.h5")
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    model_file_path = os.path.join(config.TRAINED_MODEL_DIR, model_fileName)
    return model_file_path

def train():
    ny_modellen = model.custom_model()
    callbacks = model.callbacks()
    train_generator, valid_generator = dp.train_valid_generator()

    """
       Steg för epochs
       train_generator.samples = 256
       batch_size = 16

       Fit och spara modellen

    """
    steps_per_epoch = train_generator.samples // train_generator.batch_size   
    validation_steps =  valid_generator.samples // valid_generator.batch_size

    ny_modellen.fit(
        train_generator, 
        validation_data = valid_generator,
        epochs=config.EPOCHS, 
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps, 
        callbacks=callbacks)

    #Spara modellen
    model_file_path = get_unique_model_name()
    ny_modellen.save(model_file_path)
    print(f"sparas på följande läge\n ==> {model_file_path}")

if __name__ == "__main__":
    train()






