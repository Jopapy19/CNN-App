import os
import tensorflow as tf
import numpy as np
import utils.config as config
from sklearn.preprocessing import LabelEncoder

def train_valid_generator(
    IMAGE_SIZE = config.IMAGE_SIZE[:-1],
    BATCH_SIZE = config.BATCH_SIZE,
    data_dir = config.DATA_DIR,
    data_augmentation = config.AUGMENTATION):

    datagen_kwargs = dict(
        rescale=1./255,
        validation_split=0.20
        )
    
    dataflow_kwargs = dict(
        target_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        interpolation = "bilinear"
        )

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

    valid_generator = valid_datagen.flow_from_directory(
        directory=data_dir,
        subset="validation", 
        shuffle=False,
        **dataflow_kwargs
    )


    if data_augmentation:
        train_datagen= tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2,
            heigth_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            **datagen_kwargs
        )
    else:
        train_datagen = valid_datagen

    #Träna data och  valideringdata
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        subset="training",
        shuffle=True,
        **dataflow_kwargs
    )

    return  train_generator, valid_generator
    

def hantera_input_data(input_image):
     """[Konverterar ingången till förväntad dimension]

     Args:
         input_data ([ndArray]): [bild ndArray]  
    """
    
     images = input_image
     size = config.IMAGE_SIZE[:-1]
     resized_input_img = tf.image.resize(
         images,
         size,
         preserve_aspect_ratio=False)

     """[Bild av dataförberedelse]

     Returns:
         [ndArray]: [Ändra storlek och uppdaterad dimensionerad bild]
     """
    # bild av dataförberedelse
    final_img = np.expand_dims(resized_input_img, axis=0)
    return final_img

#print(help(hantera_input_data))  # print all defined functions summary


