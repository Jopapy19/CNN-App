import os
import tensorflow as tf
import numpy as np
import config

def save_vgg_19_model(input_shape=config.IMAGE_SIZE): # Main Transfer Learning
    model = tf.keras.applications.vgg19.VGG19(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False

    )
    model.save("original_vgg19__base_model.h5")
    print("Grundmodellen sparad")

def load_base_model():
    model = tf.keras.models.load_model("original_vgg19__base_model.h5")
    print("ursprunglig basmodell laddad ")
    model.summary()
    return model


def custom_model(CLASSES=config.CLASSES, freeze_all=True, freeze_till=None):
    model = load_base_model() #Ladda upp modellen
    
    #Frysa vikter
    if freeze_all:
        for layer in model.layers:
            layer.trainable=False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in model.layers[:freeze_till]:
            layer.trainable = False
    
    #Lägg till anpassat lager.
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
        units=CLASSES,
        activation="softmax"
    )(flatten_in)

    ny_modellen = tf.keras.models.Model(
        inputs=model.input,
        outputs=prediction
    )
    print("Anpassad modellöversikt")
    ny_modellen.summary()

    ny_modellen.compile(
        optimizer = tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"]

    )
    return ny_modellen
    
if __name__ == '__main__':
    #save_vgg_19_model()   # Spara Base h5 filen
    #load_base_model()
    custom_model()    # Dense är vår sist prediktion lagret. 




