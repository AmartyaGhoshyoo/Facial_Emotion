from Common_Imports import *

input_shape = (48, 48, 3)
inputs_for_encoder = Input(shape=input_shape, name="input_image_for_encoder")

encoder_temp = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs_for_encoder, alpha=1.4)

encoder_model = Model(inputs=inputs_for_encoder, outputs=encoder_temp.output)
encoder_model.summary()

if 'encoder_temp' in locals():
    print("Full layer names in encoder_temp:")
    for layer in encoder_temp.layers:
        print(layer.name)