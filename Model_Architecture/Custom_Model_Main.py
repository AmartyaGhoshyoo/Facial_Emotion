from Common_Imports import *
from Custom_Model_Partial import conv_block_2D
from MobilenetV2_Checking import MobileNetV2
from Data_Preprocessing import X_balanced_3_channel,y_balanced
def model_training():
    def Custom_cnn(input_shape):
        inputs = Input(shape=input_shape, name="input_image")
        encoder = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs, alpha=1.4)
        E1 = encoder.get_layer("input_image").output
        E2 = encoder.get_layer("expanded_conv_depthwise_relu").output
        E3 = encoder.get_layer("block_1_expand_relu").output
        E4 = encoder.get_layer("block_2_expand_relu").output
        E5 = encoder.get_layer("block_4_expand_relu").output
        print(f'This is the shape of E1 {E1.shape}')
        print(f'This is the shape of E2 {E2.shape}')
        print(f'This is the shape of E3 {E3.shape}')
        print(f'This is the shape of E4 {E4.shape}')
        print(f'This is the shape of E5 {E5.shape}')



        E1=conv_block_2D(E1,256,'duckv2',repeat=1)
        E1=BatchNormalization(axis=-1)(E1)
        E_1=Conv2D(256,(3,3),padding='same',activation='relu',kernel_regularizer=l2(1e-3))(E1)
        E_1=BatchNormalization(axis=-1)(E_1)
        Down_E1 = MaxPooling2D((2, 2))(E_1)

        E1_E2_con = Concatenate()([Down_E1,E2])
        print(f'This is the shape of E1_E2_con {E1_E2_con.shape}')

        E2=Conv2D(256,(3,3),padding='same',activation='relu',kernel_regularizer=l2(1e-3))(E2)
        E2=BatchNormalization(axis=-1)(E2)
        # Down_E2 =MaxPooling2D((2, 2))(E2)

        E2_E3_con = Concatenate()([E2,E3])
        # E2_E3_con=BatchNormalization(axis=-1)(E2_E3_con)
        print(f'This is the shape of E2_E3_con {E2_E3_con.shape}')

        E3=Conv2D(256,(3,3),padding="same",activation='relu',kernel_regularizer=l2(1e-3))(E3)
        E3=BatchNormalization(axis=-1)(E3)
        Down_E3 = MaxPooling2D((2, 2))(E3)

        E3_E4_con = Concatenate()([Down_E3,E4])
        # E3_E4_con=BatchNormalization(axis=-1)(E3_E4_con)
        print(f'This is the shape of E3_E4_con {E3_E4_con.shape}')

        E4=Conv2D(256,(3,3),padding='same',activation='relu',kernel_regularizer=l2(1e-3))(E4)
        E4=BatchNormalization(axis=-1)(E4)
        Down_E4 = MaxPooling2D((2, 2))(E4)

        E4_E5_con = Concatenate()([Down_E4,E5])
        # E4_E5_con=BatchNormalization(axis=-1)(E4_E5_con)
        print(f'This is the shape of E4_E5_con {E4_E5_con.shape}')

        E5 = conv_block_2D(E5, 512, 'resnet', repeat=1)
        E5 = BatchNormalization(axis=-1)(E5)
        E1_E2_con=UpSampling2D(size=(2, 2))(E1_E2_con)
        E2_E3_con=UpSampling2D(size=(2, 2))(E2_E3_con)
        E3_E4_con=UpSampling2D(size=(4, 4))(E3_E4_con)
        E4_E5_con=UpSampling2D(size=(8, 8))(E4_E5_con)
        E5=UpSampling2D(size=(8, 8))(E5)
        E5=Concatenate()([E2_E3_con,E2_E3_con,E3_E4_con,E4_E5_con,E5])
        E5=BatchNormalization(axis=-1)(E5)
        print(f'This is the shape of E5 {E5.shape}')
        x = GlobalAveragePooling2D()(E5)
        x = BatchNormalization(axis=-1)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = BatchNormalization(axis=-1)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dense(8, activation='softmax')(x)
        model = Model(inputs, x , name = "Facial-Emotion")
        optimizer = Adam(learning_rate=10e-5)
        model.compile(optimizer=optimizer , loss='sparse_categorical_crossentropy', metrics= ['accuracy'])
        return model

    model = Custom_cnn((48,48,3))
    model.summary()


    from sklearn.utils import shuffle
    X_balanced_3_channel, y_balanced = shuffle(X_balanced_3_channel, y_balanced, random_state=42)

    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
    callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=1e-12, verbose=1),
            CSVLogger(csv_path),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), # Increased patience and added restore_best_weights
        ]


    results = model.fit(X_balanced_3_channel, y_balanced, validation_split=0.2, batch_size=32, epochs=200,callbacks=callbacks)
    return model


