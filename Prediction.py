from tensorflow.keras.models import model_from_json
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Multiply, Concatenate, UpSampling2D, Conv2D,Conv2DTranspose,Add,GlobalAveragePooling2D,Dense,BatchNormalization,add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
model = model_from_json(open("model_part2.json", "r").read())
model.load_weights("model_second_try.weights.h5")
