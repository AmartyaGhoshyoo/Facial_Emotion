
import math
import numpy as np
import pandas as pd
df=pd.read_csv("CK+Dataset(Very Poor)/ckextended.csv")
print(df)
print(type(df.pixels[0]))
print(df.isnull().sum())
print(df.emotion.value_counts())
print(df.Usage.value_counts())

X_train = []
y_train = []
X_test = []
y_test = []
for index, row in df.iterrows():
    k = row['pixels'].split(" ")
    if row['Usage'] == 'Training':
        X_train.append(np.array(k))
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        X_test.append(np.array(k))
        y_test.append(row['emotion'])
        
        

print(X_train[0].shape)
print(math.sqrt(2304))

X_train = np.array(X_train, dtype = 'uint8')
y_train = np.array(y_train, dtype = 'uint8')
X_test = np.array(X_test, dtype = 'uint8')
y_test = np.array(y_test, dtype = 'uint8')


print(X_train.shape)
print(y_train.shape)
classes = np.unique(y_train)
print(classes)
print(y_train.dtype)
print(type(classes[0]))


X_augment=[]
Y_augment=[]
from collections import Counter
counts=Counter(y_train)
max_count = max(counts.values())
print(counts, "max:", max_count)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range = 10,
    horizontal_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest')


for c in classes:
    X_class = X_train[y_train == c]
    n_to_generate = max_count - len(X_class)

    if n_to_generate <= 0:
        X_augment.append(X_class)
        Y_augment.append(np.full(len(X_class), c))
        continue

    gen = datagen.flow(X_class, np.full(len(X_class), c), batch_size=1)

    generated_images = []
    generated_labels = []

    for _ in range(n_to_generate):
        x_batch, y_batch = next(gen)
        generated_images.append(x_batch[0])
        generated_labels.append(y_batch[0])
    X_class_all = np.concatenate([X_class, np.array(generated_images)], axis=0)
    y_class_all = np.concatenate([np.full(len(X_class), c), np.array(generated_labels)], axis=0)

    X_augment.append(X_class_all)
    Y_augment.append(y_class_all)


X_balanced = np.concatenate(X_augment, axis=0)
y_balanced = np.concatenate(Y_augment, axis=0)

print("Before:", X_train.shape, y_train.shape)
print("After:", X_balanced.shape, y_balanced.shape)



X_test_augment = []
y_test_augment = []
from collections import Counter
test_counts = Counter(y_test)
max_count_test = max(test_counts.values())
print(test_counts, "max:", max_count_test)


for c in classes:
    X_class_test = X_test[y_test == c]
    n_to_generate_test = max_count_test - len(X_class_test) 

    if n_to_generate_test <= 0:
        X_test_augment.append(X_class_test)
        y_test_augment.append(np.full(len(X_class_test), c))
        continue

    gen_test = datagen.flow(X_class_test, np.full(len(X_class_test), c), batch_size=1)

    generated_images_test = []
    generated_labels_test = []

    for _ in range(n_to_generate_test):
        x_batch_test, y_batch_test = next(gen_test)
        generated_images_test.append(x_batch_test[0])
        generated_labels_test.append(y_batch_test[0])

    X_class_all_test = np.concatenate([X_class_test, np.array(generated_images_test)], axis=0)
    y_class_all_test = np.concatenate([np.full(len(X_class_test), c), np.array(generated_labels_test)], axis=0)

    X_test_augment.append(X_class_all_test)
    y_test_augment.append(y_class_all_test)

X_test_balanced = np.concatenate(X_test_augment, axis=0)
y_test_balanced = np.concatenate(y_test_augment, axis=0)

print("Before test augmentation:", X_test.shape, y_test.shape)
print("After test augmentation:", X_test_balanced.shape, y_test_balanced.shape)




X_balanced = np.concatenate(X_augment, axis=0)
y_balanced = np.concatenate(Y_augment, axis=0)

print("Before:", X_train.shape, y_train.shape)
print("After:", X_balanced.shape, y_balanced.shape)


X_balanced_3_channel = np.repeat(X_balanced, 3, axis=-1)
X_test_3_channel = np.repeat(X_test_balanced, 3, axis=-1)
print(f"Original X_balanced shape: {X_balanced.shape}")
print(f"New X_balanced_3_channel shape: {X_balanced_3_channel.shape}")
print(f"Original X_test shape: {X_test.shape}")
print(f"New X_test_3_channel shape: {X_test_3_channel.shape}")



r=pd.DataFrame(y_balanced)

print(r.value_counts())