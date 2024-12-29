# -*- coding: utf-8 -*-


# from keras import models, layers
from keras.datasets import boston_housing
# from keras.utils.np_utils import to_categorical
# from keras.utils import to_categorical


(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
print(f"train_data.shape={train_data.shape}")
print(f"len(train_data)={len(train_data)}")
print(f"train_labels={train_labels}")

print(f"test_data.shape={test_data.shape}")
print(f"len(test_labels)={len(test_labels)}")
print(f"test_labels={test_labels}")
