# Değişkenlerin sayısını azaltma..

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf

#####################################################################
# Tensorflow uyarı hatalarının gizlenmesi..
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#####################################################################
# Veri kümesinin okunması
print("Veri kümesi okunuyor \n")
path = "ddos.csv"
veri = pd.read_csv(path, low_memory=False)
veri = pd.DataFrame(veri)
veri.isna().sum().sum()

#####################################################################
# Önişleme
veri.replace('Infinity', -1, inplace=True)

# Bu değişkenlere gerek yok..
veri = veri.drop(' Destination Port', 1)
veri = veri.drop(' Label', 1)

# Değişken isimlerindeki boşlukların kaldırılması..
veri = veri.rename(columns={c: c.replace(' ', '') for c in veri.columns})

# Herbir özniteliğin 0-1 arasında ölçeklendirilmesi...
veri = minmax_scale(veri, axis = 1)
ncol = veri.shape[1]

# Veri kümesindeki NaN satırların silinmesi..
veri = pd.DataFrame(veri)
veri.isna().sum().sum()
veri = veri.dropna(axis='rows')

###########################################################
# Otomatik Kodlama Mimarisi..
# Kodlama boyutu..
encoding_dim = 3   

# Kodlayıcı-Kood çözücü katmanlar..
input_dim = Input(shape = (ncol, ))
encoded = Dense(ncol, activation = 'relu')(input_dim)
encoded = Dense(20, activation = 'relu')(encoded)
encoded = Dense(encoding_dim, activation = 'relu')(encoded)
decoded = Dense(20, activation = 'relu')(encoded)
decoded = Dense(ncol, activation = 'sigmoid')(decoded)

autoencoder = Model(inputs = input_dim, outputs = decoded)
autoencoder.summary()
###########################################################
# Otomatik kodlayıcının eğitilmesi
autoencoder.compile(
    metrics=['accuracy'],
    loss='mse',
    optimizer='adam')

history = autoencoder.fit(
    veri,
    veri,
    epochs = 10,
    batch_size = 100,
    verbose=2,
    validation_split = 0.2
).history

# Yukarıda elde edilen autoencoderden azaltılmış boyutları veren encoder
encoder = Model(inputs = input_dim, outputs = encoded)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(veri)
print("Yeni veri kümesi:  \n")
print(encoded_out[0:10])

