import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf

####################################################
# Tensorflow uyarı hatalarının gizlenmesi..
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print("Veri kümesi okunuyor \n")
path = "ddos.csv"
veri = pd.read_csv(path, low_memory=False)
veri = pd.DataFrame(veri)

####################################################
# Önişleme
veri.replace('Infinity', -1, inplace=True)

# Bu değişkenlere gerek yok..
veri = veri.drop(' Destination Port', 1)
veri = veri.drop(' Label', 1)

# Değişken isimlerindeki boşlukların kaldırılması..
veri = veri.rename(columns={c: c.replace(' ', '') for c in veri.columns})

# Normalleştirme işlemi..
# Herbir özniteliğin 0-1 arasında ölçeklendirilmesi..
veri = minmax_scale(veri, axis = 1)
ncol = veri.shape[1]

# Veri kümesindeki NaN satırların silinmesi..
veri = pd.DataFrame(veri)
nanVar = veri.isna().sum().sum()
print(nanVar)

# NaN satırların silinmesi..
veri = veri.dropna(axis='rows')

# Otomatik Kodlama Mimarisi..
# Kodlama boyutu..
encoding_dim = 10   

####################################################
# Kodlayıcı (encoder) katmanının tanımlanması
input_dim = Input(shape = (ncol, ))
encoded = Dense(ncol, activation = 'relu')(input_dim)
encoded = Dense(35, activation = 'relu')(encoded)
encoded = Dense(20, activation = 'relu')(encoded)
encoded = Dense(12, activation = 'relu')(encoded)
encoded = Dense(6, activation = 'relu')(encoded)

encoded =  Dense(encoding_dim, activation = 'relu')(encoded)

# Kod çözücü (decoder) katmanının tanımlanması
decoded = Dense(6, activation = 'relu')(encoded)
decoded = Dense(12, activation = 'relu')(decoded)
decoded = Dense(20, activation = 'relu')(decoded)
decoded = Dense(35, activation = 'relu')(decoded)
decoded =  Dense(ncol, activation = 'sigmoid')(decoded)

autoencoder = Model(inputs = input_dim, outputs = decoded)
autoencoder.summary()

####################################################
# Modelin eğitilmesi..
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

###########################################################
# Model hatası
plt.plot(history['loss'], linewidth=2, label='Eğitim hatası')
plt.plot(history['val_loss'], linewidth=2, label='Doğrulama Hatası')
plt.legend(loc='upper right')
plt.title('Model hataları')
plt.ylabel('Hata')
plt.xlabel('Devir')
plt.show()

###########################################################
# Yeniden yapılandırma için X_test verisi kullanılıyor..
öngörüler = autoencoder.predict(veri)
mse = np.mean(np.power(veri - öngörüler, 2), axis=1)

############################################################
# Yeniden yapılandırma grafiği
eşik = 0.06
plt.plot(mse, linewidth=0, label='Eğitim', marker='.')
plt.title('Test verisi için yeniden yapılandırma hataları')
plt.ylabel('mse')
plt.xlabel('Gözlemler')
plt.axhline(y=eşik, color="r")
plt.show()

############################################################
# Anormal kayıtların belirlenmesi ..
anormal = pd.DataFrame(öngörüler[mse > eşik ])
print("Anormal değerlerin sayısı",anormal.shape)
print("Anormal değerler: \n", anormal)
