## Bu program öncesinde 204-web trafiği-toplamlar.py
## programı çalıştırılarak günlük toplam trafik
## veri kümesinin (günlük_trafik.txt) elde edilmesi gerekir. 
## Zaman Serileri için Keras LSTM modeli
## Bu program eğitim setini belirler, modeli oluşturur
## Bu modele göre bir sonraki gün için tahmin yapar..
## Gerçek veri ile karşılaştırılır..

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf

#####################################################################
# Tensorflow uyarı mesajlarının gizlenmesi..
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#####################################################################
# Tekrarlanabilirliğin sağlanması...
from tensorflow import set_random_seed
set_random_seed(0)
np.random.seed(0)
tf.set_random_seed(0)

# tek kullanıcılı olmaya zorlayan konfigürasyon tanımlarını yapar..
from keras import backend as ker
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

# yeni bir tensorflow oturumunun açılması..
sess = tf.Session(
    graph=tf.get_default_graph(),
    config=session_conf)
ker.set_session(sess)

########################################################################
# Verinin okunması...
okunan = pd.read_csv('günlük_trafik.txt',engine='python')

# Sütün başlıkları oluşturuluyor... 
okunan.columns=["Date","Visits"]
okunan = pd.DataFrame(okunan)

#####################################################################
# Parametreler
# Değişken sayısı ..
değişken = 4

# Gün tahmin sayısı
gün = 1

# epoch
devir = 100

########################################################################
# Veri kümesinin tarih alanına göre sıralanması
okunan = okunan.sort_values(by='Date')
print("Eğitim verisi: \n", okunan.head(15))

# Okunan veri kümesinin "Visits" sütununa göre grafiği
okunan.set_index('Date',inplace=True)
okunan = okunan.iloc[0:okunan.shape[0],:]

########################################################################
# max-min normalizasyonu ile ölçeklendirme..
veri = okunan.values
sc = MinMaxScaler(feature_range = (0, 1))
veri = sc.fit_transform(veri)

########################################################################
# LSTM için otoregresif X_train ve y_train matrisleri oluşturuluyor..
X_train = []
y_train = []
satır = veri.shape[0]

for i in range(değişken, satır):
    X_train.append(veri[i-değişken:i, 0]) 
    y_train.append(veri[i, 0])      

X_train = np.array(X_train)
y_train = np.array(y_train)

# Verinin 3 boyutlu hale getirilmesi..
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Tahmin gün sayısına uygun boş diziler hazırlanıyor..
denormal = np.zeros((gün),dtype=float)

########################################################################
# LSTM Model mimarisi
import keras
model = Sequential()
yy=512
model.add(LSTM(
    yy,
    return_sequences = True,
    input_shape = (X_train.shape[1],
    1)))
model.add(LSTM(
    yy,
    return_sequences = True))
model.add(LSTM(
    yy,
    return_sequences = False))
model.add(Dense(
    1,
    activation = 'relu')
)
keras.optimizers.Adam(lr=0.001)
model.compile(optimizer = 'adam', loss = 'mse')

########################################################################
# Modelin eğitilmesi
başlangıç = X_train.shape[0]-1 
sonKayıt = X_train[başlangıç:başlangıç+1,0:değişken]

history = model.fit(X_train,
    y_train,
    batch_size = 100,
    epochs = devir, 
    verbose = 2,
    validation_split = 0.2
    )

# Hata grafiği
plt.plot(history.history["loss"])
plt.title("Model hatası")
plt.ylabel('Hata')
plt.xlabel('Günler')
plt.show()

# Doğrulama hata grafiği
plt.plot(history.history["val_loss"])
plt.title("Doğrulama hatası")
plt.ylabel('Hata')
plt.xlabel('Günler')
plt.show()

##################################################################
# Tahmin değeri
tahmin=model.predict(sonKayıt)              
denormal=sc.inverse_transform(tahmin)
print("Tahmin değeri: %.7f" % denormal)

################
gerçek_veri=okunan.iloc[okunan.shape[0]-1,:]
print("Gerçek veri: ", gerçek_veri[0])
print("Eğitim kayıt sayısı:", X_train.shape[0])

##################################################################


