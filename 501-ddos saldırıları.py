# Bu program DDos ataklarını ve normal ağ trafiğini içeren bir veri
# kümesini kullanarak sınıflandırma modeli geliştir ve
# Öngörüde bulunur..
# Veri Kümesi  :  https://www.unb.ca/cic/datasets/ids-2017.html

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
from numpy.random import seed
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils

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

#####################################################################
# Veri kümesinin okunması
print("Veri kümesi okunuyor \n")
df = pd.read_csv("ddos.csv",low_memory=False)

#####################################################################
# Önişleme
df.replace('Infinity', -1, inplace=True)

# Bu değişkene gerek yok..
df = df.drop(' Destination Port', 1)

# Değişken isimlerindeki boşlukların kaldırılması..
df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})

# Veri kümesindeki NaN satırların silinmesi..
df = pd.DataFrame(df)
df.isna().sum().sum()
df = df.dropna(axis='rows')

# Etiketlere sayısal değerlerin atanması
y = df["Label"].map({"BENIGN": 0, "DDoS": 1})

# Sınıf özniteliklerinin silinmesi
df = df.drop('Label', 1)

###########################################################
# Veri kümesinin eğitim ve test için bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    df,
    y,
    test_size = 0.2       
)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

# Normalleştirme işlemi..
# Herbir özniteliğin 0-1 arasında ölçeklendirilmesi...
ncol = X_train.shape[1]
X_col = X_train.columns

mms = MinMaxScaler()
X_train=mms.fit_transform(X_train.astype(np.float))
X_test=mms.fit_transform(X_test.astype(np.float))

X_train = pd.DataFrame(X_train, columns = X_col)
X_test = pd.DataFrame(X_test, columns = X_col)

# Sınıf özniteliği kategorik hale getiriliyor..
from sklearn.preprocessing import LabelEncoder
dönüştürüldü = LabelEncoder()
y_train = np.ravel(y_train)   # 1 boyutlu bir dizi döndürür.
y_train = dönüştürüldü.fit_transform(y_train)

y_test = np.ravel(y_test)   # 1 boyutlu bir dizi döndürür.
y_test = dönüştürüldü.fit_transform(y_test)

# Kategorik biçime dönüştürülmesi
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Sınıflar dengeli mi ?
print("Sınıfların dağılımı: \n", pd.crosstab(y_train[:,0], columns='count'))

##########################################################
# Model mimarisi..
print ("Derin öğrenme modeli çalışıyor.. \n")
model = Sequential()
model.add(Dense(
    50,
    input_dim=X_train.shape[1],      ## Öznitelik sayısı kadar..
    activation='relu'))

model.add(Dense(
    50,
    activation='relu'))

model.add(Dense(
    50,
    activation='relu'))

model.add(Dense(
    2,                              ### Sınıf sayısı kadar olmalıdır..
    activation='softmax'))

# Modelin derlenmesi
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# Modelin eğitilmesi
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    verbose=2,
    batch_size=16,
    validation_split=0.2
)

#####################################################################
# Eğitilen modelin performans grafiği..
plt.plot(history.history['accuracy']) #'acc'
plt.plot(history.history['val_accuracy']) #'val_acc
plt.title('Model Performansı')
plt.ylabel('Doğruluk')
plt.xlabel('Devir sayısı (epoch)')
plt.legend(['Eğitim', 'Test'], loc='upper left')
plt.show()

# Eğitilen modelin hata grafiği..
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Hatası')
plt.ylabel('Hata')
plt.xlabel('Devir sayısı (epoch)')
plt.legend(['Eğitim', 'Test'], loc='upper left')
plt.show()

#####################################################################
# Test verisi ile performans ölçümü
performans = model.evaluate(X_test, y_test, verbose=2)
print("\nDoğruluk: ",(performans[1]))


