import pandas as pd
import numpy as np
from tensorflow import set_random_seed
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf

#######################################################
# Tensorflow uyarı mesajlarının gizlenmesi..
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#######################################################
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


##################################################################
## Veri kümesinin okutulması ve önişlemler...
df = pd.read_csv("merged_5s.csv")

##################################################################
# önişleme
df.replace('Infinity', -1, inplace=True)

# Bu değişkenlere gerek yok..
df = df.drop('Source IP', 1)
df = df.drop(' Source Port', 1)
df = df.drop(' Destination IP', 1)
df = df.drop(' Destination Port', 1)
df = df.drop(' Protocol', 1)

# Eğer na kayıt varsa onun silinmesi..
df.dropna(inplace=True)

# Değişken isimlerindeki boşlukların kaldırılması..
df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})

# Etiketlere sayısal değerlerin atanması
y = df["label"].map({"nonTOR": 0, "TOR": 1})

###########################################################
# Veri kümesinin eğitim ve test için bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    df,
    y,
    test_size = 0.2,        
    random_state = seed(123)
)

# Sınıf değişkenlerinin veri kümesinden çıkarılması.. 
X_train = X_train.drop(['label'], axis=1)   #drop the class column
X_test = X_test.drop(["label"], axis=1)     #drop the class column
print(X_train.shape)

# Normalleştirme işlemi..
# Herbir özniteliğin 0-1 arasında ölçeklendirilmesi...
X_train_col = X_train.columns
X_train = minmax_scale(X_train, axis = 1)
X_test = minmax_scale(X_test, axis = 1)
ncol = X_train.shape[1]

X_train = pd.DataFrame(X_train, columns = X_train_col)
X_test = pd.DataFrame(X_test, columns = X_train_col)

# Kategorik niteliğin sayısal değere dönüştürmek
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
dönüştürüldü = LabelEncoder()
y_train = dönüştürüldü.fit_transform(y_train)
y_test = dönüştürüldü.fit_transform(y_test)

# Sınıflar dengeli mi ?
print("Sınıfların dağılımı: \n", pd.crosstab(y_train, columns='count'))

# Dengeleme işlemi
from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')
X_train, y_train = smote.fit_sample(X_train, y_train)
print("Dengeleme sonunda dağılım: \n", pd.crosstab(y_train, columns='count'))

# Dengeleme sonucunda kategorik biçime dönüştürülmesi
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

##########################################################
# Model mimarisi..
model = Sequential()
model.add(Dense(
    50,
    input_dim=23,           ## Öznitelik sayısı kadar..
    activation='relu'))

model.add(Dense(
    50,
    activation='relu'))

model.add(Dense(
    50,
    activation='relu'))

model.add(Dense(
    2,                      ### Sınıf sayısı kadar..
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
    epochs=100,
    verbose=2,
    batch_size=16,
    validation_split=0.2
)

#####################################################################
# Eğitim ile ilgili modelin performans grafiği..
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Performansı')
plt.ylabel('Doğruluk')
plt.xlabel('Devir sayısı')
plt.legend(['Eğitim', 'Test'], loc='upper left')
plt.show()

# Model hatası ile ilgili hata..
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Hatası')
plt.ylabel('Hata')
plt.xlabel('Devir sayısı')
plt.legend(['Eğitim', 'Test'], loc='upper left')
plt.show()

#####################################################################
# Test verisi ile performans ölçümü
print("Test verisi :", X_test.shape)
scores = model.evaluate(X_test, y_test, verbose=2)
print("\nDoğruluk: ",(scores[1]))







