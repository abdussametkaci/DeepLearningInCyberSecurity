# Veri Kümesi   :  https://www.unb.ca/cic/datasets/ids-2017.html
# Etiketsiz veriler için saldırı tespit araştırması

import pandas as pd
import numpy as np
from tensorflow import set_random_seed
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf

#####################################################################
# Tensorflow uyarı mesajlarının gizlenmesi..
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#####################################################################
# Tekrarlanabilirliğin sağlanması...
from tensorflow import set_random_seed
set_random_seed(123)
np.random.seed(1234)
tf.set_random_seed(123)

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
path = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
df = pd.read_csv(path, low_memory=False)
df = pd.DataFrame(df)
   
#####################################################################
# Önişleme
df.replace('Infinity', -1, inplace=True)

# Bu değişkene gerek yok..
df = df.drop(' Destination Port', 1)

# Eğer na içeren gözlemler varsa onların silinmesi..
# df.dropna(inplace=True, axis=1)
df.dropna(inplace=True)

# Değişken isimlerindeki boşlukların kaldırılması..
df.columns = df.columns.str.replace(" ", "", regex=True)

# Sadece "Normal" ve "Saldırı" etiketli verilerin elde edilmesi..
df['Label'] = df['Label'].str.replace('BENIGN', 'Normal', regex=True)
df['Label'] = df['Label'].str.replace('.*Brute Force', 'Saldırı', regex=True)
df['Label'] = df['Label'].str.replace('.*XSS', 'Saldırı', regex=True)
df['Label'] = df['Label'].str.replace('.*Sql Injection', 'Saldırı', regex=True)

# Etiketlerin bir vektörde saklanması
y = df["Label"]
print("Başlangıçtaki sınıflar: \n", pd.crosstab(y, columns='count'))

###########################################################
# Veri kümesinin eğitim ve test için bölünmesi
X_train, X_test,y_train,y_test = train_test_split(
    df,
    y,
    test_size=0.2)

y_test  = X_test['Label']
X_test  = X_test.drop(['Label'], axis=1)

print("Test verisindeki sınıflar: \n", pd.crosstab(y_test, columns='count'))

# Sadece Normal verinin seçilmesi işlemi... Normal olanlarla
# model eğitilecektir, test işlemi test verisiyle gerçekleşecektir.
# Bu eğitim verisinin sınıf değişkeni yoktur..
X_train = X_train[X_train.Label=="Normal"]  #Sadece normal kayıtlar
y = X_train["Label"]
X_train = X_train.drop(['Label'], axis=1)   #Sınıf sütununun silinmesi..

print("X_train verisindeki sınıflar: \n", pd.crosstab(y, columns='count'))
print(X_train.shape)

# Normalleştirme işlemi..
# Herbir özniteliğin 0-1 arasında ölçeklendirilmesi...
X_train = minmax_scale(X_train, axis = 1)
X_test = minmax_scale(X_test, axis = 1)
ncol = X_train.shape[1]

###########################################################
# Kodlama tanımlar..

# Kodlama boyutu
encoding_dim = 10

# Kodlayıcı (encoder) katmanının tanımlanması
input_dim = Input(shape = (ncol, ))
encoded = Dense(ncol, activation = 'relu')(input_dim)
encoded = Dense(50, activation = 'relu')(encoded)
encoded = Dense(50, activation = 'relu')(encoded)

encoded =  Dense(encoding_dim, activation = 'relu')(encoded)

# Kod çözücü (decoder) katmanının tanımlanması
decoded = Dense(50, activation = 'relu')(encoded)
decoded = Dense(50, activation = 'relu')(decoded)
decoded = Dense(ncol, activation = 'sigmoid')(decoded)

autoencoder = Model(inputs = input_dim, outputs = decoded)
autoencoder.summary()

###########################################################
# Autoencoderin eğitilmesi
autoencoder.compile(metrics=['accuracy'],
    loss='mean_squared_error',
    optimizer='adam')

history = autoencoder.fit(
    X_train,
    X_train,
    epochs = 5,   
    batch_size = 100,
    verbose=2,
    shuffle=True,
    validation_split = 0.2
).history

###########################################################
# Model loss
plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Doğrulama Hatası')
plt.legend(loc='upper right')
plt.title('Model hatası')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(ymin=0.70,ymax=1)
plt.show()

############################################################
###########################################################
# Yeniden yapılandırma için X_test verisi kullanılıyor..
öngörüler = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - öngörüler, 2), axis=1)

############################################################
# Anormallikler elde edildi, peki bu işlemin performansı nedir ?
# y_test vektörünü yukarıda hiç kullanmadık. Bu değeri test
# işleminin performansını öğrenmek için kullanabiliriz.
# Performansı ölçmek için ROC eğrileri kullanılır.

from sklearn.metrics import recall_score,classification_report, auc, roc_curve
print("Test verisindeki sınıflar: \n", pd.crosstab(y_test, columns='count'))

error_df = pd.DataFrame({
    "Yeniden_yapılanma_hatası": mse,
    "Gerçek_sınıflar": y_test})
error_df.Gerçek_sınıflar = error_df.Gerçek_sınıflar.map({
    "Normal": 0, "Saldırı": 1})

false_pos_rate, true_pos_rate, thresholds = roc_curve(
    error_df.Gerçek_sınıflar,
    error_df.Yeniden_yapılanma_hatası)

roc_auc = auc(false_pos_rate, true_pos_rate,)
print("AUC : ", round(roc_auc,2))

plt.plot(false_pos_rate, true_pos_rate,
         linewidth=5,
         label='AUC = %0.2f'% round(roc_auc,2))
plt.plot([0,1],[0,1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('ROC eğrisi')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




