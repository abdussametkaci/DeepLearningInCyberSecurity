'''Veri kümesi : https://www.kaggle.com/c/web-traffic-time-series-forecasting/data

Bu program kaggle platformunda yer alan wikipedia web sayfası
eri kümesini okuyarak, günlük bazda toplam trafik değerlerini
hesaplayarak bir veri kümesi biçiminde kaydeder. '''
 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Web trafik veri kümesinin okutulması..
print("Veri Kümesi okunuyor..")
veri = pd.read_csv('train_2.csv',
    low_memory=False,
    index_col='Page').T.unstack().reset_index().rename(
    columns={0:'Visits','level_1':'Date'}).dropna(subset=['Visits'])
veri.head()
print("Okunan gözlem sayısı :", veri.shape[0])
print("Okunan veri aylık ve günlük toplamlara dönüştürülüyor..")

# Tarihe göre sıralama
veri.sort_values('Date',inplace=True)

# Tarih bilgisinin pandas tarih formatına dönüştürülmesi..
veri['Date'] = pd.to_datetime(veri['Date'])

# Ay numaralarının elde edilmesi..
veri['Ay'] = veri['Date'].dt.month

# Aylık bazda gruplama..
veri_aylık = veri.groupby(['Ay']).agg({'Visits':'sum'})
print("Aylık web trafiği:\n", veri_aylık)

plt.plot(veri_aylık["Visits"])
plt.title('Aylık web trafiği')
plt.ylabel('Ziyaretler')
plt.xlabel('Ay')
plt.show()

# Toplam günlük web trafiği
toplam_günlük = veri.groupby(['Date']).agg({'Visits':'sum'})
toplam_günlük = toplam_günlük / 1000000
print("Günlük web trafiği:\n", toplam_günlük)
plt.plot(toplam_günlük["Visits"])
plt.title('Günlük toplamlar')
plt.ylabel('Ziyaretler')
plt.xlabel('Yıl-Ay')
plt.show()
plt.close()

toplam_günlük.index.name = 'Date'
toplam_günlük.reset_index(inplace=True)

# Elde edilen veri kümesinin kaydedilmesi..
import glob
dosya_var = glob.glob("günlük_trafik.txt")

if not dosya_var:
    toplam_günlük.to_csv("günlük_trafik.txt",
                         header=None, index=None, sep=',', mode='a')
else:
    print ("DİKKAT: Bu dosya zaten var !")
    
print("Günlük toplam gözlem sayısı :", toplam_günlük.shape[0])


