import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

##################################################################
## Veri kümesinin okutulması..
path= "merged_5s.csv"
veri = pd.read_csv(path, low_memory=False)

# Sınıf değişkeni..
y = veri["label"]
##################################################################
# önişleme
veri.replace('Infinity', -1, inplace=True)

# String olan bu değişkenlere gerek yok..
veri = veri.drop('Source IP', 1)
veri = veri.drop(' Destination IP', 1)

# Eğer na kayıt varsa onun silinmesi..
veri.dropna(inplace=True)

# Sınıf değişkeninin silinmesi..
y = veri["label"]
veri = veri.drop('label', 1)

# Sınıflar dengeli mi ?
print("Sınıfların dağılımı: \n", pd.crosstab(y, columns='count'))

# minority yöntemi ile dengeleme ..
smote = SMOTE(ratio='minority')
veri_min, y_min = smote.fit_sample(veri, y)
veri_min, y_min = smote.fit_sample(veri, y)

print("Minority ile dengeleme sonunda dağılım: \n",
      pd.crosstab(y_min, columns='count'))






