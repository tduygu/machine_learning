# veri dengeli mi?


import pandas as pd

# Veri setini yükle
df = pd.read_csv('files/happydata.csv')
print(df)

# Hedef sütundaki sınıf dağılımını kontrol et
class_counts = df['happy'].value_counts()

# Yüzdelik oranlarını göster
class_percentages = df['happy'].value_counts(normalize=True) * 100

# Sonuçları yazdır
print("Sınıf Dağılımı:\n", class_counts)
print("\nSınıf Dağılımı (%)\n", class_percentages)

# Dengeli olmasaydı şunlar yapılabilirdi:
# Eğer veri dengesizse:
# ✅ SMOTE (Synthetic Minority Over-sampling Technique) ile azınlık sınıfı artırılabilir.
# ✅ Ağırlıklı KNN (weight='distance') ile dengesizlik azaltılabilir.
# ✅ Stratified K-Fold Cross Validation kullanarak denge sağlanabilir.
