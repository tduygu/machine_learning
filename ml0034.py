# ödev 5
# random forest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Veri setini yükleme

data = pd.read_csv("files/anxiety_depression_data.csv")
missing_values = data.isnull().sum()
categorical_columns_with_na = [col for col in data.select_dtypes(include=['object']).columns if data[col].isnull().sum() > 0]
if categorical_columns_with_na:
    data[categorical_columns_with_na] = data[categorical_columns_with_na].fillna("Bilinmiyor")


  # Kategorik sütunları belirleme
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_columns:
  data[col] = label_encoder.fit_transform(data[col])


y = data['Life_Satisfaction_Score']
X = data.drop(columns=['Life_Satisfaction_Score'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Modeli oluşturma
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# Modeli eğitme
rf_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Tahmin yapma
y_pred = rf_model.predict(X_test)

# Hata kareleri ortalamasını hesaplama
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# R-kare skorunu hesaplama
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")
#
# Eğer modelinizin performansı beklentilerinizin altındaysa, aşağıdaki adımları düşünebilirsiniz:​
# Hiperparametre Ayarı: GridSearchCV veya RandomizedSearchCV kullanarak en iyi hiperparametreleri bulun.​
# Özellik Seçimi: Önemsiz veya düşük önem düzeyine sahip değişkenleri çıkararak modeli sadeleştirin.​
# Daha Fazla Veri: Mümkünse, daha fazla veri toplayarak modelin genelleme yeteneğini artırın.​
# Bu adımları izleyerek, Life_Satisfaction_Score özniteliğini Random Forest modeliyle başarılı bir şekilde tahmin edebilirsiniz.




