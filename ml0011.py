import numpy as np
import matplotlib.pyplot as plt

# Veri seti boyutu
complexity = np.linspace(0, 10, 100)

# Bias ve Variance eğrilerini tanımlama
bias2 = np.exp(-0.5 * complexity) * 10  # Bias^2 azalan fonksiyon
variance = np.exp(0.5 * complexity)      # Variance artan fonksiyon
error = bias2 + variance +5 # Toplam hata

# Grafik çizimi
plt.figure(figsize=(8, 6))
plt.plot(complexity, bias2, label="Bias² (Underfitting)", color="blue", linewidth=2)
plt.plot(complexity, variance, label="Variance (Overfitting)", color="red", linewidth=2)
plt.plot(complexity, error, label="Total Error", color="black", linestyle="dashed", linewidth=2)

# Denge noktasını gösteren dikey çizgi
optimal_complexity = complexity[np.argmin(error)]
plt.axvline(optimal_complexity, color="green", linestyle="dotted", linewidth=2, label="Optimal Complexity")

# Grafik özellikleri
plt.xlabel("Model Complexity")
plt.ylabel("Error")
plt.title("Bias - Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.show()
