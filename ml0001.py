import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
datas = pd.read_csv('files/veriler.csv')

# veri on isleme
print(datas)
boykilo = datas[['boy', 'kilo']]
print(boykilo)

