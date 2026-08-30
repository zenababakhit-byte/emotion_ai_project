import cv2
import numpy as np
import pandas as pd
import matplotlib
import torch

print("=" * 50)
print("GERCEK ZAMANLI DUYGU ANALIZI PROJESI")
print("GELISTIRME ORTAMI TESTI")
print("=" * 50)

print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"PyTorch: {torch.__version__}")

print("\nCUDA kullanılabilir mi?:", torch.cuda.is_available())

print("\nTum kutuphaneler basariyla calisiyor!")