import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# 加載MNIST數據集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 找到每個數字的第一個範例
examples = {}
for i in range(10):
    for j in range(len(y_train)):
        if y_train[j] == i:
            examples[i] = x_train[j]
            break

# 顯示每個數字的範例圖片
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(examples[i], cmap='gray')
    plt.title(f"Digit: {i}")
    plt.axis('off')
plt.tight_layout()
plt.show()
