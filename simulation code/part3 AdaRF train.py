# 模拟训练数据 (X, Y)
import numpy as np
X = np.random.rand(1000, 100, 100, 2)  # 1000个全息图样本，尺寸为100x100x2
Y = np.random.rand(1000, 2)  # 1000个标签，每个标签为2D坐标

# 分割数据集
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 训练模型
history = model.fit(X_train, Y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, Y_val),
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
                    ])

# 评估模型
test_loss, test_mae = model.evaluate(X_val, Y_val)
print(f"Validation Loss: {test_loss}, Validation MAE: {test_mae}")
