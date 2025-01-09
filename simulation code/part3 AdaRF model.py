import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

# 定义输入形状
input_shape = (100, 100, 2)

# 构建复杂网络模型
def create_advanced_adarf_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1: Conv + BN + ReLU + MaxPooling
    x = layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2: Conv + BN + ReLU + MaxPooling
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3: Residual Connection (Conv + BN + ReLU)
    residual = layers.Conv2D(128, kernel_size=(1, 1), padding='same', activation=None)(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.add([x, residual])  # Residual Connection
    x = tf.keras.activations.relu(x)

    # Block 4: Conv + Dropout + MaxPooling
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten
    x = layers.Flatten()(x)

    # Fully Connected Layers
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer (2D coordinates)
    outputs = layers.Dense(2, activation='linear')(x)

    model = models.Model(inputs, outputs)
    return model

# 创建模型
model = create_advanced_adarf_model(input_shape)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='mean_squared_error',
              metrics=['mae'])

# 打印模型结构
model.summary()


# 模拟训练数据 (X, Y)
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
