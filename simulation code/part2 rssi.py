import numpy as np
import matplotlib.pyplot as plt


# 定义RSSI信号衰减模型
def rssi_signal_strength(tx_power, distance, path_loss_exponent=2.0, noise_std=2.0):
    """
    模拟RSSI信号强度随距离衰减，包含路径损耗和高斯噪声。

    参数:
    - tx_power: 发射功率 (dBm)
    - distance: 标签与读写器之间的距离 (米)
    - path_loss_exponent: 路径损耗因子
    - noise_std: 噪声的标准差 (dBm)

    返回:
    - rssi: 接收信号强度 (dBm)
    """
    # 理论RSSI信号强度
    rssi = tx_power - 10 * path_loss_exponent * np.log10(distance)

    # 添加高斯噪声
    noise = np.random.normal(0, noise_std, size=len(distance))
    return rssi + noise


# 模拟参数
tx_power = -30  # 发射功率 (dBm)
distances = np.linspace(1, 100, 100)  # 距离 (1 到 100 米)
path_loss_exponent = 2.5  # 路径损耗因子（根据环境调整）
noise_std = 3.0  # 高斯噪声标准差

# 计算RSSI信号强度
rssi_values = rssi_signal_strength(tx_power, distances, path_loss_exponent, noise_std)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(distances, rssi_values, label="RSSI with Noise", color='blue', linestyle='-', marker='o', markersize=4)
plt.plot(distances, tx_power - 10 * path_loss_exponent * np.log10(distances), label="Theoretical RSSI", color='red',
         linestyle='--')
plt.xlabel("Distance (m)")
plt.ylabel("RSSI (dBm)")
plt.title("RSSI Signal Strength with Path Loss and Noise")
plt.legend()
plt.grid()
plt.show()
