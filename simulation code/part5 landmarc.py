import numpy as np
import matplotlib.pyplot as plt

# 修复后的RSSI计算函数
def calculate_rssi(distance, tx_power=-30, path_loss_exponent=2.0):
    """
    计算RSSI值
    参数:
    - distance: 距离 (米)，可以是标量或数组
    - tx_power: 发射功率 (dBm)
    - path_loss_exponent: 路径损耗因子
    返回:
    - RSSI值 (dBm)，与输入类型一致
    """
    distance = np.maximum(distance, 1e-6)  # 防止零或负值
    rssi = tx_power - 10 * path_loss_exponent * np.log10(distance)
    return rssi

# 生成参考标签和目标标签位置
def generate_positions(area_size, num_references):
    """
    生成参考标签和目标标签的位置
    参数:
    - area_size: 区域边长 (米)
    - num_references: 参考标签数量
    返回:
    - references: 参考标签的坐标数组
    - target: 目标标签的随机坐标
    """
    references = np.random.rand(num_references, 2) * area_size
    target = np.random.rand(1, 2) * area_size
    return references, target[0]

# 欧几里得距离计算
def euclidean_distance(coord1, coord2):
    """
    计算两点之间的欧几里得距离
    """
    return np.sqrt(np.sum((coord1 - coord2) ** 2))

# LANDMARC定位算法
def landmarc_position_estimation(references, target, k=3, tx_power=-30, path_loss_exponent=2.0):
    """
    LANDMARC定位算法
    参数:
    - references: 参考标签的坐标数组
    - target: 目标标签的真实坐标
    - k: 最近邻的参考标签数量
    - tx_power: 发射功率
    - path_loss_exponent: 路径损耗因子
    返回:
    - estimated_position: 估算的目标标签位置
    - distances: 各参考标签与目标标签的RSSI差异
    """
    # 计算目标标签与所有参考标签的距离
    distances = np.array([euclidean_distance(ref, target) for ref in references])
    rssi_values = np.array([calculate_rssi(dist, tx_power, path_loss_exponent) for dist in distances])

    # 计算目标标签的RSSI值
    target_rssi = calculate_rssi(distances, tx_power, path_loss_exponent)

    # 计算RSSI差异 (欧几里得距离在RSSI空间中)
    rssi_differences = np.abs(rssi_values - target_rssi)

    # 找到k个最近邻参考标签
    nearest_indices = np.argsort(rssi_differences)[:k]

    # 加权平均计算目标位置
    weights = 1 / (rssi_differences[nearest_indices] + 1e-6)  # 防止除以零
    weights /= np.sum(weights)  # 归一化权重
    estimated_position = np.sum(references[nearest_indices] * weights[:, None], axis=0)

    return estimated_position, distances

# 模拟LANDMARC系统
def landmarc_simulation(area_size=10, num_references=10, k=3):
    """
    模拟LANDMARC定位
    参数:
    - area_size: 定位区域边长 (米)
    - num_references: 参考标签数量
    - k: 最近邻参考标签数量
    返回:
    - error: 定位误差
    """
    # 生成参考标签和目标标签位置
    references, target = generate_positions(area_size, num_references)

    # 使用LANDMARC算法估算目标位置
    estimated_position, distances = landmarc_position_estimation(references, target, k=k)

    # 计算定位误差
    error = euclidean_distance(target, estimated_position)

    # 可视化结果
    plt.figure(figsize=(8, 6))
    plt.scatter(references[:, 0], references[:, 1], color='blue', label='Reference Tags')
    plt.scatter(target[0], target[1], color='red', label='Target Tag (Actual)', s=100)
    plt.scatter(estimated_position[0], estimated_position[1], color='green', label='Estimated Position', s=100)
    plt.legend()
    plt.title(f"LANDMARC Localization\nError: {error:.2f} meters")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid()
    plt.show()

    return error

# 运行LANDMARC实验
landmarc_simulation()
