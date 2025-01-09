import numpy as np
import matplotlib.pyplot as plt

# 定义参数
wavelength = 0.1  # 信号波长 (m)
antenna_positions = np.linspace(0, 1, 100)  # 天线沿轨迹的线性位置 (m)
tag_position = np.array([0.5, 0.5])  # 标签的实际位置 (x, y)


# 计算相位函数
def calculate_phase(antenna_positions, tag_position, wavelength):
    """
    计算每个天线位置接收到的信号相位。

    参数:
    - antenna_positions: 天线的位置 (沿x轴移动)
    - tag_position: 标签的实际二维位置 (x, y)
    - wavelength: 信号波长 (m)

    返回:
    - phases: 每个天线位置对应的相位 (弧度)
    """
    distances = np.sqrt((antenna_positions - tag_position[0]) ** 2 + tag_position[1] ** 2)
    phases = (2 * np.pi * distances / wavelength) % (2 * np.pi)  # 相位值归一化到 [0, 2π)
    return phases


# 模拟信号相位
phases = calculate_phase(antenna_positions, tag_position, wavelength)


# 重建位置 (示例：基于最小误差)
def estimate_position(antenna_positions, measured_phases, wavelength):
    """
    通过相位信息估算标签位置。

    参数:
    - antenna_positions: 天线的位置 (沿x轴移动)
    - measured_phases: 测得的信号相位
    - wavelength: 信号波长 (m)

    返回:
    - estimated_position: 标签的估算二维位置 (x, y)
    """
    best_position = None
    min_error = float('inf')

    # 扫描可能的标签位置
    x_range = np.linspace(0, 1, 100)
    y_range = np.linspace(0, 1, 100)
    for x in x_range:
        for y in y_range:
            test_distances = np.sqrt((antenna_positions - x) ** 2 + y ** 2)
            test_phases = (2 * np.pi * test_distances / wavelength) % (2 * np.pi)
            error = np.mean((test_phases - measured_phases) ** 2)
            if error < min_error:
                min_error = error
                best_position = (x, y)

    return best_position


# 估算标签位置
estimated_position = estimate_position(antenna_positions, phases, wavelength)

# 输出结果
print(f"Actual Position: {tag_position}")
print(f"Estimated Position: {estimated_position}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(antenna_positions, phases, label='Measured Phases', color='blue')
plt.title('Signal Phases at Different Antenna Positions')
plt.xlabel('Antenna Position (m)')
plt.ylabel('Phase (radians)')
plt.legend()
plt.grid()
plt.show()

# 绘制标签位置和估算位置
plt.figure(figsize=(8, 6))
plt.scatter(tag_position[0], tag_position[1], color='green', label='Actual Tag Position')
plt.scatter(estimated_position[0], estimated_position[1], color='red', label='Estimated Tag Position')
plt.title('Tag Position Estimation')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid()
plt.show()
