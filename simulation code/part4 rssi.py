import numpy as np
import matplotlib.pyplot as plt

# RSSI计算公式
def calculate_rssi(distance, tx_power=-30, path_loss_exponent=2.0):
    """
    计算给定距离的RSSI值
    参数:
    - distance: 目标与接收器之间的距离 (米)
    - tx_power: 发射功率 (dBm)
    - path_loss_exponent: 路径损耗因子
    返回:
    - RSSI值 (dBm)
    """
    if distance <= 0:  # 防止距离为零
        return -100
    return tx_power - 10 * path_loss_exponent * np.log10(distance)

# 导航算法
def navigate_to_target(robot_position, target_position, step_size=0.5, angle_step=np.radians(45), rssi_threshold=-18):
    """
    机器人基于RSSI的导航算法
    参数:
    - robot_position: 机器人的初始位置 [x, y]
    - target_position: 目标标签的位置 [x, y]
    - step_size: 每次移动的距离 (米)
    - angle_step: 扫描角度范围 (弧度)
    - rssi_threshold: RSSI阈值 (dBm)
    返回:
    - path: 机器人移动路径
    - steps: 移动的总步数
    """
    path = [tuple(robot_position)]  # 记录路径
    angles = [-angle_step, angle_step]  # 顺时针和逆时针方向的角度偏移

    while True:
        best_rssi = -np.inf
        best_angle = 0

        # 扫描不同方向的RSSI值
        for angle in angles:
            # 计算扫描方向的新位置
            dx = np.cos(angle) * step_size
            dy = np.sin(angle) * step_size
            test_position = [robot_position[0] + dx, robot_position[1] + dy]

            # 计算新位置的距离和RSSI值
            distance = np.linalg.norm(np.array(test_position) - np.array(target_position))
            rssi = calculate_rssi(distance)

            # 更新最优方向
            if rssi > best_rssi:
                best_rssi = rssi
                best_angle = angle

        # 更新机器人位置
        robot_position[0] += np.cos(best_angle) * step_size
        robot_position[1] += np.sin(best_angle) * step_size
        path.append(tuple(robot_position))

        # 检查是否达到目标
        if best_rssi >= rssi_threshold:
            break

    return path, len(path)

# 性能评估
def evaluate_navigation(robot_initial_positions, target_position, step_size=0.5, rssi_threshold=-18):
    """
    评估不同初始条件下的导航性能
    参数:
    - robot_initial_positions: 机器人初始位置列表 [[x1, y1], [x2, y2], ...]
    - target_position: 目标标签位置 [x, y]
    - step_size: 每次移动的距离
    - rssi_threshold: RSSI阈值
    返回:
    - results: 包含路径、步数和误差的结果字典
    """
    results = []
    for initial_position in robot_initial_positions:
        path, steps = navigate_to_target(initial_position.copy(), target_position, step_size, rssi_threshold=rssi_threshold)
        final_position = np.array(path[-1])
        goal_position = np.array(target_position)
        error = np.linalg.norm(final_position - goal_position)  # 计算欧几里得误差
        results.append({"initial_position": initial_position, "path": path, "steps": steps, "error": error})
    return results

# 模拟实验
target_position = [10, 10]  # 目标标签位置
robot_initial_positions = [
    [0, 0],  # 初始位置1
    [0, 15],  # 初始位置2
    [15, 0],  # 初始位置3
    [15, 15]  # 初始位置4
]

# 运行实验
results = evaluate_navigation(robot_initial_positions, target_position)

# 输出实验结果
for i, result in enumerate(results):
    print(f"Initial Position {i+1}: {result['initial_position']}")
    print(f"Steps Taken: {result['steps']}")
    print(f"Final Error: {result['error']:.2f} meters")

# 可视化路径
plt.figure(figsize=(8, 6))
for result in results:
    path = np.array(result["path"])
    plt.plot(path[:, 0], path[:, 1], marker='o', label=f"Path from {result['initial_position']}")

# 绘制目标标签位置
plt.scatter(target_position[0], target_position[1], color='red', label="Target Position", s=100)
plt.title("RSSI-Based Robot Navigation Paths")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid()
plt.show()
