import numpy as np
import matplotlib.pyplot as plt


def calculate_position(antenna_positions, angles):
    """
    Calculate the position of the target using AOA.

    Parameters:
    - antenna_positions: List of (x, y) coordinates of two antennas
    - angles: List of angles of arrival (in degrees) at each antenna

    Returns:
    - (x, y): Estimated position of the target
    """
    # Convert angles from degrees to radians
    angles_rad = np.radians(angles)

    # Extract antenna coordinates
    x1, y1 = antenna_positions[0]
    x2, y2 = antenna_positions[1]

    # Calculate slopes (tangents)
    tan_theta1 = np.tan(angles_rad[0])
    tan_theta2 = np.tan(angles_rad[1])

    # Solve for intersection point (x, y)
    x_position = (y2 - y1 + x1 * tan_theta1 - x2 * tan_theta2) / (tan_theta1 - tan_theta2)
    y_position = y1 + (x_position - x1) * tan_theta1

    return x_position, y_position


# Antenna positions and AOA values
antenna_positions = [(0, 0), (10, 0)]  # Coordinates of the antennas
angles = [45, 30]  # Angles of arrival (in degrees)

# Calculate the target position
x_target, y_target = calculate_position(antenna_positions, angles)
print(f"Estimated Target Position: x = {x_target:.2f}, y = {y_target:.2f}")

# Visualization
plt.figure(figsize=(8, 6))

# Plot antenna positions
for pos in antenna_positions:
    plt.scatter(pos[0], pos[1], color='red', label='Antenna' if pos == antenna_positions[0] else "")

# Plot signal paths
x_vals = np.linspace(0, 15, 100)
y_vals_1 = (x_vals - antenna_positions[0][0]) * np.tan(np.radians(angles[0])) + antenna_positions[0][1]
y_vals_2 = (x_vals - antenna_positions[1][0]) * np.tan(np.radians(angles[1])) + antenna_positions[1][1]
plt.plot(x_vals, y_vals_1, label=f'Signal Path 1 (θ1={angles[0]}°)', linestyle='--', color='blue')
plt.plot(x_vals, y_vals_2, label=f'Signal Path 2 (θ2={angles[1]}°)', linestyle='--', color='green')

# Plot target position
plt.scatter(x_target, y_target, color='blue', label='Estimated Target Position', zorder=5)

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('AOA-Based Positioning')
plt.legend()
plt.grid()
plt.show()
