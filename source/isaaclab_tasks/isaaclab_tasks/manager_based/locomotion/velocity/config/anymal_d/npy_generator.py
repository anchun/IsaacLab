import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

current_file_directory = os.path.dirname(os.path.abspath(__file__))

# 存储控制点的列表
trajectory_points = []

# 背景图路径
background_image_path = os.path.join(current_file_directory, "img", "bev_img.png")  # 替换为您的 BEV 图路径

def onclick(event):
    # 获取点击的坐标
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        # 将坐标添加到控制点列表
        trajectory_points.append([x, y])
        # 在图中标记控制点
        plt.plot(x, y, 'ro')  # 红色圆点
        plt.text(x, y, str(len(trajectory_points)), fontsize=12, color='white', ha='center')  # 添加编号
        plt.draw()

def on_close(event):
    # 当关闭图形窗口时保存控制点
    if trajectory_points:
        trajectory_points_np = np.array(trajectory_points)
        np.save(os.path.join(current_file_directory, "trajectory_points.npy"), trajectory_points_np)
        print("控制点已保存为 trajectory_points.npy")

        # 保存带有轨迹点的图像
        plt.savefig(os.path.join(current_file_directory, "img", "trajectory_with_points.png"))
        print("带有轨迹点的图像已保存为 trajectory_with_points.png")

    plt.close(event.canvas.figure)  # 关闭图形窗口

def generate_trajectory_points():
    # 创建图形
    fig, ax = plt.subplots()
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title("Click to choose waypoints")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    # 显示背景图
    background_image = mpimg.imread(background_image_path)
    ax.imshow(background_image, extent=[-3.5, 3.5, -3.5, 3.5], aspect='auto')

    # 设置相同的 x 和 y 轴比例
    ax.set_aspect('equal', adjustable='box')

    # 连接点击事件
    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_close = fig.canvas.mpl_connect('close_event', on_close)

    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.show()

if __name__ == "__main__":
    generate_trajectory_points()