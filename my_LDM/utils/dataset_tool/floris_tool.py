import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib import rcParams
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane
import floris.layout_visualization as layoutviz
import random
import pickle
import multiprocessing as mp
import time

from my_LDM.utils.dataset_tool.vtk_tool import point_data_interpolation
from my_LDM.utils.dataset_tool.path_tool import get_root_path

root_path = get_root_path()

# ======================================================== #
# ============ 生成已有案例的floris流场 ==================== #
# ======================================================== #

# ================== 风机布局 ==================
"""
to use: 
    turbine_layout[2][0] -> list of case 2, x corrdinates
    turbine_layout[2][1] -> list of case 2, y corrdinates
    turbine_layout[2][2] -> list of case 2, yaw angles
"""
turbine_layout = [
    # 第一个元素空置，这样case_num和下标对应
    # 注意yaw必须传入浮点数，否则会np会报类型加减错误
    [
        # x坐标
        # y坐标
        # yaw角度
        # 前排风机:0 后排风机:1
    ],
    [
        # case 1
        [800,  800,  1430, 2200, 2000, 2200],
        [1600, 2000, 1600, 1800, 2000, 1500],
        [float(x) for x in [0, 0, 0, 0, 0, 0]],
        [0,    0,    1,    1,    1,    1]
    ],
    [
        # case 2
        [600, 600,  600,  1600,1800, 2600, 3000],
        [800, 1600, 2400, 800, 1600, 800, 1600],
        [float(x) for x in [0, 0, 0, 0, 0, 0, 0]],
        [0,   0,    0,    1,   1,    1,    1]
    ],
    [
        # case 3
        [400, 400,  400,  400,  400,  1800, 3200],
        [800, 1000, 1380, 1630, 2400, 2400, 2400],
        [float(x) for x in [0, 0, 0, 0, 0, 0, 0]],
        [0,   0,    0,    0,    0,    1,    1]
    ],
    [
        # case 4
        [400, 400, 400, 400, 1400, 1400, 1600, 1600, 2600, 2600],
        [2160, 1900, 1520, 1140, 1900, 1520, 2160, 1140, 1900, 1520],
        [float(x) for x in [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [0,   0,   0,   0,   1,    1,    1,    1,    1,    1]
    ],
    [
        # case 5
        [400, 400, 400, 1400, 1400, 1800, 1800, 2400, 2600, 2600],
        [1980, 1700, 1420, 1550, 1850, 2000, 1400, 1700, 1900, 1500],
        [float(x) for x in [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [0,   0,   0,   1,   1,    1,    1,    1,    1,    1]
    ],
    [
        # case 6
        [600, 600, 600, 1400, 1400, 1400, 2200, 2200, 2200],
        [1000, 1380, 1760, 1000, 1380, 1760, 1000, 1380, 1760],
        [float(x) for x in [30, 30, 30, -30, 30, 30, 30, -30, 30]],
        [0,   0,   0,   1,    1,    1,    1,    1,    1]
    ],
    [
        # case 7
        [600, 600, 800, 1000, 1200, 1400, 1400, 1800, 2000, 2000, 2400],
        [400, 1200, 1000, 800, 600, 400, 1200, 1000, 600, 800, 400],
        [float(x) for x in [30, -30, -15, 0, 20, 15, -20, -10, 10, -5, 30]],
        [0,   0,   0,   0,    0,    1,    1,    1,    1,    1,    1]
    ],
    [
        # case 8
        [800, 800, 800, 1000, 1000, 1400, 1800, 2000, 2000, 2200, 2200, 2200],
        [1600, 2000, 2600, 2000, 2400, 1800, 1600, 2000, 2400, 1800, 2200, 2600],
        [float(x) for x in [30, 15, -20, 0, -30, 15, 20, 15, 15, 10, 25, 25]],
        [0,   0,   0,   0,    0,    0,    1,    1,    1,    1,    1,    1]
    ],
    [
        # case 9
        [600, 600, 600, 1400, 1400, 1400, 2200, 2200, 2200],
        [1000, 1380, 1760, 1000, 1380, 1760, 1000, 1380, 1760],
        [float(x) for x in [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [0,   0,   0,   1,    1,    1,    1,    1,    1]
    ],
    [
        # case 10
        [600, 600, 600, 1400, 1400, 2200, 2200, 2200],
        [1000, 1380, 1760, 1000, 1760, 1000, 1380, 1760],
        [float(x) for x in [0, 0, 0, 0, 0, 0, 0, 0]],
        [0,   0,   0,   1,    1,    1,    1,    1]
    ],
    [
        # case 11
        [600, 600, 1400, 2200, 2200],
        [1380, 1760, 1000, 1000, 1760],
        [float(x) for x in [0, 0, 0, 0, 0]],
        [0,   0,   1,    1,    1]
    ],
    [
        # case 12
        [600, 600, 600, 1400, 2200, 2200],
        [1000, 1380, 1760, 1380, 1000, 1380],
        [float(x) for x in [0, 0, 0, 0, 0, 0]],
        [0,   0,   0,   1,    1,    1]
    ],
]


# ================== 输入布局 floris插值到np ==================
def floris_interpolation(turbine_layout_x, turbine_layout_y, turbine_layout_yaw, grid_shape=(300, 400)):
    # 设置布局, yaw要求传入一个list包含所有的yaw，因此用[]包裹
    layout_x = turbine_layout_x
    layout_y = turbine_layout_y
    layout_yaw = [turbine_layout_yaw]
    print(len(layout_x), len(layout_y), len(layout_yaw[0]))
    # 设置floris模型
    fmodel = FlorisModel(f"{root_path}/reference_repo/floris/examples/inputs/gch.yaml")
    fmodel.set(layout_x=layout_x, layout_y=layout_y, yaw_angles=layout_yaw)
    # turbine_names = ["T1", "T2", "T3", "T4", "T9", "T10", "T75", "T78"]
    fmodel.set(
        wind_speeds=[8.00],
        wind_directions=[270.0],
        turbulence_intensities=[0.08],
    )
    # 计算
    horizontal_plane = fmodel.calculate_horizontal_plane(
        y_resolution=grid_shape[0],
        x_resolution=grid_shape[1],
        height=90.0,
        x_bounds=(0.0, 4000.0),
        y_bounds=(0.0, 3000.0),
    )
    # df_extract 是floris计算的水平面速度场数据
    # shape=(points_num, 6), 6个值分别为x1, x2, x3, u, v, w
    # 我们要把x1,x2作为平面的点坐标，u,v作为平面的对应速度分量
    # 借助写好的field_data_interpolation函数，我们只需要把df转换成numpy数组，然后传入函数即可

    # 从FLORIS封装的CutPlane对象中提取dataframe数据
    df_extract = horizontal_plane.df
    # 把dataframe转换成numpy数组
    # 其中前三列x1,x2,x3正好对应numpy_points
    # 而后三列u,v,w正好对应velocity_array
    numpy_points_floris = df_extract[["x1", "x2", "x3"]].values
    velocity_array_floris = df_extract[["u", "v", "w"]].values
    # 对数据进行插值
    interpolation_result_floris = point_data_interpolation(
        numpy_points=numpy_points_floris,
        velocity_array=velocity_array_floris,
        grid_shape=grid_shape,
    )
    # print(interpolation_result_floris.shape)
    # interpolation_result_floris = interpolation_result_floris[:, ::-1, :]
    interpolation_result_floris = interpolation_result_floris[:, :, :]
    return interpolation_result_floris


def realCase_floris_interpolation(case_num, grid_shape=(150, 200), is_saving=False, save_path=None):
    # 设置布局, yaw要求传入一个双层list，外层第一个元素包含所有的yaw，因此用[]包裹
    layout_x = turbine_layout[case_num][0]
    layout_y = turbine_layout[case_num][1]
    layout_yaw = turbine_layout[case_num][2]
    interpolation_result_floris = floris_interpolation(layout_x, layout_y, layout_yaw, grid_shape)
    # 把数据写入对应的文件夹
    if is_saving:
        np.save(save_path, interpolation_result_floris)
        print(f"case {case_num+1} data saved to {save_path}/adm_{case_num+1}.npy")
        return interpolation_result_floris
    else:
        return interpolation_result_floris


# ======================================================== #
# =============== 生成用于训练vae的随机floris流场 ============== #
# ======================================================== #


# ================== 随机生成floris布局 ==================
def generate_wind_farm_layout(expect_num_layouts: int = 50):
    """
    随机生成num_layouts条风电场布局
    expect_num_layouts: int = 50, 期望生成的风电场布局数量，实际生成的数量可能小于这个数
    return: list.shape=(一个接近但小于expect_num_layouts的数, 2)
    2为 (layout_x, layout_y) 两个list，分别记录了风机的x坐标和y坐标
    len(layout_x) == len(layout_y) 为风机数量
    """
    layouts = []
    for _ in range(expect_num_layouts):
        layout_x = []
        layout_y = []
        # ------------- 随机风机数量 ------------------ #
        # 定义三个高斯分布的参数
        mu1, sigma1 = 4, 1
        mu2, sigma2 = 6, 1
        mu3, sigma3 = 8, 2
        # 定义选择每个高斯分布的概率
        p1, p2, p3 = 0.45, 0.45, 0.1
        # 按照给定的概率选择从哪个高斯分布中生成样本
        choice = np.random.choice([1, 2, 3], p=[p1, p2, p3])
        if choice == 1:
            num_turbines_float = np.random.normal(mu1, sigma1)
        elif choice == 2:
            num_turbines_float = np.random.normal(mu2, sigma2)
        else:
            num_turbines_float = np.random.normal(mu3, sigma3)
        # 对结果进行四舍五入并转换为整数
        num_turbines = round(num_turbines_float)
        # 确保num_turbines在1到20的范围内
        num_turbines = max(1, min(num_turbines, 20))
        # ------------- 生成风机坐标 ------------------ #
        # 规则的，按照高斯分布，从下往上生成，从左往右生成
        y_positions = np.linspace(200, 2800, num_turbines)
        x_positions = np.linspace(400, 3800, num_turbines)
        for i in range(num_turbines):
            x = np.random.normal(x_positions[i], 300)
            y = np.random.normal(y_positions[i], 150)
            layout_x.append(x)
            layout_y.append(y)
        # 完全随机生成
        for _ in range(num_turbines):
            x = random.uniform(300, 3800)
            y = random.uniform(100, 2900)
            layout_x.append(x)
            layout_y.append(y)

        # ------------- 后处理部分 ------------------ #
        # 把坐标为负值的风机的坐标变为正值
        for i in range(len(layout_x)):
            if layout_x[i] < 0:
                layout_x[i] = random.uniform(300, b=600)
            if layout_y[i] < 0:
                layout_y[i] = random.uniform(300, b=2700)
        # 把坐标溢出的风机的坐标变为边界值靠里一点
        for i in range(len(layout_x)):
            if layout_x[i] > 4000:
                layout_x[i] = random.uniform(3700, b=3900)
            if layout_y[i] > 3000:
                layout_y[i] = random.uniform(2700, b=2900)
        # ------------- 相邻风机合并 ------------------ #
        # i j 双指针双循环遍历所有风机，如果距离小于100，合并
        i = 0
        while i < len(layout_x) - 1:
            j = i + 1
            while j < len(layout_x):
                dist = ((layout_x[i] - layout_x[j]) ** 2 + (layout_y[i] - layout_y[j]) ** 2) ** 0.5
                if dist < 150:
                    layout_x[i] = (layout_x[i] + layout_x[j]) / 2
                    layout_y[i] = (layout_y[i] + layout_y[j]) / 2
                    layout_x.pop(j)
                    layout_y.pop(j)
                else:
                    j += 1
            i += 1
        # 风机坐标取整
        layout_x = [int(x) for x in layout_x]
        layout_y = [int(y) for y in layout_y]
        # ------------- 随机剔除一个区域的风机 ------------------ #
        # 第一次剔除
        # x从800剔除，匹配我的实际风机设置
        if random.random() < 0.7:
            y_length = random.choice([1 / 5, 1 / 4, 1 / 3]) * 3000
            x_length = random.choice([1 / 3, 1 / 2]) * 4000
            y_start = random.uniform(0, 3000 - y_length)
            x_start = random.uniform(800, 4000 - x_length)
            i = 0
            while i < len(layout_x):
                if y_start <= layout_y[i] <= y_start + y_length and x_start <= layout_x[i] <= x_start + x_length:
                    layout_x.pop(i)
                    layout_y.pop(i)
                else:
                    i += 1
        # 第二次剔除
        # x从800剔除，匹配我的实际风机设置
        if random.random() < 0.3:
            y_length = random.choice([1 / 5, 1 / 4, 1 / 3]) * 3000
            x_length = random.choice([1 / 3, 1 / 2]) * 4000
            y_start = random.uniform(0, 3000 - y_length)
            x_start = random.uniform(800, 4000 - x_length)
            i = 0
            while i < len(layout_x):
                if (y_start <= layout_y[i] <= y_start + y_length) and (x_start <= layout_x[i] <= x_start + x_length):
                    layout_x.pop(i)
                    layout_y.pop(i)
                else:
                    i += 1
        # 第三次剔除
        # 极小概率剔除x=0~800的风机
        if random.random() < 0.1:
            y_length = random.choice([1 / 5, 1 / 4, 1 / 3]) * 3000
            x_length = random.choice([1 / 3, 1 / 2]) * 4000
            y_start = random.uniform(0, 3000 - y_length)
            x_start = random.uniform(0, 4000 - x_length)
            i = 0
            while i < len(layout_x):
                if (y_start <= layout_y[i] <= y_start + y_length) and (x_start <= layout_x[i] <= x_start + x_length):
                    layout_x.pop(i)
                    layout_y.pop(i)
                else:
                    i += 1
        # 若风机数量不为0，且小于20，则append
        if len(layout_x) > 0 and len(layout_x) <= 20:
            # ------------- 生成风机yaw ------------------ #
            # 生成yaw角度，范围为-30到30，步长为5
            layout_yaw = [
                # random.choice([-30, -25, -20, -15, -10, -5., 0, 5, 10, 15, 20, 25., 30.])
                float(random.choice([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]))
                for _ in range(len(layout_x))
            ]
            layouts.append((layout_x, layout_y, layout_yaw))
    return layouts


# ================== 随机生成floris场 每个子进程调用的函数 ==================
def process_layouts(process_id):
    grid_shape = (150, 200)

    # 生成风电场排布
    wind_farm_layouts = generate_wind_farm_layout(expect_num_layouts=100)
    data_case_num = len(wind_farm_layouts)
    with open(
        f"{root_path}/dataset/Floris_randomGenerate/{grid_shape[0]}_{grid_shape[1]}/matrix_{process_id}.pkl", "wb"
    ) as file:
        pickle.dump(wind_farm_layouts, file)

    # 保存风机速度场
    all_floris_data = []
    for layout in wind_farm_layouts:
        horizontal_plane = floris_interpolation(
            turbine_layout_x=layout[0],
            turbine_layout_y=layout[1],
            turbine_layout_yaw=layout[2],
            grid_shape=grid_shape,
        )
        all_floris_data.append(horizontal_plane)
        print(f"layout case {layout[0][0]}（这是第一个风机的x坐标） 执行完成")
    all_floris_data = np.array(all_floris_data)
    np.save(
        f"{root_path}/dataset/Floris_randomGenerate/{grid_shape[0]}_{grid_shape[1]}/floris_field_data_{process_id}.npy",
        all_floris_data,
    )

    print(f"进程 {process_id} 数据生成完成, 一共{data_case_num}个数据")
    print(f"进程 {process_id} 数据形状: {all_floris_data.shape}")


if __name__ == "__main__":
    # 计时
    start_time = time.time()

    print(f"mp.cpu_count(): {mp.cpu_count()}")
    print(f"use cpu: {mp.cpu_count() - 2}")

    num_processes = mp.cpu_count() - 2
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=process_layouts, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # # 测试debug用 单线程
    # process_layouts(0)

    # 计时
    end_time = time.time()
    print(f"总用时: {end_time - start_time} s")
