import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib import rcParams

# from path_tool import get_read_file_path_list_cell, get_read_file_path_list_point
from my_LDM.utils.dataset_tool.path_tool import get_read_file_path_list_cell, get_read_file_path_list_point


# ############################################################# #
#                     point data processing                     #
# ############################################################# #

def read_vtk_from_point(file_path):
    """
    读取vtk文件，返回点坐标和速度的numpy数组，没有进行插值数据重整
    file_path: vtu文件路径
    return: numpy_points, velocity_array
    numpy_points: 二维数组，每行一个点，每列一个坐标，以x,y的顺序排列
    velocity_array: 二维数组，每行一个点，每列一个速度分量, 以x,y的顺序排列
    """
    # 读取文件为vtk对象
    # reader = vtk.vtkXMLUnstructuredGridReader()
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()
    # 获取点数据
    points = data.GetPoints()
    numpy_points = vtk_to_numpy(points.GetData())
    # 获取点的速度
    velocity = data.GetPointData().GetArray("U")
    velocity_array = vtk_to_numpy(velocity)
    return numpy_points, velocity_array


def point_data_interpolation(numpy_points, velocity_array, grid_shape):
    """
    对点数据进行插值，返回插值后的数据
    numpy_points: 记录了点的坐标信息，每行与velocity_array一一对应。
        每行一个点，每列一个坐标，以x,y,z的顺序排列, 我们不用z坐标（全都是90）
    velocity_array: 记录了点的速度场信息。
        来自read_vtu, 二维数组=(num_point, 3)，每行一个点，每列一个速度分量,
        以x,y,z的顺序排列, 我们只用x,y两个分量
    grid_shape: 插值后的网格形状，为了和其他数据统一标准，
        我们指定为(多少行，多少列)，即（竖着的分辨率，横着的分辨率）
    return: interpolation_result
        (2, m, n)的速度矩阵；
        m, n为插值的网格形状，即grid_shape[0], grid_shape[1]，
        2代表x,y方向的速度放在最后一维的两个通道 [0]是x [1]是y
    """

    # 生成网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(
            np.min(numpy_points[:, 0]),
            np.max(numpy_points[:, 0]),
            num=grid_shape[1],
        ),
        np.linspace(
            np.min(numpy_points[:, 1]),
            np.max(numpy_points[:, 1]),
            num=grid_shape[0],
        ),
    )
    # 进行 linear 插值
    interpolation_result_x_linear = griddata(
        numpy_points[:, :2],
        velocity_array[:, 0],
        (grid_x, grid_y),
        method="linear",
        rescale=True,
    )
    interpolation_result_y_linear = griddata(
        numpy_points[:, :2],
        velocity_array[:, 1],
        (grid_x, grid_y),
        method="linear",
        rescale=True,
    )

    # 如果存在nan值, 一般来说只有cell数据才会有nan值，point不会
    if np.isnan(interpolation_result_x_linear).any() or np.isnan(interpolation_result_y_linear).any():
        print("Warning: nan value detected in linear interpolation result.")
        # 进行 nearest 插值
        interpolation_result_x_nearest = griddata(
            numpy_points[:, :2],
            velocity_array[:, 0],
            (grid_x, grid_y),
            method="nearest",
            rescale=True,
        )
        interpolation_result_y_nearest = griddata(
            numpy_points[:, :2],
            velocity_array[:, 1],
            (grid_x, grid_y),
            method="nearest",
            rescale=True,
        )
        # 找到 linear 插值结果中的 nan 值
        nan_mask_x = np.isnan(interpolation_result_x_linear)
        nan_mask_y = np.isnan(interpolation_result_y_linear)
        # 用 nearest 插值结果替换 linear 插值结果中的 nan 值
        interpolation_result_x_linear[nan_mask_x] = interpolation_result_x_nearest[nan_mask_x]
        interpolation_result_y_linear[nan_mask_y] = interpolation_result_y_nearest[nan_mask_y]

    # 将两个速度矩阵合并，生成 shape 为 (grid_shape[0], grid_shape[1], 2) 的速度矩阵
    interpolation_result = np.stack(
        [interpolation_result_x_linear, interpolation_result_y_linear],
        axis=0,
    )
    return interpolation_result


def read_case_all_vtk_from_point(case_num: int, grid_shape=(300, 400)):
    """
    返回一个算例所有时间步的插值后的速度矩阵，格式类似图像，
        shape=(num_timestep, 2, 高, 宽)
    """
    file_list = get_read_file_path_list_point(case_num)
    data = []
    for file_path in tqdm(file_list):
        numpy_points, velocity_array = read_vtk_from_point(file_path)
        interpolation_result = point_data_interpolation(numpy_points, velocity_array, grid_shape=grid_shape)
        data.append(interpolation_result)
    data = np.array(data)
    return data


# ############################################################# #
#                      cell data processing                     #
# ############################################################# #


def read_vtu_from_cell(file_path):
    # 读取文件为vtk对象
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()
    # 获取网格单元数据
    cells = data.GetCells()
    cell_array = cells.GetData()
    numpy_cells = vtk_to_numpy(cell_array).reshape(-1, 4)  # 假设网格单元是四面体或四边形
    # 获取顶点坐标数据
    points = data.GetPoints()
    numpy_points = vtk_to_numpy(points.GetData())
    # 获取单元的速度
    velocity = data.GetCellData().GetArray("U")
    velocity_array = vtk_to_numpy(velocity)
    return numpy_cells, numpy_points, velocity_array


def cell_data_interpolation(numpy_cells, numpy_points, velocity_array, grid_shape=(300, 400)):
    """
    numpy_cells, numpy_points, velocity_array: 从read_vtu_from_cell函数中获取的数据
    grid_shape: 生成的网格形状(高度，宽度)
    return: 插值后的速度矩阵，shape为(2, grid_shape[0], grid_shape[1])
    """
    # 获取单元中心点的坐标
    cell_centers = np.mean(numpy_points[numpy_cells[:, 1:]], axis=1)
    # 调用 point_data_interpolation 进行插值
    interpolation_result = point_data_interpolation(cell_centers, velocity_array, grid_shape)
    return interpolation_result


def read_case_all_vtu_from_cell(case_num: int, grid_shape=(300, 400)):
    """
    返回一个算例所有时间步的插值后的速度矩阵，格式类似图像，
        shape=(num_timestep, 2, 高, 宽)
    """
    file_list = get_read_file_path_list_cell(case_num)
    data = []
    for file_path in tqdm(file_list):
        numpy_cells, numpy_points, velocity_array = read_vtu_from_cell(file_path)
        interpolation_result = cell_data_interpolation(numpy_cells, numpy_points, velocity_array, grid_shape=grid_shape)
        data.append(interpolation_result)
    data = np.array(data)
    return data



# ############################################################# #
#                              test                             #
# ############################################################# #

def test_read_vtk_from_point():
    
    # 读取一个vtk文件
    file_path = "/media/smy/16TB/ofrun/test/export_vtk_adm_3d/adm_0/adm_0.vtu"
    # 读取vtk数据
    numpy_points, velocity_array = read_vtk_from_point(file_path)
    # 插值
    interpolation_result = point_data_interpolation(numpy_points, velocity_array, (150,200))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # im = ax.imshow(mask, cmap='gray', origin='lower')
    im = ax.imshow(interpolation_result[0, ...], cmap="jet", origin="lower")
    # 设置坐标轴标签和标题
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", fontsize=12)
    
    plt.tight_layout()
    plt.show()


# to use these funcs, see 'process_vtk_to_np.py'
    