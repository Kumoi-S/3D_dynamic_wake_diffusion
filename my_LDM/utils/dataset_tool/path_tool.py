import os
from pathlib import Path
from natsort import natsorted


# ================== 返回根目录路径 ==================
def get_root_path(root_dir_name: str = "diffusion_wake"):
    """
    获取指定名称的根目录路径。
    Args:
        root_dir_name (str, optional): 根目录的名称. Defaults to "repo_my_LDM".
    Returns:
        Path or None: 如果找到根目录，则返回其路径；否则返回 None。
    """
    # 获取当前工作目录的绝对路径
    current_path = Path.cwd()
    # 首先检查当前目录是否就是目标目录
    if current_path.name == root_dir_name:
        return current_path
    # 如果当前目录不是目标目录，则查找父目录
    for parent in current_path.parents:
        print(parent.name)
        if parent.name == root_dir_name:
            return parent
    # 如果没有找到，则返回 None
    print(f"Cannot find the root directory, current path: {current_path}")
    return current_path


# ================== 读取文件路径 ==================
# def get_read_file_path_list_cell(case_num: int, case_path: str = "/media/smy/16TB/ofrun/test/export_vtk_adm_cell/"):
#     """
#     case_num: 1-n, 对应adm_1_vtk到adm_n_vtk中的文件
#     """
#     # 生成文件夹列表
#     file_folder_list = [case_path + "/adm_" + str(i) + "/" for i in range(0, 9)]
#     # print(f"all case: {file_folder_list}")
#     # print(f"currently reading case: {file_folder_list[case_num - 1]}")
#     # 生成文件路径列表
#     return [file_folder_list[case_num - 1] + "/adm_" + str(case_num) + "_" + str(i) + ".vtu" for i in range(0, 401)]


# def get_read_file_path_list_point(case_num: int, case_path: str = "/media/smy/16TB/ofrun/test/export_vtk_adm_point/"):
#     """
#     case_num: 1-n, 对应adm_1_vtk到adm_n_vtk中的文件
#     """
#     # 生成文件夹列表
#     file_folder_list = [case_path + "/adm_" + str(i) + "/" for i in range(0, 9)]
#     # 生成文件路径列表
#     return [file_folder_list[case_num] + "/adm_" + str(case_num) + "_" + str(i) + ".vtk" for i in range(0, 401)]


def get_read_file_path_list_cell(case_num: int, case_path: str = "/media/smy/16TB/ofrun/test/export_vtk_adm_cell/"):
    return get_read_file_path_list(case_num, case_path)

def get_read_file_path_list_point(case_num: int, case_path: str = "/media/smy/16TB/ofrun/test/export_vtk_adm_point/"):
    return get_read_file_path_list(case_num, case_path)

def get_read_file_path_list_3d(case_num: int, case_path: str = "/media/smy/16TB/ofrun/test/export_vtk_adm_3d/"):
    return get_read_file_path_list(case_num, case_path)


def get_read_file_path_list(case_num: int, case_path: str):
    """
    获取指定案例文件夹中的所有文件路径

    case_num: 1-n|0, 对应adm_1到adm_n中的文件
    case_path: 存储案例文件夹的路径
    """
    # 构造目标文件夹路径
    case_folder = os.path.join(case_path, f"adm_{case_num}")

    # 获取文件夹中的所有文件并构建完整路径
    file_list = sorted([os.path.join(case_folder, f) for f in os.listdir(case_folder)])
    file_list = natsorted(file_list)

    return file_list