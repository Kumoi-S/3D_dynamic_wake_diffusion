import os
import glob
import shutil  # 导入 shutil 模块

# 是否真的删除文件, False 为测试模式，只打印要删除的文件
delete_files = True
# 针对单个算例的文件夹还是针对所有算例的文件夹
single_case = True

# 设置要删除的文件模式和路径
if single_case:
    base_path = "/media/smy/16TB/ofrun/test/example.ADM_270_11/"
    file_patterns = [
        "processor*/20*",
        "processor*/21*",
        "processor*/22*",
        "processor*/23*",
    ]
elif not single_case:
    base_path = "/media/smy/16TB/ofrun/test"
    file_patterns = [
        "example.ADM_270_*/processor*/20*",
        "example.ADM_270_*/processor*/21*",
        "example.ADM_270_*/processor*/22*",
        "example.ADM_270_*/processor*/23*",
    ]

# 查找所有匹配的文件
files_to_delete = []
for pattern in file_patterns:
    files_to_delete.extend(glob.glob(os.path.join(base_path, pattern)))

# 打印要删除的文件
print("要删除的文件：")
for file_path in files_to_delete:
    print(file_path)

# 将要删除的文件列表保存到文件
with open("./files_to_delete.txt", "w") as f:
    for file_path in files_to_delete:
        f.write(file_path + "\n")

print("要删除的文件列表已保存到 ./files_to_delete.txt")

# 根据 delete_files 变量决定是否删除文件
if delete_files:
    for file_path in files_to_delete:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"已删除文件：{file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 使用 shutil.rmtree() 删除目录
                print(f"已删除目录：{file_path}")
        except OSError as e:
            print(f"删除时出错：{file_path} - {e}")
else:
    print("未删除任何文件，请将 delete_files 设置为 True 以执行删除操作。")