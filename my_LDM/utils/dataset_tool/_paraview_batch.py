# import the simple module from paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# 1. 读取并处理数据
openfoam = OpenFOAMReader(registrationName='open.foam', FileName='/home/smy/ofrun/test/example.ADM_270_1/open.foam')
openfoam.MeshRegions = ['internalMesh']
openfoam.CellArrays = ['U']

# 合并数据块
mergeBlocks1 = MergeBlocks(registrationName='MergeBlocks1', Input=openfoam)

# 切片操作
slice1 = Slice(registrationName='Slice1', Input=mergeBlocks1)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = [2000.0, 1500.0, 90.0]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# 保存切片数据
SaveData('/home/smy/ofrun/test/export_vtk_adm_3d/adm_1/adm_1.vtk', proxy=slice1, ChooseArraysToWrite=1,
    PointDataArrays=['U'],
    CellDataArrays=['U'],
    Writetimestepsasfileseries=1,
    Lasttimestep=2)

# 2. 删除所有对象以释放内存

# 销毁 slice1
Delete(slice1)
del slice1

# 销毁 mergeBlocks1
Delete(mergeBlocks1)
del mergeBlocks1

# 销毁 openfoam
Delete(openfoam)
del openfoam

# 最后，显式调用垃圾回收（可选）
import gc
gc.collect()