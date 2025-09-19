import numpy as np


def create_from_known_params(fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height):
    """
    从已知参数创建 camera_params.npz 文件

    参数:
    fx, fy: 焦距
    cx, cy: 主点坐标
    k1, k2, p1, p2, k3: 畸变系数
    width, height: 图像尺寸
    """
    # 内参矩阵
    K = np.array( [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32 )

    # 畸变系数
    D = np.array( [k1, k2, p1, p2, k3], dtype=np.float32 )

    # 图像尺寸
    DIM = np.array( [width, height], dtype=np.int32 )

    # 保存到文件
    np.savez( 'camera_params_custom.npz', K=K, D=D, DIM=DIM )

    print( "自定义参数文件已创建" )
    print( f"K:\n{K}" )
    print( f"D: {D}" )
    print( f"DIM: {DIM}" )


# 示例：使用常见参数
create_from_known_params(
    fx=400.0, fy=400.0,  # 焦距
    cx=540.0, cy=360.0,  # 主点
    k1=-0.55, k2=0.25,  # 径向畸变
    p1=-0.01, p2=0.002,  # 切向畸变
    k3=-0.05,  # 径向畸变
    width=1080, height=720  # 图像尺寸
)