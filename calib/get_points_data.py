import cv2
import glob
import numpy as np
import os


def calibrate_camera():
    # 设置棋盘格内角点数量和每个方格的真实尺寸
    pattern = (13, 11)  # 内角点数量 (列, 行)
    square = 25e-3  # 真实边长，单位米（可根据实际棋盘格尺寸修改）

    # 准备世界坐标系中的对象点 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros( (np.prod( pattern ), 3), np.float32 )
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape( -1, 2 ) * square

    # 存储对象点和图像点的列表
    obj_points = []  # 世界坐标系中的3D点
    img_points = []  # 图像坐标系中的2D点

    # 检查snapshots目录是否存在
    snapshots_dir = './snapshots'
    if not os.path.exists( snapshots_dir ):
        print( f"错误: 目录 '{snapshots_dir}' 不存在" )
        print( "请创建snapshots目录并放入标定图像" )
        return

    # 读取所有标定图像
    image_files = glob.glob( os.path.join( snapshots_dir, '*.jpg' ) ) + \
                  glob.glob( os.path.join( snapshots_dir, '*.png' ) ) + \
                  glob.glob( os.path.join( snapshots_dir, '*.jpeg' ) )

    print( f"找到 {len( image_files )} 张标定图像" )

    if len( image_files ) == 0:
        print( "错误: 在snapshots目录中没有找到任何图像文件" )
        print( "支持的格式: .jpg, .png, .jpeg" )
        return

    # 处理每张图像
    valid_images = 0
    for i, f in enumerate( image_files ):
        print( f"处理图像 {i + 1}/{len( image_files )}: {os.path.basename( f )}" )

        # 读取图像并转换为灰度图
        img = cv2.imread( f )
        if img is None:
            print( f"  警告: 无法读取图像 {f}" )
            continue

        gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners( gray, pattern, None )

        if ret:
            # 提高角点检测精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix( gray, corners, (11, 11), (-1, -1), criteria )

            # 添加对象点和图像点
            obj_points.append( objp )
            img_points.append( corners_refined )
            valid_images += 1

            # 可视化角点
            cv2.drawChessboardCorners( img, pattern, corners_refined, ret )
            cv2.imshow( 'Chessboard Corners', img )
            cv2.waitKey( 500 )  # 显示0.5秒

            print( f"  成功检测到棋盘格角点" )
        else:
            print( f"  警告: 未能在图像中找到棋盘格" )

    cv2.destroyAllWindows()

    # 检查是否有足够的图像进行标定
    print( f"\n成功检测到 {valid_images} 张有效的棋盘格图像" )

    if valid_images < 3:
        print( "错误: 需要至少3张有效的棋盘格图像进行标定" )
        print( "请确保:" )
        print( "1. 图像中有完整的棋盘格" )
        print( "2. 棋盘格大小设置为正确的内角点数量 (当前: 9x6)" )
        print( "3. 从不同角度拍摄足够多的图像" )
        return

    print( f"使用 {valid_images} 张图像进行相机标定..." )

    # 执行相机标定
    try:
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )

        # 计算重投影误差
        mean_error = 0
        for i in range( len( obj_points ) ):
            img_points2, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], K, D
            )
            error = cv2.norm( img_points[i], img_points2, cv2.NORM_L2 ) / len( img_points2 )
            mean_error += error

        print( f"平均重投影误差: {mean_error / len( obj_points ):.5f} 像素" )
        print( '相机内参矩阵 K =' )
        print( K.round( 2 ) )
        print( '畸变系数 D =', D.flatten()[:5].round( 4 ) )

        # 保存标定结果
        np.savez( 'camera_params.npz', K=K, D=D, DIM=gray.shape[::-1] )
        print( "标定结果已保存到 camera_params_custom.npz" )

        # 显示一张图像的畸变校正效果
        if len( image_files ) > 0:
            test_img = cv2.imread( image_files[0] )
            if test_img is not None:
                h, w = test_img.shape[:2]

                # 获取最优的新相机矩阵
                new_K, roi = cv2.getOptimalNewCameraMatrix( K, D, (w, h), 1, (w, h) )

                # 校正图像
                undistorted = cv2.undistort( test_img, K, D, None, new_K )

                # 裁剪图像
                x, y, w, h = roi
                undistorted = undistorted[y:y + h, x:x + w]

                # 显示结果
                cv2.imshow( '原始图像', test_img )
                cv2.imshow( '校正后的图像', undistorted )
                cv2.waitKey( 3000 )  # 显示3秒
                cv2.destroyAllWindows()

    except Exception as e:
        print( f"标定过程中发生错误: {e}" )
        print( "请检查图像质量和棋盘格设置" )


def check_snapshots_directory():
    """检查snapshots目录并提供使用指南"""
    snapshots_dir = './snapshots'

    if not os.path.exists( snapshots_dir ):
        print( "=" * 50 )
        print( "使用指南:" )
        print( "1. 创建名为 'snapshots' 的目录" )
        print( "2. 准备一个棋盘格 (9x6 内角点)" )
        print( "3. 从不同角度拍摄10-20张棋盘格照片" )
        print( "4. 将照片放入 snapshots 目录" )
        print( "5. 重新运行此程序" )
        print( "=" * 50 )

        # 询问是否创建目录
        create = input( "是否要自动创建snapshots目录? (y/n): " )
        if create.lower() == 'y':
            os.makedirs( snapshots_dir )
            print( "已创建snapshots目录，请放入标定图像后重新运行程序" )
        return False
    return True


if __name__ == "__main__":
    print( "相机标定程序" )
    print( "=" * 30 )

    if check_snapshots_directory():
        calibrate_camera()