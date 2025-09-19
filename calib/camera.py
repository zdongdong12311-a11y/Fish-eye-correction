import cv2
import numpy as np
import argparse
import time


class CameraUndistorter:
    def __init__(self, params_path='camera_params.npz'):
        """
        初始化摄像头去畸变器
        """
        self.params_path = params_path
        self.K = None
        self.D = None
        self.DIM = None
        self.map1 = None
        self.map2 = None

        self.load_camera_params()
        self.precompute_maps()

    def load_camera_params(self):
        """加载相机参数"""
        try:
            data = np.load( self.params_path )
            self.K = data['K'].astype( np.float32 )
            self.D = data['D'].astype( np.float32 )
            # 确保DIM是整数
            self.DIM = (int( data['DIM'][0] ), int( data['DIM'][1] ))
            print( "✓ 成功加载相机参数" )
            print( f"  图像尺寸: {self.DIM}" )
            print( f"  内参矩阵 K:\n{self.K}" )
            print( f"  畸变系数 D: {self.D.flatten()}" )
        except Exception as e:
            print( f"✗ 加载相机参数失败: {e}" )
            raise

    def precompute_maps(self):
        """预计算去畸变映射表"""
        try:
            # 使用更简单的方法
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.K, self.DIM, cv2.CV_16SC2
            )
            print( "✓ 去畸变映射表预计算完成" )
        except Exception as e:
            print( f"✗ 映射表计算失败: {e}" )
            raise

    def safe_undistort(self, frame):
        """安全的去畸变处理，包含错误检查"""
        if frame is None or frame.size == 0:
            return None, None

        try:
            # 调整图像尺寸
            if frame.shape[1] != self.DIM[0] or frame.shape[0] != self.DIM[1]:
                frame_resized = cv2.resize( frame, self.DIM )
            else:
                frame_resized = frame

            # 确保图像数据类型正确
            if frame_resized.dtype != np.uint8:
                frame_resized = frame_resized.astype( np.uint8 )

            # 应用去畸变映射
            undistorted = cv2.remap(
                frame_resized, self.map1, self.map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

            return undistorted, frame_resized

        except Exception as e:
            print( f"去畸变处理错误: {e}" )
            return None, None

    def find_working_camera(self):
        """查找可用的摄像头"""
        print( "正在检测可用摄像头..." )
        for i in range( 0, 5 ):  # 尝试0-4号摄像头
            cap = cv2.VideoCapture( i, cv2.CAP_DSHOW )  # 使用DirectShow后端
            if cap.isOpened():
                # 测试读取一帧
                ret, frame = cap.read()
                if ret and frame is not None:
                    print( f"✓ 找到可用摄像头: {i}, 分辨率: {frame.shape[1]}x{frame.shape[0]}" )
                    cap.release()
                    return i
                cap.release()
            time.sleep( 0.1 )

        print( "✗ 没有找到可用的摄像头" )
        return None

    def run_realtime_undistort(self):
        """
        实时去畸变主函数
        """
        # 查找可用的摄像头
        camera_index = self.find_working_camera()
        if camera_index is None:
            return

        # 打开摄像头
        cap = cv2.VideoCapture( camera_index, cv2.CAP_DSHOW )
        if not cap.isOpened():
            print( f"✗ 无法打开摄像头 {camera_index}" )
            return

        # 设置较低的分辨率以提高兼容性
        cap.set( cv2.CAP_PROP_FRAME_WIDTH, 1080 )
        cap.set( cv2.CAP_PROP_FRAME_HEIGHT, 720 )
        cap.set( cv2.CAP_PROP_FPS, 30 )

        print( "\n🎥 摄像头实时去畸变已启动" )
        print( "=" * 40 )
        print( "快捷键: q-退出, s-保存当前帧" )
        print( "=" * 40 )

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print( "✗ 无法读取帧" )
                    time.sleep( 0.1 )
                    continue

                # 去畸变处理
                undistorted, processed_frame = self.safe_undistort( frame )

                if undistorted is None:
                    continue

                # 计算FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    print( f"FPS: {fps:.1f}" )
                    frame_count = 0
                    start_time = time.time()

                # 显示图像
                cv2.imshow( 'Original Camera', processed_frame )
                cv2.imshow( 'Undistorted View', undistorted )

                # 键盘控制
                key = cv2.waitKey( 1 ) & 0xFF
                if key == ord( 'q' ):
                    break
                elif key == ord( 's' ):
                    # 保存图像
                    timestamp = time.strftime( "%Y%m%d_%H%M%S" )
                    cv2.imwrite( f'original_{timestamp}.jpg', processed_frame )
                    cv2.imwrite( f'undistorted_{timestamp}.jpg', undistorted )
                    print( "✓ 图像已保存" )

        except Exception as e:
            print( f"处理过程中出错: {e}" )

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print( "程序已退出" )


def main():
    """主函数"""
    parser = argparse.ArgumentParser( description='摄像头实时去畸变' )
    parser.add_argument( '--params', type=str, default='camera_params_custom.npz',
                         help='相机参数文件路径 (默认: camera_params.npz)' )

    args = parser.parse_args()

    try:
        print( "正在初始化摄像头去畸变器..." )
        undistorter = CameraUndistorter( args.params )
        undistorter.run_realtime_undistort()

    except Exception as e:
        print( f"程序运行出错: {e}" )
        print( "请检查相机参数文件是否正确" )


if __name__ == "__main__":
    main()