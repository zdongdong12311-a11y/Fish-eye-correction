import cv2
import os
from pathlib import Path


def take_photos_with_keypress(save_path="C:\pythonpro\opencv\calib\snapshots"):
    """
    使用摄像头，按空格键拍照，按ESC键退出
    照片按1,2,3...顺序命名

    参数:
    save_path: 图片保存路径，默认为/home/lwx/pictures
    """
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return False

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 创建照片目录（如果不存在）
    photos_dir = Path(save_path)
    if not photos_dir.exists():
        photos_dir.mkdir(parents=True, exist_ok=True)
        print(f"已创建目录: {photos_dir}")

    # 获取已存在的照片数量，用于确定起始编号
    existing_photos = [f for f in os.listdir(photos_dir) if f.startswith("photo_") and f.endswith(".jpg")]
    photo_count = len(existing_photos)

    print("拍照说明:")
    print("- 按 SPACE(空格键) 拍照")
    print("- 按 ESC键 退出程序")
    print(f"照片保存位置: {photos_dir}")
    print(f"下一张照片编号: {photo_count + 1}")

    while True:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            print("无法读取摄像头画面")
            break

        # 显示实时画面
        cv2.imshow('Camera - Press SPACE to take photo, ESC to exit', frame)

        # 检测按键
        key = cv2.waitKey(1) & 0xFF

        # 按ESC键退出
        if key == 27:  # ESC键
            break
        # 按空格键拍照
        elif key == 32:  # 空格键
            photo_count += 1
            filename = f"photo_{photo_count}.jpg"
            filepath = photos_dir / filename

            # 保存图片
            cv2.imwrite(str(filepath), frame)
            print(f"照片已保存: {filepath}")

            # 显示刚刚拍摄的照片（短暂显示）
            cv2.imshow('Captured Photo', frame)
            cv2.waitKey(500)  # 显示0.5秒
            cv2.destroyWindow('Captured Photo')

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")
    return True


if __name__ == "__main__":
    take_photos_with_keypress()