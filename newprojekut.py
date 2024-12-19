import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque

arrow_positions = deque(maxlen=5)  # 矢印の過去位置を保存（最大5フレーム分）

# 日本語フォントのパス（必要に応じて変更）
FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"

def draw_japanese_text(image, text, position, font_size=24, color=(255, 255, 255)):
    """
    日本語テキストを画像に描画する関数
    """
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def perspective_transform(frame):
    """
    透視変換を適用して画像の歪みを補正する関数
    """
    h, w = frame.shape[:2]
    
    # 透視変換の4点を設定（例: 画像の角を指定）
    # 左上、右上、右下、左下の順に4点を選択
    pts1 = np.float32([[100, 100], [w - 100, 100], [w - 100, h - 100], [100, h - 100]])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # 透視変換行列の計算
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 透視変換の適用
    result = cv2.warpPerspective(frame, matrix, (w, h))
    
    return result

def process_frame(frame):
    """
    入力映像フレームを処理し、必要な指示や表示を更新。
    """
    global arrow_positions

    # グレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 白線検出（Canny Edge Detection + Hough Line Transform）
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    h, w, _ = frame.shape
    center_x = w // 2
    center_arrow = (center_x, h - 50)

    # 矢印の描画位置
    arrow_dx = 0

    if lines is not None:
        line_centers = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_center = (x1 + x2) // 2
            line_centers.append(line_center)
        
        if line_centers:
            avg_center = int(np.mean(line_centers))
            arrow_dx = avg_center - center_x

    # 矢印の移動をスムーズにする
    arrow_positions.append(arrow_dx)
    smoothed_arrow_dx = int(np.mean(arrow_positions))

    # 矢印を描画
    arrow_tip = (center_x + smoothed_arrow_dx, h - 100)
    cv2.arrowedLine(frame, center_arrow, arrow_tip, (0, 255, 0), 5, tipLength=0.3)

    # 色の帯検出（赤、青、緑）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    blue_mask = cv2.inRange(hsv, np.array([110, 150, 50]), np.array([130, 255, 255]))
    
    # 黄色の代わりに緑を検出
    green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))

    # ピクセル数（色の塊の量）を計算
    red_area = np.sum(red_mask > 0)
    blue_area = np.sum(blue_mask > 0)
    green_area = np.sum(green_mask > 0)

    # 色の塊を左上に表示（各色で表示）
    frame = draw_japanese_text(frame, f"赤: {red_area}", (10, 10), font_size=24, color=(0, 0, 255))
    frame = draw_japanese_text(frame, f"青: {blue_area}", (10, 40), font_size=24, color=(255, 0, 0))
    frame = draw_japanese_text(frame, f"緑: {green_area}", (10, 70), font_size=24, color=(0, 255, 0))

    # 真ん中上に指示を表示
    if red_area > 5000:
        frame = draw_japanese_text(frame, "右折してください", (center_x - 100, 50), font_size=32, color=(0, 0, 255))
    elif blue_area > 5000:
        frame = draw_japanese_text(frame, "左折してください", (center_x - 100, 50), font_size=32, color=(255, 0, 0))
    elif green_area > 5000:
        frame = draw_japanese_text(frame, "停止してください", (center_x - 100, 50), font_size=32, color=(0, 255, 0))

    # 二値化処理
    _, binary_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 透視変換を適用
    transformed_frame = perspective_transform(binary_frame)

    # 透視変換後の画像を表示するための新しいウィンドウを作成
    cv2.imshow("Transformed Perspective", transformed_frame)

    return frame

def main():
    cap = cv2.VideoCapture(0)  # Webカメラ映像取得
    if not cap.isOpened():
        print("カメラが開けませんでした")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("映像を取得できませんでした")
            break

        processed_frame = process_frame(frame)
        cv2.imshow('Processed Video', processed_frame)
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
