import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, PoseLandmarksConnections,
    HandLandmarker, HandLandmarkerOptions, HandLandmarksConnections,
    FaceLandmarker, FaceLandmarkerOptions, FaceLandmarksConnections,
    drawing_utils, RunningMode,
)
from emotion_gpu import EmotiEffLibRecognizerOnnxGPU
import time
import os
import numpy as np


# 指数移动平均平滑器，让检测结果更稳定
class EMASmoother:
    """用历史帧的数据来平滑当前帧，减少抖动"""
    def __init__(self, alpha=0.4, max_lost_frames=15):
        self.alpha = alpha  # 平滑系数，越大越灵敏
        self.max_lost_frames = max_lost_frames  # 丢失多少帧后重置
        self.smoothed = None  # 存储平滑后的结果
        self.lost_count = 0  # 连续丢失帧数

    def update(self, landmarks):
        if landmarks:
            self.lost_count = 0
            if self.smoothed is None or len(self.smoothed) != len(landmarks):
                self.smoothed = landmarks
            else:
                # 新坐标 = alpha * 当前帧 + (1-alpha) * 历史值
                for i in range(len(landmarks)):
                    self.smoothed[i].x = (
                        self.alpha * landmarks[i].x + (1 - self.alpha) * self.smoothed[i].x)
                    self.smoothed[i].y = (
                        self.alpha * landmarks[i].y + (1 - self.alpha) * self.smoothed[i].y)
                    self.smoothed[i].z = (
                        self.alpha * landmarks[i].z + (1 - self.alpha) * self.smoothed[i].z)
            return self.smoothed
        else:
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                self.smoothed = None
            return self.smoothed


# 8种情绪对应的显示颜色 (BGR格式)
_EMOTION_COLORS_8 = {
    "Anger":      (0, 0, 255),      # 红色
    "Contempt":   (120, 120, 120),  # 灰色
    "Disgust":    (0, 150, 0),      # 绿色
    "Fear":       (150, 150, 255),  # 浅蓝
    "Happiness":  (0, 255, 255),    # 黄色
    "Neutral":    (180, 180, 180),  # 浅灰
    "Sadness":    (255, 150, 50),   # 橙色
    "Surprise":   (255, 100, 255),  # 粉色
}


# 基于 blendshape 分数做情绪分类
def classify_emotion(blendshapes):
    """根据面部 blendshape 值判断当前表情"""
    s = {}
    for bs in blendshapes:
        s[bs.category_name] = bs.score

    def avg(*keys):
        return sum(s.get(k, 0) for k in keys) / len(keys)

    # 每种情绪的计算公式
    scores = {
        "Happiness": avg("mouthSmileLeft","mouthSmileRight")
               + avg("cheekSquintLeft","cheekSquintRight")*0.5,
        "Sadness": avg("browInnerUp")
             + avg("mouthFrownLeft","mouthFrownRight")*0.7
             - s.get("mouthSmileLeft",0)-s.get("mouthSmileRight",0),
        "Surprise": avg("browInnerUp")+avg("eyeWideLeft","eyeWideRight")*0.5
                  + s.get("jawOpen",0),
        "Anger": avg("browDownLeft","browDownRight")
               + avg("eyeSquintLeft","eyeSquintRight")*0.3
               + avg("mouthFrownLeft","mouthFrownRight")*0.3,
        "Fear": avg("browInnerUp")*0.5
              + avg("eyeWideLeft","eyeWideRight")*0.5
              + avg("mouthStretchLeft","mouthStretchRight")*0.3,
        "Disgust": avg("noseSneerLeft","noseSneerRight")
                 + avg("mouthUpperUpLeft","mouthUpperUpRight")*0.5
                 + s.get("browDownLeft",0)*0.3,
        "Neutral": 1.0-s.get("_neutral",0)*0.3,
    }

    best = max(scores, key=scores.get)
    best_score = scores[best]
    # 得分太低就认为是中性
    if best_score < 0.35:
        return "Neutral", best_score
    return best, best_score


# 计算人脸边界框
def get_face_bbox(face_landmarks, frame_w, frame_h, margin=0.35):
    """根据人脸关键点坐标算出矩形框，带一点边距"""
    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw, bh = x_max - x_min, y_max - y_min
    x_min = max(0, x_min - bw * margin)
    x_max = min(1, x_max + bw * margin)
    y_min = max(0, y_min - bh * margin)
    y_max = min(1, y_max + bh * margin)

    return (int(x_min * frame_w), int(y_min * frame_h),
            int((x_max - x_min) * frame_w), int((y_max - y_min) * frame_h))


# 绘制进度条
def draw_progress_bar(img, x, y, w, h, value, color, label=""):
    """画一个水平进度条，用来显示某种情绪的置信度"""
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    bar_w = int(w * min(value, 1.0))
    if bar_w > 0:
        cv2.rectangle(img, (x, y), (x + bar_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), 1)
    if label:
        cv2.putText(img, label, (x + 4, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


class ScaledLandmark:
    """把原始关键点坐标映射到裁剪后的人脸区域"""
    def __init__(self, lm, ox, oy, sx, sy, img_w, img_h):
        self.x = (lm.x * img_w - ox) / sx
        self.y = (lm.y * img_h - oy) / sy
        self.z = lm.z
        self.visibility = getattr(lm, "visibility", 1.0) or 1.0
        self.presence = getattr(lm, "presence", 1.0) or 1.0


# 每种情绪对应需要高亮的面部区域
EMOTION_FOCUS = {
    "Happiness": ["LIPS"],
    "Sadness":   ["EYEBROWS", "LIPS"],
    "Surprise":  ["EYES", "LIPS"],
    "Anger":     ["EYEBROWS", "EYES"],
    "Fear":      ["EYES"],
    "Disgust":   ["EYEBROWS", "LIPS"],
    "Neutral":   [],
    "Contempt":  ["LIPS"],
}

# 面部区域对应的关键点连接
_CONTOUR_MAP = {
    "EYES":     [FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
                 FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE],
    "EYEBROWS": [FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
                 FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW],
    "LIPS":     [FaceLandmarksConnections.FACE_LANDMARKS_LIPS],
}


def draw_emotion_focus(img, landmark_list, emotion):
    """根据当前情绪，用浅红色标出相关的面部区域"""
    focus = EMOTION_FOCUS.get(emotion, [])
    if not focus:
        return
    red = drawing_utils.DrawingSpec(color=(80, 80, 240), thickness=1, circle_radius=0)
    nodot = drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)
    for key in focus:
        for conn_list in _CONTOUR_MAP.get(key, []):
            drawing_utils.draw_landmarks(
                image=img, landmark_list=landmark_list, connections=conn_list,
                landmark_drawing_spec=nodot, connection_drawing_spec=red)


# 绘制全身骨架
def draw_pose_full(img, landmarks, w, h,
                   lm_color=(0, 255, 0), cn_color=(0, 0, 255),
                   thickness=2, circle_r=3, skip_face=False):
    """画出 33 个身体关键点和骨架连线，skip_face=True 时不画面部"""
    _FACE_IDS = {0,1,2,3,4,5,6,7,8,9,10}
    connections = PoseLandmarksConnections.POSE_LANDMARKS
    pts = {}
    for i, lm in enumerate(landmarks):
        if skip_face and i in _FACE_IDS:
            continue
        px, py = int(lm.x * w), int(lm.y * h)
        pts[i] = (px, py)
        cv2.circle(img, (px, py), circle_r, lm_color, -1)

    for conn in connections:
        a, b = conn.start, conn.end
        if skip_face and (a in _FACE_IDS or b in _FACE_IDS):
            continue
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], cn_color, thickness)


# ==================== 主程序入口 ====================
def main():
    # ----- 设置模型文件路径 -----
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    mp_dir = os.path.join(models_dir, "mediapipe")
    emotion_dir = os.path.join(models_dir, "emotion")
    pose_full = os.path.join(mp_dir, "pose_landmarker_full.task")
    pose_lite = os.path.join(mp_dir, "pose_landmarker_lite.task")
    pose_model = pose_full if os.path.exists(pose_full) else pose_lite
    hand_model = os.path.join(mp_dir, "hand_landmarker.task")
    face_model = os.path.join(mp_dir, "face_landmarker.task")
    print(f"姿态模型: {'full' if os.path.exists(pose_full) else 'lite'}")

    for path in [pose_model, hand_model, face_model]:
        if not os.path.exists(path):
            print(f"错误: 模型文件未找到: {path}")
            return

    # ----- 存放异步推理结果的变量 -----
    pose_result = None
    hand_result = None
    face_result = None

    def on_pose_result(result, _img, _ts):
        nonlocal pose_result
        pose_result = result

    def on_hand_result(result, _img, _ts):
        nonlocal hand_result
        hand_result = result

    def on_face_result(result, _img, _ts):
        nonlocal face_result
        face_result = result

    # ----- 创建 MediaPipe 检测器 (异步模式) -----
    pose_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model),
        running_mode=RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        result_callback=on_pose_result,
    ))
    hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        result_callback=on_hand_result,
    ))
    face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model),
        running_mode=RunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_face_blendshapes=True,
        result_callback=on_face_result,
    ))

    # ----- 加载深度学习表情识别模型 -----
    trained_onnx = os.path.join(emotion_dir, "enet_b0_7_fer2013.onnx")
    trained_weights = os.path.join(emotion_dir, "enet_b0_7_fer2013_weights.npz")
    if os.path.exists(trained_onnx) and os.path.exists(trained_weights):
        rec = EmotiEffLibRecognizerOnnxGPU(
            custom_onnx=trained_onnx, custom_weights=trained_weights)
        recognizers = [rec]
        print("情绪识别器: 自定义训练模型 (FER2013, 7 类)")
    else:
        recognizers = [
            EmotiEffLibRecognizerOnnxGPU(model_name="enet_b0_8_best_vgaf"),
            EmotiEffLibRecognizerOnnxGPU(model_name="enet_b0_8_best_afew"),
        ]
        print("情绪识别器: VGAF + AFEW 双模型集成")
    EMOTION_IDX = recognizers[0].idx_to_emotion_class
    EMOTION_LABELS = [EMOTION_IDX[i] for i in sorted(EMOTION_IDX.keys())]

    # ----- 绘制样式设置 -----
    left_lm = drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
    left_cn = drawing_utils.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=1)
    right_lm = drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
    right_cn = drawing_utils.DrawingSpec(color=(255, 0, 140), thickness=2, circle_radius=1)
    face_tess = drawing_utils.DrawingSpec(color=(220, 215, 235), thickness=1, circle_radius=1)

    # ----- 创建平滑器 -----
    pose_smoother = EMASmoother(alpha=0.5, max_lost_frames=25)
    face_smoother = EMASmoother(alpha=0.5, max_lost_frames=15)

    # ----- 打开摄像头 -----
    cap = None
    for idx in [1, 0]:
        try:
            cap_test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap_test.isOpened():
                cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                cam_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
                cam_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"摄像头 {idx}: {cam_w}x{cam_h}")
                for _ in range(5):
                    cap_test.read()  # 跳过前几帧，让摄像头预热
                _, test_frame = cap_test.read()
                if test_frame is not None:
                    cap = cap_test
                    break
            cap_test.release()
        except Exception:
            continue

    if cap is None:
        print("错误: 没有可用的摄像头。")
        pose_landmarker.close(); hand_landmarker.close(); face_landmarker.close()
        return

    # 设置四个面板的大小（2×2 布局）
    panel_w, panel_h = 480, 360
    print(f"面板尺寸: {panel_w}x{panel_h}, 画布: {panel_w*2}x{panel_h*2}")
    print("摄像头就绪。2×2 网格模式 (异步推理)。按 'q' 退出。")

    prev_time = time.time()
    frame_timestamp_ms = 0
    frame_count = 0

    # 情绪识别相关的状态变量
    dl_emotion_cached = None       # 当前确认的情绪
    dl_scores_cached = None        # 当前情绪分数
    dl_scores_ema = None           # EMA 平滑后的分数
    dl_ema_alpha = 0.35            # EMA 平滑系数
    dl_candidate_label = None      # 候选情绪（用于滞回判断）
    dl_candidate_streak = 0        # 候选情绪连续出现次数
    dl_hysteresis_frames = 2       # 滞回帧数，防止频繁切换

    # 创建大画布，包含四个小面板
    canvas = np.zeros((panel_h * 2, panel_w * 2, 3), dtype=np.uint8)
    panel1 = canvas[0:panel_h, 0:panel_w]                       # 左上：原始画面
    panel2 = canvas[0:panel_h, panel_w:panel_w*2]               # 右上：表情分析
    panel3 = canvas[panel_h:panel_h*2, 0:panel_w]               # 左下：身体骨架
    panel4 = canvas[panel_h:panel_h*2, panel_w:panel_w*2]       # 右下：综合视图
    pose_overlay = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    cam_fail = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cam_fail += 1
            if cam_fail > 30:
                print("摄像头连续失败, 退出")
                break
            continue
        cam_fail = 0

        # 把摄像头帧发给三个检测器，异步处理
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        pose_landmarker.detect_async(mp_image, frame_timestamp_ms)
        hand_landmarker.detect_async(mp_image, frame_timestamp_ms)
        face_landmarker.detect_async(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33

        # 取上一帧的检测结果
        cur_pose = pose_result
        cur_hand = hand_result
        cur_face = face_result

        # 对姿态和人脸做平滑处理
        smoothed_pose = None
        if cur_pose and cur_pose.pose_landmarks:
            smoothed_pose = pose_smoother.update(cur_pose.pose_landmarks[0])
        else:
            smoothed_pose = pose_smoother.update([])

        smoothed_face = None
        if cur_face and cur_face.face_landmarks:
            smoothed_face = face_smoother.update(cur_face.face_landmarks[0])
        else:
            smoothed_face = face_smoother.update([])

        # blendshape 情绪分类
        emotion_label = "Neutral"
        emotion_score = 0.0
        face_blends_raw = None
        if cur_face and cur_face.face_blendshapes:
            face_blends_raw = cur_face.face_blendshapes[0]
            emotion_label, emotion_score = classify_emotion(face_blends_raw)

        # ==================== 构建四个面板 ====================
        panel2.fill(0)

        # ---- 面板 1: 原始画面 ----
        cv2.resize(frame, (panel_w, panel_h), dst=panel1)
        cv2.putText(panel1, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if smoothed_face and face_blends_raw:
            fx, fy, fw, fh = get_face_bbox(smoothed_face, w, h, margin=0.35)
            fx, fy = max(0, fx), max(0, fy)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)

            if fw > 0 and fh > 0:
                face_crop = frame[fy:fy+fh, fx:fx+fw].copy()

                # 每 2 帧跑一次深度学习情绪推理（含水平翻转增强）
                if frame_count % 2 == 0:
                    try:
                        flipped = cv2.flip(face_crop, 1)
                        all_scores = []
                        for rec in recognizers:
                            _, s1 = rec.predict_emotions(face_crop, False)
                            _, s2 = rec.predict_emotions(flipped, False)
                            all_scores.append((s1[0] + s2[0]) * 0.5)
                        scores = sum(all_scores) / len(all_scores)

                        # 用 EMA 平滑分数
                        if dl_scores_ema is None or len(dl_scores_ema) != len(scores):
                            dl_scores_ema = scores
                        else:
                            dl_scores_ema = dl_ema_alpha * scores + (1 - dl_ema_alpha) * dl_scores_ema

                        # 滞回逻辑：新情绪需连续出现几帧才切换
                        best_i = int(np.argmax(dl_scores_ema))
                        proposed = EMOTION_LABELS[best_i]
                        if dl_emotion_cached is None:
                            dl_emotion_cached = proposed
                        elif proposed == dl_emotion_cached:
                            dl_candidate_label = None; dl_candidate_streak = 0
                        elif proposed == dl_candidate_label:
                            dl_candidate_streak += 1
                            if dl_candidate_streak >= dl_hysteresis_frames:
                                dl_emotion_cached = proposed
                                dl_candidate_label = None; dl_candidate_streak = 0
                        else:
                            dl_candidate_label = proposed; dl_candidate_streak = 1
                        dl_scores_cached = dl_scores_ema
                    except Exception:
                        pass

                # 把人脸关键点坐标映射到裁剪区域
                scaled = [ScaledLandmark(lm, fx, fy, fw, fh, w, h) for lm in smoothed_face]

                # 画人脸网格
                drawing_utils.draw_landmarks(
                    image=face_crop, landmark_list=scaled,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=face_tess,
                    connection_drawing_spec=face_tess,
                )
                # 画情绪相关区域高亮
                draw_emotion_focus(face_crop, scaled, dl_emotion_cached or "Neutral")

                cv2.resize(face_crop, (panel_w, panel_h), dst=panel2)

            # 在面板2顶部显示情绪识别结果
            cv2.rectangle(panel2, (0, 0), (panel_w, 70), (0, 0, 0), -1)
            if dl_emotion_cached is not None and dl_scores_cached is not None:
                best_idx = int(np.argmax(dl_scores_cached))
                best_score_dl = float(dl_scores_cached[best_idx])
                cv2.putText(panel2, f"DL: {dl_emotion_cached} ({best_score_dl:.0%})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(panel2, f"BS: {emotion_label} ({emotion_score:.0%})",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            # 在面板2右侧显示各情绪的置信度条
            bar_x = panel_w - 180
            n_emotions = len(EMOTION_LABELS)
            bar_h = min(20, (panel_h - 80) // n_emotions - 4)
            bar_y_start = 80
            if dl_scores_cached is not None:
                for i, name in enumerate(EMOTION_LABELS):
                    val = float(dl_scores_cached[i])
                    color = _EMOTION_COLORS_8.get(name, (180, 180, 180))
                    draw_progress_bar(panel2, bar_x, bar_y_start + i * (bar_h + 4),
                                      160, bar_h, val, color, name)
            else:
                s = {bs.category_name: bs.score for bs in face_blends_raw}
                emotion_scores_map = {
                    "Happiness": avg2(s, "mouthSmileLeft", "mouthSmileRight"),
                    "Sadness":   max(s.get("browInnerUp", 0), avg2(s, "mouthFrownLeft", "mouthFrownRight")),
                    "Surprise":  max(s.get("browInnerUp", 0), s.get("jawOpen", 0)),
                    "Anger":     max(avg2(s, "browDownLeft", "browDownRight"),
                                     avg2(s, "mouthFrownLeft", "mouthFrownRight")),
                    "Fear":      max(s.get("browInnerUp", 0), s.get("jawOpen", 0)) * 0.8,
                    "Disgust":   max(avg2(s, "noseSneerLeft", "noseSneerRight"),
                                     avg2(s, "mouthUpperUpLeft", "mouthUpperUpRight")),
                    "Neutral":   s.get("_neutral", 0),
                    "Contempt":  s.get("mouthPressLeft", 0) * 0.5,
                }
                for i, name in enumerate(EMOTION_LABELS):
                    val = emotion_scores_map.get(name, 0)
                    color = _EMOTION_COLORS_8.get(name, (180, 180, 180))
                    draw_progress_bar(panel2, bar_x, bar_y_start + i * (bar_h + 4),
                                      160, bar_h, val, color, name)
        else:
            cv2.putText(panel2, "No face detected", (panel_w // 2 - 100, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 2)

        cv2.putText(panel2, "Expression", (10, panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # ---- 面板 3: 身体骨架 ----
        panel3.fill(0)

        if smoothed_pose:
            draw_pose_full(panel3, smoothed_pose, panel_w, panel_h)
            n_visible = sum(1 for lm in smoothed_pose if getattr(lm, "visibility", 0) > 0.5)
            cv2.putText(panel3, f"Pose: {n_visible}/33", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(panel3, "Pose: NOT FOUND - step back",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if cur_hand and cur_hand.hand_landmarks:
            for i, hand_lms in enumerate(cur_hand.hand_landmarks):
                handedness = cur_hand.handedness[i][0].category_name
                lm_s, cn_s = (left_lm, left_cn) if handedness == "Left" else (right_lm, right_cn)
                drawing_utils.draw_landmarks(
                    image=panel3, landmark_list=hand_lms,
                    connections=HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=lm_s, connection_drawing_spec=cn_s,
                )
            cv2.putText(panel3, f"Hands: {len(cur_hand.hand_landmarks)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(panel3, "Body Skeleton", (10, panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # ---- 面板 4: 综合视图 ----
        cv2.resize(frame, (panel_w, panel_h), dst=panel4)

        if smoothed_pose:
            pose_overlay.fill(0)
            draw_pose_full(pose_overlay, smoothed_pose, panel_w, panel_h,
                           lm_color=(0, 220, 0), cn_color=(200, 100, 0),
                           thickness=2, skip_face=True)
            cv2.addWeighted(pose_overlay, 0.50, panel4, 1.0, 0, dst=panel4)

        if cur_hand and cur_hand.hand_landmarks:
            for i, hand_lms in enumerate(cur_hand.hand_landmarks):
                handedness = cur_hand.handedness[i][0].category_name
                lm_s, cn_s = (left_lm, left_cn) if handedness == "Left" else (right_lm, right_cn)
                drawing_utils.draw_landmarks(
                    image=panel4, landmark_list=hand_lms,
                    connections=HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=lm_s, connection_drawing_spec=cn_s,
                )

        if smoothed_face:
            # 画人脸框和前3个最可能的情绪
            fx, fy, fw, fh = get_face_bbox(smoothed_face, panel_w, panel_h, margin=0.15)
            cv2.rectangle(panel4, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            if dl_scores_cached is not None:
                order = np.argsort(dl_scores_cached)[::-1]
                tx = min(fx + fw + 6, panel_w - 90)
                ty = fy
                font_sizes = [0.55, 0.45, 0.40]
                for i, idx in enumerate(order[:3]):
                    label = EMOTION_LABELS[idx]
                    score = float(dl_scores_cached[idx])
                    fs = font_sizes[i]
                    color = _EMOTION_COLORS_8.get(label, (255,255,255))
                    cv2.putText(panel4, f"{label} {score:.0%}", (tx, ty + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1)
                    ty += int(20 + 10 * fs)

        cv2.putText(panel4, "Combined", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ---- 计算并显示帧率 ----
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time
        frame_count += 1

        cv2.putText(panel1, f"FPS: {int(fps)}", (panel_w - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(panel3, f"FPS: {int(fps)}", (panel_w - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(panel4, f"FPS: {int(fps)}", (panel_w - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 画分隔线
        cv2.line(canvas, (panel_w, 0), (panel_w, panel_h * 2), (80, 80, 80), 2)
        cv2.line(canvas, (0, panel_h), (panel_w * 2, panel_h), (80, 80, 80), 2)

        cv2.imshow("MediaPipe - Pose + Hands + Face - Press 'q' to exit", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("已退出。再见。")
            break

    # ----- 释放所有资源 -----
    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    hand_landmarker.close()
    face_landmarker.close()


def avg2(s, k1, k2):
    """计算两个 blendshape 键的平均值"""
    return (s.get(k1, 0) + s.get(k2, 0)) / 2


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出")
