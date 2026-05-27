import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, PoseLandmarksConnections,
    HandLandmarksConnections,
    GestureRecognizer, GestureRecognizerOptions,
    FaceLandmarker, FaceLandmarkerOptions, FaceLandmarksConnections,
    drawing_utils, RunningMode,
)
from emotion_gpu import EmotiEffLibRecognizerOnnxGPU
import time
import os
import ctypes
import numpy as np
from math import atan2, degrees, sqrt
import threading
from PIL import Image, ImageDraw, ImageFont
import collections
import queue



# 平滑一下关键点，减少抖动
class EMASmoother:
    def __init__(self, alpha=0.4, max_lost_frames=15):
        self.alpha = alpha
        self.max_lost_frames = max_lost_frames
        self.smoothed = None
        self.lost_count = 0

    def update(self, landmarks):
        if landmarks:
            self.lost_count = 0
            if self.smoothed is None or len(self.smoothed) != len(landmarks):
                self.smoothed = landmarks
            else:
                # new = alpha * cur + (1-alpha) * prev
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


def _get_or_create_smoother(pool, person_id, alpha, max_lost):
    """按 person_id 获取或创建 EMA 平滑器"""
    if person_id not in pool:
        pool[person_id] = EMASmoother(alpha=alpha, max_lost_frames=max_lost)
    return pool[person_id]


# 根据人脸位置分配人员ID
_PERSON_COLORS_BGR = [
    (0, 0, 255),    # 红 - person 0 (主人物，最大人脸)
    (255, 0, 0),    # 蓝 - person 1
    (0, 255, 255),  # 黄 - person 2
    (255, 0, 255),  # 品红 - person 3
]
_MAX_PEOPLE = 4
_MAX_CENTROID_DIST = 0.15  # 一帧最多漂这么远，超过就算不同人了


def assign_person_ids(face_list, pose_list, prev_face_centroids, fw, fh):
    """为检测到的人脸和姿态分配持久化 person ID。
    返回 (person_faces, person_poses, new_centroids, unmatched_poses)

    person_faces: dict[pid] = (face_lms, bbox_px, centroid, area, orig_idx)
    person_poses: dict[pid] = pose_lms or None
    unmatched_poses: list of pose_lms not matched to any face
    """
    if not face_list:
        # 人脸被遮挡时，用姿态鼻子位置维持 person ID
        if not pose_list or not prev_face_centroids:
            return {}, {}, {}, list(pose_list) if pose_list else []
        person_poses = {}
        new_centroids = {}
        used_poses = set()
        max_dist_sq = _MAX_CENTROID_DIST ** 2
        for prev_id in sorted(prev_face_centroids.keys()):
            px, py = prev_face_centroids[prev_id]
            best_dist = float("inf")
            best_idx = -1
            for pi, pose_lms in enumerate(pose_list):
                if pi in used_poses:
                    continue
                nose = pose_lms[0]
                dist = (nose.x - px) ** 2 + (nose.y - py) ** 2
                if dist < best_dist and dist < max_dist_sq:
                    best_dist = dist
                    best_idx = pi
            if best_idx >= 0:
                person_poses[prev_id] = pose_list[best_idx]
                nose = pose_list[best_idx][0]
                new_centroids[prev_id] = (nose.x, nose.y)
                used_poses.add(best_idx)
        unmatched = [pl for pi, pl in enumerate(pose_list) if pi not in used_poses]
        return {}, person_poses, new_centroids, unmatched

    # Step 1: 计算每张人脸 bbox + 质心 + 面积
    face_entries = []
    for idx, face_lms in enumerate(face_list):
        xs = [lm.x for lm in face_lms]
        ys = [lm.y for lm in face_lms]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bw, bh = x_max - x_min, y_max - y_min
        if bw <= 0 or bh <= 0:
            continue
        margin = 0.35
        x_min_m = max(0.0, x_min - bw * margin)
        x_max_m = min(1.0, x_max + bw * margin)
        y_min_m = max(0.0, y_min - bh * margin)
        y_max_m = min(1.0, y_max + bh * margin)
        cx = (x_min_m + x_max_m) / 2.0
        cy = (y_min_m + y_max_m) / 2.0
        area = (x_max_m - x_min_m) * (y_max_m - y_min_m)
        bbox_px = (int(x_min_m * fw), int(y_min_m * fh),
                   int((x_max_m - x_min_m) * fw), int((y_max_m - y_min_m) * fh))
        face_entries.append((face_lms, bbox_px, (cx, cy), area, idx))

    if not face_entries:
        return {}, {}, {}, list(pose_list) if pose_list else []

    # Step 2: 按面积降序 → person 0 = 最大人脸
    face_entries.sort(key=lambda e: e[3], reverse=True)
    if len(face_entries) > _MAX_PEOPLE:
        face_entries = face_entries[:_MAX_PEOPLE]

    # Step 3: 质心最近邻匹配，跨帧保持 ID
    person_faces = {}
    new_centroids = {}
    used_cur = set()
    max_dist_sq = _MAX_CENTROID_DIST ** 2

    if prev_face_centroids:
        current_centroids = [e[2] for e in face_entries]
        for prev_id in sorted(prev_face_centroids.keys()):
            best_dist = float("inf")
            best_idx = -1
            for i, cur_cent in enumerate(current_centroids):
                if i in used_cur:
                    continue
                dist = (cur_cent[0] - prev_face_centroids[prev_id][0]) ** 2 + \
                       (cur_cent[1] - prev_face_centroids[prev_id][1]) ** 2
                if dist < best_dist and dist < max_dist_sq:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0:
                person_faces[prev_id] = face_entries[best_idx]
                new_centroids[prev_id] = face_entries[best_idx][2]
                used_cur.add(best_idx)
    else:
        # 首帧：直接按顺序分配 ID
        for i, entry in enumerate(face_entries):
            person_faces[i] = entry
            new_centroids[i] = entry[2]
            used_cur.add(i)

    # 为新出现的人脸分配新 ID
    next_id = 0
    for i in range(len(face_entries)):
        if i in used_cur:
            continue
        while next_id in person_faces:
            next_id += 1
        if next_id >= _MAX_PEOPLE:
            break
        person_faces[next_id] = face_entries[i]
        new_centroids[next_id] = face_entries[i][2]
        used_cur.add(i)
        next_id += 1

    # Step 4: 姿态关联 — 鼻子落点在人脸 bbox 内即匹配
    person_poses = {}
    used_pose_indices = set()
    for pid, (_, bbox_px, _, _, _) in person_faces.items():
        fx_n = bbox_px[0] / fw
        fy_n = bbox_px[1] / fh
        fw_n = bbox_px[2] / fw
        fh_n = bbox_px[3] / fh
        matched = False
        for pi, pose_lms in enumerate(pose_list or []):
            if pi in used_pose_indices:
                continue
            nose = pose_lms[0]
            if fx_n <= nose.x <= fx_n + fw_n and fy_n <= nose.y <= fy_n + fh_n:
                person_poses[pid] = pose_lms
                used_pose_indices.add(pi)
                matched = True
                break
        if not matched:
            person_poses[pid] = None

    # 收集未匹配的姿态
    unmatched_poses = [pl for pi, pl in enumerate(pose_list or [])
                       if pi not in used_pose_indices]

    return person_faces, person_poses, new_centroids, unmatched_poses


# 每种情绪对应的显示颜色 (BGR)
_EMOTION_COLORS_8 = {
    "Anger":      (0, 0, 255),
    "Contempt":   (120, 120, 120),
    "Disgust":    (0, 150, 0),
    "Fear":       (150, 150, 255),
    "Happiness":  (0, 255, 255),
    "Neutral":    (180, 180, 180),
    "Sadness":    (255, 150, 50),
    "Surprise":   (255, 100, 255),
}


# 用 blendshape 分数来推断情绪
def classify_emotion(blendshapes):
    s = {}
    for bs in blendshapes:
        s[bs.category_name] = bs.score

    def avg(*keys):
        return sum(s.get(k, 0) for k in keys) / len(keys)

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
    if best_score < 0.35:  # 分数太低就默认 Neutral
        return "Neutral", best_score, s
    return best, best_score, s


# 情绪 + 手势 → 意图映射 (7x10 = 70 条)
# 带 "[失调警报]" 前缀的会在 Panel 4 触发红色警告样式
_INTENT_MAP = {
    # ---- 开心 ----
    ("Happiness", "Thumb_Up"): "强烈赞同 (Strong Approval)",
    ("Happiness", "Victory"): "庆祝喜悦 (Celebration / Joy)",
    ("Happiness", "Open_Palm"): "热情欢迎 (Warm Welcome)",
    ("Happiness", "Pointing_Up"): "兴奋发现 (Excited Discovery / Aha!)",
    ("Happiness", "ILoveYou"): "喜爱比心 (Affectionate)",
    ("Happiness", "Finger_Heart"): "甜蜜示好 (Sweet Affection)",
    ("Happiness", "Two_Handed_Heart"): "狂热崇拜 (Passionate Adoration)",
    ("Happiness", "Thumb_Down"): "[失调警报] 幸灾乐祸 (Gloating / Schadenfreude)",
    ("Happiness", "Closed_Fist"): "[失调警报] 狂躁性喜悦 (Manic Joy / Concealed Aggression)",
    ("Happiness", "Middle_Finger"): "[失调警报] 玩笑式侮辱 (Playful Banter)",

    # ---- 愤怒 ----
    ("Anger", "Thumb_Down"): "强烈否定 (Strong Rejection)",
    ("Anger", "Victory"): "挑衅式胜利 (Aggressive Taunting)",
    ("Anger", "Open_Palm"): "愤怒质问 (Angry Confrontation)",
    ("Anger", "Closed_Fist"): "暴力倾向 (Aggression / Threat Detected)",
    ("Anger", "Pointing_Up"): "警告指责 (Warning / Accusing)",
    ("Anger", "Middle_Finger"): "极度敌意 (Extreme Hostility / Insult)",
    ("Anger", "Thumb_Up"): "[失调警报] 极度嘲讽的赞同 (Sarcastic Approval)",
    ("Anger", "ILoveYou"): "[失调警报] 扭曲的占有欲 (Twisted Possessiveness)",
    ("Anger", "Finger_Heart"): "[失调警报] 恐吓性病态示好 (Threatening Sweetness)",
    ("Anger", "Two_Handed_Heart"): "[失调警报] 充满杀意的狂热 (Lethal Affection)",

    # ---- 悲伤 ----
    ("Sadness", "Thumb_Down"): "极度绝望 (Deep Despair)",
    ("Sadness", "ILoveYou"): "痛失所爱 (Heartbroken Affection)",
    ("Sadness", "Open_Palm"): "无奈接受 (Helpless Acceptance)",
    ("Sadness", "Closed_Fist"): "痛苦隐忍 (Suppressed Anguish)",
    ("Sadness", "Pointing_Up"): "悲观警告 (Pessimistic Warning)",
    ("Sadness", "Thumb_Up"): "[失调警报] 苦笑勉励 (Bitter Encouragement)",
    ("Sadness", "Victory"): "[失调警报] 充满悲剧色彩的庆祝 (Tragic Irony)",
    ("Sadness", "Middle_Finger"): "[失调警报] 绝望的控诉 (Desperate Accusation)",
    ("Sadness", "Finger_Heart"): "[失调警报] 绝望中的病态讨好 (Desperate Appeasement)",
    ("Sadness", "Two_Handed_Heart"): "[失调警报] 极度卑微的乞求 (Submissive Begging)",

    # ---- 惊讶 ----
    ("Surprise", "Thumb_Up"): "难以置信的赞赏 (Astonished Praise)",
    ("Surprise", "Victory"): "意外之喜 (Pleasant Surprise)",
    ("Surprise", "ILoveYou"): "受宠若惊 (Overwhelmed by Affection)",
    ("Surprise", "Open_Palm"): "震惊防御 (Shocked / Defensive)",
    ("Surprise", "Closed_Fist"): "惊愕防备 (Startled Readiness)",
    ("Surprise", "Pointing_Up"): "震惊提问 (Shocked Questioning)",
    ("Surprise", "Finger_Heart"): "突如其来的心动 (Sudden Infatuation)",
    ("Surprise", "Two_Handed_Heart"): "极度惊喜的感动 (Overjoyed & Moved)",
    ("Surprise", "Thumb_Down"): "[失调警报] 震惊且极度排斥 (Shocked Revulsion)",
    ("Surprise", "Middle_Finger"): "[失调警报] 防御性应激咒骂 (Startle Response / Profanity)",

    # ---- 恐惧 ----
    ("Fear", "Thumb_Down"): "极度抗拒 (Terrified Rejection)",
    ("Fear", "Open_Palm"): "害怕退缩 (Defensive / Backing Off)",
    ("Fear", "Closed_Fist"): "极度紧张 (Tense / Anxious)",
    ("Fear", "Pointing_Up"): "惊恐指认 (Panicked Warning)",
    ("Fear", "Thumb_Up"): "[失调警报] 胁迫性赞同 (Coerced Agreement)",
    ("Fear", "Victory"): "[失调警报] 害怕但顺从 (Submissive Appeasement)",
    ("Fear", "ILoveYou"): "[失调警报] 斯德哥尔摩综合征 (Trauma Bonding)",
    ("Fear", "Middle_Finger"): "[失调警报] 恐惧中的歇斯底里 (Hysterical Defiance)",
    ("Fear", "Finger_Heart"): "[失调警报] 恐惧下的强制讨好 (Fearful Fawning)",
    ("Fear", "Two_Handed_Heart"): "[失调警报] 绝望求生式臣服 (Desperate Surrender)",

    # ---- 厌恶 ----
    ("Disgust", "Thumb_Down"): "强烈厌恶 (Strong Aversion)",
    ("Disgust", "Victory"): "鄙视性嘲笑 (Condescending Victory)",
    ("Disgust", "Open_Palm"): "极其嫌弃/远离 (Repulsion / Stay Away)",
    ("Disgust", "Closed_Fist"): "压抑的反胃与愤怒 (Suppressed Disgust)",
    ("Disgust", "Pointing_Up"): "鄙视指责 (Scornful Accusation)",
    ("Disgust", "Middle_Finger"): "强烈鄙视 (Strong Contempt)",
    ("Disgust", "Thumb_Up"): "[失调警报] 虚伪敷衍 (Sarcastic Approval)",
    ("Disgust", "ILoveYou"): "[失调警报] 极度恶心的逢场作戏 (Nauseated Play-acting)",
    ("Disgust", "Finger_Heart"): "[失调警报] 极度虚伪的迎合 (Insincere Flattery)",
    ("Disgust", "Two_Handed_Heart"): "[失调警报] 讽刺性极强的做作 (Extreme Sarcastic Devotion)",

    # ---- 中性 ----
    ("Neutral", "Thumb_Up"): "赞同确认 (Approval)",
    ("Neutral", "Thumb_Down"): "拒绝不赞同 (Disapproval)",
    ("Neutral", "Victory"): "轻松搞定 (Casual Success)",
    ("Neutral", "ILoveYou"): "随性打招呼 (Casual Greeting)",
    ("Neutral", "Open_Palm"): "展示说明 (Explaining / Presenting)",
    ("Neutral", "Closed_Fist"): "专注准备 (Focused / Getting Ready)",
    ("Neutral", "Pointing_Up"): "强调重点 (Making a Point)",
    ("Neutral", "Finger_Heart"): "礼貌性比心 (Polite Heart)",
    ("Neutral", "Two_Handed_Heart"): "标准营业动作 (Standard Idol Pose)",
    ("Neutral", "Middle_Finger"): "粗鲁挑衅 (Rude Gesture / Provocation)",
}

_DISSONANCE_TAG = "[失调警报]"


# scipy 导入耗时较长，提前到模块层导入避免运行时卡顿
try:
    import scipy.signal as _scipy_signal
except ImportError:
    _scipy_signal = None


def estimate_heart_rate(rppg_buffer, fps):
    if _scipy_signal is None or len(rppg_buffer) < 120:
        return None
    g = np.array(rppg_buffer, dtype=np.float64)
    g -= g.mean()
    try:
        sos = _scipy_signal.butter(4, [0.75, 2.5], btype="band", fs=fps, output="sos")
        g = _scipy_signal.sosfiltfilt(sos, g)
    except Exception:
        return None
    n = len(g)
    fft = np.abs(np.fft.rfft(g))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mask = (freqs >= 0.75) & (freqs <= 2.5)
    if not np.any(mask):
        return None
    peak_idx = np.argmax(fft[mask])
    peak_freq = freqs[mask][peak_idx]
    bpm = int(round(peak_freq * 60))
    return bpm if 40 <= bpm <= 180 else None



def synthesize_intent(emotion, gesture):
    """情绪 + 手势 → 意图，返回双语文本，没有就返回 None"""
    if emotion is None or gesture is None:
        return None
    return _INTENT_MAP.get((emotion, gesture),
                           f"未知组合: {emotion} + {gesture}")


def _load_chinese_font(font_size):
    """尝试加载中文字体，找不到就用默认的"""
    for name in ["msyh.ttc", "simhei.ttf", "simsun.ttc", "PingFang.ttc"]:
        try:
            return ImageFont.truetype(name, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# 字体缓存，避免每帧重新加载
_FONT_CACHE = {}


def _get_cached_font(size):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = _load_chinese_font(size)
    return _FONT_CACHE[size]


# 触发红色警告卡片的关键词
_HIGH_ALERT_KW = ["[警报]", "[异常]", "[泄露]", "[预判]", "[失调警报]"]


def _is_high_alert(text):
    return any(kw in text for kw in _HIGH_ALERT_KW)


# cv2 BGR → PIL RGBA
def _bgr_to_rgba(bgr, alpha=255):
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]), alpha)


def draw_modern_hud_panel(draw, text, x, y, font,
                          text_color=(255, 255, 255, 255),
                          bg_color=(30, 30, 30, 180),
                          radius=12, pad_x=15, pad_y=10,
                          accent_color=None, card_width=None):
    """画一个圆角矩形 HUD 卡片，文字居中；可选 accent 装饰条；card_width 可统一宽度"""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    rw = max(tw + pad_x * 2, card_width) if card_width else tw + pad_x * 2
    rh = th + pad_y * 2
    draw.rounded_rectangle([x, y, x + rw, y + rh], radius=radius, fill=bg_color)
    if accent_color is not None:
        draw.rounded_rectangle([x + 6, y + 2, x + rw - 6, y + 5], radius=2, fill=accent_color)
    tx = x + (rw - tw) // 2
    ty = y + (rh - th) // 2 - bbox[1]
    draw.text((tx, ty), text, font=font, fill=text_color)


def _wrap_text_lines(draw, text, font, max_px):
    """把文字拆成1-2行，优先在空格处换行，尽量不拆单词"""
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] <= max_px:
        return [text]
    # 二分查找合适的断点
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        b = draw.textbbox((0, 0), text[:mid], font=font)
        if b[2] - b[0] <= max_px:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        return [text]
    # 尽量在空格处断，不拆单词
    space_idx = text.rfind(" ", max(0, lo - 15), lo)
    if space_idx > 0:
        lo = space_idx
    line1, rest = text[:lo].rstrip(), text[lo:].lstrip()
    # 第二行放不下就截断加省略号
    b2 = draw.textbbox((0, 0), rest, font=font)
    if b2[2] - b2[0] <= max_px:
        return [line1, rest]
    lo2, hi2 = 0, len(rest)
    while lo2 < hi2:
        mid2 = (lo2 + hi2 + 1) // 2
        b = draw.textbbox((0, 0), rest[:mid2] + "...", font=font)
        if b[2] - b[0] <= max_px:
            lo2 = mid2
        else:
            hi2 = mid2 - 1
    if lo2 > 0:
        space2 = rest.rfind(" ", max(0, lo2 - 15), lo2)
        if space2 > 0:
            lo2 = space2
    line2 = (rest[:lo2].rstrip() + "...") if lo2 > 0 else "..."
    return [line1, line2]


def draw_pil_notification_card(draw, text, center_x, top_y, font,
                               is_high_alert=None, bounds=None):
    """画通知卡片，警告用红色、普通用青色，支持2行 + 边界限制 + 左侧彩色装饰条"""
    if is_high_alert is None:
        is_high_alert = _is_high_alert(text)
    is_alert = is_high_alert
    if is_alert:
        bg = (180, 0, 20, 220)
        fg = (255, 255, 255, 255)
        accent = (255, 60, 60, 255)
        radius = 16
    else:
        bg = (10, 40, 50, 200)
        fg = (0, 255, 255, 255)
        accent = (0, 220, 220, 255)
        radius = 12
    pad_x, pad_y = 18, 14
    line_gap = 4
    left_bar_w = 4

    if bounds is not None:
        max_tw = (bounds[1] - bounds[0]) - pad_x * 2 - 8
    else:
        max_tw = 9999

    lines = _wrap_text_lines(draw, text, font, max(max_tw, 60))

    line_metrics = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    tw = max(m[2] - m[0] for m in line_metrics)
    lh = line_metrics[0][3] - line_metrics[0][1]
    total_th = lh * len(lines) + line_gap * (len(lines) - 1)
    rw = tw + pad_x * 2
    rh = total_th + pad_y * 2
    rx = center_x - rw // 2
    if bounds is not None:
        x_min, x_max = bounds
        rx = max(x_min + 4, min(rx, x_max - rw - 4))
    draw.rounded_rectangle([rx, top_y, rx + rw, top_y + rh], radius=radius, fill=bg)
    # 左侧彩色装饰条
    draw.rounded_rectangle(
        [rx + 2, top_y + 4, rx + 2 + left_bar_w, top_y + rh - 4],
        radius=left_bar_w // 2, fill=accent)
    # 每行居中绘制
    cur_y = top_y + (rh - total_th) // 2
    for ln in lines:
        b = draw.textbbox((0, 0), ln, font=font)
        lw = b[2] - b[0]
        tx = rx + (rw - lw) // 2
        draw.text((tx, cur_y - b[1]), ln, font=font, fill=fg)
        cur_y += lh + line_gap


def draw_pil_text_card(draw, text, center_x, top_y, font,
                       text_color=(255, 255, 255, 255),
                       bg_color=(30, 30, 30, 200),
                       radius=10, pad_x=14, pad_y=8, bounds=None):
    """通用文字卡片，支持2行 + 边界限制"""
    line_gap = 3

    if bounds is not None:
        max_tw = (bounds[1] - bounds[0]) - pad_x * 2 - 8
    else:
        max_tw = 9999

    lines = _wrap_text_lines(draw, text, font, max(max_tw, 50))
    line_metrics = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    tw = max(m[2] - m[0] for m in line_metrics)
    lh = line_metrics[0][3] - line_metrics[0][1]
    total_th = lh * len(lines) + line_gap * (len(lines) - 1)
    rw = tw + pad_x * 2
    rh = total_th + pad_y * 2
    rx = center_x - rw // 2
    if bounds is not None:
        x_min, x_max = bounds
        rx = max(x_min + 4, min(rx, x_max - rw - 4))
    draw.rounded_rectangle([rx, top_y, rx + rw, top_y + rh], radius=radius, fill=bg_color)
    cur_y = top_y + (rh - total_th) // 2
    for ln in lines:
        b = draw.textbbox((0, 0), ln, font=font)
        lw = b[2] - b[0]
        tx = rx + (rw - lw) // 2
        draw.text((tx, cur_y - b[1]), ln, font=font, fill=text_color)
        cur_y += lh + line_gap


def draw_pil_emotion_bars(draw, x, y, w, bar_h, gap,
                          scores_array, label_order, font_small):
    """画情绪置信度圆角条形图（渐变填充 + 百分比 + 描边）"""
    for i, name in enumerate(label_order):
        val = float(scores_array[i]) if i < len(scores_array) else 0.0
        by_ = y + i * (bar_h + gap)
        # trough
        draw.rounded_rectangle([x, by_, x + w, by_ + bar_h], radius=4,
                               fill=(0, 0, 0, 100))
        if val > 0.001:
            bw = max(4, int(w * min(val, 1.0)))
            color_bgr = _EMOTION_COLORS_8.get(name, (180, 180, 180))
            r_c, g_c, b_c = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
            # 用numpy画渐变条，8段就够了，别逐像素画
            segments = 8
            for seg in range(segments):
                t = (seg + 0.5) / segments
                sx = x + int(bw * seg / segments)
                ex = x + int(bw * (seg + 1) / segments)
                if ex <= sx:
                    continue
                fill_r = int(r_c * (0.45 + 0.55 * t))
                fill_g = int(g_c * (0.45 + 0.55 * t))
                fill_b = int(b_c * (0.45 + 0.55 * t))
                draw.rectangle([sx, by_ + 1, ex, by_ + bar_h - 1],
                               fill=(fill_r, fill_g, fill_b, 230))
            # 1px 描边
            draw.rounded_rectangle([x, by_, x + bw, by_ + bar_h], radius=4,
                                   outline=(r_c, g_c, b_c, 180), width=1)
        # 情绪名
        draw.text((x + 5, by_ + 1), name, font=font_small,
                  fill=(255, 255, 255, 210))
        # 右侧百分比
        pct_text = f"{val:.0%}"
        pct_tw = draw.textbbox((0, 0), pct_text, font=font_small)[2]
        draw.text((x + w - pct_tw - 4, by_ + 1), pct_text, font=font_small,
                  fill=(220, 220, 220, 200))


# 用PIL画中文，带个背景框，老代码也要能用
def draw_chinese_text_on_draw(draw, text, center_x, top_y, font,
                              text_color=(0, 255, 255), bg_color=(0, 0, 0)):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6
    rx = center_x - tw // 2 - pad
    ry = top_y - pad
    rw, rh = tw + pad * 2, th + pad * 2
    draw.rounded_rectangle([rx, ry, rx + rw, ry + rh], radius=8, fill=bg_color)
    tc = (text_color[2], text_color[1], text_color[0], 255)
    draw.text((rx + pad, ry + pad - bbox[1]), text, font=font, fill=tc)


# 认知-预判引擎：通过上半身动力学检测情绪失调和行为预判
_UPPER_BODY_IDS = frozenset({
    11, 12,  # 肩膀
    13, 14,  # 手肘
    15, 16,  # 手腕
    23, 24,  # 髋部
})
_WRIST_IDS = (15, 16)  # 左右手腕

# 阈值已经用肩膀宽度归一化了，远近都能用
_T_STIFF = 0.0125     # 上半身总动能低于此 → 僵直
_T_MICRO = 0.075      # 手部动能高于此 → 微颤
_T_ACCEL = 0.04       # 手腕加速度高于此 → 爆发
_T_Z_SPEED_IN = -0.05 # 手腕 Z 速度低于此 → 向镜头突进


def analyze_cognitive_and_anticipation(pose_history, emotion):
    """用3帧姿态历史做动力学分析，检出情绪失调和预判信号。返回 (警告文字, BGR颜色) 或 None"""
    if len(pose_history) != 3:
        return None

    p0, p1, p2 = pose_history

    def _to_xyz(landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)

    xyz0, xyz1, xyz2 = _to_xyz(p0), _to_xyz(p1), _to_xyz(p2)

    # 用肩宽做深度归一化的基准
    sh11 = xyz2[11]
    sh12 = xyz2[12]
    ref_length = float(np.linalg.norm(sh12 - sh11))
    if ref_length < 0.05:
        ref_length = 0.4  # 遮挡或太远时用保底值

    # 每个关节的速度和加速度
    v_t = xyz2 - xyz1
    v_t1 = xyz1 - xyz0
    a_t = v_t - v_t1
    speed_t = np.linalg.norm(v_t, axis=1)
    accel_t = np.linalg.norm(a_t, axis=1)

    # 上半身总动能，深度已经归一化了
    upper_mask = np.array([i in _UPPER_BODY_IDS for i in range(33)])
    upper_speed2 = speed_t[upper_mask] ** 2
    ek_total = float(np.sum(upper_speed2)) / (ref_length ** 2)

    # 手的指标，深度也归一化了
    hand_mask = np.array([i in _WRIST_IDS for i in range(33)])
    ek_hands = float(np.sum(speed_t[hand_mask] ** 2)) / (ref_length ** 2)
    wrist_accel = (np.max(accel_t[hand_mask]) if np.any(hand_mask) else 0.0) / ref_length
    wrist_z_vel = float(np.mean(v_t[hand_mask, 2])) / ref_length

    # 逻辑1: 情绪-动力学失调检测
    if emotion in ("Happiness", "Surprise"):
        if ek_total < _T_STIFF and ek_hands > _T_MICRO * 1.5:
            return ("[异常] 割裂姿态: 静止躯干+手部微动 (Split Posture / Concealed Emotion)",
                    (50, 0, 255))

    if emotion == "Neutral":
        if ek_hands > _T_MICRO and ek_total < _T_MICRO * 1.5:
            return ("[潜在] 隐性焦躁: 微颤 / 不安 (Hidden Agitation / Tremors)",
                    (255, 140, 0))

    # 逻辑2: 行为预判
    if emotion == "Anger":
        if wrist_accel > _T_ACCEL and wrist_z_vel < _T_Z_SPEED_IN:
            return ("[预判] 物理攻击警告: 快速突进 (Strike Anticipation)",
                    (255, 0, 0))

    if emotion in ("Fear", "Sadness"):
        if wrist_accel > _T_ACCEL:
            wrist_y_accel = np.mean(a_t[hand_mask, 1]) / ref_length
            if wrist_y_accel < -_T_ACCEL * 0.5:
                return ("[预判] 绝望抓取 / 寻求庇护 (Plea / Reach Anticipation)",
                        (0, 100, 255))

    return None


# blendshape的中文名字，显示在脸旁边
_BS_CN_LABELS = {
    "browInnerUp":         "挑眉",
    "browDownLeft":        "左皱眉",
    "browDownRight":       "右皱眉",
    "browOuterUpLeft":     "左抬眉",
    "browOuterUpRight":    "右抬眉",
    "eyeBlinkLeft":        "左眨眼",
    "eyeBlinkRight":       "右眨眼",
    "eyeSquintLeft":       "左眯眼",
    "eyeSquintRight":      "右眯眼",
    "eyeWideLeft":         "左瞪眼",
    "eyeWideRight":        "右瞪眼",
    "mouthSmileLeft":      "左微笑",
    "mouthSmileRight":     "右微笑",
    "mouthFrownLeft":      "左撇嘴",
    "mouthFrownRight":     "右撇嘴",
    "mouthDimpleLeft":     "左酒窝",
    "mouthDimpleRight":    "右酒窝",
    "mouthUpperUpLeft":    "左上唇扬",
    "mouthUpperUpRight":   "右上唇扬",
    "mouthPressLeft":      "左抿嘴",
    "mouthPressRight":     "右抿嘴",
    "noseSneerLeft":       "左鼻翼",
    "noseSneerRight":      "右鼻翼",
    "jawOpen":             "张嘴",
    "mouthPucker":         "噘嘴",
    "cheekSquintLeft":     "左脸颊",
    "cheekSquintRight":    "右脸颊",
}

# 左右对称的动作，取大的那个显示就行
_BS_MERGED = {
    "browDown":       ("browDownLeft", "browDownRight"),
    "browOuterUp":    ("browOuterUpLeft", "browOuterUpRight"),
    "eyeBlink":       ("eyeBlinkLeft", "eyeBlinkRight"),
    "eyeSquint":      ("eyeSquintLeft", "eyeSquintRight"),
    "eyeWide":        ("eyeWideLeft", "eyeWideRight"),
    "mouthSmile":     ("mouthSmileLeft", "mouthSmileRight"),
    "mouthFrown":     ("mouthFrownLeft", "mouthFrownRight"),
    "mouthDimple":    ("mouthDimpleLeft", "mouthDimpleRight"),
    "mouthUpperUp":   ("mouthUpperUpLeft", "mouthUpperUpRight"),
    "mouthPress":     ("mouthPressLeft", "mouthPressRight"),
    "noseSneer":      ("noseSneerLeft", "noseSneerRight"),
    "cheekSquint":    ("cheekSquintLeft", "cheekSquintRight"),
}
_BS_MERGED_LABELS = {
    "browDown":       "皱眉",
    "browOuterUp":    "抬眉",
    "eyeBlink":       "眨眼",
    "eyeSquint":      "眯眼",
    "eyeWide":        "瞪眼",
    "mouthSmile":     "微笑",
    "mouthFrown":     "撇嘴",
    "mouthDimple":    "酒窝",
    "mouthUpperUp":   "上唇扬",
    "mouthPress":     "抿嘴",
    "noseSneer":      "鼻翼抽动",
    "cheekSquint":    "脸颊收紧",
}


def get_facial_actions(blendshapes, top_n=4, min_score=0.06):
    """从 blendshape 列表中提取最显著的几个面部动作，返回 [(中文名, 百分比), ...]"""
    if blendshapes is None:
        return []
    s = {bs.category_name: bs.score for bs in blendshapes}

    results = []
    # 先处理独立动作
    for key, cn_name in _BS_CN_LABELS.items():
        if key in s and s[key] > min_score:
            results.append((cn_name, s[key]))

    # 合并左右对称的，取大的
    for merged, (left, right) in _BS_MERGED.items():
        val = max(s.get(left, 0.0), s.get(right, 0.0))
        if val > min_score:
            # 去掉左右单独条目，用合并条目代替
            left_name = _BS_CN_LABELS.get(left, "")
            right_name = _BS_CN_LABELS.get(right, "")
            results = [(n, v) for n, v in results
                       if n != left_name and n != right_name]
            results.append((_BS_MERGED_LABELS[merged], val))

    # 按强度降序
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def advanced_cognitive_engine(hand_landmarks, head_pose_angles,
                              current_gesture):
    """注视方向 vs 手指方向3D夹角检测。返回 (警告文字, BGR颜色) 或 None"""
    if not (head_pose_angles is not None and hand_landmarks is not None
            and current_gesture == "Pointing_Up"):
        return None

    pitch, yaw, _roll = head_pose_angles
    rad = np.deg2rad
    cp, sp = np.cos(rad(pitch)), np.sin(rad(pitch))
    cy, sy = np.cos(rad(yaw)), np.sin(rad(yaw))
    vec_gaze = np.array([-sy * cp, sp, -cy * cp], dtype=np.float64)
    vec_gaze /= np.linalg.norm(vec_gaze) + 1e-8

    for hand_lms in hand_landmarks:
        idx_tip = np.array([hand_lms[8].x, hand_lms[8].y,
                            hand_lms[8].z], dtype=np.float64)
        norm = np.linalg.norm(idx_tip)
        if norm < 1e-8:
            continue
        vec_finger = idx_tip / norm
        cos_angle = np.dot(vec_gaze, vec_finger)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = degrees(np.arccos(cos_angle))
        if angle_deg > 35:
            return ("[交互异常] 盲指 / 认知游离 (Blind Pointing / Distracted)",
                    (0, 0, 200))

    return None


# 从人脸关键点算包围盒
def get_face_bbox(face_landmarks, frame_w, frame_h, margin=0.35):
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


# 画个进度条显示置信度
def draw_progress_bar(img, x, y, w, h, value, color, label=""):
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


# 不同情绪对应的面部重点区域
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

_CONTOUR_MAP = {
    "EYES":     [FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
                 FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE],
    "EYEBROWS": [FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
                 FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW],
    "LIPS":     [FaceLandmarksConnections.FACE_LANDMARKS_LIPS],
}

# blendshape key → 面部区域映射
_BS_REGION = {}
for _k in ["browInnerUp", "browDownLeft", "browDownRight",
           "browOuterUpLeft", "browOuterUpRight"]:
    _BS_REGION[_k] = "EYEBROWS"
for _k in ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
           "eyeWideLeft", "eyeWideRight"]:
    _BS_REGION[_k] = "EYES"
for _k in ["mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
           "mouthDimpleLeft", "mouthDimpleRight", "mouthUpperUpLeft", "mouthUpperUpRight",
           "mouthPressLeft", "mouthPressRight", "jawOpen", "mouthPucker",
           "noseSneerLeft", "noseSneerRight"]:
    _BS_REGION[_k] = "LIPS"


def draw_emotion_focus(img, landmark_list, emotion):
    """高亮当前情绪相关的面部区域"""
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


def draw_action_glow(img, landmark_list, blendshapes, min_score=0.06):
    """将当前活跃的面部动作部位用红色渐变标注"""
    if blendshapes is None:
        return
    s = {bs.category_name: bs.score for bs in blendshapes}
    active_regions = set()
    for bs_key, region in _BS_REGION.items():
        val = s.get(bs_key, 0)
        # 也检查合并键
        for merged, (left, right) in _BS_MERGED.items():
            if bs_key in (left, right):
                val = max(val, s.get(left, 0), s.get(right, 0))
                break
        if val > min_score:
            active_regions.add(region)

    if not active_regions:
        return

    overlay = np.zeros_like(img)
    nodot = drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)
    # 三层渐变：外圈粗淡 → 内圈细浓
    layers = [
        (30, 20, 80, 4),    # 淡粉粗线 (BGR)
        (30, 20, 110, 2),   # 中粉
        (30, 20, 150, 1),   # 亮粉细线
    ]
    for b, g, r, t in layers:
        spec = drawing_utils.DrawingSpec(color=(b, g, r), thickness=t, circle_radius=0)
        for region in active_regions:
            for conn_list in _CONTOUR_MAP.get(region, []):
                drawing_utils.draw_landmarks(
                    image=overlay, landmark_list=landmark_list, connections=conn_list,
                    landmark_drawing_spec=nodot, connection_drawing_spec=spec)

    overlay = cv2.GaussianBlur(overlay, (5, 5), 2)
    cv2.addWeighted(overlay, 0.28, img, 1.0, 0, dst=img)


# 画全身33个关键点和连线
def draw_pose_full(img, landmarks, w, h,
                   lm_color=(0, 255, 0), cn_color=(0, 0, 255),
                   thickness=2, circle_r=3, skip_face=False,
                   prev_landmarks=None, attention_mode=False):
    """attention_mode 开启时，连线颜色/粗细按关节速度动态映射（蓝=静止, 红=快速）"""
    _FACE_IDS = {0,1,2,3,4,5,6,7,8,9,10}
    connections = PoseLandmarksConnections.POSE_LANDMARKS
    pts = {}
    for i, lm in enumerate(landmarks):
        if skip_face and i in _FACE_IDS:
            continue
        px, py = int(lm.x * w), int(lm.y * h)
        pts[i] = (px, py)
        cv2.circle(img, (px, py), circle_r, lm_color, -1)

    # 构建上一帧坐标映射，用于速度计算
    prev_pts = {}
    if attention_mode and prev_landmarks is not None:
        for i, lm in enumerate(prev_landmarks):
            if skip_face and i in _FACE_IDS:
                continue
            prev_pts[i] = (int(lm.x * w), int(lm.y * h))

    # 算最大速度用于归一化
    max_vel = 1.0
    if attention_mode and prev_pts:
        velocities = []
        for conn in connections:
            a, b = conn.start, conn.end
            if skip_face and (a in _FACE_IDS or b in _FACE_IDS):
                continue
            if a in pts and b in pts and a in prev_pts and b in prev_pts:
                da = sqrt((pts[a][0]-prev_pts[a][0])**2 + (pts[a][1]-prev_pts[a][1])**2)
                db = sqrt((pts[b][0]-prev_pts[b][0])**2 + (pts[b][1]-prev_pts[b][1])**2)
                velocities.append(da + db)
        if velocities:
            max_vel = max(max(velocities), 1.0)

    for conn in connections:
        a, b = conn.start, conn.end
        if skip_face and (a in _FACE_IDS or b in _FACE_IDS):
            continue
        if a not in pts or b not in pts:
            continue

        if attention_mode and prev_pts and a in prev_pts and b in prev_pts:
            da = sqrt((pts[a][0]-prev_pts[a][0])**2 + (pts[a][1]-prev_pts[a][1])**2)
            db = sqrt((pts[b][0]-prev_pts[b][0])**2 + (pts[b][1]-prev_pts[b][1])**2)
            vel = da + db
            t = min(vel / max_vel, 1.0)
            # 暗蓝 → 亮红
            attn_color = (int(60 + 195 * t), int(30 * (1 - t)), int(30 * (1 - t) + 225 * t))
            attn_thick = max(1, int(1 + 5 * t))
            cv2.line(img, pts[a], pts[b], attn_color, attn_thick)
        else:
            cv2.line(img, pts[a], pts[b], cn_color, thickness)


# 用solvePnP估个头的角度
# 通用3D人脸模型点
_FACE_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],       # 1    nose tip
    [0.0, -63.6, -12.5],   # 152  chin
    [-43.3, 32.7, -26.0],  # 33   left eye left corner
    [43.3, 32.7, -26.0],   # 263  right eye right corner
    [-28.9, -28.9, -25.0], # 61   left mouth corner
    [28.9, -28.9, -25.0],  # 291  right mouth corner
], dtype=np.float64)

_FACE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(face_landmarks, img_w, img_h):
    """用 solvePnP 估计头部欧拉角 (yaw, pitch, roll)，单位度"""
    img_pts = []
    for idx in _FACE_LANDMARK_IDS:
        lm = face_landmarks[idx]
        img_pts.append([lm.x * img_w, lm.y * img_h])
    img_pts = np.array(img_pts, dtype=np.float64)

    focal = img_w
    center = (img_w / 2, img_h / 2)
    cam_mat = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        _FACE_3D_POINTS, img_pts, cam_mat, None,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None, None

    rot_mat, _ = cv2.Rodrigues(rvec)
    proj_mat = np.hstack((rot_mat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
    pitch, yaw, roll = euler.flatten()
    return (pitch, yaw, roll), img_pts


def draw_head_pose_axes(img, img_pts, pose_angles, size=40):
    """在人脸上画朝向轴 (红=X, 绿=Y, 蓝=Z)"""
    if img_pts is None or pose_angles is None:
        return
    pitch, yaw, roll = pose_angles
    nose_tip = tuple(img_pts[0].astype(int))

    rad = lambda d: d * np.pi / 180
    y, p, r = rad(yaw), rad(pitch), rad(roll)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    axes = np.array([[size, 0, 0], [0, -size, 0], [0, 0, -size]], dtype=np.float64)
    projected = (R @ axes.T).T[:, :2] + nose_tip

    cv2.line(img, nose_tip, tuple(projected[0].astype(int)), (0, 0, 255), 2)
    cv2.line(img, nose_tip, tuple(projected[1].astype(int)), (0, 255, 0), 2)
    cv2.line(img, nose_tip, tuple(projected[2].astype(int)), (255, 0, 0), 2)


# 手势识别由 MediaPipe GestureRecognizer 自动完成
# 输出: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up,
#       Victory, ILoveYou, 或 None


# 三个点算夹角，b是顶点，返回角度
def calc_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    cos_a = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return degrees(atan2(sqrt(1 - cos_a**2), cos_a))


def is_middle_finger_extended(landmarks):
    """中指竖起、其他三指弯曲 → 竖中指手势"""
    mt, md = landmarks[12].y, landmarks[10].y
    it, id_ = landmarks[8].y, landmarks[6].y
    rt, rd = landmarks[16].y, landmarks[14].y
    pt, pd = landmarks[20].y, landmarks[18].y
    middle_extended = mt < md
    others_curled = (it > id_) and (rt > rd) and (pt > pd)
    return middle_extended and others_curled


# 比心手势：拇指和食指尖距离多近才算比心了
_FINGER_HEART_TOUCH_THRESHOLD = 0.10


def is_finger_heart(landmarks):
    """单手比心: 拇指尖(4)和食指尖(8)接触，拇指和食指伸展，其他三指弯曲"""
    thumb = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    index = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    if np.linalg.norm(thumb - index) >= _FINGER_HEART_TOUCH_THRESHOLD:
        return False
    # 拇指和食指得伸着，不能是握拳状态
    thumb_extended = landmarks[4].y < landmarks[3].y  # tip above IP joint
    index_extended = landmarks[8].y < landmarks[6].y  # tip above PIP joint
    if not (thumb_extended and index_extended):
        return False
    # 其他三指弯曲
    return (landmarks[12].y > landmarks[10].y and
            landmarks[16].y > landmarks[14].y and
            landmarks[20].y > landmarks[18].y)


def is_two_handed_heart(hand_landmarks_list, handedness_list):
    """双手比心: 左右拇指和食指分别接触，需要2只手"""
    if len(hand_landmarks_list) < 2:
        return False
    left_lm = right_lm = None
    for i, h in enumerate(handedness_list):
        cat = h[0].category_name
        if cat == "Left":
            left_lm = hand_landmarks_list[i]
        elif cat == "Right":
            right_lm = hand_landmarks_list[i]
    if left_lm is None or right_lm is None:
        return False
    lt = np.array([left_lm[4].x, left_lm[4].y, left_lm[4].z])
    rt = np.array([right_lm[4].x, right_lm[4].y, right_lm[4].z])
    li = np.array([left_lm[8].x, left_lm[8].y, left_lm[8].z])
    ri = np.array([right_lm[8].x, right_lm[8].y, right_lm[8].z])
    return (np.linalg.norm(lt - rt) < _FINGER_HEART_TOUCH_THRESHOLD and
            np.linalg.norm(li - ri) < _FINGER_HEART_TOUCH_THRESHOLD)


# 关节角度定义: (A, B, C, 名称, 颜色), B是顶点
_JOINT_DEFS = [
    (11, 13, 15, "L-Elbow",    (255, 200, 100)),
    (12, 14, 16, "R-Elbow",    (255, 150, 50)),
    (23, 25, 27, "L-Knee",     (100, 255, 100)),
    (24, 26, 28, "R-Knee",     (50, 200, 50)),
    (13, 11, 23, "L-Shoulder", (200, 200, 100)),
    (14, 12, 24, "R-Shoulder", (200, 150, 50)),
    (25, 23, 27, "L-Hip",      (100, 200, 255)),
    (26, 24, 28, "R-Hip",      (50, 150, 255)),
]


def draw_joint_angles(img, landmarks, w, h):
    """在骨架上标关节角度"""
    pts = {}
    for i, lm in enumerate(landmarks):
        pts[i] = (int(lm.x * w), int(lm.y * h))

    for a_idx, b_idx, c_idx, _name, color in _JOINT_DEFS:
        if a_idx not in pts or b_idx not in pts or c_idx not in pts:
            continue
        a_pt = pts[a_idx]
        b_pt = pts[b_idx]
        c_pt = pts[c_idx]
        angle = calc_angle(a_pt, b_pt, c_pt)
        if angle is None:
            continue
        text_pos = (b_pt[0] + 20, b_pt[1] - 10)
        cv2.putText(img, f"{int(angle)}", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# 块置乱加密可视化: 8x8 分块随机打乱
_SCRAMBLE_RNG = np.random.default_rng(42)
_SCRAMBLE_ORDER = None
_SCRAMBLE_FRAME_SHAPE = None


def _get_scramble_order(h, w, grid=8):
    """生成块级置乱映射，按帧尺寸缓存（向量化版本）"""
    global _SCRAMBLE_ORDER, _SCRAMBLE_FRAME_SHAPE
    if _SCRAMBLE_FRAME_SHAPE == (h, w) and _SCRAMBLE_ORDER is not None:
        return _SCRAMBLE_ORDER
    bh, bw = h // grid, w // grid
    block_indices = [(i, j) for i in range(grid) for j in range(grid)]
    shuffled = block_indices[:]
    _SCRAMBLE_RNG.shuffle(shuffled)
    # 构建块级源映射: dst_block → src_block
    src_block_y = np.empty((grid, grid), dtype=np.int32)
    src_block_x = np.empty((grid, grid), dtype=np.int32)
    for old_idx, (oi, oj) in enumerate(block_indices):
        ni, nj = shuffled[old_idx]
        src_block_y[ni, nj] = oi
        src_block_x[ni, nj] = oj
    # 用广播生成像素级映射，避免 O(H*W) 循环
    dy_idx = np.arange(h, dtype=np.int32)
    dx_idx = np.arange(w, dtype=np.int32)
    block_y_idx = np.clip(dy_idx // bh, 0, grid - 1)
    block_x_idx = np.clip(dx_idx // bw, 0, grid - 1)
    src_bky = src_block_y[block_y_idx[:, None], block_x_idx[None, :]]
    src_bkx = src_block_x[block_y_idx[:, None], block_x_idx[None, :]]
    src_y = src_bky * bh + (dy_idx[:, None] % bh)
    src_x = src_bkx * bw + (dx_idx[None, :] % bw)
    _SCRAMBLE_ORDER = np.stack([src_y, src_x], axis=-1)
    _SCRAMBLE_FRAME_SHAPE = (h, w)
    return _SCRAMBLE_ORDER


def apply_scramble(frame):
    """对帧做 8x8 块置乱（模拟可学习图像加密）"""
    h, w = frame.shape[:2]
    idx = _get_scramble_order(h, w)
    scrambled = frame.copy()
    dst_y = np.arange(h)[:, None]
    dst_x = np.arange(w)[None, :]
    src_y = idx[:, :, 0]
    src_x = idx[:, :, 1]
    scrambled[dst_y, dst_x] = frame[src_y, src_x]
    return scrambled


def avg2(s, k1, k2):
    """取两个blendshape分数的均值"""
    return (s.get(k1, 0) + s.get(k2, 0)) / 2


def _draw_sidebar_alert_block(pil_draw, text, font, max_px, sx, sw, sy,
                                bg_fill, text_fill):
    """侧边栏告警卡片: 返回更新后的 sy 坐标"""
    lines = _wrap_text_lines(pil_draw, text, font, max_px)
    lh = pil_draw.textbbox((0, 0), "Ag", font=font)[3]
    row_h = lh * len(lines) + 2 * (len(lines) - 1) + 10
    pil_draw.rounded_rectangle([sx + 8, sy, sx + sw - 8, sy + row_h],
                                radius=6, fill=bg_fill)
    cy = sy + (row_h - lh * len(lines)) // 2
    for ln in lines:
        pil_draw.text((sx + 14, cy), ln, font=font, fill=text_fill)
        cy += lh + 2
    return sy + row_h + 4


def _draw_section_divider(draw, text, at_y, font, sx, sw):
    """侧边栏分组标题：文字居中，两侧细线"""
    tw_s = draw.textbbox((0, 0), text, font=font)[2]
    lx = sx + (sw - tw_s) // 2
    bar_y = at_y + 7
    draw.line([(sx + 18, bar_y), (lx - 6, bar_y)], fill=(100, 100, 120, 120), width=1)
    draw.line([(lx + tw_s + 6, bar_y), (sx + sw - 18, bar_y)], fill=(100, 100, 120, 120), width=1)
    draw.text((lx, at_y), text, font=font, fill=(160, 160, 180, 200))


# ==================== 主程序 ====================
def main():
    # 模型路径
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    mp_dir = os.path.join(models_dir, "mediapipe")
    emotion_dir = os.path.join(models_dir, "emotion")
    pose_full = os.path.join(mp_dir, "pose_landmarker_full.task")
    pose_lite = os.path.join(mp_dir, "pose_landmarker_lite.task")
    pose_model = pose_full if os.path.exists(pose_full) else pose_lite
    gesture_model = os.path.join(mp_dir, "gesture_recognizer.task")
    face_model = os.path.join(mp_dir, "face_landmarker.task")
    print(f"Pose model: {'full' if os.path.exists(pose_full) else 'lite'}")

    for path in [pose_model, gesture_model, face_model]:
        if not os.path.exists(path):
            print(f"Error: model not found: {path}")
            return

    # 异步回调结果存储
    pose_result = None
    gesture_result = None
    face_result = None

    def on_pose_result(result, _img, _ts):
        nonlocal pose_result
        pose_result = result

    def on_gesture_result(result, _img, _ts):
        nonlocal gesture_result
        gesture_result = result

    def on_face_result(result, _img, _ts):
        nonlocal face_result
        face_result = result

    # 初始化三个检测器，都是异步回调的
    pose_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model),
        running_mode=RunningMode.LIVE_STREAM,
        num_poses=4,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        result_callback=on_pose_result,
    ))
    gesture_recognizer = GestureRecognizer.create_from_options(GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=gesture_model),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.4,
        result_callback=on_gesture_result,
    ))
    face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model),
        running_mode=RunningMode.LIVE_STREAM,
        num_faces=4,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_face_blendshapes=True,
        result_callback=on_face_result,
    ))

    # 加载 DL 情绪识别器
    trained_onnx = os.path.join(emotion_dir, "enet_b0_7_fer2013.onnx")
    trained_weights = os.path.join(emotion_dir, "enet_b0_7_fer2013_weights.npz")
    if os.path.exists(trained_onnx) and os.path.exists(trained_weights):
        rec = EmotiEffLibRecognizerOnnxGPU(
            custom_onnx=trained_onnx, custom_weights=trained_weights)
        recognizers = [rec]
        print("Emotion recognizer: custom trained model (FER2013, 7-class)")
    else:
        recognizers = [
            EmotiEffLibRecognizerOnnxGPU(model_name="enet_b0_8_best_vgaf"),
            EmotiEffLibRecognizerOnnxGPU(model_name="enet_b0_8_best_afew"),
        ]
        print("Emotion recognizer: VGAF + AFEW dual-model ensemble")
    EMOTION_IDX = recognizers[0].idx_to_emotion_class
    EMOTION_LABELS = [EMOTION_IDX[i] for i in sorted(EMOTION_IDX.keys())]

    # 绘图样式
    left_lm = drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3)
    left_cn = drawing_utils.DrawingSpec(color=(0, 140, 255), thickness=3, circle_radius=2)
    right_lm = drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=3)
    right_cn = drawing_utils.DrawingSpec(color=(255, 0, 140), thickness=3, circle_radius=2)
    face_tess = drawing_utils.DrawingSpec(color=(180, 210, 220), thickness=1, circle_radius=1)

    # 每个人单独一个平滑器
    pose_smoothers: dict = {}
    face_smoothers: dict = {}
    prev_face_centroids: dict = {}  # 跨帧人员 ID 追踪

    # 打开摄像头
    cap = None
    for idx in [1, 0]:
        try:
            cap_test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap_test.isOpened():
                cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                cam_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
                cam_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera {idx}: {cam_w}x{cam_h}")
                for _ in range(5):
                    cap_test.read()  # 预热摄像头
                _, test_frame = cap_test.read()
                if test_frame is not None:
                    cap = cap_test
                    break
            cap_test.release()
        except Exception:
            continue

    if cap is None:
        print("Error: no camera available.")
        pose_landmarker.close(); gesture_recognizer.close(); face_landmarker.close()
        return

    # 根据屏幕分辨率自适应面板大小
    try:
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except Exception:
        screen_w, screen_h = 1920, 1080
    sidebar_w = 260
    margin = 0
    panel_w = (screen_w - sidebar_w - margin) // 2
    panel_h = (screen_h - margin) // 2
    panel_w = max(panel_w, 400)  # 保证最小可用尺寸
    panel_h = max(panel_h, 280)
    print(f"Panel: {panel_w}x{panel_h}, Canvas: {panel_w*2+sidebar_w}x{panel_h*2}")
    print("Camera ready. 2x2 grid mode. Press 'q' to quit.")
    print("Modes: [e] scramble  [n] OOD noise  [a] ST-GCN attention")

    prev_time = time.time()
    frame_timestamp_ms = 0
    frame_count = 0

    # DL 情绪状态
    dl_emotion_cached = None
    dl_scores_cached = None
    dl_scores_ema = None
    dl_ema_alpha = 0.35
    dl_candidate_label = None      # 滞后候选标签
    dl_candidate_streak = 0        # 候选连续帧数
    dl_hysteresis_frames = 2

    # 姿态历史 (3帧窗口, 用于动力学分析)
    pose_history = collections.deque(maxlen=3)

    # rPPG 心率缓冲
    rppg_buffer = collections.deque(maxlen=120)
    current_bpm = None
    bpm_display = None

    # 视图开关
    scramble_mode = False
    noise_mode = False
    attention_mode = False
    multi_person_mode = False  # 多人识别开关，默认单人

    # 人脸框平滑一下，不然panel2会闪
    _smooth_fx = _smooth_fy = _smooth_fw = _smooth_fh = None
    _prev_face_display = None  # 上一帧的人脸显示，防黑闪
    _face_lost_frames = 0      # 连续丢脸帧计数
    _pose_lost_frames = 0      # 连续丢姿态帧计数
    # 手部追踪缓冲，快速移动的时候不会丢
    _hand_buffer = {}           # {"Left": (landmarks, lost_frames), "Right": (landmarks, lost_frames)}

    # 画布: 2x2 布局 + 右侧栏
    canvas = np.zeros((panel_h * 2, panel_w * 2 + sidebar_w, 3), dtype=np.uint8)
    panel1 = canvas[0:panel_h, 0:panel_w]                       # 左上: 原图
    panel2 = canvas[0:panel_h, panel_w:panel_w*2]               # 右上: 表情
    panel3 = canvas[panel_h:panel_h*2, 0:panel_w]               # 左下: 骨架
    panel4 = canvas[panel_h:panel_h*2, panel_w:panel_w*2]       # 右下: 合成
    sidebar = canvas[0:panel_h*2, panel_w*2:panel_w*2+sidebar_w]  # 右侧栏
    pose_overlay = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # DL 常驻消费者线程 + 队列
    dl_queue = queue.Queue(maxsize=1)
    dl_thread_lock = threading.Lock()

    def dl_worker_loop():
        while True:
            crop_img = dl_queue.get()
            try:
                flipped = cv2.flip(crop_img, 1)
                all_scores = []
                for rec in recognizers:
                    _, s1 = rec.predict_emotions(crop_img, False)
                    _, s2 = rec.predict_emotions(flipped, False)
                    all_scores.append((s1[0] + s2[0]) * 0.5)
                scores = sum(all_scores) / len(all_scores)
                with dl_thread_lock:
                    nonlocal dl_scores_ema, dl_emotion_cached, dl_scores_cached
                    nonlocal dl_candidate_label, dl_candidate_streak
                    if dl_scores_ema is None or len(dl_scores_ema) != len(scores):
                        dl_scores_ema = scores
                    else:
                        dl_scores_ema = (dl_ema_alpha * scores
                                         + (1 - dl_ema_alpha) * dl_scores_ema)
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
            except Exception as e:
                print(f"DL inference error: {e}")
            finally:
                pass

    threading.Thread(target=dl_worker_loop, daemon=True).start()

    cam_fail = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cam_fail += 1
            if cam_fail > 30:
                print("Camera failed repeatedly, exiting")
                break
            continue
        cam_fail = 0

        # 线程安全：一次性拷贝 DL 共享状态，避免竞态条件
        with dl_thread_lock:
            dl_emotion = dl_emotion_cached
            dl_scores = dl_scores_cached.copy() if dl_scores_cached is not None else None

        # OOD 噪声注入 (在 MediaPipe / DL 处理前)
        if noise_mode:
            n_rects = np.random.randint(8, 13)
            for _ in range(n_rects):
                rw = np.random.randint(40, 121)
                rh = np.random.randint(40, 121)
                rx = np.random.randint(0, max(1, frame.shape[1] - rw))
                ry = np.random.randint(0, max(1, frame.shape[0] - rh))
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 0), -1)

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 异步推给 MediaPipe
        pose_landmarker.detect_async(mp_image, frame_timestamp_ms)
        gesture_recognizer.recognize_async(mp_image, frame_timestamp_ms)
        face_landmarker.detect_async(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33

        # 用上一帧的结果（异步会晚一帧）
        cur_pose = pose_result
        cur_hand = gesture_result
        cur_face = face_result

        # --- 多人 ID 分配 ---
        pose_list = cur_pose.pose_landmarks if cur_pose else []
        face_list = cur_face.face_landmarks if cur_face else []
        blendshapes_list = cur_face.face_blendshapes if cur_face else []

        person_faces, person_poses, prev_face_centroids, unmatched_poses = \
            assign_person_ids(face_list, pose_list, prev_face_centroids, w, h)

        # 单人模式：只保留 person 0
        if not multi_person_mode:
            person_faces = {0: person_faces[0]} if 0 in person_faces else {}
            person_poses = {0: person_poses[0]} if 0 in person_poses else {}
            prev_face_centroids = {0: prev_face_centroids[0]} if 0 in prev_face_centroids else {}
            unmatched_poses = []

        _pil_person_count = len(person_faces) or len(person_poses)

        # --- 平滑每个人的骨架和人脸 ---
        smoothed_poses = {}
        for pid in sorted(person_poses.keys()):
            pl = person_poses[pid]
            s = _get_or_create_smoother(pose_smoothers, pid, 0.5, 30)
            # 手臂交叉的时候MediaPipe会把左右搞反，检测一下要不要交换
            if pl is not None and s.smoothed is not None and len(pl) >= 17 and len(s.smoothed) >= 17:
                _pairs = [(11, 12), (13, 14), (15, 16)]  # 肩/肘/腕
                _cur_dist, _swap_dist = 0.0, 0.0
                _n = 0
                for _li, _ri in _pairs:
                    if (getattr(pl[_li], "visibility", 0) > 0.5
                            and getattr(pl[_ri], "visibility", 0) > 0.5
                            and getattr(s.smoothed[_li], "visibility", 0) > 0.5
                            and getattr(s.smoothed[_ri], "visibility", 0) > 0.5):
                        _cur_dist += ((pl[_li].x - s.smoothed[_li].x)**2
                                      + (pl[_li].y - s.smoothed[_li].y)**2
                                      + (pl[_ri].x - s.smoothed[_ri].x)**2
                                      + (pl[_ri].y - s.smoothed[_ri].y)**2)
                        _swap_dist += ((pl[_li].x - s.smoothed[_ri].x)**2
                                       + (pl[_li].y - s.smoothed[_ri].y)**2
                                       + (pl[_ri].x - s.smoothed[_li].x)**2
                                       + (pl[_ri].y - s.smoothed[_li].y)**2)
                        _n += 1
                if _n > 0 and _swap_dist < _cur_dist * 0.6:
                    _pl = list(pl)
                    for _li, _ri in _pairs:
                        _pl[_li], _pl[_ri] = _pl[_ri], _pl[_li]
                    pl = _pl
            # alpha调高点跟手快，加个单帧位移上限防骨架乱飞
            s.alpha = 0.90
            if pl is not None and s.smoothed is not None and len(pl) == len(s.smoothed):
                _max_step = 0.06  # 一帧最多动这么多，超过就是抽风
                for i in range(len(pl)):
                    if getattr(pl[i], "visibility", 0) > 0.5:
                        _dx = pl[i].x - s.smoothed[i].x
                        _dy = pl[i].y - s.smoothed[i].y
                        _d = (_dx * _dx + _dy * _dy) ** 0.5
                        if _d > _max_step:
                            _scale = _max_step / _d
                            pl[i].x = s.smoothed[i].x + _dx * _scale
                            pl[i].y = s.smoothed[i].y + _dy * _scale
            smoothed_poses[pid] = s.update(pl) if pl is not None else s.update([])

        smoothed_faces = {}
        for pid in sorted(person_faces.keys()):
            face_lms, _, _, _, _ = person_faces[pid]
            s = _get_or_create_smoother(face_smoothers, pid, 0.5, 15)
            smoothed_faces[pid] = s.update(face_lms)

        # --- 每个人的表情从blendshape拿 ---
        person_emotions = {}
        person_blends_raw = {}
        for pid, (_, _, _, _, orig_idx) in person_faces.items():
            if blendshapes_list and orig_idx < len(blendshapes_list):
                bs = blendshapes_list[orig_idx]
                el, es, mbs = classify_emotion(bs)
                person_emotions[pid] = (el, es, mbs)
                person_blends_raw[pid] = bs

        # --- 主人物 (person 0) 的便捷引用 ---
        primary_face = smoothed_faces.get(0)
        primary_pose = smoothed_poses.get(0)
        emotion_label = person_emotions.get(0, ("Neutral", 0.0, {}))[0]
        emotion_score = person_emotions.get(0, ("Neutral", 0.0, {}))[1]
        face_blends_raw = person_blends_raw.get(0)

        # --- 头部姿态估计 (仅 person 0) ---
        head_pose_angles = None
        head_pose_img_pts = None
        head_pose_text = ""
        if primary_face and len(primary_face) > 263:
            head_pose_angles, head_pose_img_pts = estimate_head_pose(primary_face, w, h)
            if head_pose_angles is not None:
                p, y, r = head_pose_angles
                dir_y = "R" if y > 5 else ("L" if y < -5 else "C")
                dir_p = "U" if p < -5 else ("D" if p > 5 else "L")
                dir_r = "TR" if r > 5 else ("TL" if r < -5 else "C")
                head_pose_text = f"Yaw:{y:+.0f}{dir_y}  Pitch:{p:+.0f}{dir_p}  Roll:{r:+.0f}{dir_r}"

        # 构建四个面板
        panel2.fill(0)

        # PIL后处理用的数据变量
        _pil_emotion_scores = {}          # pid -> scores array
        _pil_p2_no_face = False           # "No face detected" flag
        _pil_p3_pose_texts = {}           # pid -> (text, color_bgr)
        _pil_p3_hand_count = 0
        _pil_p3_gesture_labels = []
        _pil_p3_attention = False
        _pil_p4_face_bboxes = []          # list of (pid, fx, fy, fw, fh, label, score, color)
        _pil_p4_top_emotion = None
        _pil_p4_emotion_hint = None
        _pil_p4_facial_actions = []    # 脸框旁持续显示的面部动作
        _pil_notification = None
        _pil_cognitive_alert = None
        _pil_advanced_alert = None
        _pil_bpm_text = None
        _pil_mode_list = []
        _pil_person_count = len(person_faces) or len(person_poses)

        # 左上角：原始画面（或者加密模式下的乱码）
        if scramble_mode:
            scrambled = apply_scramble(frame)
            cv2.resize(scrambled, (panel_w, panel_h), dst=panel1)
        else:
            cv2.resize(frame, (panel_w, panel_h), dst=panel1)

        # 人脸丢了别急着算无人，等15帧再说
        if primary_face and face_blends_raw:
            _face_lost_frames = 0
        else:
            _face_lost_frames += 1
        _face_stable_lost = _face_lost_frames >= 15

        fxp4 = fyp4 = fwp4 = fhp4 = 0
        if primary_face and face_blends_raw:
            fx, fy, fw, fh = get_face_bbox(primary_face, w, h, margin=0.35)
            fx, fy = max(0, fx), max(0, fy)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)

            # 平滑一下bbox不然panel2会闪
            if _smooth_fx is None:
                _smooth_fx, _smooth_fy = fx, fy
                _smooth_fw, _smooth_fh = fw, fh
            else:
                alpha = 0.35
                _smooth_fx += (fx - _smooth_fx) * alpha
                _smooth_fy += (fy - _smooth_fy) * alpha
                _smooth_fw += (fw - _smooth_fw) * alpha
                _smooth_fh += (fh - _smooth_fh) * alpha
            fx = int(_smooth_fx)
            fy = int(_smooth_fy)
            fw = int(_smooth_fw)
            fh = int(_smooth_fh)

            # 算一下总览里的人脸框，缩放一下，边距小一点
            sf = panel_w / w  # 缩放因子
            fxp4, fyp4 = int(fx * sf), int(fy * sf)
            fwp4, fhp4 = int(fw * sf), int(fh * sf)

            if fw > 0 and fh > 0:
                face_crop = frame[fy:fy+fh, fx:fx+fw].copy()

                # rPPG：取额头区域绿色通道均值
                fh_crop, fw_crop = face_crop.shape[:2]
                roi_x1 = int(fw_crop * 0.30)
                roi_x2 = int(fw_crop * 0.70)
                roi_y2 = max(1, int(fh_crop * 0.30))
                if roi_x2 > roi_x1 and roi_y2 > 0:
                    forehead_roi = face_crop[0:roi_y2, roi_x1:roi_x2]
                    g_mean = float(np.mean(forehead_roi[:, :, 1]))
                    rppg_buffer.append(g_mean)

                # 丢给DL去推理，不阻塞主线程，队列满了就跳过
                if frame_count % 2 == 0:
                    try:
                        dl_queue.put_nowait(face_crop.copy())
                    except queue.Full:
                        pass

                scaled = [ScaledLandmark(lm, fx, fy, fw, fh, w, h) for lm in primary_face]

                # 画面部网格 + 情绪高亮
                drawing_utils.draw_landmarks(
                    image=face_crop, landmark_list=scaled,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=face_tess,
                    connection_drawing_spec=face_tess,
                )
                draw_emotion_focus(face_crop, scaled, dl_emotion or "Neutral")

                # 在裁剪脸上画头部姿态坐标轴
                if head_pose_angles is not None and head_pose_img_pts is not None:
                    crop_img_pts = []
                    for pt in head_pose_img_pts:
                        crop_img_pts.append([
                            (pt[0] - fx) * panel_w / fw,
                            (pt[1] - fy) * panel_h / fh,
                        ])
                    draw_head_pose_axes(face_crop, np.array(crop_img_pts), head_pose_angles, size=30)

                cv2.resize(face_crop, (panel_w, panel_h), dst=panel2)
                _prev_face_display = panel2.copy()

            # 收集情绪分数，供侧边栏柱状图用 (person 0)
            if dl_scores is not None:
                _pil_emotion_scores[0] = dl_scores
            elif face_blends_raw is not None:
                s = {bs.category_name: bs.score for bs in face_blends_raw}
                _bs_raw = [
                    avg2(s, "mouthSmileLeft", "mouthSmileRight"),
                    max(s.get("browInnerUp", 0), avg2(s, "mouthFrownLeft", "mouthFrownRight")),
                    max(s.get("browInnerUp", 0), s.get("jawOpen", 0)),
                    max(avg2(s, "browDownLeft", "browDownRight"),
                        avg2(s, "mouthFrownLeft", "mouthFrownRight")),
                    max(s.get("browInnerUp", 0), s.get("jawOpen", 0)) * 0.8,
                    max(avg2(s, "noseSneerLeft", "noseSneerRight"),
                        avg2(s, "mouthUpperUpLeft", "mouthUpperUpRight")),
                    s.get("_neutral", 0),
                    s.get("mouthPressLeft", 0) * 0.5,
                ]
                _pil_emotion_scores[0] = np.array(_bs_raw[:len(EMOTION_LABELS)], dtype=np.float64)
        elif _face_stable_lost:
            _pil_p2_no_face = True
        if not (primary_face and face_blends_raw):
            if _prev_face_display is not None:
                np.copyto(panel2, _prev_face_display)

        # 面板3：身体骨骼 + 关节角度 + 手势
        panel3.fill(0)

        prev_pose_from_history = pose_history[-2] if len(pose_history) >= 2 else None
        # --- 多人骨架渲染 ---
        for pid in sorted(smoothed_poses.keys()):
            sp = smoothed_poses[pid]
            if sp is None:
                continue
            color = _PERSON_COLORS_BGR[pid % len(_PERSON_COLORS_BGR)]
            cn_color = tuple(int(c * 0.5) for c in color)
            draw_pose_full(panel3, sp, panel_w, panel_h,
                           lm_color=color, cn_color=cn_color,
                           thickness=3, skip_face=False,
                           prev_landmarks=(prev_pose_from_history if pid == 0 else None),
                           attention_mode=attention_mode)
            if pid == 0:
                draw_joint_angles(panel3, sp, panel_w, panel_h)
            n_visible = sum(1 for lm in sp if getattr(lm, "visibility", 0) > 0.5)
            _pil_p3_pose_texts[pid] = (f"P{pid}: {n_visible}/33", color)

        # 未关联到人脸的姿态用灰色绘制
        for up_lms in unmatched_poses:
            draw_pose_full(panel3, up_lms, panel_w, panel_h,
                           lm_color=(150, 150, 150), cn_color=(100, 100, 100),
                           thickness=1, skip_face=False)

        if smoothed_poses or unmatched_poses:
            _pose_lost_frames = 0
        else:
            _pose_lost_frames += 1
        if _pose_lost_frames >= 12:
            _pil_p3_pose_texts[-1] = ("Pose: NOT FOUND - step back", (0, 0, 255))

        current_active_gestures = set()
        two_hand_heart = False
        if (cur_hand and cur_hand.hand_landmarks
                and len(cur_hand.hand_landmarks) == 2
                and is_two_handed_heart(cur_hand.hand_landmarks, cur_hand.handedness)):
            two_hand_heart = True
            current_active_gestures.add("Two_Handed_Heart")

        # 手部追踪缓冲：更新检测到的手，丢失的手保留最多8帧渐隐
        if cur_hand and cur_hand.hand_landmarks:
            _pil_p3_hand_count = len(cur_hand.hand_landmarks)
            detected = set()
            for i, hand_lms in enumerate(cur_hand.hand_landmarks):
                handedness = cur_hand.handedness[i][0].category_name
                hand_conf = cur_hand.handedness[i][0].score
                # 脸被误识别成手了，看手腕有没有贴到脸上
                _face_lms = smoothed_faces.get(0)
                if _face_lms is not None and len(_face_lms) > 14:
                    # 取鼻尖和上唇的中间点当脸的中心
                    _cx = (_face_lms[1].x + _face_lms[13].x) * 0.5
                    _cy = (_face_lms[1].y + _face_lms[13].y) * 0.5
                    _wx, _wy = hand_lms[0].x, hand_lms[0].y
                    if (_cx - _wx) ** 2 + (_cy - _wy) ** 2 < 0.0049:
                        continue
                detected.add(handedness)
                _hand_buffer[handedness] = (hand_lms, 0, hand_conf)
            for _h in list(_hand_buffer.keys()):
                if _h not in detected:
                    lms, lost, conf = _hand_buffer[_h]
                    lost += 1
                    if lost <= 8:
                        _hand_buffer[_h] = (lms, lost, conf)
                    else:
                        del _hand_buffer[_h]
        else:
            _pil_p3_hand_count = 0
            for _h in list(_hand_buffer.keys()):
                lms, lost, conf = _hand_buffer[_h]
                lost += 1
                if lost <= 8:
                    _hand_buffer[_h] = (lms, lost, conf)
                else:
                    del _hand_buffer[_h]

        # 渲染缓冲中的所有手
        for handedness, (hand_lms, lost, hand_conf) in list(_hand_buffer.items()):
            lm_s, cn_s = (left_lm, left_cn) if handedness == "Left" else (right_lm, right_cn)
            if lost > 0:
                fade = max(0.25, 1.0 - lost / 9.0)
                lm_s = drawing_utils.DrawingSpec(
                    color=tuple(int(c * fade) for c in lm_s.color),
                    thickness=lm_s.thickness, circle_radius=lm_s.circle_radius)
                cn_s = drawing_utils.DrawingSpec(
                    color=tuple(int(c * fade) for c in cn_s.color),
                    thickness=cn_s.thickness, circle_radius=cn_s.circle_radius)
            drawing_utils.draw_landmarks(
                image=panel3, landmark_list=hand_lms,
                connections=HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=lm_s, connection_drawing_spec=cn_s,
            )
            # 手腕与胳膊骨架连线
            primary_pose = smoothed_poses.get(0)
            if primary_pose is not None:
                wrist_idx = 15 if handedness == "Left" else 16
                pw = primary_pose[wrist_idx]
                px_w, py_w = int(pw.x * panel_w), int(pw.y * panel_h)
                hw = hand_lms[0]
                hx_w, hy_w = int(hw.x * panel_w), int(hw.y * panel_h)
                cv2.line(panel3, (px_w, py_w), (hx_w, hy_w),
                         cn_s.color, cn_s.thickness)

        # 手势识别，只认当前帧检测到的手
        if cur_hand and cur_hand.hand_landmarks:
            for i, hand_lms in enumerate(cur_hand.hand_landmarks):
                handedness = cur_hand.handedness[i][0].category_name
                hand_conf = cur_hand.handedness[i][0].score
                if hand_conf < 0.55:
                    continue
                ml_gesture = None
                if (cur_hand and cur_hand.gestures
                        and i < len(cur_hand.gestures)
                        and cur_hand.gestures[i]):
                    cat = cur_hand.gestures[i][0]
                    if cat.category_name != "None" and cat.score > 0.6:
                        ml_gesture = cat.category_name
                if two_hand_heart:
                    ml_gesture = "Two_Handed_Heart"
                elif is_finger_heart(hand_lms):
                    ml_gesture = "Finger_Heart"
                elif is_middle_finger_extended(hand_lms):
                    ml_gesture = "Middle_Finger"
                if ml_gesture:
                    current_active_gestures.add(ml_gesture)
                    wrist = hand_lms[0]
                    _pil_p3_gesture_labels.append(
                        (handedness, ml_gesture, int(wrist.x * panel_w), int(wrist.y * panel_h))
                    )

        if attention_mode:
            _pil_p3_attention = True

        # 面板4：融合视图
        cv2.resize(frame, (panel_w, panel_h), dst=panel4)

        # 把骨架画到总览上，单人蓝色半透明，多人每人一个颜色
        if smoothed_poses:
            pose_overlay.fill(0)
            for pid in sorted(smoothed_poses.keys()):
                sp = smoothed_poses[pid]
                if sp is None:
                    continue
                if multi_person_mode:
                    color = _PERSON_COLORS_BGR[pid % len(_PERSON_COLORS_BGR)]
                else:
                    color = (200, 50, 0)  # 半透明蓝 (BGR)
                cn_color = tuple(int(c * 0.7) for c in color)
                draw_pose_full(pose_overlay, sp, panel_w, panel_h,
                               lm_color=color, cn_color=cn_color,
                               thickness=2, skip_face=True,
                               prev_landmarks=(prev_pose_from_history if pid == 0 else None),
                               attention_mode=attention_mode)
            cv2.addWeighted(pose_overlay, 0.50, panel4, 1.0, 0, dst=panel4)

        for handedness, (hand_lms, lost, hand_conf) in list(_hand_buffer.items()):
            lm_s, cn_s = (left_lm, left_cn) if handedness == "Left" else (right_lm, right_cn)
            if lost > 0:
                fade = max(0.25, 1.0 - lost / 9.0)
                lm_s = drawing_utils.DrawingSpec(
                    color=tuple(int(c * fade) for c in lm_s.color),
                    thickness=lm_s.thickness, circle_radius=lm_s.circle_radius)
                cn_s = drawing_utils.DrawingSpec(
                    color=tuple(int(c * fade) for c in cn_s.color),
                    thickness=cn_s.thickness, circle_radius=cn_s.circle_radius)
            drawing_utils.draw_landmarks(
                image=panel4, landmark_list=hand_lms,
                connections=HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=lm_s, connection_drawing_spec=cn_s,
            )

        # 多人脸框 + person 标签
        sf = panel_w / w
        for pid in sorted(person_faces.keys()):
            face_lms, _, _, _, _ = person_faces[pid]
            sm_face = smoothed_faces.get(pid)
            if sm_face is None:
                continue
            fx_f, fy_f, fw_f, fh_f = get_face_bbox(sm_face, w, h, margin=0)
            fxp = int(fx_f * sf)
            fyp = int(fy_f * sf)
            fwp = int(fw_f * sf)
            fhp = int(fh_f * sf)
            if multi_person_mode:
                color = _PERSON_COLORS_BGR[pid % len(_PERSON_COLORS_BGR)]
                cv2.putText(panel4, f"P{pid}", (fxp, fyp - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            else:
                color = (0, 220, 0)  # 单人模式绿框
            cv2.rectangle(panel4, (fxp, fyp), (fxp + fwp, fyp + fhp), color, 2)

        # P0的DL情绪标签（人脸框上面那个循环已经画了）
        if 0 in smoothed_faces and primary_face and dl_scores is not None:
            best_idx = int(np.argmax(dl_scores))
            _pil_p4_top_emotion = (
                EMOTION_LABELS[best_idx],
                float(dl_scores[best_idx]),
                _EMOTION_COLORS_8.get(EMOTION_LABELS[best_idx], (255, 255, 255)),
                fxp4, fyp4, fwp4, fhp4,
            )

        # 次要人物的 blendshape 情绪标签
        for pid in sorted(person_faces.keys()):
            if pid == 0 or pid not in person_emotions:
                continue
            em_label, em_score, _ = person_emotions[pid]
            sm_face = smoothed_faces.get(pid)
            if sm_face:
                fx_f, fy_f, fw_f, fh_f = get_face_bbox(sm_face, w, h, margin=0)
                fxp_s = int(fx_f * sf)
                fyp_s = int(fy_f * sf)
                fwp_s = int(fw_f * sf)
                color = _PERSON_COLORS_BGR[pid % len(_PERSON_COLORS_BGR)]
                _pil_p4_face_bboxes.append((pid, fxp_s, fyp_s + fhp_s + 4,
                                            em_label, em_score, color))

        # 把表情和手势合在一起猜意图
        dominant_emotion = dl_emotion or emotion_label
        dominant_gesture = next(iter(current_active_gestures), None) if current_active_gestures else None
        intent = synthesize_intent(dominant_emotion, dominant_gesture)
        if intent:
            _pil_notification = (f"意图: {intent}", _DISSONANCE_TAG in intent)
        elif dominant_emotion:
            _pil_p4_emotion_hint = f"Emotion: {dominant_emotion}"

        # ---- 认知与预判警报 ----
        if len(pose_history) == 3:
            cognitive = analyze_cognitive_and_anticipation(pose_history, dominant_emotion)
            if cognitive is not None:
                _pil_cognitive_alert = cognitive  # (alert_text, bg_color_bgr)

        # FPS计算
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time
        frame_count += 1

        # rPPG心率，每15帧算一次
        if frame_count % 15 == 0 and len(rppg_buffer) >= 120:
            raw_bpm = estimate_heart_rate(rppg_buffer, fps)
            if raw_bpm is not None:
                if current_bpm is None:
                    current_bpm = float(raw_bpm)
                else:
                    current_bpm = current_bpm * 0.7 + raw_bpm * 0.3
                bpm_int = int(round(current_bpm))
                bpm_color = (0, 255, 100) if bpm_int < 90 else (0, 200, 255) if bpm_int < 110 else (0, 100, 255)
                bpm_display = (f"BPM: {bpm_int}", bpm_color)
        _pil_bpm_text = bpm_display


        # ---- 脸上的动作标签 ----
        _pil_p4_facial_actions = []
        if face_blends_raw:
            _pil_p4_facial_actions = get_facial_actions(face_blends_raw, top_n=4)

        # ---- 看你眼睛看的方向和手指指的方向差多少 ----
        adv_result = None
        if face_blends_raw:
            cur_hand_list = cur_hand.hand_landmarks if (cur_hand and cur_hand.hand_landmarks) else None
            adv_result = advanced_cognitive_engine(
                cur_hand_list, head_pose_angles,
                dominant_gesture,
            )

        # 高级警报覆盖通知
        if adv_result is not None:
            _pil_advanced_alert = adv_result  # (alert_text, bg_color_bgr)

        # 收集各模式状态
        for mode_name, is_on, color in [
            ("NOISE", noise_mode, (255, 100, 100)),
            ("SCRAMBLE", scramble_mode, (100, 100, 255)),
            ("ATTN", attention_mode, (255, 200, 50)),
            ("MULTI", multi_person_mode, (100, 255, 200)),
        ]:
            if is_on:
                _pil_mode_list.append((mode_name, color))

        # 拼合画布
        canvas[:panel_h, :panel_w] = panel1
        canvas[:panel_h, panel_w:panel_w * 2] = panel2
        canvas[panel_h:panel_h * 2, :panel_w] = panel3
        canvas[panel_h:panel_h * 2, panel_w:panel_w * 2] = panel4

        # 统一用PIL画标题、文字、圆角卡片
        pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA))
        pil_draw = ImageDraw.Draw(pil_canvas)

        # 准备不同大小的字体
        font_panel = _get_cached_font(17)
        font_info = _get_cached_font(20)
        font_small = _get_cached_font(16)
        font_tiny = _get_cached_font(14)
        font_emotion = _get_cached_font(18)

        # 面板边界，防止文字溢出
        _p2_bounds = (panel_w + 8, panel_w * 2 - 8)
        _p4_bounds = (panel_w + 8, panel_w * 2 - 8)

        # 面板标题位置：对角放置，不挡画面

        # 辅助函数：先量文字宽度，再画卡片
        def _title_card_w(text, font, pad_x=14):
            bbox = pil_draw.textbbox((0, 0), text, font=font)
            return (bbox[2] - bbox[0]) + pad_x * 2

        # 统一四个面板标题卡片的宽度，取最宽的为准
        _titles = ["Original", "Expression", "Body Skeleton", "Combined"]
        _uniform_title_w = max(_title_card_w(t, font_panel) for t in _titles)

        # 面板1：标题右下角，FPS左上角
        p1_title = "Original [ENCRYPTED]" if scramble_mode else "Original"
        draw_modern_hud_panel(pil_draw, p1_title, panel_w - _uniform_title_w - 8, panel_h - 36,
                              font_panel, pad_x=14, pad_y=8, radius=10,
                              text_color=(255, 100, 100, 255) if scramble_mode else (200, 200, 200, 255),
                              bg_color=(80, 0, 0, 200) if scramble_mode else (30, 30, 30, 180),
                              accent_color=(200, 50, 50, 255) if scramble_mode else (0, 200, 200, 255),
                              card_width=_uniform_title_w)
        # FPS 药丸徽章：颜色按帧率分档
        if fps >= 25:
            fps_color = (0, 180, 80)
        elif fps >= 15:
            fps_color = (200, 160, 0)
        else:
            fps_color = (200, 40, 40)
        fps_str = f"FPS {int(fps)}"
        fps_bbox = pil_draw.textbbox((0, 0), fps_str, font=font_tiny)
        fps_tw, fps_th = fps_bbox[2] - fps_bbox[0], fps_bbox[3] - fps_bbox[1]
        fps_pad = 10
        pil_draw.rounded_rectangle(
            [10, 6, 10 + fps_tw + fps_pad * 2, 6 + fps_th + fps_pad],
            radius=fps_th // 2 + 4,
            fill=(fps_color[2] // 6, fps_color[1] // 6, fps_color[0] // 6, 200))
        pil_draw.text((10 + fps_pad, 6 + fps_pad // 2 - fps_bbox[1]), fps_str,
                      font=font_tiny, fill=(fps_color[2], fps_color[1], fps_color[0], 255))

        # 面板2：标题左下角
        draw_modern_hud_panel(pil_draw, "Expression", panel_w + 8, panel_h - 36,
                              font_panel, pad_x=14, pad_y=8, radius=10,
                              text_color=(200, 200, 200, 255), bg_color=(30, 30, 30, 180),
                              accent_color=(200, 50, 200, 255), card_width=_uniform_title_w)

        # 面板3：标题右上角
        draw_modern_hud_panel(pil_draw, "Body Skeleton", panel_w - _uniform_title_w - 8, panel_h + 8,
                              font_panel, pad_x=14, pad_y=8, radius=10,
                              text_color=(200, 200, 200, 255), bg_color=(30, 30, 30, 180),
                              accent_color=(50, 200, 50, 255), card_width=_uniform_title_w)

        # 面板4：标题左上角，BPM右上角
        draw_modern_hud_panel(pil_draw, "Combined", panel_w + 8, panel_h + 8,
                              font_panel, pad_x=14, pad_y=8, radius=10,
                              text_color=(200, 200, 200, 255), bg_color=(30, 30, 30, 180),
                              accent_color=(200, 150, 0, 255), card_width=_uniform_title_w)
        if _pil_bpm_text is not None:
            bpm_str, bpm_bgr = _pil_bpm_text
            bpm_tw = pil_draw.textbbox((0, 0), bpm_str, font=font_info)[2]
            pil_draw.text((panel_w * 2 - bpm_tw - 14, panel_h + 10),
                          bpm_str, font=font_info, fill=(255, 20, 20, 255))
        if _pil_p3_attention:
            pil_draw.text((panel_w - 200, panel_h * 2 - 28),
                          "ST-GCN ATTENTION ON", font=font_tiny,
                          fill=(0, 165, 255, 200))

        # 面板4：人脸检测框旁边显示Top-1情绪
        if _pil_p4_top_emotion:
            lbl, scr, clr, fx, fy, fw, fh = _pil_p4_top_emotion
            # 情绪标签放人脸框右边，放不下就放下面
            label_text = f"{lbl} {scr:.0%}"
            lbbox = pil_draw.textbbox((0, 0), label_text, font=font_emotion)
            ltw = lbbox[2] - lbbox[0]
            if fx + fw + ltw + 16 < panel_w:
                lx = fx + fw + 8
                ly = fy + 8
            else:
                lx = fx + 8
                ly = fy + fh + 6
            # 限制在面板4范围内
            lx = max(4, min(int(lx), panel_w - ltw - 12))
            ly = max(4, min(int(ly), panel_h - 28))
            lx_canvas = panel_w + lx  # offset into panel 4
            ly_canvas = panel_h + ly
            pil_draw.rounded_rectangle(
                [lx_canvas - 4, ly_canvas - 2, lx_canvas + ltw + 4, ly_canvas + 24],
                radius=6, fill=(clr[2] // 4, clr[1] // 4, clr[0] // 4, 180))
            pil_draw.text((lx_canvas, ly_canvas), label_text,
                          font=font_emotion, fill=(clr[2], clr[1], clr[0], 255))

            # 动作标签放情绪下面
            if _pil_p4_facial_actions:
                action_y = ly_canvas + 26
                for action_name, action_val in _pil_p4_facial_actions:
                    action_text = f"{action_name}: {action_val:.0%}"
                    pil_draw.text((lx_canvas, action_y), action_text,
                                  font=font_tiny, fill=(0, 255, 200, 255))
                    action_y += 15

        # 次要人物的情绪标签
        for (pid, fx_s, fy_s, em_lbl, em_sc, em_clr) in _pil_p4_face_bboxes:
            label_s = f"P{pid} {em_lbl} {em_sc:.0%}"
            sb = pil_draw.textbbox((0, 0), label_s, font=font_tiny)
            stw = sb[2] - sb[0]
            sx_l = max(4, min(int(fx_s), panel_w - stw - 8))
            sy_l = max(4, min(int(fy_s), panel_h - 18))
            sx_c = panel_w + sx_l
            sy_c = panel_h + sy_l
            pil_draw.rounded_rectangle(
                [sx_c - 3, sy_c - 1, sx_c + stw + 3, sy_c + 18],
                radius=4, fill=(em_clr[2] // 5, em_clr[1] // 5, em_clr[0] // 5, 160))
            pil_draw.text((sx_c, sy_c), label_s,
                          font=font_tiny, fill=(em_clr[2], em_clr[1], em_clr[0], 220))

        # 通知/意图卡片
        displayed_notification = None
        if _pil_advanced_alert is not None:
            displayed_notification = _pil_advanced_alert
        elif _pil_notification is not None:
            displayed_notification = _pil_notification

        if displayed_notification is not None:
            alert_text, alert_info = displayed_notification
            if isinstance(alert_info, bool):
                draw_pil_notification_card(pil_draw, alert_text,
                                           panel_w + panel_w // 2, panel_h + 36,
                                           font_info, is_high_alert=alert_info,
                                           bounds=_p4_bounds)
            else:
                draw_pil_text_card(pil_draw, alert_text,
                                   panel_w + panel_w // 2, panel_h + 36,
                                   font_info,
                                   text_color=(255, 255, 255, 255),
                                   bg_color=(alert_info[2], alert_info[1], alert_info[0], 220),
                                   radius=12, pad_x=16, pad_y=10,
                                   bounds=_p4_bounds)

        # 底部的认知警报
        if _pil_cognitive_alert is not None:
            c_alert, c_bg = _pil_cognitive_alert
            draw_pil_text_card(pil_draw, c_alert,
                               panel_w + panel_w // 2, panel_h * 2 - 68,
                               font_small,
                               text_color=(255, 255, 255, 255),
                               bg_color=(c_bg[2], c_bg[1], c_bg[0], 200),
                               radius=10, pad_x=12, pad_y=6,
                               bounds=_p4_bounds)

        # 底部的状态标签，描边风格
        mode_x = 12
        for mode_name, mode_color in _pil_mode_list:
            bbox = pil_draw.textbbox((0, 0), mode_name, font=font_tiny)
            mw, mh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            r, g, b = mode_color[2], mode_color[1], mode_color[0]
            pil_draw.rounded_rectangle(
                [mode_x, panel_h * 2 - mh - 16, mode_x + mw + 18, panel_h * 2 - 8],
                radius=6, fill=(r // 4, g // 4, b // 4, 140))
            pil_draw.rounded_rectangle(
                [mode_x, panel_h * 2 - mh - 16, mode_x + mw + 18, panel_h * 2 - 8],
                radius=6, outline=(r, g, b, 220), width=1)
            pil_draw.text((mode_x + 9, panel_h * 2 - mh - 14), mode_name,
                          font=font_tiny, fill=(r, g, b, 255))
            mode_x += mw + 28

        # 右侧栏：情绪监控面板
        sx = panel_w * 2
        sw = sidebar_w
        _max_side_px = sw - 32  # 侧边栏可用文字宽度

        # 侧边栏背景
        pil_draw.rounded_rectangle([sx + 4, 4, sx + sw - 4, panel_h * 2 - 4],
                                    radius=12, fill=(18, 20, 28, 220))

        # 侧边栏标题
        side_title_w = pil_draw.textbbox((0, 0), "Deep Emotion Analysis", font=font_panel)
        side_tw = side_title_w[2] - side_title_w[0]
        pil_draw.text((sx + (sw - side_tw) // 2, 14), "Deep Emotion Analysis",
                      font=font_panel, fill=(255, 255, 255, 255))

        sy = 52

        # 人数计数 + 模式标识
        mode_label = "Multi" if multi_person_mode else "Single"
        mode_color = (100, 255, 200) if multi_person_mode else (200, 200, 100)
        mode_str = f"{mode_label}  |  People: {_pil_person_count}" if _pil_person_count > 0 else mode_label
        mode_tw = pil_draw.textbbox((0, 0), mode_str, font=font_tiny)[2]
        pil_draw.text((sx + (sw - mode_tw) // 2, 36), mode_str,
                      font=font_tiny, fill=(mode_color[0], mode_color[1], mode_color[2], 220))

        if _pil_p2_no_face or _pil_person_count == 0:
            pil_draw.text((sx + 14, sy + 20), "No person detected",
                          font=font_info, fill=(255, 100, 100, 255))
        else:
            # 深度学习情绪识别结果 (P0)
            if dl_emotion is not None and dl_scores is not None:
                _draw_section_divider(pil_draw, "── STATUS ──", sy, font_tiny, sx, sw)
                sy += 18
                best_idx = int(np.argmax(dl_scores))
                best_score_dl = float(dl_scores[best_idx])
                color_dl = _EMOTION_COLORS_8.get(dl_emotion, (0, 255, 0))
                pil_draw.rounded_rectangle([sx + 10, sy, sx + sw - 10, sy + 28],
                                            radius=6, fill=(color_dl[2] // 4, color_dl[1] // 4, color_dl[0] // 4, 120))
                r_dl, g_dl, b_dl = int(color_dl[2]), int(color_dl[1]), int(color_dl[0])
                pil_draw.ellipse([sx + 18, sy + 8, sx + 26, sy + 16], fill=(r_dl, g_dl, b_dl, 255))
                pil_draw.text((sx + 32, sy + 4), f"P0 DL: {dl_emotion}",
                              font=font_small, fill=(r_dl, g_dl, b_dl, 255))
                score_x = sx + sw - 16 - pil_draw.textbbox((0, 0), f"{best_score_dl:.0%}", font=font_small)[2]
                pil_draw.text((score_x, sy + 4), f"{best_score_dl:.0%}",
                              font=font_small, fill=(r_dl, g_dl, b_dl, 200))
                sy += 32

            # P0 blendshape 情绪
            if 0 in person_emotions:
                p0_em, p0_sc, _ = person_emotions[0]
                color_bs = _EMOTION_COLORS_8.get(p0_em, (0, 200, 255))
                pil_draw.rounded_rectangle([sx + 10, sy, sx + sw - 10, sy + 28],
                                            radius=6, fill=(color_bs[2] // 4, color_bs[1] // 4, color_bs[0] // 4, 120))
                r_bs, g_bs, b_bs = int(color_bs[2]), int(color_bs[1]), int(color_bs[0])
                pil_draw.ellipse([sx + 18, sy + 8, sx + 26, sy + 16], fill=(r_bs, g_bs, b_bs, 255))
                pil_draw.text((sx + 32, sy + 4), f"P0 BS: {p0_em}",
                              font=font_small, fill=(r_bs, g_bs, b_bs, 255))
                score_x = sx + sw - 16 - pil_draw.textbbox((0, 0), f"{p0_sc:.0%}", font=font_small)[2]
                pil_draw.text((score_x, sy + 4), f"{p0_sc:.0%}",
                              font=font_small, fill=(r_bs, g_bs, b_bs, 200))
                sy += 32

            # 次要人物 blendshape 情绪
            for pid in sorted(person_emotions.keys()):
                if pid == 0:
                    continue
                p_em, p_sc, _ = person_emotions[pid]
                p_color = _PERSON_COLORS_BGR[pid % len(_PERSON_COLORS_BGR)]
                pil_draw.rounded_rectangle([sx + 10, sy, sx + sw - 10, sy + 24],
                                            radius=5, fill=(p_color[2] // 5, p_color[1] // 5, p_color[0] // 5, 100))
                pil_draw.ellipse([sx + 18, sy + 6, sx + 24, sy + 12],
                                 fill=(int(p_color[2]), int(p_color[1]), int(p_color[0]), 255))
                pil_draw.text((sx + 30, sy + 2), f"P{pid}: {p_em}",
                              font=font_tiny, fill=(int(p_color[2]), int(p_color[1]), int(p_color[0]), 220))
                score_x = sx + sw - 16 - pil_draw.textbbox((0, 0), f"{p_sc:.0%}", font=font_tiny)[2]
                pil_draw.text((score_x, sy + 2), f"{p_sc:.0%}",
                              font=font_tiny, fill=(int(p_color[2]), int(p_color[1]), int(p_color[0]), 180))
                sy += 26

            # 每个人的身体状态
            _draw_section_divider(pil_draw, "── BODY ──", sy, font_tiny, sx, sw)
            sy += 18
            for pid in sorted(_pil_p3_pose_texts.keys()):
                txt, clr = _pil_p3_pose_texts[pid]
                pil_draw.text((sx + 14, sy), txt, font=font_small,
                              fill=(clr[2], clr[1], clr[0], 255))
                sy += 20 if pid >= 0 else 22
            if _pil_p3_hand_count > 0:
                for h, g, gx, gy in _pil_p3_gesture_labels:
                    label = f"{'R' if h[0] == 'R' else 'L'} Hand: {g}"
                    pil_draw.text((sx + 14, sy), label,
                                  font=font_tiny, fill=(0, 255, 255, 200))
                    sy += 16
                if not _pil_p3_gesture_labels:
                    pil_draw.text((sx + 14, sy), f"Hands: {_pil_p3_hand_count}",
                                  font=font_tiny, fill=(0, 255, 255, 200))
                    sy += 16

            # 头部姿态
            if head_pose_text:
                ht = f"Head: {head_pose_text}"
                h_lines = _wrap_text_lines(pil_draw, ht, font_tiny, _max_side_px)
                for hl in h_lines:
                    pil_draw.text((sx + 14, sy), hl,
                                  font=font_tiny, fill=(200, 200, 100, 255))
                    sy += 16
                sy += 2

            # 心率显示
            if _pil_bpm_text is not None:
                bpm_str, bpm_bgr = _pil_bpm_text
                bpm_rgba = (bpm_bgr[2], bpm_bgr[1], bpm_bgr[0], 255)
                pil_draw.rounded_rectangle([sx + 10, sy, sx + sw - 10, sy + 28],
                                            radius=8, fill=(bpm_bgr[2] // 5, bpm_bgr[1] // 5, bpm_bgr[0] // 5, 140))
                bpm_tw = pil_draw.textbbox((0, 0), bpm_str, font=font_small)[2]
                pil_draw.text((sx + (sw - bpm_tw) // 2, sy + 3), bpm_str,
                              font=font_small, fill=bpm_rgba)
                sy += 32

            # 分隔线 → EMOTIONS 分组标题
            _draw_section_divider(pil_draw, "── EMOTIONS ──", sy, font_tiny, sx, sw)
            sy += 18

            # 情绪分数条 (P0)
            if _pil_emotion_scores.get(0) is not None:
                bar_w = sw - 28
                remaining_h = panel_h * 2 - sy - 20
                bar_h = min(20, remaining_h // 8 - 5)
                gap = max(2, bar_h // 8)
                if bar_h >= 6:
                    draw_pil_emotion_bars(pil_draw, sx + 14, sy, bar_w, bar_h, gap,
                                          _pil_emotion_scores[0], EMOTION_LABELS, font_tiny)
                    sy += (bar_h + gap) * 8 + 6

            # 多人颜色图例
            if _pil_person_count > 1:
                _draw_section_divider(pil_draw, "── PEOPLE ──", sy, font_tiny, sx, sw)
                sy += 16
                for pid in sorted(person_faces.keys()):
                    color = _PERSON_COLORS_BGR[pid % len(_PERSON_COLORS_BGR)]
                    r_c, g_c, b_c = int(color[2]), int(color[1]), int(color[0])
                    pil_draw.ellipse([sx + 14, sy + 4, sx + 22, sy + 12], fill=(r_c, g_c, b_c, 255))
                    em_info = person_emotions.get(pid, (None, 0.0, {}))
                    label = f"P{pid}" + (f"  {em_info[0]}" if em_info[0] else "")
                    pil_draw.text((sx + 28, sy), label, font=font_tiny, fill=(r_c, g_c, b_c, 220))
                    sy += 16
                sy += 4

            # 认知分析区
            sy = max(sy, panel_h * 2 - 200)
            has_cognitive = False

            # 意图/手势
            if _pil_notification is not None:
                notif_text, is_alert = _pil_notification
                short = notif_text.replace("意图: ", "").replace("[失调警报]", "[!]")
                text_fill = (255, 80, 80, 255) if is_alert else (0, 220, 220, 255)
                sy = _draw_sidebar_alert_block(pil_draw, short, font_tiny, _max_side_px,
                                               sx, sw, sy, (30, 30, 40, 180), text_fill)
                has_cognitive = True

            # 认知警报
            if _pil_cognitive_alert is not None:
                c_alert, c_bg = _pil_cognitive_alert
                sy = _draw_sidebar_alert_block(pil_draw, c_alert, font_tiny, _max_side_px,
                                               sx, sw, sy,
                                               (c_bg[2] // 4, c_bg[1] // 4, c_bg[0] // 4, 180),
                                               (c_bg[2], c_bg[1], c_bg[0], 255))
                has_cognitive = True

            # 高级认知警报
            if _pil_advanced_alert is not None:
                adv_alert, adv_bg = _pil_advanced_alert
                sy = _draw_sidebar_alert_block(pil_draw, adv_alert, font_tiny, _max_side_px,
                                               sx, sw, sy,
                                               (adv_bg[2] // 4, adv_bg[1] // 4, adv_bg[0] // 4, 180),
                                               (255, 255, 255, 255))
                has_cognitive = True

            if not has_cognitive:
                pil_draw.text((sx + 14, sy + 4), "Awaiting signal...",
                              font=font_tiny, fill=(120, 120, 140, 180))

        # 面板彩色边框
        _panel_borders = [
            (0, 0, panel_w, panel_h, (0, 200, 200, 80)),           # Cyan
            (panel_w, 0, panel_w, panel_h, (200, 50, 200, 80)),    # Magenta
            (0, panel_h, panel_w, panel_h, (50, 200, 50, 80)),     # Green
            (panel_w, panel_h, panel_w, panel_h, (200, 150, 0, 80)),  # Orange
        ]
        for bx, by_, bw_, bh_, bc in _panel_borders:
            pil_draw.rectangle([bx, by_, bx + bw_ - 1, by_ + bh_ - 1], outline=bc, width=1)

        # 画网格线
        pil_draw.line([(panel_w, 0), (panel_w, panel_h * 2)], fill=(80, 80, 80, 200), width=2)
        pil_draw.line([(0, panel_h), (panel_w * 2, panel_h)], fill=(80, 80, 80, 200), width=2)

        # RGBA转回BGR给OpenCV显示，用asarray共享内存不拷贝
        canvas = cv2.cvtColor(np.asarray(pil_canvas), cv2.COLOR_RGBA2BGR)

        cv2.imshow("MediaPipe - Pose + Hands + Face - Press 'q' to exit", canvas)

        # 键盘：退出 + 模式切换
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit. Goodbye.")
            break
        elif key == ord("e"):
            scramble_mode = not scramble_mode
            print(f"Scramble mode: {'ON' if scramble_mode else 'OFF'}")
        elif key == ord("n"):
            noise_mode = not noise_mode
            print(f"Noise mode: {'ON' if noise_mode else 'OFF'}")
        elif key == ord("a"):
            attention_mode = not attention_mode
            print(f"Attention mode: {'ON' if attention_mode else 'OFF'}")
        elif key == ord("m"):
            multi_person_mode = not multi_person_mode
            print(f"Multi-person mode: {'ON' if multi_person_mode else 'OFF (single)'}")

        # 存姿态历史 (仅 person 0)，用于时序动力学分析
        if primary_pose is not None:
            pose_history.append(primary_pose)
        else:
            pose_history.clear()

        # 定期清理过期的 smoother (每 150 帧 ≈ 5 秒)
        if frame_count % 150 == 0:
            current_pids = set(person_faces.keys()) | set(person_poses.keys())
            for pid in list(pose_smoothers.keys()):
                if pid not in current_pids:
                    del pose_smoothers[pid]
            for pid in list(face_smoothers.keys()):
                if pid not in person_faces:
                    del face_smoothers[pid]

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    gesture_recognizer.close()
    face_landmarker.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nQuit")
