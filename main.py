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
import scipy.signal as signal
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
        return "Neutral", best_score
    return best, best_score


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


def estimate_heart_rate(rppg_buffer, fps):
    if len(rppg_buffer) < 120:
        return None
    g = np.array(rppg_buffer, dtype=np.float64)
    g -= g.mean()
    try:
        sos = signal.butter(4, [0.75, 2.5], btype="band", fs=fps, output="sos")
        g = signal.sosfiltfilt(sos, g)
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
                          radius=12, pad_x=15, pad_y=10):
    """画一个圆角矩形 HUD 卡片，文字居中"""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    rw, rh = tw + pad_x * 2, th + pad_y * 2
    draw.rounded_rectangle([x, y, x + rw, y + rh], radius=radius, fill=bg_color)
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
    """画通知卡片，警告用红色、普通用青色，支持2行 + 边界限制"""
    if is_high_alert is None:
        is_high_alert = _is_high_alert(text)
    is_alert = is_high_alert
    if is_alert:
        bg = (180, 0, 20, 220)
        fg = (255, 255, 255, 255)
        radius = 16
    else:
        bg = (10, 40, 50, 200)
        fg = (0, 255, 255, 255)
        radius = 12
    pad_x, pad_y = 18, 14
    line_gap = 4

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
    """画情绪置信度圆角条形图"""
    for i, name in enumerate(label_order):
        val = float(scores_array[i]) if i < len(scores_array) else 0.0
        by_ = y + i * (bar_h + gap)
        # trough
        draw.rounded_rectangle([x, by_, x + w, by_ + bar_h], radius=4,
                               fill=(0, 0, 0, 100))
        if val > 0.001:
            bw = int(w * min(val, 1.0))
            if bw > 4:
                color_bgr = _EMOTION_COLORS_8.get(name, (180, 180, 180))
                color_rgba = (int(color_bgr[2]), int(color_bgr[1]),
                              int(color_bgr[0]), 220)
                draw.rounded_rectangle([x, by_, x + bw, by_ + bar_h],
                                       radius=4, fill=color_rgba)
        draw.text((x + 5, by_ + 1), name, font=font_small,
                  fill=(255, 255, 255, 200))


# 在 PIL draw 上画带背景的中文文字（向后兼容旧代码）
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

# 深度无关阈值（已用肩宽归一化）
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

    # 速度和加速度（逐关节）
    v_t = xyz2 - xyz1
    v_t1 = xyz1 - xyz0
    a_t = v_t - v_t1
    speed_t = np.linalg.norm(v_t, axis=1)
    accel_t = np.linalg.norm(a_t, axis=1)

    # 上半身总动能（深度归一化）
    upper_mask = np.array([i in _UPPER_BODY_IDS for i in range(33)])
    upper_speed2 = speed_t[upper_mask] ** 2
    ek_total = float(np.sum(upper_speed2)) / (ref_length ** 2)

    # 手部指标（深度归一化）
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


# 注视-手指拓扑 + 微表情泄漏检测
_MICRO_KEYS = {
    "noseSneerLeft":   "瞬时厌恶抽动 (Disgust leak)",
    "noseSneerRight":  "瞬时厌恶抽动 (Disgust leak)",
    "mouthDimpleLeft":  "瞬时讥笑抽动 (Smirk leak)",
    "mouthDimpleRight": "瞬时讥笑抽动 (Smirk leak)",
    "browInnerUp":     "瞬时悲伤/恐惧抽动 (Sadness/Fear leak)",
}


def advanced_cognitive_engine(hand_landmarks, head_pose_angles,
                              current_bs, micro_bs_history,
                              current_emotion, current_gesture):
    """注视方向 vs 手指方向的3D夹角 + 微表情微分泄漏。返回 (警告文字, BGR颜色) 或 None"""
    alerts = []

    # 逻辑1: 注视-手指3D夹角
    if (head_pose_angles is not None and hand_landmarks is not None
            and current_gesture == "Pointing_Up"):
        pitch, yaw, _roll = head_pose_angles
        rad = np.deg2rad
        cp, sp = np.cos(rad(pitch)), np.sin(rad(pitch))
        cy, sy = np.cos(rad(yaw)), np.sin(rad(yaw))
        vec_gaze = np.array([-sy * cp, sp, -cy * cp], dtype=np.float64)
        vec_gaze /= np.linalg.norm(vec_gaze) + 1e-8

        # 以头部为原点，计算食指指尖方向
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
                alerts.append((
                    "[交互异常] 盲指 / 认知游离 (Blind Pointing / Distracted)",
                    (0, 0, 200),
                ))
                break

    # 逻辑2: 微表情瞬时微分（中性表情下检测泄漏）
    if current_bs is not None:
        micro_bs_history.append(dict(current_bs))
        if len(micro_bs_history) >= 3 and current_emotion == "Neutral":
            bs0 = micro_bs_history[0]
            for key, label in _MICRO_KEYS.items():
                delta = abs(current_bs.get(key, 0.0) - bs0.get(key, 0.0))
                if delta > 0.12:
                    alerts.append((f"[微表情泄露] {label}", (150, 0, 150)))
                    break

    return alerts[0] if alerts else None


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


# 画置信度进度条（cv2版本，用于向后兼容）
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


# 画全身33个姿态关键点 + 连线（支持 ST-GCN 注意力热力图）
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


# 头部姿态估计（solvePnP）
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


# 三点算角度（b 是顶点），返回度数
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


# 手指比心：拇指和食指尖距离阈值（归一化坐标）
_FINGER_HEART_TOUCH_THRESHOLD = 0.10


def is_finger_heart(landmarks):
    """单手比心: 拇指尖(4)和食指尖(8)接触，其他三指弯曲"""
    thumb = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    index = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    if np.linalg.norm(thumb - index) >= _FINGER_HEART_TOUCH_THRESHOLD:
        return False
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
_SCRAMBLE_RNG = np.random.RandomState(42)
_SCRAMBLE_ORDER = None
_SCRAMBLE_FRAME_SHAPE = None


def _get_scramble_order(h, w, grid=8):
    """生成块级置乱映射，按帧尺寸缓存"""
    global _SCRAMBLE_ORDER, _SCRAMBLE_FRAME_SHAPE
    if _SCRAMBLE_FRAME_SHAPE == (h, w) and _SCRAMBLE_ORDER is not None:
        return _SCRAMBLE_ORDER
    bh, bw = h // grid, w // grid
    block_indices = [(i, j) for i in range(grid) for j in range(grid)]
    shuffled = block_indices[:]
    _SCRAMBLE_RNG.shuffle(shuffled)
    # 构建像素级索引映射
    new_to_old = np.empty((h, w, 2), dtype=np.int32)
    for old_idx, (oi, oj) in enumerate(block_indices):
        ni, nj = shuffled[old_idx]
        y0_dst, x0_dst = ni * bh, nj * bw
        y0_src, x0_src = oi * bh, oj * bw
        for dy in range(bh):
            for dx in range(bw):
                new_to_old[y0_dst + dy, x0_dst + dx] = [y0_src + dy, x0_src + dx]
    _SCRAMBLE_ORDER = new_to_old
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

    # 初始化三个检测器（异步回调模式）
    pose_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model),
        running_mode=RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        result_callback=on_pose_result,
    ))
    gesture_recognizer = GestureRecognizer.create_from_options(GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=gesture_model),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        result_callback=on_gesture_result,
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
    left_lm = drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
    left_cn = drawing_utils.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=1)
    right_lm = drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
    right_cn = drawing_utils.DrawingSpec(color=(255, 0, 140), thickness=2, circle_radius=1)
    face_tess = drawing_utils.DrawingSpec(color=(220, 215, 235), thickness=1, circle_radius=1)

    # EMA 平滑器
    pose_smoother = EMASmoother(alpha=0.5, max_lost_frames=25)
    face_smoother = EMASmoother(alpha=0.5, max_lost_frames=15)

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

    # 微表情历史 (用于微分泄漏检测)
    micro_bs_history = collections.deque(maxlen=4)

    # 视图开关
    scramble_mode = False
    noise_mode = False
    attention_mode = False

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
            except Exception:
                pass  # 推理失败就跳过这一帧
            finally:
                dl_queue.task_done()

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

        # 使用前一帧的结果（异步延迟一帧）
        cur_pose = pose_result
        cur_hand = gesture_result
        cur_face = face_result

        # EMA 平滑
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

        # Blendshape 情绪推断
        emotion_label = "Neutral"
        emotion_score = 0.0
        face_blends_raw = None
        micro_bs = None
        if cur_face and cur_face.face_blendshapes:
            face_blends_raw = cur_face.face_blendshapes[0]
            emotion_label, emotion_score = classify_emotion(face_blends_raw)
            # 存微表情历史，给导数分析用
            micro_bs = {bs.category_name: bs.score for bs in face_blends_raw}
            micro_bs_history.append(micro_bs)

        # 头部姿态估计
        head_pose_angles = None
        head_pose_img_pts = None
        head_pose_text = ""
        if smoothed_face and len(smoothed_face) > 263:
            head_pose_angles, head_pose_img_pts = estimate_head_pose(smoothed_face, w, h)
            if head_pose_angles is not None:
                p, y, r = head_pose_angles
                dir_y = "R" if y > 5 else ("L" if y < -5 else "C")
                dir_p = "U" if p < -5 else ("D" if p > 5 else "L")
                dir_r = "TR" if r > 5 else ("TL" if r < -5 else "C")
                head_pose_text = f"Yaw:{y:+.0f}{dir_y}  Pitch:{p:+.0f}{dir_p}  Roll:{r:+.0f}{dir_r}"

        # 构建四个面板
        panel2.fill(0)

        # PIL后处理用的数据变量
        _pil_emotion_scores = None     # scores array for emotion bars, or None
        _pil_p2_no_face = False        # "No face detected" flag
        _pil_p3_pose_text = None       # (text, color_bgr) for panel 3 pose info
        _pil_p3_hand_count = 0         # number of hands detected
        _pil_p3_gesture_labels = []    # list of (handedness, gesture_name, wrist_x, wrist_y)
        _pil_p3_attention = False      # ST-GCN注意力模式标志
        _pil_p4_top_emotion = None     # (label, score, color_bgr, fx, fy, fw, fh) top-1 following face bbox
        _pil_p4_emotion_hint = None    # fallback emotion text when no intent
        _pil_notification = None       # (text, is_alert) for intent notification card
        _pil_cognitive_alert = None    # (text, bg_color_bgr) for cognitive alert

        _pil_advanced_alert = None     # (text, bg_color_bgr) for advanced alert
        _pil_bpm_text = None           # (text, color_bgr) for BPM display
        _pil_mode_list = []            # list of (text, color_bgr) for active modes

        # 面板1：原始画面（或加密模式下的乱码）
        if scramble_mode:
            scrambled = apply_scramble(frame)
            cv2.resize(scrambled, (panel_w, panel_h), dst=panel1)
        else:
            cv2.resize(frame, (panel_w, panel_h), dst=panel1)

        if smoothed_face and face_blends_raw:
            fx, fy, fw, fh = get_face_bbox(smoothed_face, w, h, margin=0.35)
            fx, fy = max(0, fx), max(0, fy)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)

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

                # 提交DL推理任务到常驻线程（非阻塞）
                if frame_count % 2 == 0 and not dl_queue.full():
                    dl_queue.put(face_crop.copy())

                scaled = [ScaledLandmark(lm, fx, fy, fw, fh, w, h) for lm in smoothed_face]

                # 画面部网格 + 情绪高亮
                drawing_utils.draw_landmarks(
                    image=face_crop, landmark_list=scaled,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=face_tess,
                    connection_drawing_spec=face_tess,
                )
                draw_emotion_focus(face_crop, scaled, dl_emotion_cached or "Neutral")

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

            # 收集情绪分数，供侧边栏柱状图用
            if dl_scores_cached is not None:
                _pil_emotion_scores = dl_scores_cached
            else:
                s = {bs.category_name: bs.score for bs in face_blends_raw}
                _pil_emotion_scores = np.array([
                    avg2(s, "mouthSmileLeft", "mouthSmileRight"),                    # Happiness
                    max(s.get("browInnerUp", 0), avg2(s, "mouthFrownLeft", "mouthFrownRight")),  # Sadness
                    max(s.get("browInnerUp", 0), s.get("jawOpen", 0)),              # Surprise
                    max(avg2(s, "browDownLeft", "browDownRight"),
                        avg2(s, "mouthFrownLeft", "mouthFrownRight")),               # Anger
                    max(s.get("browInnerUp", 0), s.get("jawOpen", 0)) * 0.8,        # Fear
                    max(avg2(s, "noseSneerLeft", "noseSneerRight"),
                        avg2(s, "mouthUpperUpLeft", "mouthUpperUpRight")),           # Disgust
                    s.get("_neutral", 0),                                            # Neutral
                    s.get("mouthPressLeft", 0) * 0.5,                               # Contempt
                ], dtype=np.float64)
        else:
            _pil_p2_no_face = True

        # 面板3：身体骨骼 + 关节角度 + 手势
        panel3.fill(0)

        prev_pose_from_history = pose_history[-2] if len(pose_history) >= 2 else None
        if smoothed_pose:
            draw_pose_full(panel3, smoothed_pose, panel_w, panel_h,
                           prev_landmarks=prev_pose_from_history, attention_mode=attention_mode)
            draw_joint_angles(panel3, smoothed_pose, panel_w, panel_h)
            n_visible = sum(1 for lm in smoothed_pose if getattr(lm, "visibility", 0) > 0.5)
            _pil_p3_pose_text = (f"Pose: {n_visible}/33", (0, 255, 0))
        else:
            _pil_p3_pose_text = ("Pose: NOT FOUND - step back", (0, 0, 255))

        current_active_gestures = set()
        two_hand_heart = False
        if (cur_hand and cur_hand.hand_landmarks
                and len(cur_hand.hand_landmarks) == 2
                and is_two_handed_heart(cur_hand.hand_landmarks, cur_hand.handedness)):
            two_hand_heart = True
            current_active_gestures.add("Two_Handed_Heart")

        if cur_hand and cur_hand.hand_landmarks:
            _pil_p3_hand_count = len(cur_hand.hand_landmarks)
            for i, hand_lms in enumerate(cur_hand.hand_landmarks):
                handedness = cur_hand.handedness[i][0].category_name
                lm_s, cn_s = (left_lm, left_cn) if handedness == "Left" else (right_lm, right_cn)
                drawing_utils.draw_landmarks(
                    image=panel3, landmark_list=hand_lms,
                    connections=HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=lm_s, connection_drawing_spec=cn_s,
                )
                ml_gesture = None
                if (gesture_result and gesture_result.gestures
                        and i < len(gesture_result.gestures)
                        and gesture_result.gestures[i]):
                    cat = gesture_result.gestures[i][0]
                    if cat.category_name != "None" and cat.score > 0.5:
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

        if smoothed_pose:
            pose_overlay.fill(0)
            draw_pose_full(pose_overlay, smoothed_pose, panel_w, panel_h,
                           lm_color=(0, 220, 0), cn_color=(200, 100, 0),
                           thickness=2, skip_face=True,
                           prev_landmarks=prev_pose_from_history, attention_mode=attention_mode)
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
            fx, fy, fw, fh = get_face_bbox(smoothed_face, panel_w, panel_h, margin=0.15)
            cv2.rectangle(panel4, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            if dl_scores_cached is not None:
                best_idx = int(np.argmax(dl_scores_cached))
                _pil_p4_top_emotion = (
                    EMOTION_LABELS[best_idx],
                    float(dl_scores_cached[best_idx]),
                    _EMOTION_COLORS_8.get(EMOTION_LABELS[best_idx], (255, 255, 255)),
                    fx, fy, fw, fh,
                )

        # 多模态意图融合（情绪 + 手势）
        dominant_emotion = dl_emotion_cached or emotion_label
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

        # rPPG心率估计（每15帧算一次，EMA平滑）
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


        # ---- 高级认知引擎（视线-手指 + 微表情）----
        adv_result = None
        if face_blends_raw:
            cur_hand_list = cur_hand.hand_landmarks if (cur_hand and cur_hand.hand_landmarks) else None
            adv_result = advanced_cognitive_engine(
                cur_hand_list, head_pose_angles,
                micro_bs, micro_bs_history,
                dominant_emotion, dominant_gesture,
            )


        # 高级警报覆盖通知
        if adv_result is not None:
            _pil_advanced_alert = adv_result  # (alert_text, bg_color_bgr)

        # 收集各模式状态
        for mode_name, is_on, color in [
            ("NOISE", noise_mode, (255, 100, 100)),
            ("SCRAMBLE", scramble_mode, (100, 100, 255)),
            ("ATTN", attention_mode, (255, 200, 50)),
        ]:
            if is_on:
                _pil_mode_list.append((mode_name, color))

        # 拼合画布
        canvas[:panel_h, :panel_w] = panel1
        canvas[:panel_h, panel_w:panel_w * 2] = panel2
        canvas[panel_h:panel_h * 2, :panel_w] = panel3
        canvas[panel_h:panel_h * 2, panel_w:panel_w * 2] = panel4

        # 统一用PIL做后处理（标题/文字/圆角卡片）
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

        # 面板1：标题右下角，FPS左上角
        p1_title = "Original [ENCRYPTED]" if scramble_mode else "Original"
        p1t_w = _title_card_w(p1_title, font_panel)
        draw_modern_hud_panel(pil_draw, p1_title, panel_w - p1t_w - 8, panel_h - 36,
                              font_panel, pad_x=14, pad_y=8, radius=10,
                              text_color=(255, 100, 100, 255) if scramble_mode else (200, 200, 200, 255),
                              bg_color=(80, 0, 0, 200) if scramble_mode else (30, 30, 30, 180))
        # FPS放左上角
        fps_str = f"FPS: {int(fps)}"
        pil_draw.text((14, 10), fps_str, font=font_tiny, fill=(0, 255, 0, 255))

        # 面板2：标题左下角
        p2t_w = _title_card_w("Expression", font_panel)
        draw_modern_hud_panel(pil_draw, "Expression", panel_w + 8, panel_h - 36,
                              font_panel, pad_x=14, pad_y=8, radius=10,
                              text_color=(200, 200, 200, 255), bg_color=(30, 30, 30, 180))

        # 面板3：标题右上角
        p3t_w = _title_card_w("Body Skeleton", font_panel)
        draw_modern_hud_panel(pil_draw, "Body Skeleton", panel_w - p3t_w - 8, panel_h + 8,
                              font_panel, pad_x=14, pad_y=8, radius=10,
                              text_color=(200, 200, 200, 255), bg_color=(30, 30, 30, 180))

        # 面板4：标题左上角，BPM右上角
        draw_modern_hud_panel(pil_draw, "Combined", panel_w + 8, panel_h + 8,
                              font_panel, pad_x=14, pad_y=8, radius=10,
                              text_color=(200, 200, 200, 255), bg_color=(30, 30, 30, 180))
        if _pil_bpm_text is not None:
            bpm_str, bpm_bgr = _pil_bpm_text
            bpm_tw = pil_draw.textbbox((0, 0), bpm_str, font=font_info)[2]
            pil_draw.text((panel_w * 2 - bpm_tw - 14, panel_h + 10),
                          bpm_str, font=font_info, fill=(255, 20, 20, 255))
        if _pil_p3_hand_count > 0:
            pil_draw.text((12, panel_h + 64),
                          f"Hands: {_pil_p3_hand_count}", font=font_small,
                          fill=(0, 255, 255, 255))
        for h, g, gx, gy in _pil_p3_gesture_labels:
            gx_canvas = max(8, min(gx, panel_w - 8))
            gy_canvas = max(panel_h + 20, min(panel_h + gy - 15, panel_h * 2 - 18))
            gest_text = f"{h[0]}:{g}"  # abbreviate handedness
            tw3 = pil_draw.textbbox((0, 0), gest_text, font=font_tiny)[2]
            tx3 = gx_canvas - tw3 // 2
            tx3 = max(4, min(tx3, panel_w - tw3 - 4))
            pil_draw.text((tx3, gy_canvas), gest_text,
                          font=font_tiny, fill=(0, 255, 255, 220))
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

        if _pil_p4_emotion_hint and not _pil_notification:
            pil_draw.text((panel_w + 12, panel_h + 66), _pil_p4_emotion_hint,
                          font=font_tiny, fill=(180, 180, 180, 200))

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

        # 认知警报（面板4底部）
        if _pil_cognitive_alert is not None:
            c_alert, c_bg = _pil_cognitive_alert
            draw_pil_text_card(pil_draw, c_alert,
                               panel_w + panel_w // 2, panel_h * 2 - 68,
                               font_small,
                               text_color=(255, 255, 255, 255),
                               bg_color=(c_bg[2], c_bg[1], c_bg[0], 200),
                               radius=10, pad_x=12, pad_y=6,
                               bounds=_p4_bounds)

        # 模式状态标签（画布底部）
        mode_x = 12
        for mode_name, mode_color in _pil_mode_list:
            bbox = pil_draw.textbbox((0, 0), mode_name, font=font_tiny)
            mw, mh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pil_draw.rounded_rectangle(
                [mode_x, panel_h * 2 - mh - 16, mode_x + mw + 18, panel_h * 2 - 8],
                radius=6, fill=(mode_color[2], mode_color[1], mode_color[0], 180))
            pil_draw.text((mode_x + 9, panel_h * 2 - mh - 14), mode_name,
                          font=font_tiny, fill=(255, 255, 255, 255))
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

        if _pil_p2_no_face:
            pil_draw.text((sx + 14, sy + 20), "No face detected",
                          font=font_info, fill=(255, 100, 100, 255))
        else:
            # 深度学习情绪识别结果
            if dl_emotion_cached is not None and dl_scores_cached is not None:
                best_idx = int(np.argmax(dl_scores_cached))
                best_score_dl = float(dl_scores_cached[best_idx])
                color_dl = _EMOTION_COLORS_8.get(dl_emotion_cached, (0, 255, 0))
                pil_draw.rounded_rectangle([sx + 10, sy, sx + sw - 10, sy + 26],
                                            radius=6, fill=(color_dl[2] // 4, color_dl[1] // 4, color_dl[0] // 4, 120))
                pil_draw.text((sx + 16, sy + 2), f"DL: {dl_emotion_cached}",
                              font=font_small, fill=(color_dl[2], color_dl[1], color_dl[0], 255))
                score_x = sx + sw - 16 - pil_draw.textbbox((0, 0), f"{best_score_dl:.0%}", font=font_small)[2]
                pil_draw.text((score_x, sy + 2), f"{best_score_dl:.0%}",
                              font=font_small, fill=(color_dl[2], color_dl[1], color_dl[0], 200))
                sy += 30

            # 融合变形情绪
            color_bs = _EMOTION_COLORS_8.get(emotion_label, (0, 200, 255))
            pil_draw.rounded_rectangle([sx + 10, sy, sx + sw - 10, sy + 26],
                                        radius=6, fill=(color_bs[2] // 4, color_bs[1] // 4, color_bs[0] // 4, 120))
            pil_draw.text((sx + 16, sy + 2), f"BS: {emotion_label}",
                          font=font_small, fill=(color_bs[2], color_bs[1], color_bs[0], 255))
            score_x = sx + sw - 16 - pil_draw.textbbox((0, 0), f"{emotion_score:.0%}", font=font_small)[2]
            pil_draw.text((score_x, sy + 2), f"{emotion_score:.0%}",
                          font=font_small, fill=(color_bs[2], color_bs[1], color_bs[0], 200))
            sy += 30

            # 姿态信息
            if _pil_p3_pose_text is not None:
                txt, clr = _pil_p3_pose_text
                pil_draw.text((sx + 14, sy), txt, font=font_small,
                              fill=(clr[2], clr[1], clr[0], 255))
                sy += 22
            if _pil_p3_hand_count > 0:
                pil_draw.text((sx + 14, sy), f"Hands: {_pil_p3_hand_count}",
                              font=font_tiny, fill=(0, 255, 255, 200))
                sy += 18

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

            # 分隔线
            sy += 4
            pil_draw.line([(sx + 14, sy), (sx + sw - 14, sy)],
                          fill=(80, 80, 100, 150), width=1)
            sy += 8

            # 情绪分数条
            if _pil_emotion_scores is not None:
                bar_w = sw - 28
                remaining_h = panel_h * 2 - sy - 20
                bar_h = min(20, remaining_h // 8 - 5)
                gap = max(2, bar_h // 8)
                if bar_h >= 6:
                    draw_pil_emotion_bars(pil_draw, sx + 14, sy, bar_w, bar_h, gap,
                                          _pil_emotion_scores, EMOTION_LABELS, font_tiny)
                    sy += (bar_h + gap) * 8 + 6

            # 认知分析区
            sy = max(sy, panel_h * 2 - 200)
            has_cognitive = False

            # 意图/手势
            if _pil_notification is not None:
                notif_text, is_alert = _pil_notification
                short = notif_text.replace("意图: ", "").replace("[失调警报]", "[!]")
                lines = _wrap_text_lines(pil_draw, short, font_tiny, _max_side_px)
                lh = pil_draw.textbbox((0, 0), "Ag", font=font_tiny)[3]
                row_h = lh * len(lines) + 2 * (len(lines) - 1) + 10
                color = (255, 80, 80, 255) if is_alert else (0, 220, 220, 255)
                pil_draw.rounded_rectangle([sx + 8, sy, sx + sw - 8, sy + row_h],
                                            radius=6, fill=(30, 30, 40, 180))
                cy = sy + (row_h - lh * len(lines)) // 2
                for ln in lines:
                    pil_draw.text((sx + 14, cy), ln, font=font_tiny, fill=color)
                    cy += lh + 2
                sy += row_h + 4
                has_cognitive = True

            # 认知警报
            if _pil_cognitive_alert is not None:
                c_alert, c_bg = _pil_cognitive_alert
                lines = _wrap_text_lines(pil_draw, c_alert, font_tiny, _max_side_px)
                lh = pil_draw.textbbox((0, 0), "Ag", font=font_tiny)[3]
                row_h = lh * len(lines) + 2 * (len(lines) - 1) + 10
                c_rgba = (c_bg[2], c_bg[1], c_bg[0], 255)
                pil_draw.rounded_rectangle([sx + 8, sy, sx + sw - 8, sy + row_h],
                                            radius=6,
                                            fill=(c_bg[2] // 4, c_bg[1] // 4, c_bg[0] // 4, 180))
                cy = sy + (row_h - lh * len(lines)) // 2
                for ln in lines:
                    pil_draw.text((sx + 14, cy), ln, font=font_tiny, fill=c_rgba)
                    cy += lh + 2
                sy += row_h + 4
                has_cognitive = True

            # 高级认知警报
            if _pil_advanced_alert is not None:
                adv_alert, adv_bg = _pil_advanced_alert
                lines = _wrap_text_lines(pil_draw, adv_alert, font_tiny, _max_side_px)
                lh = pil_draw.textbbox((0, 0), "Ag", font=font_tiny)[3]
                row_h = lh * len(lines) + 2 * (len(lines) - 1) + 10
                pil_draw.rounded_rectangle([sx + 8, sy, sx + sw - 8, sy + row_h],
                                            radius=6,
                                            fill=(adv_bg[2] // 4, adv_bg[1] // 4, adv_bg[0] // 4, 180))
                cy = sy + (row_h - lh * len(lines)) // 2
                for ln in lines:
                    pil_draw.text((sx + 14, cy), ln, font=font_tiny,
                                  fill=(255, 255, 255, 255))
                    cy += lh + 2
                has_cognitive = True

            if not has_cognitive:
                pil_draw.text((sx + 14, sy + 4), "Awaiting signal...",
                              font=font_tiny, fill=(120, 120, 140, 180))

        # 画网格线
        pil_draw.line([(panel_w, 0), (panel_w, panel_h * 2)], fill=(80, 80, 80, 200), width=2)
        pil_draw.line([(0, panel_h), (panel_w * 2, panel_h)], fill=(80, 80, 80, 200), width=2)

        # RGBA转回BGR，给OpenCV显示
        canvas = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGBA2BGR)

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

        # 存姿态历史，用于时序动力学分析
        if smoothed_pose is not None:
            pose_history.append(smoothed_pose)
        else:
            pose_history.clear()

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    gesture_recognizer.close()
    face_landmarker.close()


def avg2(s, k1, k2):
    """取两个blendshape分数的均值"""
    return (s.get(k1, 0) + s.get(k2, 0)) / 2


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nQuit")
