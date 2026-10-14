# -*- coding: utf-8 -*-
"""
lane_detector.py
================
تشخیص کامل خطوط جاده، مسیر خودرو، لاین هر خودرو، فاصله‌ها (HUD)
و تابلوهای راهنمایی.

خروجی‌ها:
  - همه‌ی خطوط لاین بزرگراه (قرمز) و تکمیل خطوط خط‌چین
  - مسیر خودروی خودی با گرادینت آبی
  - لاین هر خودرو + خط چپ و راست همان لاین
  - فاصله تا خودروی جلویی، چپ و راست روی HUD
  - تشخیص تابلوها (روش رنگ + شکل)

نکته: متن‌های روی تصویر انگلیسی هستند چون OpenCV از فونت فارسی پشتیبانی نمی‌کند.

استفاده‌ی ساده:
    detector = LaneDetector()
    out, info = detector.process(frame, vehicles)

که در آن vehicles لیستی از دیکشنری‌هایی مثل
    {"box": (x1, y1, x2, y2), "cls": "car", "id": 7}
است (یا فقط لیستی از (x1, y1, x2, y2)).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]

FONT = cv2.FONT_HERSHEY_SIMPLEX

# رنگ‌ها به صورت BGR
RED = (0, 0, 255)
GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
GRAY = (200, 200, 200)

# عرض واقعی تقریبی اشیاء (متر) برای تخمین فاصله
REAL_WIDTH_M = {
    "car": 1.8,
    "truck": 2.5,
    "bus": 2.5,
    "van": 2.0,
    "motorcycle": 0.8,
    "motorbike": 0.8,
    "bicycle": 0.6,
}


def _rng(*vals: int) -> np.ndarray:
    return np.array(vals, dtype=np.uint8)


# ----------------------------------------------------------------------
# ساختارهای داده
# ----------------------------------------------------------------------
@dataclass
class LaneLine:
    """خط لاین به صورت x = a*y + b (چون خطوط جاده تقریباً عمودی‌اند)."""

    a: float
    b: float
    hits: int = 1
    missed: int = 0
    virtual: bool = False  # خطی که فقط از روی فاصله‌ی لاین‌ها حدس زده شده

    def x_at(self, y: float) -> float:
        return self.a * y + self.b


@dataclass
class Vehicle:
    box: Box
    cls: str = "car"
    track_id: Optional[int] = None
    distance: Optional[float] = None   # فاصله‌ی طولی (متر)
    lateral: Optional[float] = None    # فاصله‌ی جانبی از مرکز خودرو (متر)
    lane: Optional[int] = None         # اندیس لاین از چپ (۰-based)
    rel_lane: Optional[int] = None     # نسبت به لاین خودی (۰=همان لاین، -۱=چپ، +۱=راست)
    left_line: Optional[LaneLine] = None
    right_line: Optional[LaneLine] = None

    @property
    def cx(self) -> float:
        return 0.5 * (self.box[0] + self.box[2])

    @property
    def bottom(self) -> float:
        return self.box[3]


# ----------------------------------------------------------------------
# کلاس اصلی
# ----------------------------------------------------------------------
class LaneDetector:
    def __init__(
        self,
        roi_top: float = 0.45,        # بالای ناحیه‌ی جاده (نسبت به ارتفاع تصویر)
        hood_ratio: float = 0.0,      # درصدی از پایین تصویر که کاپوت خودرو است
        ego_x_ratio: float = 0.5,     # موقعیت افقی دوربین/خودروی خودی در تصویر
        hfov_deg: float = 70.0,       # زاویه‌ی دید افقی دوربین (برای تخمین فاصله)
        focal_px: Optional[float] = None,  # اگر فاصله‌ی کانونی را می‌دانید مستقیم بدهید
        lane_width_m: float = 3.5,
        smooth: float = 0.3,          # ضریب هموارسازی زمانی خطوط
        min_hits: int = 2,            # حداقل تعداد فریم تأیید برای نمایش خط
        max_missed: int = 20,         # تعداد فریمی که خط گم‌شده هنوز نگه داشته می‌شود
        fill_gaps: bool = True,       # ساخت خطوط لاین جا‌افتاده
        enable_signs: bool = True,
        enable_hud: bool = True,
        warn_dist: float = 20.0,      # متر
        danger_dist: float = 10.0,    # متر
        occlusion: str = "body",     # "none": همه چیز روی خودروها هم رسم شود | "body": فقط بدنه‌ی خودرو پوشانده شود | "full": کل باکس
        debug: bool = False,
    ):
        self.roi_top = roi_top
        self.hood_ratio = hood_ratio
        self.ego_x_ratio = ego_x_ratio
        self.hfov_deg = hfov_deg
        self.focal_px = focal_px
        self.lane_width_m = lane_width_m
        self.smooth = smooth
        self.min_hits = max(1, min_hits)
        self.max_missed = max_missed
        self.fill_gaps = fill_gaps
        self.enable_signs = enable_signs
        self.enable_hud = enable_hud
        self.warn_dist = warn_dist
        self.danger_dist = danger_dist
        self.occlusion = occlusion
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        """پاک کردن حافظه‌ی زمانی (برای شروع ویدیوی جدید)."""
        self._tracks: List[LaneLine] = []
        self._lines: List[LaneLine] = []
        self._vp: Optional[Tuple[float, float]] = None
        self._vehicles: List[Vehicle] = []
        self._signs: List[Dict[str, Any]] = []
        self._sign_store: List[Dict[str, Any]] = []
        self._ego_idx: Optional[int] = None
        self._ego_lines: Tuple[Optional[LaneLine], Optional[LaneLine]] = (None, None)
        self._targets: Dict[str, Optional[Vehicle]] = {"front": None, "left": None, "right": None}
        self._raw_lines = None
        self.info: Dict[str, Any] = {}

    # ==================================================================
    # API عمومی
    # ==================================================================
    def detect(self, frame, vehicle_boxes=None):
        """
        تحلیل کامل فریم. مثل نسخه‌ی قبل (mask, lines) برمی‌گرداند
        ولی مدل لاین‌ها، تابلوها و فاصله‌ها هم داخل شیء ذخیره می‌شوند.
        """
        h, w = frame.shape[:2]
        self._vehicles = self._to_vehicles(vehicle_boxes)

        mask = self._marking_mask(frame, self._vehicles)

        # اتصال بخش‌های خط‌چین به هم
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))

        edges = cv2.Canny(mask, 50, 150)
        s = h / 720.0
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=max(15, int(25 * s)),
            maxLineGap=max(10, int(20 * s)),
        )
        self._raw_lines = lines

        self._update_lane_model(lines, (h, w))
        self._analyze_scene(frame)
        return mask, lines

    def draw_lanes(self, frame, lines=None, vehicle_boxes=None):
        """
        رسم خروجی نهایی. پارامترهای lines و vehicle_boxes فقط برای سازگاری با
        نسخه‌ی قبل هستند؛ داده‌ها از آخرین فراخوانی detect() گرفته می‌شوند.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        s = max(0.6, w / 1280.0)

        veh_mask = self._occlusion_mask((h, w), self._vehicles)
        vx, vy = self._vp if self._vp else (w * self.ego_x_ratio, h * self.roi_top)
        y_bottom = int(h * (1.0 - self.hood_ratio)) - 1
        y_line_top = int(min(0.60 * h, max(0.30 * h, vy + 0.03 * h)))
        y_path_top = int(min(0.70 * h, max(0.40 * h, vy + 0.10 * h)))

        # ۱) مسیر خودروی خودی با گرادینت آبی
        self._draw_ego_path(out, y_path_top, y_bottom, veh_mask)

        # ۲) همه‌ی خطوط لاین (قرمز)
        self._draw_lane_lines(out, y_line_top, y_bottom, veh_mask, s)

        # ۳) خط چپ و راست هر خودرو (ضخیم‌تر) + شماره‌ی لاین
        self._draw_vehicle_lanes(out, y_line_top, y_bottom, veh_mask, s)

        # ۴) تابلوها
        for sg in self._signs:
            x1, y1, x2, y2 = sg["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), sg["color"], max(2, int(2 * s)))
            self._label(out, sg["label"], (x1, y1 - 4), bg=sg["color"], scale=0.45 * s)

        # ۵) خطوط ارتباطی + HUD
        if self.enable_hud:
            self._draw_connectors(out, y_bottom, s)
            self._draw_hud(out, s)

        # حالت دیباگ: خطوط خام Hough
        if self.debug and self._raw_lines is not None:
            for l in self._raw_lines:
                x1, y1, x2, y2 = (int(v) for v in l[0])
                cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

        return out

    def process(self, frame, vehicles=None):
        """detect + draw در یک مرحله. خروجی: (تصویر نهایی، دیکشنری اطلاعات)"""
        _, lines = self.detect(frame, vehicles)
        out = self.draw_lanes(frame, lines, vehicles)
        return out, self.info

    # ==================================================================
    # ورودی خودروها
    # ==================================================================
    @staticmethod
    def _to_vehicles(items) -> List[Vehicle]:
        out: List[Vehicle] = []
        if items is None:
            return out
        for it in items:
            if isinstance(it, Vehicle):
                out.append(it)
                continue
            cls, tid = "car", None
            if isinstance(it, dict):
                box = it.get("box", it.get("bbox"))
                cls = str(it.get("cls", it.get("label", it.get("name", "car")))).lower()
                tid = it.get("id", it.get("track_id"))
            else:
                box = it
            if cls in ("person", "pedestrian"):
                continue  # عابر پیاده نادیده گرفته می‌شود
            x1, y1, x2, y2 = (float(v) for v in list(box)[:4])
            out.append(Vehicle((x1, y1, x2, y2), cls, tid))
        return out

    @staticmethod
    def _vehicle_mask(shape, vehicles, pad: int = 0):
        h, w = shape[:2]
        m = np.zeros((h, w), np.uint8)
        for v in vehicles:
            x1, y1, x2, y2 = (int(round(c)) for c in v.box)
            cv2.rectangle(
                m,
                (max(0, x1 - pad), max(0, y1 - pad)),
                (min(w - 1, x2 + pad), min(h - 1, y2 + pad)),
                255,
                -1,
            )
        return m

    def _occlusion_mask(self, shape, vehicles):
        """
        ناحیه‌ای از خودرو که مسیر آبی و خطوط نباید روی آن رسم شوند.
        فقط بدنه‌ی خودرو پوشانده می‌شود تا مسیر و خطوط زیر و اطراف خودرو دیده شوند.
        """
        h, w = shape[:2]
        m = np.zeros((h, w), np.uint8)
        if self.occlusion == "none":
            return m
        for v in vehicles:
            x1, y1, x2, y2 = v.box
            bw, bh = x2 - x1, y2 - y1
            if self.occlusion == "body":
                x1, x2 = x1 + 0.12 * bw, x2 - 0.12 * bw
                y1, y2 = y1 + 0.20 * bh, y2 - 0.25 * bh
            cv2.rectangle(
                m,
                (max(0, int(x1)), max(0, int(y1))),
                (min(w - 1, int(x2)), min(h - 1, int(y2))),
                255,
                -1,
            )
        return m

    def _y_ref(self, h: int) -> int:
        return int(0.92 * h)

    def _focal(self, w: int) -> float:
        if self.focal_px:
            return float(self.focal_px)
        return (w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    # ==================================================================
    # ماسک خط‌کشی‌ها
    # ==================================================================
    def _marking_mask(self, frame, vehicles):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # خطوط سفید و زرد
        white = cv2.inRange(hsv, _rng(0, 0, 160), _rng(180, 75, 255))
        yellow = cv2.inRange(hsv, _rng(12, 60, 80), _rng(40, 255, 255))
        mask = cv2.bitwise_or(white, yellow)

        # محدود کردن به جاده
        y_bot = int(h * (1.0 - self.hood_ratio)) - 1
        roi = np.zeros_like(mask)
        polygon = np.array(
            [[
                (0, y_bot),
                (int(w * 0.12), int(h * self.roi_top)),
                (int(w * 0.88), int(h * self.roi_top)),
                (w - 1, y_bot),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(roi, polygon, 255)
        mask = cv2.bitwise_and(mask, roi)

        # اگر خطوط کم‌نور باشند (سایه، شب، هوای ابری) آستانه‌ی رنگی کم می‌آورد.
        # در این حالت لکه‌های روشن‌تر از اطراف (Top-hat) هم اضافه می‌شوند.
        roi_px = max(1, int(np.count_nonzero(roi)))
        if np.count_nonzero(mask) / roi_px < 0.006:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kk = max(9, int(0.03 * w)) | 1
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (kk, kk)))
            vals = tophat[roi > 0]
            thr = float(np.clip(np.percentile(vals, 96), 18, 60)) if vals.size else 30.0
            bright = cv2.bitwise_and(cv2.inRange(tophat, int(thr), 255), cv2.inRange(gray, 90, 255))
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(bright, roi))

        # حذف ناحیه‌ی خودروها
        mask[self._vehicle_mask(mask.shape, vehicles) > 0] = 0

        # حذف سطوح روشن و پهن (آسمان، بتن، دیوار): خط‌کشی‌ها باریک‌اند.
        # برای سرعت، روی نسخه‌ی کوچک‌شده کار می‌کنیم.
        q = 4
        small = cv2.resize(mask, (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA)
        _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
        k = max(5, int(0.07 * w) // q) | 1
        big = cv2.morphologyEx(small, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if big.any():
            big = cv2.dilate(big, np.ones((3, 3), np.uint8))
            big = cv2.resize(big, (w, h), interpolation=cv2.INTER_NEAREST)
            mask[big > 0] = 0
        return mask

    # ==================================================================
    # تابلوها (رنگ + شکل)
    # ==================================================================
    def _find_signs(self, frame):
        h, w = frame.shape[:2]
        vy = self._vp[1] if self._vp else h * self.roi_top

        # تابلوهای کنار/بالای جاده بالاتر از ارتفاع دوربین‌اند، پس بالای افق دیده می‌شوند
        y0 = int(0.03 * h)
        y1 = int(min(0.75 * h, max(0.35 * h, vy + 0.10 * h)))
        if y1 - y0 < 20:
            return []

        hsv = cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2HSV)
        masks = {
            "red": cv2.bitwise_or(
                cv2.inRange(hsv, _rng(0, 110, 70), _rng(10, 255, 255)),
                cv2.inRange(hsv, _rng(170, 110, 70), _rng(180, 255, 255)),
            ),
            "blue": cv2.inRange(hsv, _rng(100, 150, 50), _rng(128, 255, 255)),
            "yellow": cv2.inRange(hsv, _rng(18, 120, 130), _rng(34, 255, 255)),
            "green": cv2.inRange(hsv, _rng(55, 70, 30), _rng(95, 255, 255)),
        }
        draw_colors = {"red": (0, 0, 255), "blue": (255, 120, 0),
                       "yellow": (0, 215, 255), "green": (0, 170, 0)}

        veh = self._vehicle_mask((h, w), self._vehicles, pad=4)[y0:y1]
        min_a = max(120, int(1.2e-4 * w * h))
        max_a = int(0.03 * w * h)
        k_close = np.ones((5, 5), np.uint8)
        k_open = np.ones((3, 3), np.uint8)

        dets: List[Dict[str, Any]] = []
        for color, m in masks.items():
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_a or area > max_a:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if veh[min(y + bh // 2, veh.shape[0] - 1), min(x + bw // 2, veh.shape[1] - 1)]:
                    continue  # چراغ و بدنه‌ی خودروها
                hull_area = cv2.contourArea(cv2.convexHull(c))
                peri = cv2.arcLength(c, True)
                if hull_area <= 0 or peri <= 0:
                    continue
                solidity = area / hull_area
                if solidity < 0.85:
                    continue
                aspect = bw / float(bh)
                extent = area / float(bw * bh)
                circ = 4.0 * math.pi * area / (peri * peri)
                nv = len(cv2.approxPolyDP(c, 0.04 * peri, True))

                label = None
                if color == "red":
                    if 0.6 <= aspect <= 1.6 and nv == 3 and 0.35 <= extent <= 0.65:
                        label = "WARNING SIGN"
                    elif 0.7 <= aspect <= 1.4 and circ >= 0.85:
                        label = "PROHIBITORY / STOP SIGN"
                elif color == "blue":
                    if 0.6 <= aspect <= 1.6 and nv <= 6 and extent >= 0.85:
                        label = "INFO SIGN"
                    elif 0.7 <= aspect <= 1.4 and circ >= 0.8:
                        label = "MANDATORY SIGN"
                elif color == "yellow":
                    if 0.6 <= aspect <= 1.6 and nv in (3, 4) and extent >= 0.35:
                        label = "WARNING SIGN (YELLOW)"
                elif color == "green":
                    if nv <= 6 and extent >= 0.88 and 1.1 <= aspect <= 5.0 and area >= 2 * min_a:
                        label = "GUIDE SIGN"

                if label:
                    dets.append({
                        "bbox": (x, y + y0, x + bw, y + bh + y0),
                        "label": label,
                        "color": draw_colors[color],
                        "area": area,
                    })

        # حذف تشخیص‌های تکراری
        dets.sort(key=lambda d: -d["area"])
        kept: List[Dict[str, Any]] = []
        for d in dets:
            if all(self._iou(d["bbox"], k["bbox"]) < 0.3 for k in kept):
                kept.append(d)
        return kept

    @staticmethod
    def _iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    # ==================================================================
    # پایداری زمانی برای تابلوها
    # ==================================================================
    def _persist(self, store, dets, max_missed: int = 6):
        min_hits = max(1, self.min_hits - 1)
        for t in store:
            t["matched"] = False
        for d in dets:
            bx = d["bbox"]
            cxd, cyd = 0.5 * (bx[0] + bx[2]), 0.5 * (bx[1] + bx[3])
            sz = max(bx[2] - bx[0], bx[3] - bx[1])
            best, bd = None, 1e9
            for t in store:
                if t["matched"] or t.get("label") != d.get("label"):
                    continue
                tb = t["bbox"]
                dist = math.hypot(cxd - 0.5 * (tb[0] + tb[2]), cyd - 0.5 * (tb[1] + tb[3]))
                lim = 0.6 * max(sz, tb[2] - tb[0], tb[3] - tb[1]) + 10
                if dist < lim and dist < bd:
                    best, bd = t, dist
            if best is not None:
                best.update(d)
                best["hits"] += 1
                best["missed"] = 0
                best["matched"] = True
            else:
                store.append({**d, "hits": 1, "missed": 0, "matched": True})
        for t in store:
            if not t["matched"]:
                t["missed"] += 1
        store[:] = [t for t in store if t["missed"] <= max_missed]
        return [t for t in store if t["hits"] >= min_hits]

    # ==================================================================
    # مدل لاین‌ها
    # ==================================================================
    def _update_lane_model(self, lines, shape) -> None:
        h, w = shape
        y_ref = self._y_ref(h)

        # ۱) استخراج خطوط از روی قطعه‌های Hough (رأی‌گیری روی پارامتر خط)
        tol = max(14.0, 0.022 * w)
        measured = self._extract_lines(lines, h, w, y_ref)

        # ۴) تطبیق با خطوط قبلی (ردیابی زمانی)
        for t in self._tracks:
            t._matched = False  # type: ignore[attr-defined]
        measured.sort(key=lambda m: -m[2])
        for a, b, _ in measured:
            xr = a * y_ref + b
            best, bd = None, 1e9
            for t in self._tracks:
                if getattr(t, "_matched", False):
                    continue
                d = abs(t.x_at(y_ref) - xr)
                if d < tol * 1.5 and d < bd:
                    best, bd = t, d
            if best is not None:
                al = self.smooth
                best.a = (1 - al) * best.a + al * a
                best.b = (1 - al) * best.b + al * b
                best.hits += 1
                best.missed = 0
                best._matched = True  # type: ignore[attr-defined]
            else:
                nt = LaneLine(a, b)
                nt._matched = True  # type: ignore[attr-defined]
                self._tracks.append(nt)
        for t in self._tracks:
            if not getattr(t, "_matched", False):
                t.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

        # ۵) حذف خطوط تکراری که روی هم افتاده‌اند
        self._tracks.sort(key=lambda t: t.x_at(y_ref))
        dedup: List[LaneLine] = []
        for t in self._tracks:
            if dedup and abs(t.x_at(y_ref) - dedup[-1].x_at(y_ref)) < 1.4 * tol \
                    and abs(t.a - dedup[-1].a) < 0.35:
                if t.hits > dedup[-1].hits:
                    dedup[-1] = t
                continue
            dedup.append(t)
        self._tracks = dedup

    def _extract_lines(self, lines, h: int, w: int, y_ref: int):
        """
        قطعه‌های Hough را به نقاط تبدیل می‌کند و با رأی‌گیری روی (شیب، محل برخورد با ردیف مرجع)
        کامل‌ترین خطوط را یکی‌یکی پیدا می‌کند. قطعه‌های جدا از هم (خط‌چین) روی یک خط جمع می‌شوند.
        خروجی: لیست (a, b, طول تقریبی) با x = a*y + b
        """
        if lines is None or len(lines) == 0:
            return []

        step = 4.0
        pts = []
        for l in lines:
            x1, y1, x2, y2 = (float(v) for v in l[0])
            dy, dx = y1 - y2, x1 - x2
            length = math.hypot(dx, dy)
            if abs(dy) < 6 or length < 12:
                continue
            if abs(dx / dy) > 3.3:  # تقریباً افقی
                continue
            n = max(2, int(length / step))
            t = np.linspace(0.0, 1.0, n)
            pts.append(np.stack([x1 - dx * t, y1 - dy * t], axis=1))
        if not pts:
            return []
        P = np.concatenate(pts, axis=0)
        if len(P) > 8000:
            P = P[:: int(math.ceil(len(P) / 8000.0))]

        a_grid = np.arange(-3.3, 3.3001, 0.025)
        A = len(a_grid)
        binw = max(3.0, 0.004 * w)
        x_min = -float(w)
        nb = int(3.0 * w / binw) + 2

        def votes(pp):
            X = pp[:, 0:1] + a_grid[None, :] * (y_ref - pp[:, 1:2])
            b = np.floor((X - x_min) / binw).astype(np.int64)
            ok = (b >= 0) & (b < nb)
            flat = (np.arange(A, dtype=np.int64)[None, :] * nb + b)[ok]
            return np.bincount(flat, minlength=A * nb).reshape(A, nb).astype(np.float32)

        acc = votes(P)
        alive = np.ones(len(P), bool)
        min_len = 0.08 * h
        min_pts = max(6, int(min_len / step))
        tol_in = max(10.0, 0.02 * w)

        found = []
        for _ in range(16):
            if alive.sum() < min_pts:
                break
            smooth = cv2.boxFilter(acc, -1, (3, 3), normalize=False)
            ai, bi = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
            if smooth[ai, bi] < min_pts:
                break
            a0 = float(a_grid[ai])
            x_ref0 = x_min + (bi + 0.5) * binw
            b0 = x_ref0 - a0 * y_ref

            idx = np.nonzero(alive & (np.abs(P[:, 0] - (a0 * P[:, 1] + b0)) <= tol_in))[0]
            if len(idx) >= 2 and np.ptp(P[idx, 1]) > 1:
                a1, b1 = np.polyfit(P[idx, 1], P[idx, 0], 1)
                idx2 = np.nonzero(alive & (np.abs(P[:, 0] - (a1 * P[:, 1] + b1)) <= tol_in))[0]
                if len(idx2) >= 2 and np.ptp(P[idx2, 1]) > 1:
                    a2, b2 = np.polyfit(P[idx2, 1], P[idx2, 0], 1)
                    idx, a0, b0 = idx2, float(a2), float(b2)
                else:
                    a0, b0 = float(a1), float(b1)
            if len(idx) == 0:
                acc[max(0, ai - 1):ai + 2, max(0, bi - 1):bi + 2] = 0
                continue

            span = float(np.ptp(P[idx, 1]))
            if len(idx) >= min_pts and span >= 0.05 * h and abs(a0) <= 3.5:
                found.append((a0, b0, len(idx) * step))
            acc -= votes(P[idx])
            alive[idx] = False
            acc[max(0, ai - 1):ai + 2, max(0, bi - 1):bi + 2] = 0
        return found

    def _fill_gaps(self, lines: List[LaneLine], y_ref: int, w: int) -> List[LaneLine]:
        """اگر فاصله‌ی دو خط مجاور ~ چند برابر عرض لاین باشد، خطوط جاافتاده را می‌سازد."""
        if len(lines) < 2:
            return lines
        xs = np.array([l.x_at(y_ref) for l in lines])
        gaps = np.diff(xs)
        valid = gaps[gaps > 0.07 * w]
        if valid.size == 0:
            return lines
        lane_w = float(valid.min())

        out = [lines[0]]
        for i in range(1, len(lines)):
            gap = xs[i] - xs[i - 1]
            missing = int(round(gap / lane_w)) - 1
            if gap > 1.7 * lane_w and 1 <= missing <= 4:
                for k in range(1, missing + 1):
                    t = k / (missing + 1)
                    out.append(LaneLine(
                        (1 - t) * lines[i - 1].a + t * lines[i].a,
                        (1 - t) * lines[i - 1].b + t * lines[i].b,
                        hits=1, virtual=True,
                    ))
            out.append(lines[i])
        return out

    def _update_vp(self, lines: List[LaneLine], h: int, w: int) -> None:
        """نقطه‌ی گریز از تقاطع خطوط لاین (کمترین مربعات) + هموارسازی."""
        good = [l for l in lines if not l.virtual]
        vp = None
        if len(good) >= 2 and np.ptp([l.a for l in good]) > 0.2:
            A, B, W = [], [], []
            for l in good:
                n = math.sqrt(1.0 + l.a * l.a)
                A.append([1.0 / n, -l.a / n])
                B.append(l.b / n)
                W.append(math.sqrt(max(1, l.hits)))
            A, B, W = np.array(A), np.array(B), np.array(W)
            sol, *_ = np.linalg.lstsq(A * W[:, None], B * W, rcond=None)
            vx, vy = float(sol[0]), float(sol[1])
            if 0.05 * h < vy < 0.75 * h and -0.5 * w < vx < 1.5 * w:
                vp = (vx, vy)
        if vp is None:
            if self._vp is None:
                self._vp = (w * self.ego_x_ratio, h * self.roi_top)
            return
        if self._vp is None:
            self._vp = vp
        else:
            self._vp = (0.85 * self._vp[0] + 0.15 * vp[0], 0.85 * self._vp[1] + 0.15 * vp[1])

    @staticmethod
    def _locate(lines: List[LaneLine], x: float, y: float):
        """
        تعیین لاین یک نقطه. خروجی: (idx, خط چپ, خط راست)
        idx=-1 یعنی چپ‌تر از همه‌ی خطوط، idx=n-1 یعنی راست‌تر از همه.
        """
        n = len(lines)
        if n < 2:
            return None, None, None
        pairs = sorted(((l.x_at(y), l) for l in lines), key=lambda p: p[0])
        xs = [p[0] for p in pairs]
        i = bisect_right(xs, x) - 1
        left = pairs[i][1] if 0 <= i < n else None
        right = pairs[i + 1][1] if 0 <= i + 1 < n else None
        return i, left, right

    # ==================================================================
    # تحلیل صحنه: لاین هر خودرو + فاصله‌ها + تابلوها
    # ==================================================================
    def _analyze_scene(self, frame) -> None:
        h, w = frame.shape[:2]
        y_ref = self._y_ref(h)

        vis = [t for t in self._tracks if t.hits >= self.min_hits and t.missed <= self.max_missed]
        vis.sort(key=lambda l: l.x_at(y_ref))
        if self.fill_gaps:
            vis = self._fill_gaps(vis, y_ref, w)
        self._lines = vis
        self._update_vp(vis, h, w)

        n_lanes = max(len(vis) - 1, 0)
        ego_x = w * self.ego_x_ratio
        idx, left, right = self._locate(vis, ego_x, y_ref)
        ego_valid = idx is not None and 0 <= idx <= n_lanes - 1
        self._ego_idx = idx if ego_valid else None
        self._ego_lines = (left, right) if ego_valid else (None, None)

        # لاین و فاصله‌ی هر خودرو
        f = self._focal(w)
        for v in self._vehicles:
            bw = max(1.0, v.box[2] - v.box[0])
            v.distance = f * REAL_WIDTH_M.get(v.cls, 1.8) / bw
            v.lateral = (v.cx - ego_x) * v.distance / f
            v.lane, v.left_line, v.right_line = self._locate(vis, v.cx, min(v.bottom, h - 1))
            v.rel_lane = (v.lane - idx) if (ego_valid and v.lane is not None) else None

        def pick(cands):
            return min(cands, key=lambda v: v.distance) if cands else None

        vs = self._vehicles
        if ego_valid:
            front = pick([v for v in vs if v.rel_lane == 0])
            lft = pick([v for v in vs if v.rel_lane == -1])
            rgt = pick([v for v in vs if v.rel_lane == 1])
        else:  # بدون لاین: با فاصله‌ی جانبی حدس می‌زنیم
            half = self.lane_width_m / 2.0
            front = pick([v for v in vs if abs(v.lateral) <= half])
            lft = pick([v for v in vs if -1.5 * self.lane_width_m <= v.lateral < -half])
            rgt = pick([v for v in vs if half < v.lateral <= 1.5 * self.lane_width_m])
        self._targets = {"front": front, "left": lft, "right": rgt}

        # تابلوها
        if self.enable_signs:
            self._signs = self._persist(self._sign_store, self._find_signs(frame))
        else:
            self._signs = []

        def vinfo(v: Optional[Vehicle]):
            if v is None:
                return None
            lane_no = v.lane + 1 if (v.lane is not None and 0 <= v.lane < n_lanes) else None
            return {
                "id": v.track_id, "cls": v.cls,
                "box": tuple(int(c) for c in v.box),
                "lane": lane_no, "rel_lane": v.rel_lane,
                "distance_m": round(v.distance, 1), "lateral_m": round(v.lateral, 1),
            }

        self.info = {
            "lane_count": n_lanes,
            "ego_lane": (self._ego_idx + 1) if self._ego_idx is not None else None,
            "vanishing_point": tuple(round(c, 1) for c in self._vp) if self._vp else None,
            "lane_lines": [{"x_bottom": round(l.x_at(y_ref), 1), "virtual": l.virtual} for l in vis],
            "vehicles": [vinfo(v) for v in vs],
            "front": vinfo(front), "left": vinfo(lft), "right": vinfo(rgt),
            "signs": [{"bbox": tuple(int(c) for c in s_["bbox"]), "label": s_["label"]}
                      for s_ in self._signs],
        }

    # ==================================================================
    # توابع رسم
    # ==================================================================
    @staticmethod
    def _blend(out, alpha_u8, color, strength: float = 1.0) -> None:
        """ترکیب رنگ روی پیکسل‌هایی که ماسک آنها غیرصفر است."""
        ys, xs = np.nonzero(alpha_u8)
        if ys.size == 0:
            return
        a = (alpha_u8[ys, xs].astype(np.float32) / 255.0 * strength)[:, None]
        px = out[ys, xs].astype(np.float32)
        col = np.array(color, np.float32)
        out[ys, xs] = (px * (1.0 - a) + col * a).astype(np.uint8)

    @staticmethod
    def _label(img, text, org, bg=(0, 0, 0), fg=(255, 255, 255), scale=0.5, thick=1) -> None:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        x = int(max(0, min(org[0], img.shape[1] - tw - 6)))
        y = int(max(th + 6, org[1]))
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), bg, -1)
        cv2.putText(img, text, (x + 3, y - 3), FONT, scale, fg, thick, cv2.LINE_AA)

    def _dist_color(self, d: Optional[float]):
        if d is None:
            return GRAY
        if d <= self.danger_dist:
            return RED
        if d <= self.warn_dist:
            return ORANGE
        return GREEN

    def _ego_polygon(self, w, h, y_top, y_bottom):
        l, r = self._ego_lines
        if l is not None and r is not None:
            return np.array([
                (l.x_at(y_top), y_top), (r.x_at(y_top), y_top),
                (r.x_at(y_bottom), y_bottom), (l.x_at(y_bottom), y_bottom),
            ], np.int32)

        # بدون لاین: مسیر پیش‌فرض که به نقطه‌ی گریز همگرا می‌شود
        vx, vy = self._vp if self._vp else (w * self.ego_x_ratio, h * self.roi_top)
        ego_x = w * self.ego_x_ratio
        xs = [self._lines[i + 1].x_at(y_bottom) - self._lines[i].x_at(y_bottom)
              for i in range(len(self._lines) - 1)]
        half_b = 0.5 * (float(np.median(xs)) if xs else 0.30 * w)
        k = (y_top - vy) / max(1.0, (y_bottom - vy))
        half_t = half_b * k
        x_top = vx + (ego_x - vx) * k
        return np.array([
            (x_top - half_t, y_top), (x_top + half_t, y_top),
            (ego_x + half_b, y_bottom), (ego_x - half_b, y_bottom),
        ], np.int32)

    def _draw_ego_path(self, out, y_top, y_bottom, veh_mask) -> None:
        h, w = out.shape[:2]
        if y_bottom - y_top < 10:
            return
        poly = self._ego_polygon(w, h, y_top, y_bottom)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        mask[veh_mask > 0] = 0

        r0, r1 = max(0, y_top), min(h - 1, y_bottom) + 1
        n = r1 - r0
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)          # ۰=دور، ۱=نزدیک
        far = np.array([255, 215, 130], np.float32)              # آبی روشن (BGR)
        near = np.array([255, 90, 10], np.float32)               # آبی پررنگ (BGR)
        colors = far[None, :] * (1 - t[:, None]) + near[None, :] * t[:, None]
        alpha_row = 0.15 + 0.55 * t

        a = (mask[r0:r1].astype(np.float32) / 255.0) * alpha_row[:, None]
        region = out[r0:r1].astype(np.float32)
        region = region * (1 - a[..., None]) + colors[:, None, :] * a[..., None]
        out[r0:r1] = region.astype(np.uint8)

    def _draw_lane_lines(self, out, y_top, y_bottom, veh_mask, s) -> None:
        h, w = out.shape[:2]
        alpha = np.zeros((h, w), np.uint8)
        n_seg = 16
        ys = np.linspace(y_bottom, y_top, n_seg + 1)
        for ln in self._lines:
            for k in range(n_seg):
                if ln.virtual and k % 2 == 1:
                    continue  # خطوط حدسی به صورت خط‌چین
                t = k / n_seg
                thick = max(1, int(round((6.0 * (1 - t) + 1.0 * t) * s)))
                p1 = (int(ln.x_at(ys[k])), int(ys[k]))
                p2 = (int(ln.x_at(ys[k + 1])), int(ys[k + 1]))
                cv2.line(alpha, p1, p2, 255, thick, cv2.LINE_AA)
        alpha[veh_mask > 0] = 0
        self._blend(out, alpha, RED, 0.92)

    def _draw_vehicle_lanes(self, out, y_top, y_bottom, veh_mask, s) -> None:
        h, w = out.shape[:2]
        alpha = np.zeros((h, w), np.uint8)
        n_lanes = max(len(self._lines) - 1, 0)
        for v in self._vehicles:
            if v.lane is None:
                continue
            ya = int(max(y_top, v.bottom - 0.12 * h))
            yb = int(min(y_bottom, v.bottom + 0.05 * h))
            if yb <= ya:
                continue
            for ln in (v.left_line, v.right_line):
                if ln is not None:
                    cv2.line(alpha, (int(ln.x_at(ya)), ya), (int(ln.x_at(yb)), yb),
                             255, max(3, int(10 * s)), cv2.LINE_AA)
        alpha[veh_mask > 0] = 0
        self._blend(out, alpha, RED, 1.0)

        for v in self._vehicles:
            if v.lane is None:
                continue
            txt = f"L{v.lane + 1}" if 0 <= v.lane < n_lanes else "OUT"
            self._label(out, txt, (v.box[0], v.box[3] + 18 * s), bg=(60, 60, 60), scale=0.45 * s)

    def _draw_connectors(self, out, y_bottom, s) -> None:
        h, w = out.shape[:2]
        anchor = (int(w * self.ego_x_ratio), int(y_bottom - 0.04 * h))
        for v in self._targets.values():
            if v is None:
                continue
            col = self._dist_color(v.distance)
            p = (int(v.cx), int(min(v.bottom, h - 1)))
            cv2.line(out, anchor, p, col, max(1, int(2 * s)), cv2.LINE_AA)
            cv2.circle(out, p, max(3, int(5 * s)), col, -1, cv2.LINE_AA)
            self._label(out, f"{v.distance:.1f} m", (v.box[0], v.box[1] - 6), bg=col, scale=0.6 * s, thick=2)

    def _draw_hud(self, out, s) -> None:
        h, w = out.shape[:2]
        pw, ph = int(520 * s), int(112 * s)
        x0 = (w - pw) // 2
        y0 = h - ph - int(18 * s)

        ov = out.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + pw, y0 + ph), (25, 25, 25), -1)
        cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)
        cv2.rectangle(out, (x0, y0), (x0 + pw, y0 + ph), (255, 220, 0), max(1, int(2 * s)))

        n_lanes = self.info.get("lane_count", 0)
        ego = self.info.get("ego_lane")
        header = f"LANE {ego}/{n_lanes}" if ego else "LANE --"
        (tw, th), _ = cv2.getTextSize(header, FONT, 0.55 * s, 1)
        cv2.putText(out, header, (x0 + (pw - tw) // 2, y0 + th + int(8 * s)),
                    FONT, 0.55 * s, (255, 220, 0), 1, cv2.LINE_AA)

        for i, (name, key) in enumerate((("LEFT", "left"), ("FRONT", "front"), ("RIGHT", "right"))):
            v = self._targets[key]
            d = v.distance if v else None
            col = self._dist_color(d)
            cx = x0 + int(pw * (2 * i + 1) / 6)

            (nw, _), _ = cv2.getTextSize(name, FONT, 0.5 * s, 1)
            cv2.putText(out, name, (cx - nw // 2, y0 + int(52 * s)), FONT, 0.5 * s,
                        (220, 220, 220), 1, cv2.LINE_AA)

            val = f"{d:.1f} m" if d is not None else "--"
            (vw, _), _ = cv2.getTextSize(val, FONT, 0.95 * s, 2)
            cv2.putText(out, val, (cx - vw // 2, y0 + int(92 * s)), FONT, 0.95 * s, col, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------
# دمو: اجرا روی ویدیو با YOLO
# ----------------------------------------------------------------------
def _demo(source: str, weights: str = "yolov8n.pt", out_path: str = "lane_demo.mp4") -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    keep = {"bicycle", "car", "motorcycle", "bus", "truck"}
    detector = LaneDetector()

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, verbose=False)[0]
        vehicles = []
        for b in res.boxes:
            name = model.names[int(b.cls)]
            if name in keep:
                vehicles.append({
                    "box": b.xyxy[0].tolist(),
                    "cls": name,
                    "id": int(b.id) if b.id is not None else None,
                })
        out, _ = detector.process(frame, vehicles)
        writer.write(out)
        cv2.imshow("Lane Detector", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    _demo(sys.argv[1] if len(sys.argv) > 1 else 0)