"""SchoolGuard — phần mềm hỗ trợ phát hiện nguy cơ bạo lực học đường.

Chạy: python main2.py | python main2.py --help
Mọi chức năng giám sát chạy trong tệp này; trọng số AI vẫn là tệp riêng.
Điểm AI không phải kết luận về hành vi; mọi sự kiện cần người kiểm tra.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import importlib.util
import io
import json
import logging
import logging.handlers
import math
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from feedback_review import ALERT_TEXT, HUMAN_GUIDANCE, ReviewServer, bounded_notification_text, notification_payload
from incident_workflow import IncidentMixin, MaintenanceLoop
from school_policy import (POLICY_DEFAULTS, MODE_LABELS, validate_policy, detection_at,
                           ScenarioMonitor, CameraHealth, patrol_slot, patrol_grid)
from quick_evidence import QuickClip

ROOT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=7))
LOG = logging.getLogger("schoolguard")
cv2 = np = None  # Thư viện nặng chỉ nạp khi thực sự giám sát.
VERSION = "2.2.6"


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]", "_", value, flags=re.UNICODE)[:70] or "camera"


def redact(value: Any) -> str:
    text = str(value)
    text = re.sub(r"(rtsp[s]?|https?)://[^\s/@]+:[^\s/@]+@", r"\1://***@", text)
    text = re.sub(r"https://discord(?:app)?\.com/api/webhooks/[^\s]+", "[Discord webhook]", text)
    text = re.sub(r"https://api\.telegram\.org/bot[^/\s]+", "[Telegram bot]", text)
    text = re.sub(r"([?&](?:token|password|key|auth)=)[^&\s]+", r"\1***", text, flags=re.I)
    return text


def runtime_imports() -> None:
    global cv2, np
    if cv2 is None:
        import cv2 as opencv
        import numpy as numpy
        cv2, np = opencv, numpy


DEFAULT_CONFIG = {
    "sources": [{"label": "Camera 1", "source": 0, "roi": None}],
    "model": {"backend": "auto", "path": "models/violence_resnet18.pt",
              "keras_path": "model_gop.h5", "input_size": 128, "sequence_length": 10,
              "feature_weights": "", "device": "cpu", "threads": 2,
              "video_context_enabled": False, "video_context_kind": "r3d", "video_context_backbone": "models/r3d_18_kinetics400.pt",
              "video_context_head": "models/action_recall_public_20261016.pt"},
    "detection": {"threshold": 0.75, "interaction_threshold": .60, "release_threshold": 0.45, "sustain_seconds": .6,
                  "max_gap_seconds": .25,
                  "release_seconds": 1.5, "cooldown_seconds": 30, "analysis_fps": 5,
                  "require_interaction": True, "motion_threshold": 2.8, "short_kicks_enabled": False,
                  "short_punches_enabled": False},
    "person": {"enabled": True, "path": "models/yolov8s-pose.pt", "confidence": 0.20, "imgsz": 640,
               "body_occlusion": True, "min_body_confidence": .10, "enhanced": False},
    "evidence": {"enabled": True, "pre_seconds": 5, "post_seconds": 8, "fps": 5,
                 "width": 854, "buffer_mb": 32, "min_free_mb": 250,
                 "continuous": False, "segment_seconds": 900},
    "display": {"boxes": True, "heatmap": False, "privacy": False, "sound": True},
    "notifications": {"enabled": False, "discord_env": "SCHOOLGUARD_DISCORD_WEBHOOK",
                      "telegram_token_env": "SCHOOLGUARD_TELEGRAM_TOKEN",
                      "telegram_chat_env": "SCHOOLGUARD_TELEGRAM_CHAT_ID"},
    "storage": {"folder": "schoolguard_data"},
    "feedback": {"enabled": True, "host": "127.0.0.1", "port": 8765, "public_base_url": ""},
    "school_policy": copy.deepcopy(POLICY_DEFAULTS),
    "quick_clip": {"enabled": True},
}


def merge_config(base: dict, patch: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(cfg: dict) -> dict:
    # Old configurations cannot reactivate the removed audio/ASR feature.
    cfg.pop("audio", None)
    validate_policy(cfg['school_policy'], cfg['sources'])
    # Legacy response settings are preserved on disk but no longer used.
    if type(cfg['quick_clip'].get('enabled')) is not bool:
        raise ValueError('Bật/tắt clip ngắn cần true/false.')
    for section in ("model", "detection", "person", "evidence", "display", "notifications", "storage"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"{section} phải là một đối tượng cấu hình.")
    for section, keys in {"person": ("enabled",), "detection": ("require_interaction",),
                          "evidence": ("enabled", "continuous"), "display": ("boxes", "heatmap", "privacy", "sound"),
                          "notifications": ("enabled",)}.items():
        for key in keys:
            if not isinstance(cfg[section][key], bool):
                raise ValueError(f"{section}.{key} phải là true/false.")
    if not isinstance(cfg["person"].get("body_occlusion", True), bool):
        raise ValueError("person.body_occlusion phải là true/false.")
    if not isinstance(cfg["person"].get("enhanced", False), bool):
        raise ValueError("person.enhanced phải là true/false.")
    if type(cfg['detection'].get('short_kicks_enabled', False)) is not bool:
        raise ValueError('detection.short_kicks_enabled phải là true/false.')
    if type(cfg['detection'].get('short_punches_enabled', False)) is not bool:
        raise ValueError('detection.short_punches_enabled phải là true/false.')
    if type(cfg['model'].get('video_context_enabled', False)) is not bool:
        raise ValueError('model.video_context_enabled phải là true/false.')
    if cfg['model'].get('video_context_kind', 'r3d') not in ('r3d', 'vit_cls_research'):
        raise ValueError('model.video_context_kind cần là r3d hoặc vit_cls_research.')
    if cfg['model'].get('video_context_enabled'):
        for key in ('video_context_backbone', 'video_context_head'):
            if not isinstance(cfg['model'].get(key), str) or not cfg['model'][key].strip():
                raise ValueError('Thiếu đường dẫn mô hình chuỗi hành động: '+key)
    feedback = cfg.get("feedback")
    if not isinstance(feedback, dict) or not isinstance(feedback.get("enabled"), bool):
        raise ValueError("feedback.enabled phải là true/false.")
    if (isinstance(feedback.get("port"), bool) or not isinstance(feedback.get("port"), int)
            or not 1 <= feedback["port"] <= 65535):
        raise ValueError("feedback.port cần là số nguyên từ 1 đến 65535.")
    if feedback.get("host") not in ("127.0.0.1", "localhost", "0.0.0.0") or not isinstance(feedback.get("public_base_url"), str):
        raise ValueError("Cấu hình máy chủ phản hồi không hợp lệ.")
    def number(section, key, low, high):
        value = cfg[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{section}.{key} phải nằm trong [{low}, {high}].")
    for s, k, lo, hi in [
        ("detection", "threshold", .05, .99), ("detection", "release_threshold", 0, .98),
        ("detection", "interaction_threshold", .05, .99), ("detection", "max_gap_seconds", 0, .5),
        ("detection", "sustain_seconds", .1, 60), ("detection", "release_seconds", .1, 60),
        ("detection", "cooldown_seconds", 0, 3600), ("detection", "analysis_fps", 1, 30),
        ("detection", "motion_threshold", .1, 30), ("person", "confidence", .05, .95),
        ("person", "imgsz", 128, 1280), ("model", "input_size", 32, 1024),
        ("model", "sequence_length", 1, 120),
        ("evidence", "pre_seconds", 0, 30), ("evidence", "post_seconds", 0, 60),
        ("evidence", "fps", 1, 15), ("evidence", "width", 320, 1920),
        ("evidence", "buffer_mb", 1, 128), ("evidence", "min_free_mb", 0, 100000),
        ("evidence", "segment_seconds", 30, 7200), ("model", "threads", 1, 16),
    ]:
        number(s, k, lo, hi)
    if "min_body_confidence" in cfg["person"]:
        number("person", "min_body_confidence", .05, .5)
    if cfg["detection"]["release_threshold"] >= cfg["detection"]["threshold"]:
        raise ValueError("Ngưỡng kết thúc phải nhỏ hơn ngưỡng cảnh báo.")
    if not cfg["detection"]["release_threshold"] < cfg["detection"]["interaction_threshold"] <= cfg["detection"]["threshold"]:
        raise ValueError("Ngưỡng khi có tương tác phải lớn hơn ngưỡng kết thúc và không lớn hơn ngưỡng cảnh báo.")
    if cfg["model"]["backend"] not in ("auto", "resnet", "keras"):
        raise ValueError("Backend phải là auto, resnet hoặc keras.")
    sources = cfg["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
        raise ValueError("Cấu hình cần từ 1 đến 8 nguồn.")
    labels = set()
    for spec in sources:
        if not isinstance(spec, dict) or not isinstance(spec.get("source"), (str, int)) or isinstance(spec.get("source"), bool) or not str(spec.get("source", "")).strip():
            raise ValueError("Nguồn video không được để trống.")
        if isinstance(spec["source"], int) and spec["source"] < 0:
            raise ValueError("Chỉ số webcam phải không âm.")
        label = str(spec.get("label", "Camera")).strip()
        if not label or label in labels:
            raise ValueError("Tên camera phải có nội dung và không trùng nhau.")
        spec["label"] = label
        labels.add(label)
        roi = spec.get("roi")
        if roi is not None:
            if not isinstance(roi, (list, tuple)) or len(roi) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in roi):
                raise ValueError("ROI cần 4 tọa độ chuẩn hóa.")
            if not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
                raise ValueError("ROI phải nằm trong [0,1] và có diện tích dương.")
    return cfg


PERSON_PROFILES = {
    "Chính xác": ("models/yolov8s-pose.pt", 640),
    "Camera mờ (ưu tiên chi tiết)": ("models/yolov8s-pose.pt", 960),
    "Nhanh": ("models/yolov8n-pose.pt", 640),
}


def person_profile_for(cfg):
    """Không coi trọng số tùy chỉnh/cỡ ảnh khác là preset rồi ghi đè khi Áp dụng."""
    for name, (path, size) in PERSON_PROFILES.items():
        if resolve_path(cfg["path"]) == resolve_path(path) and cfg["imgsz"] == size:
            return name
    return "Tùy chỉnh"


def load_config(filename: Optional[str] = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    target = Path(filename) if filename else ROOT / "main2_config.json"
    if filename is None and not target.exists():
        target = ROOT / "main1_config.json"  # Kế thừa chỉ đọc; lưu luôn vào main2.
    if target.exists():
        if target.suffix.lower() in (".yaml", ".yml"):
            import yaml
            patch = yaml.safe_load(target.read_text(encoding="utf-8-sig"))
        else:
            patch = json.loads(target.read_text(encoding="utf-8-sig"))
        if not isinstance(patch, dict):
            raise ValueError("Cấu hình phải là một đối tượng JSON/YAML.")
        # Đọc định dạng cấu hình cũ khi người dùng chỉ định --config config.yaml.
        if "source" in patch:
            old = patch
            patch = {"sources": [{"label": x.get("label", f"Camera {i+1}"), "source": x["url"]}
                                  for i, x in enumerate(old["source"].get("multi", []))]
                     or [{"label": "Camera 1", "source": old["source"].get("default", 0)}]}
            patch["model"] = {"backend": "keras", "keras_path": old.get("model", {}).get("seq_model_path", "model_gop.h5")}
            if "confidence_threshold" in old.get("detection", {}):
                patch["detection"] = {"threshold": old["detection"]["confidence_threshold"] / 100,
                                      "cooldown_seconds": old["detection"].get("cooldown_seconds", 30)}
            # Kênh gửi dùng biến môi trường; không sao chép thông tin bí mật vào cấu hình mới.
        cfg = merge_config(cfg, patch)
    elif filename:
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {target}")
    return validate_config(cfg)


def resolve_path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def atomic_json(target: Path, data: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _dpapi(data: bytes, decrypt=False) -> bytes:
    """Windows DPAPI: chỉ tài khoản Windows đã lưu có thể giải mã."""
    if os.name != "nt":
        raise RuntimeError("Trên hệ điều hành này hãy dùng biến môi trường cho thông báo.")
    import ctypes
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_byte))]
    buf = ctypes.create_string_buffer(data)
    incoming = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)))
    outgoing = Blob()
    crypt = ctypes.windll.crypt32
    operation = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    if not operation(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)):
        raise OSError("Không xử lý được thông tin xác thực bằng Windows DPAPI.")
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        ctypes.windll.kernel32.LocalFree(outgoing.data)


def load_credentials() -> dict:
    target = ROOT / ".schoolguard" / "credentials.dpapi"
    if not target.exists():
        return {}
    try:
        return json.loads(_dpapi(target.read_bytes(), decrypt=True).decode("utf-8"))
    except (OSError, ValueError, RuntimeError):
        LOG.warning("Không đọc được thông tin xác thực cục bộ; kiểm tra tài khoản Windows.")
        return {}


def save_credentials(**values):
    data = load_credentials()
    data.update(values)
    target = ROOT / ".schoolguard" / "credentials.dpapi"
    target.parent.mkdir(parents=True, exist_ok=True)
    encrypted = _dpapi(json.dumps(data).encode("utf-8"))
    temp = target.with_suffix(".tmp")
    temp.write_bytes(encrypted)
    os.replace(temp, target)


def notification_credentials(cfg: dict) -> dict:
    data = load_credentials()
    return {"discord_webhook": os.environ.get(cfg["discord_env"]) or data.get("discord_webhook", ""),
            "telegram_token": os.environ.get(cfg["telegram_token_env"]) or data.get("telegram_token", ""),
            "telegram_chat_id": os.environ.get(cfg["telegram_chat_env"]) or data.get("telegram_chat_id", "")}


def discover_telegram(cfg: dict) -> dict:
    """Đọc getUpdates/getMe; không gửi tin, không xóa webhook hoặc cập nhật offset."""
    import requests
    token = notification_credentials(cfg).get("telegram_token")
    if not token:
        raise ValueError("Chưa có token Telegram.")
    session = requests.Session()
    me = session.get(f"https://api.telegram.org/bot{token}/getMe", timeout=(5, 15)).json()
    if not me.get("ok"):
        raise ValueError("Telegram từ chối token; cần token hợp lệ.")
    updates = session.get(f"https://api.telegram.org/bot{token}/getUpdates", params={"timeout": 0, "limit": 100}, timeout=(5, 15)).json()
    if not updates.get("ok"):
        raise ValueError("Không đọc được cập nhật bot; kiểm tra bot có đang dùng webhook khác.")
    chats = {}
    for update in updates.get("result", []):
        for key in ("message", "channel_post", "my_chat_member", "chat_member"):
            chat = update.get(key, {}).get("chat", {})
            if chat.get("type") in ("group", "supergroup"):
                chats[str(chat["id"])] = {"id": str(chat["id"]), "title": chat.get("title", "Nhóm"), "type": chat["type"]}
    return {"bot": me["result"].get("username"), "groups": list(chats.values())}


class EventStore(IncidentMixin):
    """SQLite WAL, khóa luồng, truy vấn tham số và nhật ký thay đổi trạng thái."""
    STATUSES = ("Chưa xem", "Đúng là xô xát", "Báo nhầm", "Đã xử lý")

    def __init__(self, folder: Path):
        self.folder = folder.resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.folder / "events.sqlite3", check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
              id TEXT PRIMARY KEY, created TEXT NOT NULL, camera TEXT NOT NULL,
              kind TEXT NOT NULL, score REAL, raw_score REAL, people INTEGER,
              reason TEXT, status TEXT NOT NULL DEFAULT 'Chưa xem', notes TEXT DEFAULT '',
              snapshot TEXT DEFAULT '', clip TEXT DEFAULT '', clip_state TEXT DEFAULT '',
              position REAL, model TEXT DEFAULT '');
            CREATE INDEX IF NOT EXISTS event_time ON events(created DESC);
            CREATE TABLE IF NOT EXISTS audit (
              id INTEGER PRIMARY KEY, created TEXT, event_id TEXT, action TEXT, detail TEXT);
        """)
        self.conn.commit()
        self.feedback = None
        self.init_incidents()
        with self.conn:
            self.conn.execute("UPDATE events SET status='Đúng là xô xát' WHERE status='Đã xác nhận'")
            self.conn.execute("UPDATE events SET status='Báo nhầm' WHERE status='Báo giả'")

    def get(self, eid):
        with self.lock:
            row = self.conn.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
            return dict(row) if row is not None else None

    def add(self, camera: str, kind: str, score: Optional[float], raw_score: Optional[float],
            people: int, reason: str, position: float = 0, model: str = "", event_id: Optional[str] = None) -> str:
        eid = event_id or uuid.uuid4().hex
        for value in (score, raw_score):
            if value is not None and not math.isfinite(value):
                raise ValueError("Điểm sự kiện không hữu hạn.")
        with self.lock, self.conn:
            self.conn.execute("INSERT INTO events(id,created,camera,kind,score,raw_score,people,reason,position,model) VALUES(?,?,?,?,?,?,?,?,?,?)",
                              (eid, now_iso(), camera, kind, score, raw_score, people, reason, position, model))
        return eid

    def update(self, eid: str, **fields) -> None:
        allowed = {"snapshot", "clip", "clip_state", "status", "notes"}
        if not fields or not set(fields) <= allowed:
            raise ValueError("Trường cập nhật không hợp lệ.")
        if "status" in fields:
            fields["status"] = {"Đã xác nhận": "Đúng là xô xát", "Báo giả": "Báo nhầm"}.get(fields["status"], fields["status"])
        if "status" in fields and fields["status"] not in self.STATUSES:
            raise ValueError("Trạng thái không hợp lệ.")
        with self.lock, self.conn:
            if not self.conn.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                raise ValueError("Sự kiện không còn tồn tại.")
            self.conn.execute("UPDATE events SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?", (*fields.values(), eid))
            if 'status' in fields and fields['status']!='Báo nhầm':
                self.conn.execute('DELETE FROM learning_reviews WHERE event_id=?',(eid,))
            self.conn.execute("INSERT INTO audit(created,event_id,action,detail) VALUES(?,?,?,?)",
                              (now_iso(), eid, "update", json.dumps(fields, ensure_ascii=False)))

    def rows(self, search: str = "", status: str = "Tất cả", limit: Optional[int] = 500) -> list[dict]:
        sql, args = "SELECT * FROM events WHERE (camera LIKE ? OR reason LIKE ? OR notes LIKE ?)", [f"%{search}%"] * 3
        if status != "Tất cả":
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY created DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def stats(self) -> dict:
        with self.lock:
            return dict(self.conn.execute("SELECT status,COUNT(*) FROM events GROUP BY status").fetchall())

    def export(self, target: Path, search: str = "", status: str = "Tất cả") -> int:
        rows = self.rows(search, status, None)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".json":
            atomic_json(target, {"exported": now_iso(), "version": VERSION, "events": rows})
        elif target.suffix.lower() == ".html":
            cols = ["created", "camera", "kind", "score", "people", "status", "reason", "notes"]
            cells = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in rows)
            target.write_text("<!doctype html><html lang='vi'><meta charset='utf-8'><title>Báo cáo SchoolGuard</title>"
                              "<style>body{font:14px system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}"
                              "th{background:#e8eef4}</style><h1>Báo cáo sự kiện SchoolGuard</h1>"
                              f"<p>Xuất: {html.escape(now_iso())} · {len(rows)} sự kiện. Điểm AI cần người xác minh.</p>"
                              "<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>" + cells + "</table></html>", encoding="utf-8")
        elif target.suffix.lower() == ".csv":
            cols = list(rows[0]) if rows else ["id", "created", "camera", "kind", "score", "status"]
            with target.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols)
                writer.writeheader()
                # Ngăn dữ liệu camera/ghi chú trở thành công thức trong bảng tính.
                for r in rows:
                    writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v for k, v in r.items()})
        else:
            raise ValueError("Chọn định dạng .csv, .json hoặc .html.")
        return len(rows)

    def close(self):
        with self.lock:
            self.conn.close()


class AlertGate:
    """Xác nhận theo thời gian, hysteresis và cooldown; không sửa điểm mô hình."""
    def __init__(self):
        self.reset()
        self.last_alert = -math.inf

    def reset(self):
        self.high_since = self.low_since = self.gap_since = None
        self.active = False

    def update(self, t: float, score: Optional[float], supported: bool, cfg: dict, support_since=None) -> tuple[str, bool]:
        if score is None or not math.isfinite(score):
            self.reset()
            return "Chưa có kết quả AI", False
        cutoff = cfg.get("interaction_threshold", cfg["threshold"]) if supported and cfg["require_interaction"] else cfg["threshold"]
        qualifies = score >= cutoff and (supported or not cfg["require_interaction"])
        if qualifies:
            self.gap_since = None
            self.low_since = None
            if self.high_since is None:
                self.high_since = t
            # Verifier đã kiểm tra AI và tương tác cùng cặp trong cửa sổ này.
            # Dùng thời gian bằng chứng thật để không chờ xác nhận hai lần.
            if (supported and support_since is not None and math.isfinite(support_since)
                    and 0 <= t-support_since <= 1.6):
                self.high_since = min(self.high_since, support_since)
            if t - self.high_since + 1e-6 >= cfg["sustain_seconds"]:
                self.active = True
        else:
            # Cho phép một khung AI hụt ngắn khi bằng chứng tương tác vẫn liên tục.
            # Mất người/tương tác hoặc điểm xuống thấp sẽ xóa tích lũy ngay.
            if supported and score >= cfg["release_threshold"] and self.high_since is not None:
                self.gap_since = t if self.gap_since is None else self.gap_since
                if t-self.gap_since > cfg.get("max_gap_seconds", 0):
                    self.high_since = None
            else:
                self.high_since = self.gap_since = None
            if score < cfg["release_threshold"] or not supported and cfg["require_interaction"]:
                if self.low_since is None:
                    self.low_since = t
                if t - self.low_since >= cfg["release_seconds"]:
                    self.active = False
            else:
                self.low_since = None
        fire = self.active and qualifies and t - self.last_alert >= cfg["cooldown_seconds"]
        if fire:
            self.last_alert = t
        if self.active:
            return ALERT_TEXT, fire
        if score >= cfg["threshold"] and not supported and cfg["require_interaction"]:
            return "AI nghi ngờ · thiếu bằng chứng tương tác", False
        if self.high_since is not None or score >= cfg["release_threshold"]:
            return "Đang theo dõi", False
        return "Chưa thấy dấu hiệu", False


@dataclass
class PersonBox:
    """Thông tin bounding box và trạng thái phân tích của từng người."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: int = 0
    motion_score: float = 0.0
    is_fighting: bool = False
    is_victim: bool = False
    is_attacker: bool = False
    fight_partner_id: Optional[int] = None
    keypoints: Any = None
    observed: bool = True
    local_motion: float = 0.0
    body_speed: float = 0.0
    display_tracked: bool = False

    @property
    def int_box(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_prone(self) -> bool:
        """Kiểm tra cơ bản: chiều ngang >= 1.25 chiều cao."""
        return self.width >= self.height * 1.25 and self.width >= 45.0 and (self.height >= 25.0)

    def check_is_prone(self, frame_w: int=0, frame_h: int=0) -> bool:
        """
        Ước lượng tư thế nằm từ hình dạng hộp; cần người xác minh:
        - Tỉ lệ chiều ngang lớn hơn chiều cao rõ rệt (người nằm dài trên sàn)
        - Lọc bớt cận cảnh phỏng vấn và hộp quá lớn
        - Đáy box phải tiếp xúc với mặt sàn (ở nửa dưới khung hình)
        """
        if self.width < self.height * 1.3 or self.width < 55.0 or self.height < 25.0:
            return False
        if frame_w > 0 and frame_h > 0:
            box_area = self.width * self.height
            frame_area = frame_w * frame_h
            if box_area > frame_area * 0.18:
                return False
            if self.width > frame_w * 0.45 or self.height > frame_h * 0.45:
                return False
            if self.y2 < frame_h * 0.38:
                return False
            if self.y1 < frame_h * 0.1:
                return False
        return True

def _distinct_body_poses(a: PersonBox, b: PersonBox) -> bool:
    """Hai bộ khớp thân khác vị trí có thể thuộc hai người chồng hộp nhau."""
    if a.keypoints is None or b.keypoints is None:
        return False
    distances = []
    for index in (5, 6, 11, 12):
        if index >= len(a.keypoints) or index >= len(b.keypoints):
            continue
        pa, pb = a.keypoints[index], b.keypoints[index]
        if len(pa) >= 3 and len(pb) >= 3 and pa[2] >= .35 and pb[2] >= .35:
            distances.append(math.hypot(float(pa[0]-pb[0]), float(pa[1]-pb[1])))
    return len(distances) >= 3 and sorted(distances)[len(distances)//2] > .18*max(30, min(a.height, b.height))


def _nms_boxes(boxes: List[PersonBox], iou_threshold: float=0.6) -> List[PersonBox]:
    """
    Non-Maximum Suppression thông minh:
    - Loại bỏ theo IoU trùng lặp
    - Loại bỏ theo Intersection-over-Minimum (IoM / Containment): triệt tiêu ghost box
      nhưng vẫn giữ được 2 người khi đang giằng co, vật lộn (grapple/tackle).
    """
    if len(boxes) <= 1:
        return boxes
    boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        rem = []
        for b in boxes:
            ix1 = max(best.x1, b.x1)
            iy1 = max(best.y1, b.y1)
            ix2 = min(best.x2, b.x2)
            iy2 = min(best.y2, b.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = best.area + b.area - inter
            iou = inter / union if union > 0 else 0.0
            min_area = min(best.area, b.area)
            iom = inter / min_area if min_area > 0 else 0.0
            if (iou < iou_threshold and iom < 0.82) or _distinct_body_poses(best, b):
                rem.append(b)
        boxes = rem
    return keep

class SimpleTracker:
    """
    Tracker gán ID dựa trên vị trí và làm mượt hộp người.
    Hệ số smoothing 0.82 giúp box bám dính tức thì vào người mà không giật lắc.
    """

    def __init__(self, max_disappeared: int=8, smoothing_factor: float=0.82):
        self.next_id = 1
        self.tracks: Dict[int, PersonBox] = {}
        self.disappeared: Dict[int, int] = {}
        self.smoothing = smoothing_factor
        self.max_disappeared = max_disappeared
        self.person_enhancement_tick = 0

    def update(self, detected_boxes: List[PersonBox]) -> List[PersonBox]:
        if not detected_boxes:
            for tid in list(self.tracks.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
            active = [self.tracks[tid] for tid in self.tracks if self.disappeared[tid] <= 2]
            for b in active:
                b.observed = False
            return _nms_boxes(active, iou_threshold=0.6)
        if not self.tracks:
            for b in detected_boxes:
                b.track_id = self.next_id
                self.tracks[self.next_id] = b
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return _nms_boxes(list(self.tracks.values()), iou_threshold=0.6)
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid] for tid in track_ids]
        det_boxes = detected_boxes
        used_tracks = set()
        used_dets = set()
        matches = []
        for t_idx, tb in enumerate(track_boxes):
            tc = tb.center
            diag = math.hypot(tb.width, tb.height)
            for d_idx, db in enumerate(det_boxes):
                dc = db.center
                ix1 = max(tb.x1, db.x1)
                iy1 = max(tb.y1, db.y1)
                ix2 = min(tb.x2, db.x2)
                iy2 = min(tb.y2, db.y2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = tb.area + db.area - inter
                iou = inter / union if union > 0 else 0.0
                dist = math.hypot(tc[0] - dc[0], tc[1] - dc[1])
                if iou > 0.05:
                    cost = 1.0 - iou
                    matches.append((cost, t_idx, d_idx))
                elif dist <= max(60.0, diag * .9):
                    cost = 1.0 + dist / max(100.0, diag)
                    matches.append((cost, t_idx, d_idx))
        matches.sort(key=lambda x: x[0])
        for cost, t_idx, d_idx in matches:
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            tid = track_ids[t_idx]
            old_b = self.tracks[tid]
            new_b = det_boxes[d_idx]
            displacement = math.dist(old_b.center, new_b.center)
            # Khi người chuyển động nhanh, giảm trễ hộp thay vì kéo hộp về vị trí cũ.
            s = max(self.smoothing, .96) if displacement > .08*max(30, old_b.height) else self.smoothing
            smooth_box = PersonBox(x1=old_b.x1 * (1 - s) + new_b.x1 * s, y1=old_b.y1 * (1 - s) + new_b.y1 * s, x2=old_b.x2 * (1 - s) + new_b.x2 * s, y2=old_b.y2 * (1 - s) + new_b.y2 * s, confidence=new_b.confidence, track_id=tid, keypoints=new_b.keypoints)
            self.tracks[tid] = smooth_box
            self.disappeared[tid] = 0
            used_tracks.add(t_idx)
            used_dets.add(d_idx)
        for d_idx, b in enumerate(det_boxes):
            if d_idx not in used_dets:
                b.track_id = self.next_id
                self.tracks[self.next_id] = b
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in used_tracks:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
        active_tracks = [self.tracks[tid] for tid in list(self.tracks.keys()) if self.disappeared.get(tid, 0) <= 2]
        for b in active_tracks:
            b.observed = self.disappeared.get(b.track_id, 0) == 0
        return _nms_boxes(active_tracks, iou_threshold=0.6)

    def reset(self):
        """Xóa sạch các track khi chuyển cảnh."""
        self.tracks.clear()
        self.disappeared.clear()
        self.next_id = 1
        self.person_enhancement_tick = 0


class InteractionVerifier:
    """Bằng chứng theo cặp và thời gian; giao nhau/đi cạnh nhau không đủ để báo động.

    Đây là bộ lọc tín hiệu, không phải mô hình xác định ý định hoặc người có lỗi.
    Điểm phân loại hình ảnh được giữ nguyên cho người xem đối chiếu.
    """
    def __init__(self):
        self.history, self.pose_history, self.pairs = {}, {}, {}
        self.last_time = None
        self.short_kicks = None
        self.short_punches = None

    def reset(self):
        self.__init__()

    @staticmethod
    def joint(box, index):
        points = box.keypoints
        if points is None or len(points) <= index or len(points[index]) < 3:
            return None
        point = np.asarray(points[index][:3], dtype=float)
        if not np.isfinite(point).all() or point[2] < .35:
            return None
        return point[:2]

    def observe(self, box, t):
        history = self.history.setdefault(box.track_id, deque(maxlen=16))
        history.append((t, np.asarray(box.center), max(30.0, box.height)))
        while history and t-history[0][0] > 2.4:
            history.popleft()
        walking = False
        if len(history) >= 4 and history[-1][0]-history[0][0] >= .6:
            positions = np.asarray([r[1] for r in history])
            steps = np.diff(positions, axis=0)
            lengths = np.linalg.norm(steps, axis=1)
            path = float(lengths.sum())
            net = float(np.linalg.norm(positions[-1]-positions[0]))
            scale = float(np.mean([r[2] for r in history]))
            duration = history[-1][0]-history[0][0]
            walking = path > .12*scale and net/max(1e-6, path) > .90 and net/scale/duration > .10
        current = {j: p for j in (5, 6, 9, 10, 11, 12, 15, 16)
                   if (p := self.joint(box, j)) is not None}
        previous = self.pose_history.get(box.track_id)
        speeds = {}
        if previous and .04 <= t-previous[0] <= .65:
            # Hai khung phải so cùng khớp thân; hông mất không được đổi sang
            # tâm hộp rồi coi độ lệch anchor là một cú đánh/đá.
            shared = [j for j in (11, 12) if j in current and j in previous[1]]
            if not shared:
                shared = [j for j in (5, 6) if j in current and j in previous[1]]
            if shared:
                anchor = np.mean([current[j] for j in shared], axis=0)
                old_anchor = np.mean([previous[1][j] for j in shared], axis=0)
                speeds = {j: min(12.0, float(np.linalg.norm((p-anchor)/max(30., box.height)
                                         -(previous[1][j]-old_anchor)/max(30., previous[2]))/(t-previous[0])))
                          for j, p in current.items() if j in (9, 10, 15, 16) and j in previous[1]}
        self.pose_history[box.track_id] = (t, current, box.height)
        return walking, speeds

    def strike(self, attacker, target, speeds):
        # Chỉ xét cổ tay ở trên hông hoặc chân nâng cao; bước chân qua nhau không là cú đá.
        max_speed = 0.0
        for joint in (9, 10, 15, 16):
            point = self.joint(attacker, joint)
            if point is None or speeds.get(joint, 0) < .85:
                continue
            if joint >= 15 and point[1] > attacker.y1+.72*attacker.height:
                continue
            if joint <= 10 and point[1] > attacker.y1+.78*attacker.height:
                continue
            margin = .10*max(30, target.height)
            if (target.x1-margin <= point[0] <= target.x2+margin and
                    target.y1-margin <= point[1] <= target.y1+.78*target.height):
                max_speed = max(max_speed, speeds[joint])
        return max_speed

    def touching_torso(self, a, b):
        torso = [self.joint(b, j) for j in (5, 6, 11, 12)]
        torso = [p for p in torso if p is not None]
        if len(torso) < 3:
            return None
        hull = cv2.convexHull(np.asarray(torso, dtype=np.float32))
        for j in (9, 10):
            hand = self.joint(a, j)
            if hand is not None and cv2.pointPolygonTest(hull, (float(hand[0]), float(hand[1])), True) >= -.06*max(30, b.height):
                return True
        return False

    def update(self, t, boxes, frame_shape, visual, detection=None):
        detection = detection or DEFAULT_CONFIG["detection"]
        if self.last_time is not None and (t <= self.last_time or t-self.last_time > 1.0):
            self.reset()
        self.last_time = t
        for b in boxes:
            b.is_fighting = b.is_attacker = b.is_victim = False
            b.fight_partner_id = None
        fresh = [b for b in boxes if b.observed and not b.display_tracked and b.height >= 25]
        motion = {b.track_id: self.observe(b, t) for b in fresh}
        supported, passing, contacts, details, confirmed_since = False, False, 0, [], None
        for i, a in enumerate(fresh):
            for b in fresh[i+1:]:
                height = max(30, (max(a.height, a.width)+max(b.height, b.width))/2)
                gap = max(0, max(a.x1, b.x1)-min(a.x2, b.x2))
                distance = math.dist(a.center, b.center)/height
                walk_a, speed_a = motion[a.track_id]
                walk_b, speed_b = motion[b.track_id]
                geometry = (gap <= .18*height and distance <= 1.05
                            and min(max(a.height, a.width), max(b.height, b.width))/max(a.height, a.width, b.height, b.width) >= .35)
                strike_speed = max(self.strike(a, b, speed_a), self.strike(b, a, speed_b)) if geometry else 0.0
                cropped = max(a.y2, b.y2) >= frame_shape[0]-5
                near = geometry and (abs(a.y2-b.y2) <= .35*height or cropped or strike_speed >= .85)
                key = tuple(sorted((a.track_id, b.track_id)))
                history = self.pairs.setdefault(key, deque(maxlen=16))
                walk = walk_a and walk_b or walk_a and b.body_speed < .08 or walk_b and a.body_speed < .08
                strike_speed = strike_speed if near else 0.0
                overlap = max(0, min(a.x2, b.x2)-max(a.x1, b.x1))/max(20, min(a.width, b.width))
                local = max(a.local_motion, b.local_motion)
                # Giằng co cần chuyển động bên trong cả hai hộp, không chỉ dịch chuyển toàn thân.
                touch_a, touch_b = self.touching_torso(a, b), self.touching_torso(b, a)
                touch = touch_a is True or touch_b is True
                # Nếu pose bị che khi người nằm/ngã, vẫn cho tín hiệu giằng co
                # từ residual mạnh; hộp đứng cạnh/chồng nhau không dùng fallback này.
                prone_occlusion = ((a.width > 1.2*a.height or b.width > 1.2*b.height)
                                   and (touch_a is None or touch_b is None)
                                   and min(a.local_motion, b.local_motion) >= .10
                                   and max(a.local_motion, b.local_motion) >= .30)
                grapple = near and overlap >= .25 and min(a.local_motion, b.local_motion) >= .12 and not walk and (touch or prone_occlusion)
                action = near and (strike_speed >= .85 and local >= .04 or grapple)
                history.append({"t": t, "near": near, "action": action, "strike": strike_speed, "walk": walk, "visual": visual})
                while history and t-history[0]["t"] > 1.6:
                    history.popleft()
                span = history[-1]["t"]-history[0]["t"]
                all_events = [r for r in history if r["near"] and r["action"]]
                strong_strikes = sum(r["strike"] >= 1.8 for r in all_events) >= 3
                # Không dùng chuyển động đi bộ trước đó để xác nhận sau khi người dừng.
                events = [r for r in all_events if not r["walk"] or strong_strikes]
                # Hai tín hiệu rời nhau không tạo thành chuỗi tương tác liên tục.
                # Cho phép hụt một lần lấy mẫu ở tốc độ phân tích đã chọn.
                gap_limit = max(.45, 2.1/detection["analysis_fps"])
                for event_index in range(len(events)-1, 0, -1):
                    if events[event_index]["t"]-events[event_index-1]["t"] > gap_limit:
                        events = events[event_index:]
                        break
                near_fraction = sum(r["near"] for r in history)/len(history)
                passing |= near and walk and not strong_strikes
                contacts += int(near)
                # Thời lượng phải thuộc chính chuỗi hành động của cùng cặp người,
                # không tính thời gian đứng sát nhau trước hai cử động ngắn.
                action_span = events[-1]["t"]-events[0]["t"] if events else 0.
                ready = (near and action_span+1e-6 >= .6 and len(events) >= 3 and near_fraction >= .6
                         and t-events[-1]["t"] <= gap_limit and (not walk or strong_strikes))
                if ready:
                    supported = True
                    a.is_fighting = b.is_fighting = True
                    a.fight_partner_id, b.fight_partner_id = b.track_id, a.track_id
                    details.append({"tracks": list(key), "duration": round(action_span, 2),
                                    "action_samples": len(events), "action_times": [r["t"] for r in events],
                                    "kind": "tay/chân chuyển động sát người" if strike_speed else "giằng co"})
                    consistent = [r for r in events if r["visual"] is not None and r["visual"] >= detection["interaction_threshold"]]
                    if len(consistent) >= 2:
                        earliest = consistent[0]["t"]
                        confirmed_since = earliest if confirmed_since is None else min(confirmed_since, earliest)
        if detection.get('short_kicks_enabled', False):
            if self.short_kicks is None:
                from short_contact import RaisedKickMonitor
                self.short_kicks = RaisedKickMonitor(self.joint)
            short_pairs = self.short_kicks.update(t, fresh, visual, detection['interaction_threshold'])
            by_id = {b.track_id: b for b in fresh}
            for pair in short_pairs:
                a, b = (by_id[tid] for tid in pair['tracks'])
                a.is_fighting = b.is_fighting = supported = True
                a.fight_partner_id, b.fight_partner_id = b.track_id, a.track_id
                details.append(pair)
                earliest = pair['confirmed_since']
                confirmed_since = earliest if confirmed_since is None else min(confirmed_since, earliest)
        else:
            self.short_kicks = None
        if detection.get('short_punches_enabled', False):
            if self.short_punches is None:
                from short_contact import RaisedPunchMonitor
                self.short_punches = RaisedPunchMonitor(self.joint)
            by_id = {b.track_id: b for b in fresh}
            for pair in self.short_punches.update(t, fresh, visual, detection['interaction_threshold']):
                a, b = (by_id[tid] for tid in pair['tracks'])
                a.is_fighting = b.is_fighting = supported = True
                a.fight_partner_id, b.fight_partner_id = b.track_id, a.track_id
                details.append(pair)
                earliest = pair['confirmed_since']
                confirmed_since = earliest if confirmed_since is None else min(confirmed_since, earliest)
        else:
            self.short_punches = None
        live_ids = {b.track_id for b in fresh}
        for tid in list(self.history):
            if self.history[tid] and t-self.history[tid][-1][0] > 2.4:
                self.history.pop(tid, None)
                self.pose_history.pop(tid, None)
        for key in list(self.pairs):
            if not self.pairs[key] or t-self.pairs[key][-1]["t"] > 1.6:
                self.pairs.pop(key, None)
        if supported:
            reason = "Có tương tác mạnh lặp lại giữa các người trong cùng đoạn"
        elif passing:
            reason = "Chuyển động đi ngang/đi cạnh nhau; chưa có tương tác mạnh"
        elif len(live_ids) < 2:
            reason = "Chưa đủ dữ liệu để xác minh tương tác: chưa thấy đủ hai người"
        elif contacts and not any(speeds for _, speeds in motion.values()):
            reason = "Chưa đủ dữ liệu để xác minh tương tác: thiếu diễn biến khớp cơ thể"
        elif contacts:
            reason = "Người ở gần nhau; chưa đủ diễn biến tay/chân hoặc giằng co"
        else:
            reason = "Chưa có tương tác gần giữa các người"
        uncertainties = []
        missing_pose = sum(sum(self.joint(b, j) is not None for j in (5, 6, 11, 12)) < 3 for b in fresh)
        if missing_pose:
            uncertainties.append("Khớp cơ thể thiếu/bị che khuất; cần xem camera")
        if len(fresh) >= 6 or contacts >= 4:
            uncertainties.append("Đông người/chen lấn; khó tách tương tác, cần người kiểm tra")
        if uncertainties:
            reason += "; " + "; ".join(uncertainties)
        return {"supported": supported, "passing": passing, "contact_pairs": contacts,
                "confirmed_since": confirmed_since,
                "observed_people": len(fresh), "pairs": details, "reason": reason,
                "uncertainties": uncertainties, "requires_human_review": bool(uncertainties or supported)}

def analyze_fighting_clusters(boxes: List[PersonBox], orig_w: int=0, orig_h: int=0, motion_threshold: float=2.8, is_global_violence: bool=False, ai_conf: float=0.0) -> List[PersonBox]:
    """
        Phân tích cụm ẩu đả thông minh (Smart Combat Analysis):
        - Nhận diện tư thế nạn nhân ngã/nằm sàn chuẩn xác với check_is_prone(orig_w, orig_h).
        - Nếu có nạn nhân nằm sàn và đối tượng đứng sát gần: đánh dấu ngay VICTIM & ATTACKER.
        - Phát hiện cặp ẩu đả đang giằng co / xô xát khi có va chạm và chuyển động thực tế.
        """
    if not boxes:
        return boxes
    for b in boxes:
        b.is_fighting = False
        b.is_victim = False
        b.is_attacker = False
        b.fight_partner_id = None
    n = len(boxes)
    fighting_ids = set()
    is_elevated = is_global_violence or ai_conf >= 50.0
    prone_candidates = [b for b in boxes if b.check_is_prone(orig_w, orig_h)]
    standing_candidates = [b for b in boxes if b not in prone_candidates]
    if prone_candidates:
        for vb in prone_candidates:
            matched_attacker = False
            for sb in standing_candidates:
                if sb.height > vb.height * 3.5 or sb.height > vb.width * 2.5:
                    continue
                diff_y2 = abs(sb.y2 - vb.y2)
                max_ground_diff = max(45.0, sb.height * (0.42 if is_elevated else 0.28))
                if diff_y2 > max_ground_diff:
                    continue
                dist_x = abs(sb.center[0] - vb.center[0])
                max_x_reach = max(vb.width * 0.75, sb.width * 1.1) * (1.3 if is_elevated else 1.0)
                if dist_x > max_x_reach:
                    continue
                dist = math.hypot(vb.center[0] - sb.center[0], vb.center[1] - sb.center[1])
                max_reach = max(110.0, sb.height * (1.15 if is_elevated else 0.85))
                if dist > max_reach:
                    continue
                if ai_conf < 25.0 and (not is_elevated):
                    continue
                req_att_motion = 0.8 if is_elevated else 1.8
                if is_elevated or sb.motion_score >= req_att_motion or vb.motion_score >= 1.2:
                    vb.is_victim = True
                    vb.is_fighting = True
                    sb.is_attacker = True
                    sb.is_fighting = True
                    vb.fight_partner_id = sb.track_id
                    sb.fight_partner_id = vb.track_id
                    fighting_ids.add(sb.track_id)
                    fighting_ids.add(vb.track_id)
                    matched_attacker = True
                    break
            pass
    for i in range(n):
        b1 = boxes[i]
        c1 = b1.center
        h1 = b1.height
        for j in range(i + 1, n):
            b2 = boxes[j]
            c2 = b2.center
            h2 = b2.height
            avg_h = max(30.0, (h1 + h2) / 2.0)
            avg_w = max(20.0, (b1.width + b2.width) / 2.0)
            diff_ground = abs(b1.y2 - b2.y2)
            diff_top = abs(b1.y1 - b2.y1)
            if diff_ground > avg_h * 0.22:
                continue
            if ai_conf < 25.0 and (not is_elevated) and (not is_global_violence):
                if diff_ground > avg_h * 0.13 or diff_top > avg_h * 0.16:
                    continue
            x_overlap = max(0.0, min(b1.x2, b2.x2) - max(b1.x1, b2.x1))
            box_gap = max(0.0, max(b1.x1, b2.x1) - min(b1.x2, b2.x2))
            min_motion = min(b1.motion_score, b2.motion_score)
            pair_motion = max(b1.motion_score, b2.motion_score)
            if x_overlap > 0:
                overlap_ratio = x_overlap / avg_w
                if overlap_ratio < 0.22:
                    if min_motion < 1.9 or pair_motion < 3.8 or min_motion + pair_motion < 5.8:
                        continue
                if pair_motion < 2.2 and min_motion < 1.3 and (ai_conf < 35.0) and (not is_elevated):
                    continue
                if min_motion >= 1.6 and pair_motion >= 2.2 and (min_motion + pair_motion >= 4.0):
                    b1.is_fighting = True
                    b2.is_fighting = True
                    b1.fight_partner_id = b2.track_id
                    b2.fight_partner_id = b1.track_id
                    fighting_ids.add(b1.track_id)
                    fighting_ids.add(b2.track_id)
            else:
                if box_gap > avg_h * 0.15:
                    continue
                if (ai_conf >= 55.0 or is_elevated) and (pair_motion >= 3.0 and min_motion >= 2.0 and (pair_motion + min_motion >= 5.2)):
                    b1.is_fighting = True
                    b2.is_fighting = True
                    b1.fight_partner_id = b2.track_id
                    b2.fight_partner_id = b1.track_id
                    fighting_ids.add(b1.track_id)
                    fighting_ids.add(b2.track_id)
    if not fighting_ids and n == 1 and is_global_violence:
        b = boxes[0]
        ar = b.width / max(1.0, b.height)
        if ar >= 0.7 and b.motion_score >= 4.2 and (ai_conf >= 85.0):
            b.is_fighting = True
            fighting_ids.add(b.track_id)
    return boxes

def person_box_style(box: PersonBox, alert_active: bool, alert_track_ids,
                     previous_frame: bool = False, *, display_number: int = 1):
    """Màu/nhãn hiển thị; không đổi các cờ bằng chứng của tracker/AI."""
    label = f"NGUOI #{display_number}"
    eligible = (box.observed and box.is_fighting if alert_track_ids is None
                else box.track_id in alert_track_ids and (box.observed or previous_frame))
    if alert_active and eligible:
        suffix = "CANH BAO | AI KHUNG TRUOC" if previous_frame else "CANH BAO"
        return (0, 25, 240), f"{label} [{suffix}]", (0, 10, 190)
    if box.display_tracked:
        return (0, 215, 65), f"{label} [DANG BAM]", (0, 130, 35)
    if previous_frame or not box.observed:
        suffix = "AI KHUNG TRUOC" if previous_frame else "CHUA CAP NHAT"
        return (150, 150, 150), f"{label} [{suffix}]", (75, 75, 75)
    if box.is_fighting:
        return (0, 190, 240), f"{label} [TUONG TAC?]", (0, 110, 150)
    return (0, 215, 65), label, (0, 130, 35)


def draw_tactical_boxes(frame: np.ndarray, boxes: List[PersonBox], orig_w: int, orig_h: int,
                        is_global_violence: bool, conf: float=0.0, *,
                        alert_track_ids=None, previous_frame: bool=False) -> np.ndarray:
    if not boxes:
        return frame
    dh, dw = frame.shape[:2]
    sx = dw / float(orig_w)
    sy = dh / float(orig_h)
    sc = dw / 800.0
    fs = max(0.32, 0.44 * sc)
    corner_len = max(12, int(20 * sc))
    c_th = max(3, int(3.5 * sc))
    mapped_boxes = []
    # Compact numbers describe the people visible in THIS camera. Keep their
    # relative track order stable even when confidence changes detector order.
    # Never rewrite track_id: temporal histories and alert membership use it.
    ordered_boxes = sorted(boxes, key=lambda b: (b.track_id, b.x1, b.y1))
    for display_number, b in enumerate(ordered_boxes, 1):
        x1 = int(b.x1 * sx)
        y1 = int(b.y1 * sy)
        x2 = int(b.x2 * sx)
        y2 = int(b.y2 * sy)
        mapped_boxes.append((b, x1, y1, x2, y2, display_number))
    drawn_pairs = set()
    if is_global_violence and not previous_frame:
        for b, x1, y1, x2, y2, _ in mapped_boxes:
            eligible = b.observed and (b.is_fighting if alert_track_ids is None else b.track_id in alert_track_ids)
            if eligible and b.is_fighting and b.fight_partner_id:
                pair_key = tuple(sorted([b.track_id, b.fight_partner_id]))
                if pair_key not in drawn_pairs:
                    drawn_pairs.add(pair_key)
                    partner = next((item for item in mapped_boxes if item[0].track_id == b.fight_partner_id), None)
                    if (partner and partner[0].observed and partner[0].is_fighting
                            and (alert_track_ids is None or partner[0].track_id in alert_track_ids)):
                        _, px1, py1, px2, py2, _ = partner
                        c1 = ((x1 + x2) // 2, (y1 + y2) // 2)
                        c2 = ((px1 + px2) // 2, (py1 + py2) // 2)
                        dist_px = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                        box_h = max(y2 - y1, py2 - py1)
                        partner_box = partner[0]
                        min_box_h = min(y2 - y1, py2 - py1)
                        x_ov = max(0, min(x2, px2) - max(x1, px1))
                        if x_ov > 15 and dist_px < max(80.0, min_box_h * 0.85) and (abs(y2 - py2) < min_box_h * 0.35):
                            cv2.line(frame, c1, c2, (0, 0, 255), max(2, int(2.5 * sc)), cv2.LINE_AA)
                            mid_x = (c1[0] + c2[0]) // 2
                            mid_y = (c1[1] + c2[1]) // 2
                            cv2.putText(frame, 'TUONG TAC', (mid_x - int(24 * sc), mid_y - int(6 * sc)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, (0, 0, 255), 1, cv2.LINE_AA)
    occupied_tags = []
    for b, x1, y1, x2, y2, display_number in mapped_boxes:
        primary_col, label_text, tag_bg = person_box_style(
            b, is_global_violence, alert_track_ids, previous_frame, display_number=display_number)
        cv2.rectangle(frame, (x1, y1), (x2, y2), primary_col, 1)
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), primary_col, c_th)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), primary_col, c_th)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), primary_col, c_th)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), primary_col, c_th)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), primary_col, c_th)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), primary_col, c_th)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), primary_col, c_th)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), primary_col, c_th)
        dot_r = max(2, int(2.5 * sc))
        cv2.circle(frame, (x1, y1), dot_r, primary_col, -1)
        cv2.circle(frame, (x2, y1), dot_r, primary_col, -1)
        cv2.circle(frame, (x1, y2), dot_r, primary_col, -1)
        cv2.circle(frame, (x2, y2), dot_r, primary_col, -1)
        (tw, tht), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        pad = 5
        tag_h = tht + pad * 2
        lx1 = max(0, min(x1, dw - tw - pad * 2 - 8))
        lx2 = lx1 + tw + pad * 2 + 8
        ly1 = max(0, y1 - tag_h - 2)
        ly2 = ly1 + tag_h
        for ox1, oy1, ox2, oy2 in occupied_tags:
            if not (lx2 < ox1 or lx1 > ox2 or ly2 < oy1 or (ly1 > oy2)):
                ly1 = min(dh - tag_h, oy2 + 2)
                ly2 = ly1 + tag_h
                break
        occupied_tags.append((lx1, ly1, lx2, ly2))
        tag_overlay = frame[ly1:ly2, lx1:lx2]
        if tag_overlay.size > 0:
            t_bg = np.full_like(tag_overlay, tag_bg, dtype=np.uint8)
            cv2.addWeighted(tag_overlay, 0.2, t_bg, 0.8, 0, tag_overlay)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), primary_col, 1)
        cv2.rectangle(frame, (lx1, ly1), (lx1 + max(3, int(4 * sc)), ly2), (255, 255, 255), -1)
        cv2.putText(frame, label_text, (lx1 + pad + int(4 * sc), ly2 - pad - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

class SceneCutDetector:
    """
    Bộ phát hiện chuyển cảnh / đổi góc quay (Shot / Scene Cut Detector) siêu tốc:
    - Tính Mean Absolute Difference (MAD) trên frame thu nhỏ 64x64.
    - Tính tương quan biểu đồ màu BGR (Color Histogram Correlation).
    - Phát hiện chuyển cảnh khi MAD >= 26.0 hoặc Correlation <= 0.60.
    """

    def __init__(self, mad_threshold: float=26.0, corr_threshold: float=0.6):
        self.mad_threshold = mad_threshold
        self.corr_threshold = corr_threshold
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None

    def check_cut(self, frame: np.ndarray) -> bool:
        # This detector is also used by standalone evaluation tools before a
        # VisualEngine exists. Missing lazy imports must not disable resets.
        runtime_imports()
        try:
            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            if self._prev_gray is None or self._prev_hist is None:
                self._prev_gray = gray
                self._prev_hist = hist
                return False
            mad = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
            corr = float(cv2.compareHist(hist, self._prev_hist, cv2.HISTCMP_CORREL))
            self._prev_gray = gray
            self._prev_hist = hist
            # Chuyển động mạnh/rung camera có MAD lớn nhưng vẫn cùng cảnh.
            # Yêu cầu cả đổi cấu trúc ảnh và đổi histogram để xóa lịch sử.
            return mad >= self.mad_threshold and corr <= self.corr_threshold
        except Exception:
            return False

@dataclass
class VisualState:
    features: deque = field(default_factory=deque)
    smooth: Optional[float] = None
    frame_time: Optional[float] = None
    image_smooth: Optional[float] = None


class VisualEngine:
    """Chia sẻ trọng số, nhưng mỗi camera giữ chuỗi và EMA độc lập."""
    def __init__(self, cfg: dict):
        runtime_imports()
        self.cfg, self.lock = cfg, threading.Lock()
        backend = cfg["backend"]
        if backend == "auto":
            backend = "resnet" if resolve_path(cfg["path"]).is_file() else "keras"
        self.backend, self.extractor, self.temporal = backend, None, None
        model_path = resolve_path(cfg["path"] if backend == "resnet" else cfg["keras_path"])
        if not model_path.is_file():
            raise FileNotFoundError(f"Thiếu trọng số {model_path.name}; chọn mô hình trong Cài đặt.")
        self.identity = model_path.name + ":" + hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        if backend == "resnet":
            import torch
            from torchvision.models import resnet18
            from torchvision import transforms
            torch.set_num_threads(int(cfg["threads"]))
            self.torch = torch
            requested = cfg.get("device", "cpu")
            self.device = "cuda" if requested in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
            self.model = resnet18(weights=None)  # Không tải mô hình mạng khi đã có checkpoint.
            self.model.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(self.model.fc.in_features, 2))
            checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=True)
            preprocessing = {}
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                architecture = checkpoint.get("architecture", "resnet18")
                if architecture not in ("resnet18", "resnet18_temporal"):
                    raise ValueError("Checkpoint này không dùng kiến trúc ResNet18.")
                preprocessing = checkpoint.get("preprocessing", {})
                weights = checkpoint["state_dict"]
            else:
                weights = checkpoint
            self.model.load_state_dict(weights)
            if isinstance(checkpoint, dict) and 'image_selector' in checkpoint:
                from confidence_head import ConfidenceFallbackHead
                self.model.fc = ConfidenceFallbackHead(self.model.fc, checkpoint['image_selector'])
            self.model.to(self.device).eval()
            if isinstance(checkpoint, dict) and checkpoint.get("architecture") == "resnet18_temporal":
                from temporal_visual import TemporalHead, CausalScorer, WINDOW, MIN_FRAMES, SAMPLE_FPS
                settings = checkpoint.get("temporal", {})
                if any(settings.get(k) != v for k, v in (("window", WINDOW), ("min_frames", MIN_FRAMES), ("sample_fps", SAMPLE_FPS))):
                    raise ValueError("Cấu hình chuỗi hình ảnh không tương thích với checkpoint.")
                self.temporal_weight = float(settings.get("weight", 1.0))
                if not math.isfinite(self.temporal_weight) or not 0 < self.temporal_weight <= 1:
                    raise ValueError("Trọng số mô hình chuỗi không hợp lệ.")
                head = TemporalHead().to(self.device)
                head.load_state_dict(checkpoint["temporal_state_dict"])
                self.temporal = CausalScorer(head, float(settings["offset"]), self.device)
                # Share the loaded backbone; no second image network or extra image pass.
                self.feature_model = torch.nn.Sequential(*list(self.model.children())[:-1]).eval()
            if preprocessing.get("mode") == "letterbox":
                from PIL import Image, ImageOps
                size = int(preprocessing.get("size", 160))
                pad = tuple(preprocessing.get("pad_rgb", [124, 116, 104]))
                if not 96 <= size <= 512 or len(pad) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in pad):
                    raise ValueError("Thông số ảnh của checkpoint không hợp lệ.")
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda image: ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR,
                                                                 color=pad, centering=(.5, .5))),
                    transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
                ])
            elif preprocessing.get("mode") in (None, "center_crop"):
                self.transform = transforms.Compose([
                    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
                ])
            else:
                raise ValueError("Không hỗ trợ tiền xử lý ảnh của checkpoint.")
            self.seq_len = 1
        else:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            import tensorflow as tf
            self.tf = tf
            try:
                tf.config.threading.set_intra_op_parallelism_threads(int(cfg["threads"]))
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except RuntimeError:
                LOG.info("TensorFlow đã khởi tạo; giữ cấu hình luồng hiện tại.")
            try:
                self.model = tf.keras.models.load_model(str(model_path), compile=False)
            except (TypeError, ValueError):
                # Keras 3 -> Keras 2: sửa bản config cục bộ, không monkey-patch lớp toàn cục.
                import h5py
                with h5py.File(model_path, "r") as fh:
                    config = json.loads(fh.attrs["model_config"])
                major = int(getattr(tf.keras, "__version__", "2").split(".")[0])
                def compatible(node):
                    if isinstance(node, dict):
                        if isinstance(node.get("config"), dict):
                            c = node["config"]
                            c.pop("optional", None)
                            c.pop("quantization_config", None)
                            if node.get("class_name") == "DepthwiseConv2D":
                                c.pop("groups", None)
                            if node.get("class_name") == "InputLayer" and major < 3 and "batch_shape" in c:
                                c["batch_input_shape"] = c.pop("batch_shape")
                        for value in node.values():
                            compatible(value)
                    elif isinstance(node, list):
                        for value in node:
                            compatible(value)
                compatible(config)
                self.model = tf.keras.models.model_from_json(json.dumps(config))
                self.model.load_weights(str(model_path))
            shape = self.model.input_shape
            if isinstance(shape, list):
                raise ValueError("Chưa hỗ trợ mô hình có nhiều tensor đầu vào.")
            self.image_shape = shape[1:3] if len(shape) == 4 else (cfg["input_size"], cfg["input_size"])
            self.image_shape = tuple(int(x or cfg["input_size"]) for x in self.image_shape)
            self.seq_len = int(shape[1] or cfg["sequence_length"]) if len(shape) == 3 else 1
            if len(shape) == 3:
                weights = resolve_path(cfg.get("feature_weights", ""))
                if not cfg.get("feature_weights") or not weights.is_file():
                    raise ValueError("Mô hình LSTM cần model.feature_weights (MobileNetV2 ImageNet cục bộ).")
                self.extractor = tf.keras.applications.MobileNetV2(include_top=False, weights=None,
                                     input_shape=(*self.image_shape, 3), pooling="avg")
                self.extractor.load_weights(str(weights))
            elif len(shape) != 4:
                raise ValueError(f"Không hỗ trợ hình dạng đầu vào: {shape}")
        self.video_context = None
        if cfg.get('video_context_enabled', False):
            from action_context import sha256 as action_sha256
            backbone_path = resolve_path(cfg['video_context_backbone'])
            head_path = resolve_path(cfg['video_context_head'])
            if cfg.get('video_context_kind', 'r3d') == 'vit_cls_research':
                from vit_runtime import CLSContext, FrozenCLSHead
                head = FrozenCLSHead(head_path, action_sha256(backbone_path), allow_research=True)
                context = CLSContext(backbone_path, max_streams=32)
            else:
                from action_context import CausalActionContext
                from action_head import FrozenActionHead
                head = FrozenActionHead(head_path, action_sha256(backbone_path))
                context = CausalActionContext(weights=backbone_path, threads=cfg['threads'], score_interval=.5,
                                              sample_fps=head.sample_fps, min_frames=head.min_frames,
                                              min_span=head.min_span)
            head.assert_sampling(context)
            self.video_context = (context, head)
            self.identity += '+action:'+hashlib.sha256(head_path.read_bytes()).hexdigest()[:12]
        LOG.info("Mô hình %s sẵn sàng (%s)", self.identity, backend)

    def score(self, frame, state: VisualState) -> tuple[Optional[float], Optional[float]]:
        if frame.size == 0:
            return None, None
        with self.lock:
            if self.backend == "resnet":
                from PIL import Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
                with self.torch.inference_mode():
                    if self.temporal is not None:
                        feature = self.feature_model(tensor).flatten(1)
                        raw, smooth = self.temporal.score(feature[0], state, state.frame_time)
                        image = float(self.torch.softmax(self.model.fc(feature), dim=1)[0, 1].cpu())
                        state.image_smooth = image if state.image_smooth is None else .4*image+.6*state.image_smooth
                        if raw is None:
                            return None, None
                        weight = self.temporal_weight
                        return weight*raw+(1-weight)*image, weight*smooth+(1-weight)*state.image_smooth
                    raw = float(self.torch.softmax(self.model(tensor), dim=1)[0, 1].cpu())
            else:
                h, w = frame.shape[:2]
                size = min(h, w)
                crop = frame[(h-size)//2:(h+size)//2, (w-size)//2:(w+size)//2]
                rgb = cv2.cvtColor(cv2.resize(crop, self.image_shape[::-1]), cv2.COLOR_BGR2RGB)
                tensor = rgb.astype(np.float32)[None] / 255.0
                if self.extractor is not None:
                    state.features.append(self.extractor(tensor, training=False).numpy()[0])
                    while len(state.features) > self.seq_len:
                        state.features.popleft()
                    if len(state.features) < self.seq_len:
                        return None, None
                    tensor = np.asarray(state.features, dtype=np.float32)[None]
                prediction = np.asarray(self.model(tensor, training=False)).reshape(-1)
                raw = float(prediction[1] if len(prediction) == 2 else prediction[0])
            if not math.isfinite(raw) or not 0 <= raw <= 1:
                raise ValueError("Mô hình trả điểm không hợp lệ; không thể cảnh báo.")
            state.smooth = raw if state.smooth is None else .4 * raw + .6 * state.smooth
            return raw, state.smooth


class PeopleEngine:
    def __init__(self, cfg: dict):
        self.cfg, self.lock = cfg, threading.Lock()
        self.model, self.reason = None, "Nhận diện người đã tắt"
        if not cfg["enabled"]:
            return
        target = resolve_path(cfg["path"])
        fallback_pose, fallback_note = False, ""
        if not target.is_file() and target.name == "yolov8s-pose.pt" and (ROOT/"models/yolov8n-pose.pt").is_file():
            target = ROOT/"models/yolov8n-pose.pt"
            fallback_note = "Dùng pose nano dự phòng; tải pose nhỏ để tăng độ chi tiết"
        if not target.is_file() and target.name == "yolov8n-pose.pt" and (ROOT/"yolov8n.pt").is_file():
            target = ROOT/"yolov8n.pt"
            fallback_pose = True
        if not target.is_file():
            self.reason = f"Thiếu {target.name}; nhận diện người chưa khả dụng"
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(target))
            self.reason = "Chưa có pose; dùng hộp người, thiếu tín hiệu tay/chân" if fallback_pose else fallback_note
        except Exception as exc:
            self.reason = "Không nạp được nhận diện người: " + redact(exc)
            LOG.warning(self.reason)

    def parse_results(self, results, shape, reflected=False):
        height, width = shape[:2]
        confidence = self.cfg["confidence"]
        boxes = []
        swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        for result in results:
            points = result.keypoints.data.cpu().numpy() if getattr(result, "keypoints", None) is not None else None
            for index, box in enumerate(result.boxes):
                xy = list(map(float, box.xyxy[0].tolist()))
                score = float(box.conf[0])
                if len(xy) != 4 or not all(math.isfinite(v) for v in (*xy, score)) or not 0 <= score <= 1:
                    continue
                if reflected:
                    xy[0], xy[2] = width-xy[2], width-xy[0]
                xy = [max(0., min(float(width), xy[0])), max(0., min(float(height), xy[1])),
                      max(0., min(float(width), xy[2])), max(0., min(float(height), xy[3]))]
                if xy[2] <= xy[0] or xy[3] <= xy[1]:
                    continue
                pose = points[index].copy() if points is not None and index < len(points) else None
                if pose is not None:
                    if pose.ndim != 2 or pose.shape[1] < 3:
                        pose = None
                    else:
                        pose = pose[:, :3].copy()
                        if reflected:
                            if len(pose) != 17:
                                continue
                            pose = pose[swap].copy()
                            pose[:, 0] = width-pose[:, 0]
                        valid = (np.isfinite(pose).all(axis=1) & (pose[:, 0] >= 0) & (pose[:, 0] <= width)
                                 & (pose[:, 1] >= 0) & (pose[:, 1] <= height) & (pose[:, 2] >= 0) & (pose[:, 2] <= 1))
                        pose[~valid] = 0
                if score < confidence:
                    visible = [p[:2] for p in pose[5:17] if p[2] >= .35] if pose is not None else []
                    if len(visible) < 3:
                        continue
                    span = float(np.ptp(np.asarray(visible), axis=0).max())
                    if span < max(8, .12*(xy[3]-xy[1])):
                        continue
                boxes.append(PersonBox(*xy, confidence=score, keypoints=pose))
        return boxes


    def extra_body_supported(self, box):
        if box.confidence < max(.35, self.cfg["confidence"]) or box.keypoints is None:
            return False
        visible = [p[:2] for p in box.keypoints[5:17] if p[2] >= .45]
        return (len(visible) >= 4 and float(np.ptp(np.asarray(visible), axis=0).max())
                >= max(12, .20*box.height))


    @staticmethod
    def additional_observation(box, existing):
        for old in existing:
            inter = max(0., min(box.x2,old.x2)-max(box.x1,old.x1))*max(0.,min(box.y2,old.y2)-max(box.y1,old.y1))
            union = box.area+old.area-inter
            iou, iom = inter/max(1e-9,union), inter/max(1e-9,min(box.area,old.area))
            if iou >= .70 or ((iou >= .40 or iom >= .70) and not _distinct_body_poses(box, old)):
                return False
        return True


    def detect(self, frame, tracker):
        if self.model is None:
            return []
        body_occlusion = self.cfg.get("body_occlusion", True)
        confidence = self.cfg["confidence"]
        minimum = min(confidence, self.cfg.get("min_body_confidence", .10)) if body_occlusion else confidence
        enhanced = self.cfg.get("enhanced", False)
        # Cadence belongs to this camera's tracker, never the shared model.
        tick = getattr(tracker, "person_enhancement_tick", 0)
        tracker.person_enhancement_tick = tick+1
        with self.lock:
            warmed_threads = getattr(self, "_warm_start_threads", None)
            if warmed_threads is not None:
                import torch
                torch.set_num_threads(warmed_threads)
                self._warm_start_threads = None
            args = dict(classes=[0], conf=minimum, imgsz=int(self.cfg["imgsz"]), device="cpu", verbose=False)
            boxes = _nms_boxes(self.parse_results(self.model(frame, **args), frame.shape))
            if enhanced and tick % 3 == 0:
                extra = self.parse_results(self.model(cv2.flip(frame, 1), **args), frame.shape, reflected=True)
                for box in sorted(extra, key=lambda b:b.confidence, reverse=True):
                    if self.extra_body_supported(box) and self.additional_observation(box, boxes):
                        boxes.append(box)
        # Two detector views of one image are still one observation in time.
        return tracker.update(_nms_boxes(boxes))


class LoadingCancelled(RuntimeError):
    pass


@dataclass
class EngineBundle:
    visual: Any
    people: Any
    key: Any
    reusable: bool
    stages: dict
    warm_threads: Optional[int] = None

    def activate(self):
        # Trước tối ưu: ResNet đặt threads, rồi YOLO đặt chính sách CPU ở
        # lần detect đầu. Không mang thay đổi luồng của warm-up sang phiên.
        if self.warm_threads is not None and hasattr(self.visual, "torch"):
            self.visual.torch.set_num_threads(int(self.visual.cfg["threads"]))
            self.people._warm_start_threads = self.warm_threads

    def status(self):
        missing = []
        if self.people.cfg["enabled"] and self.people.model is None:
            missing.append("Nhận diện người chưa khả dụng: " + self.people.reason)
        return "; ".join(missing) or "AI sẵn sàng · chưa mở camera"


class ModelCache:
    """Một bộ trọng số thuộc ứng dụng; không giữ tracker/EMA/trạng thái camera."""
    def __init__(self):
        self._load_lock, self._state_lock = threading.Lock(), threading.Lock()
        self._bundle, self._generation, self._closed = None, 0, False
        self.loads, self.hits = 0, 0

    @staticmethod
    def key_for(cfg):
        def frozen(value):
            if isinstance(value, dict):
                return tuple((k, frozen(v)) for k, v in sorted(value.items()))
            if isinstance(value, (list, tuple)):
                return tuple(frozen(v) for v in value)
            return value
        paths = [resolve_path(cfg["model"][k]) for k in ("path", "keras_path", "feature_weights")
                 if cfg["model"].get(k)]
        if cfg['model'].get('video_context_enabled'):
            paths.extend(resolve_path(cfg['model'][k]) for k in ('video_context_backbone','video_context_head'))
        if cfg["person"]["enabled"]:
            paths += [resolve_path(cfg["person"]["path"]), ROOT/"models/yolov8n-pose.pt", ROOT/"yolov8n.pt"]
        signatures = []
        for path in sorted(set(p.resolve() for p in paths)):
            try:
                stat = path.stat()
                signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            except OSError:
                signature = None
            signatures.append((str(path), signature))
        return frozen({s: cfg[s] for s in ("model", "person")}), tuple(signatures)

    def clear(self):
        with self._state_lock:
            self._bundle = None
            self._generation += 1

    def close(self):
        with self._state_lock:
            self._closed = True
            self._bundle = None
            self._generation += 1

    def get(self, cfg, emit=None):
        snapshot = copy.deepcopy(cfg)
        with self._load_lock:
            key = self.key_for(snapshot)
            with self._state_lock:
                if self._closed:
                    raise LoadingCancelled("Ứng dụng đã yêu cầu đóng.")
                generation = self._generation
                if self._bundle is not None and self._bundle.key == key:
                    self.hits += 1
                    if emit:
                        emit("Dùng bộ AI đã nạp")
                    return self._bundle
                self._bundle = None
            stages = {}
            def stage(name, factory):
                with self._state_lock:
                    if self._closed or generation != self._generation:
                        raise LoadingCancelled("Đã hủy lần nạp mô hình cũ.")
                if emit:
                    emit(name)
                started = time.monotonic()
                result = factory()
                stages[name] = time.monotonic()-started
                return result
            visual = stage("Đang nạp AI hình ảnh", lambda: VisualEngine(snapshot["model"]))
            people = stage("Đang nạp AI người và tư thế", lambda: PeopleEngine(snapshot["person"]))
            reusable = not snapshot["person"]["enabled"] or people.model is not None
            warm_threads = None
            if reusable:
                def warm():
                    nonlocal warm_threads
                    runtime_imports()
                    image = np.zeros((360, 640, 3), dtype=np.uint8)
                    if visual.seq_len == 1:
                        visual.score(image, VisualState())
                    if people.model is not None:
                        people.detect(image, SimpleTracker())
                        if hasattr(visual, "torch"):
                            warm_threads = visual.torch.get_num_threads()
                            visual.torch.set_num_threads(int(visual.cfg["threads"]))
                stage("Đang chuẩn bị lần suy luận đầu", warm)
            if key != self.key_for(snapshot):
                raise RuntimeError("Tệp mô hình đã thay đổi trong lúc nạp; bấm Bắt đầu để nạp lại.")
            bundle = EngineBundle(visual, people, key, reusable, stages, warm_threads)
            with self._state_lock:
                if self._closed or generation != self._generation:
                    raise LoadingCancelled("Đã hủy lần nạp mô hình cũ.")
                self.loads += 1
                if reusable:
                    self._bundle = bundle
            return bundle


class MotionField:
    """Chuyển động tương đối, trừ vector nền để giảm ảnh hưởng rung camera."""
    def __init__(self):
        self.prev = self.mag = None

    def update(self, frame, boxes, dt: float):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(32, w//4), max(32, h//4)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        for box in boxes:
            box.motion_score = box.local_motion = box.body_speed = 0.0
        if self.prev is not None and self.prev.shape == gray.shape and .02 <= dt <= 1.0:
            flow = cv2.calcOpticalFlowFarneback(self.prev, gray, None, .5, 2, 11, 2, 5, 1.1, 0)
            background = np.ones(gray.shape, dtype=bool)
            for box in boxes:
                x1, x2 = max(0, int(box.x1*gray.shape[1]/w)), min(gray.shape[1], int(box.x2*gray.shape[1]/w)+1)
                y1, y2 = max(0, int(box.y1*gray.shape[0]/h)), min(gray.shape[0], int(box.y2*gray.shape[0]/h)+1)
                background[y1:y2, x1:x2] = False
            vectors = flow[background] if background.sum() > flow.shape[0]*flow.shape[1]*.1 else flow.reshape(-1, 2)
            flow -= np.median(vectors, axis=0)
            # Quy về khoảng cách giữa 2 frame ở 12.5 Hz, tương đương cấu hình cũ.
            self.mag = np.linalg.norm(flow, axis=2) * (.08 / max(.02, dt))
            mh, mw = self.mag.shape
            for box in boxes:
                x1, x2 = max(0, int(box.x1*mw/w)), min(mw, int(box.x2*mw/w)+1)
                y1, y2 = max(0, int(box.y1*mh/h)), min(mh, int(box.y2*mh/h)+1)
                region = self.mag[y1:y2, x1:x2]
                box.motion_score = float(np.percentile(region, 75)) if region.size else 0
                region_flow = flow[y1:y2, x1:x2]
                if region_flow.size:
                    scale = max(8, box.height*mh/h)*dt
                    magnitude = np.linalg.norm(region_flow, axis=2)
                    moving = magnitude > .04*scale
                    vectors = region_flow[moving]
                    # Bỏ pixel nền tĩnh trong hộp trước khi ước lượng dịch chuyển
                    # toàn thân; tránh xem người đi ngang nền tĩnh như giằng co.
                    translation = np.median(vectors, axis=0) if len(vectors) >= 6 else np.zeros(2)
                    residual = np.linalg.norm(vectors-translation, axis=1) if len(vectors) >= 6 else np.zeros(1)
                    box.local_motion = float(np.percentile(residual, 80)/scale)
                    box.body_speed = float(np.linalg.norm(translation)/scale)
        else:
            self.mag = None
        self.prev = gray

    def heatmap(self, shape):
        if self.mag is None:
            return None
        gray = np.clip(self.mag * 35, 0, 255).astype(np.uint8)
        return cv2.resize(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), (shape[1], shape[0]))


def parse_source(value):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    if text.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return text
    return str(resolve_path(text))


class Capture:
    """Một chủ sở hữu VideoCapture; stream chỉ giữ frame mới nhất, file phát theo FPS."""
    def __init__(self, source, stop: threading.Event, frame_hook=None):
        self.source, self.stop = parse_source(source), stop
        self.frame_hook = frame_hook
        self.pause = threading.Event()
        self.stream = isinstance(self.source, int) or "://" in str(self.source)
        self.lock = threading.Lock()
        self.frame, self.seq, self.stamp, self.position = None, 0, 0.0, 0.0
        self.status, self.eof, self.fps = "Đang kết nối", False, 25.0
        self.thread = threading.Thread(target=self.run, daemon=True, name="capture")
        self.thread.start()

    def run(self):
        cap, backoff = None, 1.0
        try:
            while not self.stop.is_set():
                if self.pause.is_set() and not self.stream:
                    self.stop.wait(.05)
                    continue
                if cap is None:
                    if isinstance(self.source, int):
                        cap = cv2.VideoCapture(self.source)
                    elif self.stream:
                        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG,
                              [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
                    else:
                        cap = cv2.VideoCapture(self.source)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        self.status = "Không mở được nguồn" if not self.stream else "Mất kết nối · đang thử lại"
                        if not self.stream:
                            break
                        self.stop.wait(backoff)
                        backoff = min(15, backoff*2)
                        continue
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    self.fps = fps if math.isfinite(fps) and 1 <= fps <= 120 else 25.0
                    backoff = 1.0
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if not self.stream:
                        # Preserve the last genuine file frame until analysis
                        # and pending causal-video inference have drained.
                        self.status = "Đã phát hết video"
                        break
                    with self.lock:
                        self.frame = None
                    self.status = "Mất tín hiệu · đang kết nối lại"
                    cap.release()
                    cap = None
                    self.stop.wait(backoff)
                    continue
                with self.lock:
                    self.frame, self.seq, self.stamp = frame, self.seq+1, time.monotonic()
                    self.position = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if not self.stream else 0
                if self.frame_hook:
                    self.frame_hook(self.stamp,frame)
                self.status = "Đang giám sát"
                if not self.stream:
                    self.stop.wait(max(0, 1/self.fps-(time.monotonic()-started)))
        except Exception as exc:
            self.status = "Lỗi nguồn: " + redact(exc)
            LOG.error(self.status)
        finally:
            if cap is not None:
                cap.release()
            self.eof = True

    def latest(self, copy_frame=True):
        with self.lock:
            snapshot = self.seq, self.stamp, self.position, self.frame
        # Capture công bố một ndarray mới mỗi lần đọc; không sửa frame cũ.
        # Sao chép ngoài khóa để GUI không chặn luồng thu hình.
        seq, stamp, position, frame = snapshot
        return seq, stamp, position, frame.copy() if copy_frame and frame is not None else frame


def privacy_frame(frame, boxes):
    result = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.int_box
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y1 + max(1, int((y2-y1)*.32)))
        if x2 > x1 and y2 > y1:
            region = result[y1:y2, x1:x2]
            tiny = cv2.resize(region, (max(1, (x2-x1)//18), max(1, (y2-y1)//18)))
            result[y1:y2, x1:x2] = cv2.resize(tiny, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
    return result


def save_jpeg(frame, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise OSError("Không mã hóa được ảnh JPEG.")
    target.write_bytes(data.tobytes())  # Hỗ trợ đường dẫn Unicode trên Windows.


class EvidenceRecorder:
    def __init__(self, cfg: dict, folder: Path, store: EventStore, camera: str):
        self.cfg, self.folder, self.store, self.camera = cfg, folder, store, camera
        self.buffer, self.bytes = deque(), 0
        self.writer = None
        self.event_id, self.last_buffer = None, -math.inf
        self.start = self.until = self.written = 0
        self.writer_size = None

    def disk_ok(self):
        return shutil.disk_usage(self.folder).free >= self.cfg["min_free_mb"] * 1024**2

    def feed(self, stamp: float, frame):
        if not self.cfg["enabled"] or stamp-self.last_buffer < 1/self.cfg["fps"]:
            return
        self.last_buffer = stamp
        h, w = frame.shape[:2]
        scale = min(1, self.cfg["width"]/w)
        frame = cv2.resize(frame, (max(2, int(w*scale)//2*2), max(2, int(h*scale)//2*2)))
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        raw = data.tobytes()
        self.buffer.append((stamp, raw))
        self.bytes += len(raw)
        while self.buffer and (stamp-self.buffer[0][0] > self.cfg["pre_seconds"] or self.bytes > self.cfg["buffer_mb"]*1024**2):
            self.bytes -= len(self.buffer.popleft()[1])
        if self.writer is not None:
            if not self.disk_ok():
                self.close("Dừng clip: dung lượng trống thấp")
                return
            self._write(stamp, frame)
            if stamp >= self.until:
                self.close("Đã lưu")

    def _write(self, stamp, frame):
        if frame.shape[1::-1] != self.writer_size:
            frame = cv2.resize(frame, self.writer_size)
        target_count = int(max(0, stamp-self.start)*self.cfg["fps"])+1
        # Resample thời gian thực: không làm clip chạy nhanh khi suy luận chậm.
        while self.written < target_count:
            self.writer.write(frame)
            self.written += 1

    def trigger(self, eid: str, stamp: float):
        if not self.cfg["enabled"]:
            return
        if self.writer is not None:
            self.store.update(eid, clip_state="Sự kiện đã có clip đang ghi")
            return
        if not self.buffer or not self.disk_ok():
            self.store.update(eid, clip_state="Không đủ bộ đệm hoặc dung lượng trống")
            return
        target = self.folder / "clips" / f"{safe_name(self.camera)}_{eid}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imdecode(np.frombuffer(self.buffer[0][1], np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), self.cfg["fps"], (w, h))
        if not writer.isOpened():
            writer.release()
            self.store.update(eid, clip_state="Không mở được bộ ghi MP4")
            return
        self.writer, self.event_id = writer, eid
        self.writer_size = (w, h)
        self.start, self.until, self.written = self.buffer[0][0], stamp+self.cfg["post_seconds"], 0
        self.store.update(eid, clip=str(target), clip_state="Đang ghi")
        for t, raw in self.buffer:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            self._write(t, cv2.resize(img, (w, h)))

    def close(self, status="Đã lưu · kết thúc nguồn"):
        if self.writer is not None:
            self.writer.release()
            self.store.update(self.event_id, clip_state=status)
            self.writer, self.event_id = None, None


class NotificationQueue:
    """Một worker/session cho mỗi kênh; camera chỉ xếp hàng, không chờ mạng/encode."""
    def __init__(self, cfg: dict, offline=False, feedback=None, store=None):
        self.feedback = feedback
        self.store = store
        self.cfg, self.enabled = cfg, cfg["enabled"] and not offline
        self.stop, self.queues, self.threads = threading.Event(), {}, []
        self.thread, self.items = None, None
        self.metrics_lock = threading.Lock()
        self.metrics = {}
        if self.enabled:
            credentials = notification_credentials(cfg)
            endpoints = {}
            webhook = credentials["discord_webhook"]
            if webhook.startswith(("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")):
                endpoints["Discord"] = (webhook, "")
            if credentials["telegram_token"] and credentials["telegram_chat_id"]:
                endpoints["Telegram"] = ("https://api.telegram.org/bot"+credentials["telegram_token"], credentials["telegram_chat_id"])
            for channel, endpoint in endpoints.items():
                items = queue.Queue(maxsize=16)
                self.queues[channel] = items
                self.metrics[channel] = {"delivered": 0, "failed": 0, "dropped": 0, "max_queue_delay_ms": 0.0, "last_delivery_ms": None}
                worker = threading.Thread(target=self.run, args=(channel, endpoint, items), daemon=True, name="notify-"+channel)
                self.threads.append(worker)
                worker.start()
            self.thread = self.threads[0] if self.threads else None
            self.enabled = bool(self.threads)

    def send(self, event: dict, frame):
        if not self.enabled or event.get('reminder') or event.get('accepted'):
            return
        event = dict(event)
        if self.feedback is not None and not event.get('no_buttons'):
            event["review_links"] = self.feedback.links(event["id"])
        payload = (event, frame.copy() if frame is not None else None, time.monotonic())
        for channel, items in self.queues.items():
            if event.get('target_channels') is not None and channel not in event['target_channels']:continue
            try:
                items.put_nowait(payload)
            except queue.Full:
                # Sự kiện cũ vẫn ở SQLite; ưu tiên cảnh báo mới thay vì chờ backlog dài.
                try:
                    items.get_nowait()
                    items.task_done()
                except queue.Empty:
                    pass
                with self.metrics_lock:
                    self.metrics[channel]["dropped"] += 1
                try:
                    items.put_nowait(payload)
                except queue.Full:
                    pass
                LOG.warning("%s: hàng đợi đầy; bỏ thông báo cũ, sự kiện vẫn lưu cục bộ.", channel)

    def snapshot_stats(self):
        with self.metrics_lock:
            return copy.deepcopy(self.metrics)

    def run(self, channel, endpoint, items):
        import requests
        session = requests.Session()
        try:
            while not self.stop.is_set() or not items.empty():
                try:
                    event, frame, enqueued = items.get(timeout=.1)
                except queue.Empty:
                    continue
                # Ignore legacy incident jobs that were already queued before upgrade.
                if event.get('reminder') or event.get('accepted'):
                    items.task_done()
                    continue
                delay = 1000*(time.monotonic()-enqueued)
                image = None
                if frame is not None:
                    try:
                        h, w = frame.shape[:2]
                        if w > 640:
                            frame = cv2.resize(frame, (640, max(1, round(h*640/w))))
                        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        image = encoded.tobytes() if ok else None
                    except Exception:
                        LOG.warning("%s: chưa encode được ảnh; gửi nội dung cảnh báo.", channel)
                score = f"{event['score']:.1%}" if event["score"] is not None else "không áp dụng"
                text = (f"SchoolGuard: {event.get('notice', ALERT_TEXT)}\nLoại sự kiện: {event['kind']}\nCamera: {event['camera']}\n"
                        f"Điểm AI: {score}\n{event['created']}\nMã: {event['id']}\n{event['reason']}\n{HUMAN_GUIDANCE}")
                video = None
                media_path=event.get('quick_clip')
                if media_path:
                    try:
                        target=Path(media_path)
                        if target.stat().st_size<=45*1024**2:
                            video=target.read_bytes()
                    except OSError:
                        LOG.warning('Clip ngắn chưa đọc được; giữ cảnh báo chữ/ảnh.')
                text = bounded_notification_text(text, 950 if image or video else 1900 if channel == "Discord" else 4000)
                if channel == "Discord":
                    url = endpoint[0]
                    payload = notification_payload("Discord", text, event.get("review_links"))
                    if image or video:
                        # Multipart cần payload_json để Discord đọc các trường object.
                        kwargs = {"data": {"payload_json": json.dumps(payload, ensure_ascii=False)},
                                  "files": {"files[0]": ("context.mp4", video, "video/mp4") if video else ("event.jpg", image, "image/jpeg")}}
                    else:
                        kwargs = {"json": payload}
                    kwargs['params']={'wait':'true'}
                    if payload.get("components"):
                        kwargs["params"]["with_components"] = "true"
                else:
                    has_media=bool(image or video)
                    url = endpoint[0]+("/sendVideo" if video else "/sendPhoto" if image else "/sendMessage")
                    kwargs = {"data": {"chat_id": endpoint[1], "caption" if has_media else "text": text[:950] if has_media else text[:4000]},
                              "files": {"video": ("context.mp4", video, "video/mp4")} if video else {"photo": ("event.jpg", image, "image/jpeg")} if image else None}
                    if video:kwargs['data']['supports_streaming']='true'
                    keyboard = notification_payload("Telegram", text, event.get("review_links"))
                    if keyboard:
                        kwargs["data"]["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
                delivered = False
                for attempt in range(3):
                    try:
                        response = session.post(url, timeout=(2, 4), **kwargs)
                        if response.status_code == 429:
                            body = response.json()
                            retry = body.get("retry_after", body.get("parameters", {}).get("retry_after", 1))
                            if self.stop.wait(max(.1, float(retry))):
                                break
                            continue
                        if response.status_code in (400, 401, 403, 404):
                            LOG.warning("%s từ chối cảnh báo (%s); kiểm tra cấu hình/quyền.", channel, response.status_code)
                            break
                        response.raise_for_status()
                        if channel == "Telegram" and not response.json().get("ok"):
                            LOG.warning("Telegram chưa chấp nhận cảnh báo.")
                            break
                        delivered = True
                        if self.store:
                            try:
                                body=response.json()
                                message_id=body.get('id') if channel=='Discord' else body.get('result',{}).get('message_id')
                                if message_id is not None:
                                    destination=hashlib.sha256(repr(endpoint).encode()).hexdigest()
                                    self.store.save_receipt(event['id'],channel,message_id,bool(image or video),destination)
                            except ValueError:pass
                        LOG.info("Đã gửi %s cho sự kiện %s", channel, event["id"])
                        break
                    except (requests.RequestException, ValueError, TypeError):
                        LOG.warning("Gửi %s thất bại (%d/3)", channel, attempt+1)
                        if self.stop.wait(.25*2**attempt):
                            break
                with self.metrics_lock:
                    stats = self.metrics[channel]
                    stats["delivered" if delivered else "failed"] += 1
                    stats["max_queue_delay_ms"] = round(max(stats["max_queue_delay_ms"], delay), 1)
                    stats["last_delivery_ms"] = round(1000*(time.monotonic()-enqueued), 1) if delivered else None
                items.task_done()
        finally:
            session.close()

    def close(self):
        self.stop.set()
        for worker in self.threads:
            worker.join()


def beep():
    try:
        import winsound
        winsound.Beep(1100, 180)
    except (ImportError, RuntimeError):
        pass


class PreviewBoxTracker:
    """Track display copies at video rate; never feed them back into detection.

    Actual detector observations refresh the 1.5 s lifetime. Optical flow only
    moves the outline, so an indefinitely moving background cannot keep it alive.
    """
    def __init__(self, max_age=1.5):
        self.max_age = max_age
        self.reset()

    def reset(self, epoch=None):
        self.epoch = epoch
        self.boxes, self.seen = {}, {}
        self.gray = self.shape = None
        self.seq, self.stamp, self.analysis_seq = -1, 0., -1
        self.cut_stamp = 0.
        self.scene = SceneCutDetector()

    @staticmethod
    def gray_frame(frame):
        h, w = frame.shape[:2]
        scale = min(1., 416 / w, 256 / h)
        small = cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale)))) if scale < 1 else frame
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def move_box(box, before, after, shape):
        """Robust translation from body texture, checked by backward flow."""
        if before is None or before.shape != after.shape:
            return False
        h, w = shape[:2]
        sy, sx = before.shape[0]/h, before.shape[1]/w
        mask = np.zeros(before.shape, np.uint8)
        x1, y1 = max(0, round(box.x1*sx)+2), max(0, round(box.y1*sy)+2)
        x2, y2 = min(before.shape[1], round(box.x2*sx)-2), min(before.shape[0], round(box.y2*sy)-2)
        if x2 <= x1 or y2 <= y1:
            return False
        mask[y1:y2, x1:x2] = 255
        points = cv2.goodFeaturesToTrack(before, maxCorners=30, qualityLevel=.02,
                                         minDistance=4, mask=mask, blockSize=5)
        if points is None or len(points) < 4:
            return False
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, .03))
        moved, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, **options)
        if moved is None:
            return False
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(after, before, moved, None, **options)
        if backward is None:
            return False
        old, new = points.reshape(-1, 2), moved.reshape(-1, 2)
        good = (status.ravel().astype(bool) & back_status.ravel().astype(bool)
                & np.isfinite(new).all(axis=1)
                & (np.linalg.norm(backward.reshape(-1, 2)-old, axis=1) < 1.5))
        if np.count_nonzero(good) < 4:
            return False
        shifts = new[good]-old[good]
        translation = np.median(shifts, axis=0)
        inliers = np.linalg.norm(shifts-translation, axis=1) <= 3.
        if np.count_nonzero(inliers) < 4 or inliers.mean() < .5:
            return False
        dx, dy = np.median(shifts[inliers], axis=0)
        if math.hypot(dx, dy) > max(20., .75*math.hypot(box.width*sx, box.height*sy)):
            return False
        box.x1 += float(dx)/sx; box.x2 += float(dx)/sx
        box.y1 += float(dy)/sy; box.y2 += float(dy)/sy
        return True

    @staticmethod
    def display_copy(box):
        result = copy.copy(box)
        result.observed = result.is_fighting = result.is_attacker = result.is_victim = False
        result.fight_partner_id = result.keypoints = None
        result.motion_score = result.local_motion = result.body_speed = 0.
        result.display_tracked = True
        return result

    def project(self, frame, seq, stamp, analyzed, epoch, now, roi=None):
        shape = frame.shape
        if self.epoch != epoch or self.shape != shape or seq < self.seq or stamp < self.stamp:
            self.reset(epoch)
        self.shape = shape
        gray = self.gray_frame(frame) if (seq, stamp) != (self.seq, self.stamp) else self.gray
        if (seq, stamp) != (self.seq, self.stamp):
            if self.scene.check_cut(frame):
                self.boxes.clear(); self.seen.clear()
                self.cut_stamp = stamp
            else:
                for box in self.boxes.values():
                    self.move_box(box, self.gray, gray, shape)
            self.gray, self.seq, self.stamp = gray, seq, stamp
        if analyzed.analysis_seq != self.analysis_seq:
            self.analysis_seq = analyzed.analysis_seq
            if self.cut_stamp <= analyzed.source_stamp <= stamp and 0 <= now-analyzed.source_stamp <= self.max_age:
                anchor_gray = self.gray_frame(analyzed.frame)
                for box in analyzed.boxes:
                    if not box.observed:
                        continue  # Coasting detector tracks do not renew display lifetime.
                    projected = self.display_copy(box)
                    old = self.boxes.get(box.track_id)
                    if (seq, stamp) != (analyzed.source_seq, analyzed.source_stamp):
                        followed = self.move_box(projected, anchor_gray, gray, shape)
                        if not followed and old is not None:
                            projected.x1, projected.y1, projected.x2, projected.y2 = old.x1, old.y1, old.x2, old.y2
                    if old is not None:
                        blend = .95 if math.dist(old.center, projected.center) > .2*max(30, old.height) else .65
                        for key in ('x1', 'y1', 'x2', 'y2'):
                            setattr(projected, key, getattr(old, key)*(1-blend)+getattr(projected, key)*blend)
                    self.boxes[box.track_id] = projected
                    self.seen[box.track_id] = analyzed.source_stamp
        h, w = shape[:2]
        for tid, box in list(self.boxes.items()):
            box.x1, box.x2 = max(0., box.x1), min(float(w), box.x2)
            box.y1, box.y2 = max(0., box.y1), min(float(h), box.y2)
            cx, cy = box.center
            if (not 0 <= now-self.seen[tid] <= self.max_age or box.width < 4 or box.height < 4
                    or roi and not (roi[0]*w <= cx < roi[2]*w and roi[1]*h <= cy < roi[3]*h)):
                del self.boxes[tid]; del self.seen[tid]
        # Detector NMS already distinguished the people. Repeating NMS without
        # their current poses would hide one of two overlapping tracked bodies.
        return [copy.copy(box) for box in self.boxes.values()]


@dataclass
class CameraView:
    frame: Any = None
    boxes: list = field(default_factory=list)
    heatmap: Any = None
    score: Optional[float] = None
    raw: Optional[float] = None
    status: str = "Đang khởi tạo"
    reason: str = ""
    fps: float = 0
    latency: float = 0
    position: float = 0
    source_seq: int = -1
    source_stamp: float = 0.0
    privacy_applied: bool = False
    annotation_epoch: int = 0
    analysis_seq: int = -1
    preview: bool = False
    annotations_valid: bool = True
    prediction_age: float = 0.0
    alert_active: bool = False
    alert_track_ids: Tuple[int, ...] = ()


class CameraWorker:
    def __init__(self, spec, cfg, visual, people, store, notifications, action_worker=None):
        self.spec, self.cfg, self.visual, self.people = copy.deepcopy(spec), cfg, visual, people
        self.store, self.notifications = store, notifications
        self.stop, self.pause = threading.Event(), threading.Event()
        self.lock, self.view = threading.Lock(), CameraView()
        self.commands = queue.Queue(maxsize=16)
        self.state, self.tracker = VisualState(), SimpleTracker()
        self.gate, self.motion, self.scene = AlertGate(), MotionField(), SceneCutDetector()
        self.interaction = InteractionVerifier()
        self.scenarios, self.health = ScenarioMonitor(), CameraHealth()
        self.quick = None
        self.recorder = EvidenceRecorder(cfg["evidence"], store.folder, store, spec["label"])
        self.thread = threading.Thread(target=self.run, daemon=True, name="analysis-"+spec["label"])
        self.capture = None
        self.failure = ""
        self.active_recording = None
        self.processed, self.peak, self.total_latency, self.skipped = 0, None, 0.0, 0
        self.preview_epoch = 0
        self.preview_tracker, self.preview_lock = PreviewBoxTracker(), threading.Lock()
        self.completed_at = deque(maxlen=150)
        self.analysis_started_at = None
        self.action_monitor = None
        if action_worker is not None:
            from action_runtime import ActionMonitor
            self.action_monitor = ActionMonitor(action_worker, spec['label'], cooldown=cfg['detection']['cooldown_seconds'])
        self.action_scene, self.action_capture_lock = SceneCutDetector(), threading.Lock()
        self.action_frame_id, self.action_due, self.action_failure = 0, None, ''



    def snapshot_view(self, preview=False):
        with self.lock:
            view = copy.copy(self.view)
            epoch = self.preview_epoch
            completed = tuple(self.completed_at)
            started = self.analysis_started_at
            roi = self.spec.get("roi")
        now = time.monotonic()
        view.prediction_age = max(0, now-view.source_stamp) if view.source_stamp > 0 else 0
        if started is not None:
            view.fps = sum(t > now-5 for t in completed)/max(.5, min(5, now-started))
        if not preview:
            return view  # SOS/bằng chứng luôn lấy đúng khung đã phân tích.
        capture = self.capture
        if capture is None:
            with self.preview_lock:
                self.preview_tracker.reset()
            view.frame, view.boxes, view.heatmap = None, [], None
            view.alert_active, view.alert_track_ids, view.annotations_valid = False, (), False
            return view
        seq, stamp, position, frame = capture.latest(copy_frame=False)
        live = (frame is not None and not capture.eof and math.isfinite(stamp)
                and stamp > 0 and now >= stamp and (self.pause.is_set() or now-stamp <= 3))
        if not live:
            with self.preview_lock:
                self.preview_tracker.reset()
            view.frame, view.boxes, view.heatmap = None, [], None
            view.alert_active, view.alert_track_ids, view.annotations_valid = False, (), False
            view.status = capture.status
            view.score = view.raw = None
            return view
        ttl = min(.4, max(.08, 1.1/self.cfg["detection"]["analysis_fps"]))
        usable = (not self.pause.is_set() and not self.stop.is_set() and epoch == view.annotation_epoch
                 and view.frame is not None and view.frame.shape == frame.shape
                 and seq >= view.source_seq >= 0 and view.source_stamp > 0
                 and math.isfinite(view.source_stamp)
                 and 0 <= stamp-view.source_stamp and 0 <= now-view.source_stamp)
        valid = usable and stamp-view.source_stamp <= ttl and now-view.source_stamp <= ttl
        view.annotations_valid = valid
        if not valid:
            view.alert_active, view.alert_track_ids = False, ()
        analyzed_boxes = view.boxes
        protected = self.cfg["display"]["privacy"]
        can_track = usable and (not protected or view.privacy_applied)
        with self.preview_lock:
            if can_track:
                # Privacy mode tracks only the protected analyzed image, never raw preview.
                shown_boxes = self.preview_tracker.project(
                    view.frame if protected else frame,
                    view.source_seq if protected else seq,
                    view.source_stamp if protected else stamp,
                    view, epoch, now, roi)
            else:
                self.preview_tracker.reset()
                shown_boxes = []
        if self.cfg["display"]["privacy"]:
            # Raw preview không có mask của chính khung đó. Chỉ dùng ảnh
            # đã bảo vệ đúng epoch; off→on/ROI phải chờ ảnh mới.
            if (not view.privacy_applied or epoch != view.annotation_epoch or view.frame is None
                    or view.frame.shape != frame.shape):
                view.frame, view.boxes, view.heatmap = None, [], None
                view.alert_active, view.alert_track_ids, view.annotations_valid = False, (), False
                view.status = "Đang chờ khung đã che vùng đầu"
            else:
                view.boxes = shown_boxes
                if not valid:
                    view.heatmap = None
                else:
                    originals = {b.track_id: b for b in analyzed_boxes if b.observed}
                    for index, box in enumerate(view.boxes):
                        if box.track_id in originals:
                            exact = copy.deepcopy(originals[box.track_id])
                            exact.x1, exact.y1, exact.x2, exact.y2 = box.x1, box.y1, box.x2, box.y2
                            view.boxes[index] = exact
            if self.pause.is_set():
                view.status = "Tạm dừng phân tích"
            return self.guard_preview(view, epoch)
        same = seq == view.source_seq and stamp == view.source_stamp
        view.frame, view.position = frame, position
        view.source_seq, view.source_stamp = seq, stamp
        view.preview, view.annotations_valid = not same, valid
        originals = {b.track_id: b for b in view.boxes if b.observed} if valid and same else {}
        view.boxes = shown_boxes
        for index, box in enumerate(view.boxes):
            if box.track_id in originals:
                exact = copy.deepcopy(originals[box.track_id])
                exact.x1, exact.y1, exact.x2, exact.y2 = box.x1, box.y1, box.x2, box.y2
                view.boxes[index] = exact
        if not valid or not same:
            view.heatmap = None
        if self.pause.is_set():
            view.status = "Tạm dừng phân tích"
        if view.preview:
            view.reason = "Đang bám người theo video; chờ AI cập nhật. " + view.reason
        return self.guard_preview(view, epoch)

    def guard_preview(self, view, epoch):
        """A reset/pause during optical flow must invalidate the returned overlay."""
        with self.lock:
            changed = epoch != self.preview_epoch
            invalid = changed or self.pause.is_set() or self.stop.is_set()
        if invalid:
            view.boxes, view.heatmap = [], None
            view.alert_active, view.alert_track_ids, view.annotations_valid = False, (), False
            if changed and self.cfg["display"]["privacy"]:
                view.frame = None
        return view

    def publish_view(self, view, completed=False):
        """Công bố UI ngắn dưới khóa; không hồi sinh kết quả sau pause/reset."""
        with self.lock:
            if completed:
                finished = time.monotonic()
                self.completed_at.append(finished)
                view.fps = sum(t > finished-5 for t in self.completed_at)/max(.5, min(5, finished-self.analysis_started_at))
            if (view.annotation_epoch != self.preview_epoch or self.pause.is_set() or self.stop.is_set()):
                return False
            self.view = view
            return True

    def invalidate_preview(self):
        with self.lock:
            self.preview_epoch += 1

    def command(self, name):
        if name == "reset":
            self.invalidate_preview()
        try:
            self.commands.put_nowait(name)
        except queue.Full:
            LOG.warning("Quá nhiều thao tác camera; vui lòng chờ.")

    def roi(self):
        with self.lock:
            return self.spec.get("roi")

    def set_roi(self, roi):
        with self.lock:
            self.spec["roi"] = roi
        self.command("reset")

    def reset(self, reset_action=True):
        self.invalidate_preview()
        self.state, self.motion = VisualState(), MotionField()
        self.tracker.reset()
        self.interaction.reset()
        self.scenarios.reset()
        if self.quick:self.quick.reset()
        self.gate.reset()
        self.recorder.buffer.clear()
        self.recorder.bytes = 0
        self.recorder.close("Đã lưu · đổi cảnh/vùng giám sát")
        if reset_action and self.action_monitor is not None:
            with self.action_capture_lock:
                self.action_monitor.reset()
                self.action_scene = SceneCutDetector()
                self.action_due, self.action_failure = None, ''

    def feed_capture(self, stamp, frame):
        if self.quick:
            self.quick.feed(stamp, frame)
        if self.action_monitor is None or self.pause.is_set() or self.action_failure:
            return
        try:
            with self.action_capture_lock:
                self.action_frame_id += 1
                native_cuts = getattr(self.action_monitor.worker.extractor, 'scene_cadence', '') == 'native'
                if not native_cuts and self.action_due is not None and stamp+1e-8 < self.action_due:
                    return
                period = self.action_monitor.worker.sample_interval
                if not native_cuts:
                    self.action_due = stamp+period if self.action_due is None else self.action_due+period*(math.floor(max(0.,stamp-self.action_due)/period+1e-8)+1)
                with self.lock:
                    roi = self.spec.get('roi')
                h,w = frame.shape[:2]
                x1,y1,x2,y2 = (int(roi[0]*w),int(roi[1]*h),int(roi[2]*w),int(roi[3]*h)) if roi else (0,0,w,h)
                region = frame[y1:y2,x1:x2]
                self.action_monitor.feed(region, stamp, self.action_frame_id, bool(self.action_scene.check_cut(region)))
        except Exception as exc:
            self.action_failure = 'AI chuỗi hành động chưa khả dụng: '+redact(exc)
            LOG.error('%s | %s', self.spec['label'], self.action_failure)

    def poll_action(self, stamp):
        if self.action_monitor is None:
            return None
        evidence = self.action_monitor.poll(stamp)
        if self.action_monitor.error:
            self.action_failure = 'AI chuỗi hành động gặp lỗi: '+self.action_monitor.error
            LOG.error('%s | %s', self.spec['label'], self.action_failure)
        if evidence is not None:
            evidence['fire'] = bool(evidence['fire'] and stamp-self.gate.last_alert >= self.cfg['detection']['cooldown_seconds'])
            if evidence['fire']:
                self.gate.last_alert = stamp
        return evidence

    def event(self, frame, view, kind):
        eid = self.store.add(self.spec["label"], kind, view.score, view.raw, len(view.boxes),
                             view.reason, view.position, self.visual.identity)
        if kind != "Ảnh thủ công":
            # Dispatch ngay sau khi có mã sự kiện; không chờ ghi ảnh hoặc flush clip trước.
            self.notifications.send({"id": eid, "camera": self.spec["label"], "kind": kind,
                                     "score": view.score, "created": now_iso(), "reason": view.reason}, frame)
            if self.quick:self.quick.trigger(eid,view.source_stamp or time.monotonic())
        if self.cfg["evidence"].get("save_snapshots", True) and self.recorder.disk_ok():
            target = self.store.folder / "snapshots" / f"{safe_name(self.spec['label'])}_{eid}.jpg"
            try:
                save_jpeg(frame, target)
                self.store.update(eid, snapshot=str(target))
            except OSError as exc:
                LOG.warning("Không lưu được ảnh: %s", redact(exc))
        elif self.cfg["evidence"].get("save_snapshots", True):
            self.store.update(eid, notes="Dung lượng trống thấp; chưa lưu ảnh.")
        if kind != "Ảnh thủ công":
            self.recorder.trigger(eid, time.monotonic())
            if self.cfg["display"]["sound"]:
                threading.Thread(target=beep, daemon=True).start()
        LOG.warning("%s | %s | sự kiện %s", self.spec["label"], kind, eid)

    def quick_completed(self,eid,path,metadata):
        atomic_json(path.with_suffix('.json'),metadata)
        row=self.store.get(eid)
        if row:
            row['quick_clip']=str(path)
            row['notice']='🎞️ Clip ngữ cảnh 3 giây · cần người kiểm tra'
            self.notifications.send(row,None)

    def run(self):
        continuous_start, continuous_frames = 0, 0
        try:
            self.analysis_started_at = time.monotonic()
            if self.stop.is_set():
                return
            self.quick=QuickClip(self.store.folder,self.quick_completed,lambda:bool(self.cfg['display']['privacy']),
                                 self.cfg['quick_clip']['enabled'],self.cfg['evidence']['min_free_mb'])
            self.capture = Capture(self.spec["source"], self.stop, self.feed_capture)
            last_seq, prev_stamp, last_frame = -1, 0.0, None
            lost = False
            lost_since=None
            lost_alerted=False
            eof_action_deadline = None
            while not self.stop.is_set():
                started = time.monotonic()
                while not self.commands.empty():
                    command = self.commands.get_nowait()
                    if command == "reset":
                        self.reset()
                        last_frame = None
                    elif last_frame is not None:
                        self.event(last_frame, self.snapshot_view(), "SOS thủ công" if command == "sos" else "Ảnh thủ công")
                seq, stamp, position, frame = self.capture.latest()
                if frame is None or not self.pause.is_set() and not self.capture.eof and time.monotonic()-stamp > 3:
                    if lost_since is None:lost_since=time.monotonic()
                    if (self.cfg['school_policy']['camera_health'] and not lost_alerted
                            and time.monotonic()-lost_since>=self.cfg['school_policy']['health_seconds'] and not self.capture.eof):
                        lost_alerted=True
                        missing_view=CameraView(score=None,raw=None,reason='Không nhận được khung hình mới; kiểm tra kết nối và nguồn điện camera.')
                        self.event(np.zeros((240,320,3),np.uint8),missing_view,'⚠️ Camera mất tín hiệu · '+self.spec['label'])
                    if not lost:
                        self.reset()
                        lost = True
                    with self.lock:
                        self.view.status = self.capture.status
                        self.view.score = self.view.raw = None
                        self.view.frame = None
                    if self.capture.eof:
                        break
                    self.stop.wait(.1)
                    continue
                if seq == last_seq or self.pause.is_set():
                    if self.capture.eof:
                        if self.action_monitor is not None and last_frame is not None:
                            eof_action_deadline = eof_action_deadline or time.monotonic()+1.5
                            action = self.poll_action(time.monotonic())
                            if action is not None and action['fire']:
                                tail = self.snapshot_view()
                                tail.score,tail.status,tail.reason = action['score'],ALERT_TEXT,action_reason(action)
                                tail.source_stamp,tail.alert_active = action['second'],True
                                self.publish_view(tail)
                                self.event(last_frame,tail,ALERT_TEXT)
                            if time.monotonic() < eof_action_deadline:
                                self.stop.wait(.05)
                                continue
                        break
                    with self.lock:
                        if self.pause.is_set():
                            self.view.status = "Tạm dừng phân tích · nguồn tiếp tục phát"
                    self.stop.wait(.05)
                    continue
                if lost:
                    lost, prev_stamp = False, 0
                lost_since=None;lost_alerted=False
                if last_seq >= 0:
                    self.skipped += max(0, seq-last_seq-1)
                last_seq = seq
                with self.lock:
                    roi, epoch = self.spec.get("roi"), self.preview_epoch
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = (int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)) if roi else (0, 0, w, h)
                region = frame[y1:y2, x1:x2]
                if region.size == 0:
                    raise ValueError("Vùng giám sát quá nhỏ.")
                if self.scene.check_cut(region):
                    self.reset(reset_action=False)  # Capture owns video-context shot boundaries.
                    epoch += 1  # Các reset từ GUI vẫn khiến kết quả đang chạy hết hạn.
                self.state.frame_time = stamp
                raw, score = self.visual.score(region, self.state)
                self.processed += 1
                if raw is not None:
                    self.peak = max(self.peak or 0, raw)
                boxes = self.people.detect(frame, self.tracker)
                all_boxes = copy.deepcopy(boxes)
                self.motion.update(frame, boxes, stamp-prev_stamp if prev_stamp else .2)
                prev_stamp = stamp
                boxes = [b for b in boxes if x1 <= b.center[0] < x2 and y1 <= b.center[1] < y2]
                detection,policy_mode=detection_at(self.cfg['detection'],self.cfg['school_policy'])
                interaction = self.interaction.update(stamp, boxes, frame.shape, score, detection=detection)
                supported = interaction["supported"]
                reason = (f"AI: {score:.1%}; {interaction['observed_people']} người; " + interaction["reason"]) if score is not None else "Đang tích lũy chuỗi hình ảnh"
                if self.people.reason:
                    reason += "; " + self.people.reason
                status, fire = self.gate.update(stamp, score, supported, detection, interaction["confirmed_since"])
                if fire and self.action_monitor is not None:
                    self.action_monitor.last_alert = stamp
                action = self.poll_action(stamp)
                action_high = action is not None and action['score'] >= .5
                if action_high:
                    status = ALERT_TEXT
                    reason += '; '+action_reason(action)
                    if action['fire']:
                        fire = True
                        score = max(score or 0., action['score'])
                if self.action_failure:
                    reason += '; '+self.action_failure
                scenario=self.scenarios.update(stamp,boxes,frame.shape,self.cfg['school_policy'],policy_mode,
                    self.spec.get('hidden_zones',[]),self.gate.active and supported)
                reason += '; chế độ: '+MODE_LABELS[policy_mode]
                if self.gate.active:
                    status = ALERT_TEXT if supported else "Theo dõi sau cảnh báo · cần xác minh"
                elif interaction["passing"]:
                    status = "Người đi ngang/đi cạnh nhau · chưa đủ bằng chứng cảnh báo"
                if action_high:
                    status = ALERT_TEXT
                if scenario['occlusion_hold']:
                    status='Giữ cảnh báo: nhóm vừa xung đột bị đám đông che khuất · cần kiểm tra'
                if self.people.model is None and not self.cfg["detection"]["require_interaction"]:
                    reason += "; cảnh báo chỉ dựa vào hình ảnh"
                protected = bool(self.cfg["display"]["privacy"])
                public = privacy_frame(frame, all_boxes) if protected else frame.copy()
                heatmap = self.motion.heatmap(public.shape) if self.cfg["display"]["heatmap"] else None
                view = CameraView(public, boxes, heatmap, score, raw, status, reason,
                                  0, (time.monotonic()-started)*1000, position,
                                  source_seq=seq, source_stamp=stamp, privacy_applied=protected,
                                  annotation_epoch=epoch, analysis_seq=seq,
                                  alert_active=self.gate.active or scenario['occlusion_hold'] or action_high,
                                  alert_track_ids=tuple(sorted({b.track_id for b in boxes if b.observed and b.is_fighting})))
                self.total_latency += view.latency
                self.publish_view(view)  # GUI đọc được kết quả trước I/O lưu/gửi sự kiện.
                self.recorder.feed(stamp, public)
                if fire:
                    self.event(public, view, ALERT_TEXT)
                for signal in scenario['signals']:
                    scenario_view=copy.copy(view);scenario_view.reason=signal['reason']
                    scenario_view.status=signal['kind']+' · cần người kiểm tra'
                    with self.lock:
                        self.view.status=scenario_view.status
                    self.event(public,scenario_view,signal['kind'])
                if self.cfg['school_policy']['camera_health']:
                    health=self.health.update(stamp,frame,self.cfg['school_policy']['health_seconds'])
                    if health['reason']:
                        with self.lock:self.view.status='⚠️ Camera: '+health['reason']
                    if health['fire']:
                        health_view=copy.copy(view);health_view.reason=health['reason']+'; có thể do môi trường hoặc camera bị che; cần kiểm tra.'
                        self.event(public,health_view,'⚠️ Camera có hình bất thường · '+self.spec['label'])
                if self.cfg["evidence"]["continuous"] and self.recorder.disk_ok():
                    if self.active_recording is None or stamp-continuous_start >= self.cfg["evidence"]["segment_seconds"]:
                        if self.active_recording:
                            self.active_recording.release()
                        target = self.store.folder / "recordings" / f"{safe_name(self.spec['label'])}_{uuid.uuid4().hex}.mp4"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        self.active_recording = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                                                  self.cfg["detection"]["analysis_fps"], (w, h))
                        if not self.active_recording.isOpened():
                            raise OSError("Không mở được bộ ghi liên tục.")
                        continuous_start, continuous_frames = stamp, 0
                    count = int((stamp-continuous_start)*self.cfg["detection"]["analysis_fps"])+1
                    while continuous_frames < count:
                        self.active_recording.write(public)
                        continuous_frames += 1
                elif self.active_recording:
                    self.active_recording.release()
                    self.active_recording = None
                self.publish_view(view, completed=True)
                last_frame = public
                self.stop.wait(max(0, 1/self.cfg["detection"]["analysis_fps"]-(time.monotonic()-started)))
            if self.capture.status.startswith(("Không mở", "Lỗi")):
                self.failure = self.capture.status
        except Exception as exc:
            self.failure = redact(exc)
            LOG.error("Camera %s: %s", self.spec["label"], self.failure)
            with self.lock:
                self.view.status = "Lỗi: " + self.failure
        finally:
            self.stop.set()
            try:
                cleanup = [("clip sự kiện", self.recorder.close)]
                if self.active_recording:
                    cleanup.append(("ghi liên tục", self.active_recording.release))
                for name, finish in cleanup:
                    try:
                        finish()
                    except Exception as exc:
                        self.failure = self.failure or f"Không hoàn tất {name}: " + redact(exc)
                        LOG.error("Camera %s: %s", self.spec["label"], self.failure)
            finally:
                for child in (self.capture.thread if self.capture else None,):
                    if child is not None:
                        child.join()
                if self.quick:self.quick.close()
                if self.action_monitor is not None:self.action_monitor.close()


class Monitor:
    def __init__(self, cfg, store, offline=False, engine_cache=None, progress=None, cancel=None):
        self.cfg, self.store = cfg, store
        if cfg['model'].get('video_context_enabled') and cfg['model'].get('video_context_kind') == 'vit_cls_research':
            raise ValueError('ViT chuỗi đang thử nghiệm trên video cục bộ; chưa mở cho luồng giám sát trực tiếp.')
        if cancel is not None and cancel.is_set():
            raise LoadingCancelled("Đã hủy khởi động camera.")
        if engine_cache is None:
            self.visual, self.people = VisualEngine(cfg["model"]), PeopleEngine(cfg["person"])
        else:
            bundle = engine_cache.get(cfg, progress)
            if cancel is not None and cancel.is_set():
                raise LoadingCancelled("Đã hủy khởi động camera.")
            bundle.activate()
            self.visual, self.people = bundle.visual, bundle.people
        if cancel is not None and cancel.is_set():
            raise LoadingCancelled("Đã hủy khởi động camera.")
        self.notifications = NotificationQueue(cfg["notifications"], offline, feedback=getattr(store, "feedback", None),store=store)
        self.action_worker = make_action_worker(self.visual, max_cameras=len(cfg['sources']))
        missing = []
        if cfg["person"]["enabled"] and self.people.model is None:
            missing.append(self.people.reason + "; thiếu kiểm chứng tương tác người")
        self.ai_notice = "; ".join(missing)
        self.workers = []
        self.closed, self.created = False, now_iso()
        self.close_lock = threading.Lock()
        self.maintenance=MaintenanceLoop(store,self.patrol)

    def patrol(self,now):
        slot=patrol_slot(self.cfg['school_policy'],now)
        if not slot or not self.workers:return
        selected=[w for w in self.workers if w.spec.get('hidden_zones')]
        if not selected or not self.store.patrol_once(slot):return
        frame,details=patrol_grid(selected,time.monotonic())
        if frame is None:return
        reason='; '.join(r['camera']+(': đang cập nhật' if r['fresh'] else ': cũ/tạm dừng/mất nguồn') for r in details)
        eid=self.store.add('Nhóm camera góc khuất','Kiểm tra định kỳ',None,None,0,reason)
        target=self.store.folder/'patrol'/f'{eid}.jpg';save_jpeg(frame,target)
        self.store.update(eid,snapshot=str(target))
        row=self.store.get(eid);row['notice']='🔔 Đến giờ kiểm tra định kỳ: Mời thầy/cô lướt xem nhanh tình hình các góc khuất hiện tại'
        row['no_buttons']=True
        self.notifications.send(row,frame)

    def start(self, sources):
        if self.closed:
            raise RuntimeError("Phiên giám sát đã đóng.")
        for spec in sources:
            worker = CameraWorker(spec, self.cfg, self.visual, self.people, self.store, self.notifications, action_worker=self.action_worker)
            self.workers.append(worker)
            worker.thread.start()

    def request_stop(self):
        for worker in self.workers:
            worker.stop.set()

    def close(self):
        # Chỉ một chủ sở hữu kết thúc phiên. join chạy trên luồng nền GUI;
        # không cho phiên mới dùng AI khi luồng cũ còn đang suy luận.
        with self.close_lock:
            self._close()

    def _close(self):
        if self.closed:
            return
        self.request_stop()
        self.maintenance.close()
        for worker in self.workers:
            worker.thread.join()
            # Ranh giới sở hữu vẫn đúng nếu analysis lỗi trong cleanup.
            for child in (worker.capture.thread if worker.capture else None,):
                if child is not None:
                    child.join()
        self.notifications.close()
        if getattr(self, 'action_worker', None) is not None:
            self.action_worker.close(timeout=None)  # Models cannot be reused while an old inference runs.
        self.closed = True
        session = {"started": self.created, "ended": now_iso(), "model": self.visual.identity,
                   "notification_delivery": self.notifications.snapshot_stats(),
                   "cameras": [{"camera": w.spec["label"], "analyzed_frames": w.processed,
                                "skipped_frames": w.skipped, "visual_peak": w.peak,
                                "average_latency_ms": round(w.total_latency/max(1, w.processed), 1),
                                "error": w.failure} for w in self.workers]}
        try:
            target = self.store.folder / "sessions" / f"{uuid.uuid4().hex}.json"
            atomic_json(target, session)
            LOG.info("Đã lưu báo cáo phiên: %s", target)
        except OSError as exc:
            LOG.warning("Không lưu được báo cáo phiên: %s", redact(exc))
        for camera in session["cameras"]:
            LOG.info("%s: %d khung phân tích, %.1f ms/khung", camera["camera"], camera["analyzed_frames"], camera["average_latency_ms"])


def make_action_worker(visual, max_cameras=1, synchronous=False):
    context = getattr(visual, 'video_context', None)
    if context is None:
        return None
    from action_runtime import ActionWorker
    return ActionWorker(*context, max_cameras=max_cameras, synchronous=synchronous,
                        max_prepared_frame_bytes=getattr(context[0], 'max_prepared_frame_bytes', 3*112*112*4))


def action_reason(evidence):
    return ('AI chuỗi hành động nghi có xô xát; cần người đến kiểm tra. '
            'AI không kết luận hành vi hay xác định người có lỗi.')


def analyze_clip(filename: Path, cfg: dict, engine=None, cancel=None, people_engine=None,
                 observer=None, emit=None) -> dict:
    """Phân tích chuỗi hình ảnh; cảnh báo cần người xem và xác minh."""
    runtime_imports()
    if not filename.is_file():
        raise FileNotFoundError("Không tìm thấy video cần phân tích.")
    engine = engine or VisualEngine(cfg["model"])
    people_engine = (people_engine or PeopleEngine(cfg["person"])) if cfg["detection"]["require_interaction"] else None
    cancel, state = cancel or threading.Event(), VisualState()
    tracker, motion, interaction = SimpleTracker(), MotionField(), InteractionVerifier()
    scene, gate, alerts = SceneCutDetector(), AlertGate(), []
    scenarios=ScenarioMonitor()
    detection,policy_mode=detection_at(cfg['detection'],cfg['school_policy'])
    action_worker = make_action_worker(engine, synchronous=True)
    action_monitor = None
    if action_worker is not None:
        from action_runtime import ActionMonitor
        action_monitor = ActionMonitor(action_worker, 'offline', cooldown=detection['cooldown_seconds'])
    action_scene, action_due, action_origin, action_timeline = SceneCutDetector(), 0., None, []
    action_decode_audit = None
    if action_worker is not None and action_worker.cls_context:
        try:
            from explore_vit_causal import audit_source
            from train_vit_causal import validate_decode_audit
            action_decode_audit = audit_source(filename)
            validate_decode_audit(action_decode_audit, {'frame_ids': np.empty((0,4), dtype=np.int64),
                                                       'frame_seconds': np.empty((0,4))})
        except BaseException:
            action_monitor.close()
            action_worker.close()
            raise
    cap = cv2.VideoCapture(str(filename))
    if not cap.isOpened():
        if action_monitor is not None:
            action_monitor.close()
        if action_worker is not None:
            action_worker.close()
        cap.release()
        raise OSError("Không mở được video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 25
    step, index, timeline = max(1, round(fps/cfg["detection"]["analysis_fps"])), 0, []
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_second, last_progress, previous_analysis = -1.0, time.monotonic(), None
    scene_epoch = 0
    try:
        while not cancel.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            second = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            # Dùng timestamp decoder cho video tốc độ khung biến thiên;
            # fallback khi backend không trả timestamp tăng dần.
            if not math.isfinite(second) or second <= last_second:
                second = max(index/fps, last_second+1/fps)
            if action_worker is not None and action_worker.cls_context:
                # All evidence and shared cooldowns use the audited native clock.
                second = index/fps
            last_second = second
            if action_monitor is not None:
                if action_origin is None:
                    action_origin = second
                source_second = index/fps if action_worker.cls_context else second-action_origin
                if action_worker.cls_context or source_second+1e-8 >= action_due:
                    sample_fps = action_worker.sample_fps
                    action_due = (math.floor(source_second*sample_fps+1e-8)+1)/sample_fps
                    action_monitor.feed(frame, source_second, index, bool(action_scene.check_cut(frame)))
                    action = action_monitor.poll(source_second)
                    if action_monitor.error:
                        raise RuntimeError('Causal action inference failed: '+action_monitor.error)
                    if action is not None:
                        # The video branch is a separately calibrated causal
                        # hypothesis, and never invents person/attacker boxes.
                        action['fire'] = bool(action['fire'] and second-gate.last_alert >= detection['cooldown_seconds'])
                        action_timeline.append(action)
                        if action['fire']:
                            gate.last_alert = second
                            alerts.append({'second': round(second,3), 'visual': action['score'],
                                           'reason': action_reason(action), 'kind': 'causal_video_context'})
            if index % step == 0:
                if scene.check_cut(frame):
                    scene_epoch += 1
                    state = VisualState()
                    tracker.reset()
                    motion, interaction = MotionField(), InteractionVerifier()
                    gate.reset()
                    scenarios.reset()
                    previous_analysis = None
                state.frame_time = second
                raw, smooth = engine.score(frame, state)
                # Keep pose/motion history during sequence warm-up, as live cameras do.
                boxes = people_engine.detect(frame, tracker) if people_engine is not None else []
                dt = second-previous_analysis if previous_analysis is not None else 1/cfg["detection"]["analysis_fps"]
                previous_analysis = second
                motion.update(frame, boxes, dt)
                evidence = interaction.update(second, boxes, frame.shape, smooth, detection=detection)
                status, fire = gate.update(second, smooth, evidence["supported"], detection, evidence["confirmed_since"])
                scenario=scenarios.update(second,boxes,frame.shape,cfg['school_policy'],policy_mode,conflict=gate.active and evidence['supported'])
                if raw is not None:
                    timeline.append({"second": round(second, 3), "visual": raw, "smooth": smooth,
                                     "interaction": evidence, "automatic_alert": fire or bool(scenario['signals']), "status": status,
                                     'scenario':scenario, "frame_index": index, "scene_epoch": scene_epoch,
                                     "alert_active": bool(gate.active or scenario['occlusion_hold']),
                                     "alert_track_ids": sorted(b.track_id for b in boxes if b.observed and b.is_fighting)})
                    if observer is not None:
                        observer(frame, timeline[-1], boxes)
                    if emit is not None and time.monotonic()-last_progress >= 10:
                        emit(f"{filename.name}: đã phân tích đến {second:.1f}s, {len(timeline)} khung lấy mẫu, {len(alerts)} cảnh báo")
                        last_progress = time.monotonic()
                if fire:
                    if action_monitor is not None:
                        action_monitor.last_alert = index/fps if action_worker.cls_context else second-action_origin
                    alerts.append({"second": round(second, 3), "visual": smooth, "reason": evidence["reason"]})
                for signal in scenario['signals']:
                    alerts.append({'second':round(second,3),'visual':smooth,'reason':signal['reason'],'kind':signal['kind']})
            index += 1
        if action_decode_audit is not None and not cancel.is_set() and index != action_decode_audit['actual_decoded_count']:
            raise ValueError('Số khung giải mã đã thay đổi sau khi kiểm tra thời gian video.')
    finally:
        cap.release()
        if action_monitor is not None:
            action_monitor.close()
        if action_worker is not None:
            action_worker.close()
    visual = sum(r["visual"] for r in timeline)/len(timeline) if timeline else None
    result = {"file": filename.name, "created": now_iso(), "model": engine.identity,
              "duration_seconds": round(last_second+1/fps, 2) if index else 0, "visual_mean": visual,
              "decoded_frames": index, "reported_frames": reported_frames, "source_fps": fps,
              "visual_peak": max((r["visual"] for r in timeline), default=None),
              "timeline": timeline,
              "automatic_alerts": alerts,
              "action_timeline": action_timeline,
              "action_context_enabled": action_monitor is not None,
              "action_decode_audit": action_decode_audit,
              "person_model": cfg["person"]["path"] if people_engine is not None else "interaction filter disabled",
              "cancelled": cancel.is_set(), "note": HUMAN_GUIDANCE}
    fallback = visual_decision(result["visual_peak"], cfg)
    result["visual_signal"] = dict(fallback)
    if alerts:
        fallback = {"label": ALERT_TEXT, "risk": "high", "corroborated": False}
    elif cfg["detection"]["require_interaction"] and fallback["risk"] == "high":
        fallback = {"label": "AI nghi ngờ · chưa đủ bằng chứng tương tác để báo động", "risk": "review", "corroborated": False}
    result["decision"] = fallback
    return result


JOBS = {"build-dataset": "build_dataset.py", "train-database": "retrain_database.py", "verify-videos": "verify_schoolguard_videos.py",
        "verify-occlusion": "verify_person_occlusion.py", "train-visual": "train_model.py",
        "train-sequence": "train.py", "evaluate": "evaluate.py", "verify": "verify_pipeline.py"}


def job_command(name: str, extra: list[str]) -> list[str]:
    if name not in JOBS:
        raise ValueError("Công việc không hợp lệ.")
    script = ROOT / JOBS[name]
    if not script.is_file():
        raise FileNotFoundError(f"Thiếu công cụ dự án {script.name}.")
    return [sys.executable, "-u", str(script), *extra]


def doctor(cfg: dict) -> dict:
    modules = ["numpy", "cv2", "PIL", "tkinter", "torch", "torchvision", "ultralytics",
               "tensorflow", "requests", "yaml"]
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "modules": {name: importlib.util.find_spec(name) is not None for name in modules},
            "weights": {name: resolve_path(value).is_file() for name, value in
                        [("resnet", cfg["model"]["path"]), ("keras", cfg["model"]["keras_path"]), ("person", cfg["person"]["path"])]},
            "storage": str(resolve_path(cfg["storage"]["folder"])),
            "external_notifications": cfg["notifications"]["enabled"]}


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def http_bytes(url: str, max_bytes: int = 10*1024**2, headers=None) -> bytes:
    """Tải có timeout/retry/giới hạn; không đọc nguyên kho nhiều GB nếu server bỏ Range."""
    import requests
    last = None
    for attempt in range(3):
        try:
            with requests.get(url, headers=headers, timeout=(10, 40), stream=True) as response:
                response.raise_for_status()
                if headers and "Range" in headers and response.status_code != 206:
                    raise OSError("Nguồn không hỗ trợ HTTP Range; dừng để tránh tải cả kho lớn.")
                parts, total = [], 0
                for part in response.iter_content(64*1024):
                    total += len(part)
                    if total > max_bytes:
                        raise OSError("Tệp tải vượt giới hạn dung lượng.")
                    parts.append(part)
                return b"".join(parts)
        except (requests.RequestException, OSError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(1+attempt)
    raise OSError("Tải dữ liệu thất bại: " + redact(last))


def prepare_pose_model(model_name=None):
    name = model_name or Path(DEFAULT_CONFIG["person"]["path"]).name
    if name not in ("yolov8n-pose.pt", "yolov8s-pose.pt"):
        raise ValueError("Chọn pose nano hoặc pose small được hỗ trợ.")
    target = ROOT/"models"/name
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"+name
    if not target.is_file():
        data = http_bytes(url, 40*1024**2)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".download.pt")
        temporary.write_bytes(data)
        from ultralytics import YOLO
        if YOLO(str(temporary)).task != "pose":
            raise ValueError("Trọng số tải không phải mô hình pose.")
        temporary.replace(target)
    result = {"source": url, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "bytes": target.stat().st_size,
              "purpose": "Keypoints người; chưa phải mô hình phân loại ý định hay bạo lực học đường"}
    atomic_json(target.with_suffix(".source.json"), result)
    return result


def visual_decision(visual, cfg: dict) -> dict:
    if visual is None:
        return {"label": "Chưa đủ dữ liệu", "risk": "unknown", "corroborated": False}
    if visual >= cfg["detection"]["threshold"]:
        return {"label": "AI nghi ngờ · cần xác minh diễn biến", "risk": "high", "corroborated": False}
    return {"label": "Chưa đủ bằng chứng xô xát", "risk": "review" if visual >= cfg["detection"]["release_threshold"] else "low", "corroborated": False}


def analyze_folder(folder: Path, cfg: dict, emit=print, cancel=None) -> dict:
    cancel = cancel or threading.Event()
    if not folder.is_dir():
        raise ValueError("Thư mục video không tồn tại.")
    videos = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError("Không tìm thấy video trong thư mục.")
    engine = VisualEngine(cfg["model"])
    people_engine = PeopleEngine(cfg["person"]) if cfg["detection"]["require_interaction"] else None
    results = []
    target = resolve_path(cfg["storage"]["folder"])/"reports"/("folder_"+uuid.uuid4().hex)
    target.mkdir(parents=True)
    for i, video in enumerate(videos, 1):
        if cancel.is_set():
            break
        emit(f"Video {i}/{len(videos)}: {video.name}")
        try:
            result = analyze_clip(video, cfg, engine=engine, cancel=cancel, people_engine=people_engine)
            output = target/(safe_name(video.stem)+"_"+uuid.uuid4().hex[:8]+".json")
            atomic_json(output, result)
            row = {"video": str(video.relative_to(folder)), "decision": result["decision"]["label"],
                   "risk": result["decision"]["risk"], "visual_peak": result["visual_peak"],
                   "automatic_alerts": len(result["automatic_alerts"]),
                   "report": str(output), "error": ""}
        except Exception as exc:
            row = {"video": str(video.relative_to(folder)), "error": redact(exc), "risk": "error", "decision": "Không phân tích được"}
        results.append(row)
        atomic_json(target/"summary.json", {"folder": str(folder), "results": results, "cancelled": cancel.is_set()})
        emit(row["decision"])
    atomic_json(target/"summary.json", {"folder": str(folder), "videos": len(videos),
                                       "processed": len(results), "results": results, "cancelled": cancel.is_set()})
    with (target/"summary.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        keys = ["video", "decision", "risk", "visual_peak", "automatic_alerts", "report", "error"]
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: "'"+v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v for k, v in r.items()} for r in results])
    return {"videos": len(videos), "processed": len(results), "report_folder": str(target), "results": results}


class SchoolGuardApp:
    """Tk chỉ được dùng từ luồng giao diện; AI/camera/tác vụ chạy nền."""
    def __init__(self, cfg, store, offline=False, smoke_seconds=0):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        from PIL import Image, ImageTk
        self.tk, self.ttk, self.dialog, self.messagebox = tk, ttk, filedialog, messagebox
        self.Image, self.ImageTk = Image, ImageTk
        self.cfg, self.store, self.offline = cfg, store, offline
        # Khung thứ hai chỉ được mở qua thao tác Thêm khung, kể cả cấu hình có nhiều nguồn.
        self.pane_count = 1
        self.monitor, self.bus = None, queue.Queue()
        self.starting = self.stopping = self.closing = False
        self.model_cache = ModelCache()
        self.preload_thread = self.boot_thread = None
        self.preloading, self.preload_requested = False, False
        self.preload_enabled = False
        self.model_generation = self.boot_generation = 0
        self.start_cancel = threading.Event()
        self.smoke_seconds = smoke_seconds
        self.task_thread = None
        self.task_cancel = threading.Event()
        self.cards, self.images = {}, {}
        self.render_cursor = 0
        self.root = tk.Tk()
        self.root.title("SchoolGuard 2 · Giám sát an toàn học đường")
        self.root.geometry("1320x850")
        self.root.minsize(1000, 700)
        self.root.configure(bg="#101923")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background="#172330", foreground="#e5eef5")
        style.configure("TFrame", background="#172330")
        style.configure("TLabel", background="#172330", foreground="#e5eef5")
        style.configure("TButton", padding=(12, 7), background="#294156", foreground="#ffffff")
        style.map("TButton", background=[("active", "#365e78"), ("disabled", "#25313c")])
        style.configure("Accent.TButton", background="#17785f")
        style.configure("Danger.TButton", background="#954246")
        style.configure("TEntry", fieldbackground="#223445", foreground="white", padding=5)
        style.configure("TCombobox", fieldbackground="#223445", foreground="white", arrowcolor="white")
        style.map("TCombobox", fieldbackground=[("readonly", "#223445")], foreground=[("readonly", "white")])
        style.configure("Treeview", fieldbackground="#172330", background="#172330", foreground="#e5eef5", rowheight=30)
        style.configure("Treeview.Heading", background="#294156", foreground="white", padding=7)
        style.map("Treeview", background=[("selected", "#255f72")])
        style.configure("TNotebook.Tab", padding=(18, 9), background="#223445")
        style.map("TNotebook.Tab", background=[("selected", "#326176")])
        header = ttk.Frame(self.root, padding=(20, 16))
        header.pack(fill="x")
        ttk.Label(header, text="SchoolGuard 2", font=("Segoe UI", 24, "bold")).pack(side="left")
        ttk.Label(header, text="Giám sát an toàn học đường\nAI hỗ trợ cảnh báo · người phụ trách xác minh", foreground="#a9bfce").pack(side="left", padx=24)
        self.health = tk.StringVar(value="Chưa nạp AI · chưa mở camera")
        ttk.Label(header, textvariable=self.health, foreground="#73d9b3").pack(side="right")
        self.book = ttk.Notebook(self.root)
        self.book.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        tabs = []
        for name in ("Giám sát", "Sự kiện & bằng chứng", "Cài đặt", "Hướng dẫn"):
            tab = ttk.Frame(self.book, padding=14)
            self.book.add(tab, text=name)
            tabs.append(tab)
        self.build_monitor_tab(tabs[0])
        self.build_events_tab(tabs[1])
        self.build_settings_tab(tabs[2])
        self.build_help_tab(tabs[3])
        self.footer = tk.StringVar(value="Dữ liệu: " + str(store.folder))
        ttk.Label(self.root, textvariable=self.footer, padding=(16, 8), foreground="#9eb4c5").pack(fill="x")
        self.last_events = 0
        self.refresh_sources()
        self.refresh_events()
        self.root.bind("<Control-s>", lambda e: self.command_selected("snapshot"))
        self.root.bind("<F9>", lambda e: self.command_selected("sos"))
        self.root.after(100, self.poll)
        if smoke_seconds:
            self.root.after(int(smoke_seconds*1000), self.close)


    def build_monitor_tab(self, parent):
        ttk = self.ttk
        sources = ttk.Frame(parent)
        sources.pack(fill="x")
        columns = ("name", "source", "roi")
        self.source_tree = ttk.Treeview(sources, columns=columns, show="headings", height=3, selectmode="browse")
        for col, title, width in zip(columns, ("Camera / khu vực", "Nguồn (thông tin đăng nhập được che)", "Vùng giám sát"), (200, 650, 180)):
            self.source_tree.heading(col, text=title)
            self.source_tree.column(col, width=width)
        self.source_tree.pack(fill="x")
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=10)
        self.start_button = ttk.Button(controls, text="▶ Bắt đầu", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="left")
        ttk.Button(controls, text="■ Dừng", command=self.stop).pack(side="left", padx=5)
        ttk.Button(controls, text="Tạm dừng / Tiếp tục", command=self.toggle_pause).pack(side="left", padx=5)
        ttk.Button(controls, text="Chụp ảnh", command=lambda: self.command_selected("snapshot")).pack(side="left", padx=5)
        ttk.Button(controls, text="SOS thủ công · F9", style="Danger.TButton", command=lambda: self.command_selected("sos")).pack(side="right")
        editing = ttk.Frame(parent)
        editing.pack(fill="x", pady=(0, 10))
        for text, method in [("＋ Thêm khung", self.add_pane), ("Mở camera", self.open_camera),
                             ("Nguồn RTSP / URL", self.add_source), ("Chọn video", self.add_video), ("Thư mục video", self.browse_video_folder), ("Xóa khung", self.remove_source)]:
            ttk.Button(editing, text=text, command=method).pack(side="left", padx=(0, 6))
        ttk.Button(editing, text="⚡ Chạy 20-30 video kiểm tra", style="Accent.TButton", command=self.run_fresh_eval).pack(side="left", padx=(10, 6))
        self.roi_armed = False
        self.display_vars = {}
        options = ttk.Frame(parent)
        options.pack(fill="x", pady=(0, 6))
        for title, command in [("Vẽ vùng giám sát", self.arm_roi), ("Toàn khung hình", self.clear_roi),
                               ('Vẽ góc khuất',self.arm_hidden_zone),('Xóa góc khuất',self.clear_hidden_zones)]:
            ttk.Button(options, text=title, command=command).pack(side="left", padx=(0, 6))
        for key, title in [("boxes", "Khung người"), ("heatmap", "Chuyển động"), ("privacy", "Che vùng đầu"), ("sound", "Còi")]:
            variable = self.tk.BooleanVar(value=self.cfg["display"][key])
            self.display_vars[key] = variable
            ttk.Checkbutton(options, text=title, variable=variable, command=self.update_display).pack(side="left", padx=4)
        self.video_scroll = self.tk.Canvas(parent, bg="#101923", highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.video_scroll.yview)
        self.video_scroll.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.video_scroll.pack(fill="both", expand=True)
        self.video_grid = ttk.Frame(self.video_scroll)
        self.grid_window = self.video_scroll.create_window(0, 0, anchor="nw", window=self.video_grid)
        self.video_grid.bind("<Configure>", lambda e: self.video_scroll.configure(scrollregion=self.video_scroll.bbox("all")))
        self.video_scroll.bind("<Configure>", self.resize_video_grid)
        self.video_grid.columnconfigure(0, weight=1)
        self.video_grid.columnconfigure(1, weight=1)

    def resize_video_grid(self, event=None):
        width = event.width if event else self.video_scroll.winfo_width()
        height = event.height if event else self.video_scroll.winfo_height()
        rows = math.ceil(self.pane_count/2) if self.pane_count > 1 else 1
        self.video_scroll.itemconfigure(self.grid_window, width=max(200, width),
                                        height=max(height, rows*310) if self.pane_count > 1 else max(280, height))
        for card in self.cards.values():
            wrap = max(220, width//(2 if self.pane_count > 1 else 1)-40)
            card["title_label"].configure(wraplength=wrap)
            card["detail_label"].configure(wraplength=wrap)

    def refresh_sources(self):
        self.source_tree.delete(*self.source_tree.get_children())
        for card in self.cards.values():
            card["frame"].destroy()
        self.cards.clear()
        self.images.clear()
        self.pane_count = min(self.pane_count, len(self.cfg["sources"]))
        self.source_tree.configure(height=min(3, self.pane_count))
        for row in range(4):
            self.video_grid.rowconfigure(row, weight=1 if row < math.ceil(self.pane_count/2) else 0)
        for i, spec in enumerate(self.cfg["sources"][:self.pane_count]):
            self.source_tree.insert("", "end", iid=str(i), values=(spec["label"], redact(spec["source"]), "Đã chọn" if spec.get("roi") else "Toàn khung"))
            block = self.ttk.Frame(self.video_grid, padding=8)
            block.grid(row=i//2, column=i%2, columnspan=2 if self.pane_count == 1 else 1, sticky="nsew", padx=4, pady=4)
            title = self.tk.StringVar(value=spec["label"] + " · chưa chạy")
            title_label = self.ttk.Label(block, textvariable=title, wraplength=540, font=("Segoe UI", 11, "bold"))
            title_label.pack(fill="x", pady=(0, 6))
            canvas = self.tk.Canvas(block, height=240, bg="#0b121a", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.create_text(260, 110, text="Chọn video hoặc Mở camera, rồi bắt đầu giám sát", fill="#698197", tags="hint")
            canvas.bind("<Configure>", lambda e, c=canvas: c.coords("hint", e.width//2, e.height//2))
            detail = self.tk.StringVar(value="Nguồn: " + redact(spec["source"]))
            detail_label = self.ttk.Label(block, textvariable=detail, wraplength=540, foreground="#a4bbcc")
            detail_label.pack(fill="x", pady=4)
            card = {"frame": block, "canvas": canvas, "title": title, "detail": detail,
                    "title_label": title_label, "detail_label": detail_label,
                    "mapping": None, "image": None, "drag": None, "rectangle": None,
                    "render_key": None, "last_source": None, "rendered_at": deque(),
                    "display_started_at": None, "display_fps": 0.0, "render_count": 0,
                    "label_key": None, "label_at": 0.0}
            self.cards[i] = card
            canvas.bind("<ButtonPress-1>", lambda e, idx=i: self.roi_start(e, idx))
            canvas.bind("<B1-Motion>", lambda e, idx=i: self.roi_drag(e, idx))
            canvas.bind("<ButtonRelease-1>", lambda e, idx=i: self.roi_end(e, idx))
        if self.cfg["sources"]:
            self.source_tree.selection_set("0")
        self.resize_video_grid()

    def selected_index(self):
        chosen = self.source_tree.selection()
        return int(chosen[0]) if chosen else 0

    def can_edit_sources(self):
        if self.monitor is not None or self.starting or self.stopping:
            self.footer.set("Dừng giám sát trước khi thêm/xóa nguồn.")
            return False
        return True

    def install_source(self, spec, new_frame=False):
        candidate = copy.deepcopy(self.cfg)
        selected = self.selected_index()
        used = {s["label"] for i, s in enumerate(candidate["sources"]) if new_frame or i != self.selected_index()}
        while spec["label"] in used:
            spec["label"] += " (2)"
        if new_frame:
            if self.pane_count >= 8:
                self.footer.set("Tối đa 8 khung giám sát.")
                return False
            if len(candidate["sources"]) >= 8:
                candidate["sources"][self.pane_count] = spec
            else:
                candidate["sources"].insert(self.pane_count, spec)
        else:
            candidate["sources"][self.selected_index()] = spec
        validate_config(candidate)
        self.cfg["sources"] = candidate["sources"]
        if new_frame:
            self.pane_count += 1
        self.refresh_sources()
        self.source_tree.selection_set(str(self.pane_count-1 if new_frame else selected))
        return True

    def add_pane(self):
        if not self.can_edit_sources():
            return
        menu = self.tk.Menu(self.root, tearoff=False)
        menu.add_command(label="Thêm khung video", command=lambda: self.add_video(new_frame=True))
        menu.add_command(label="Thêm khung camera", command=lambda: self.open_camera(new_frame=True))
        menu.add_command(label="Thêm khung RTSP / URL", command=lambda: self.add_source(new_frame=True))
        for i, spec in enumerate(self.cfg["sources"][self.pane_count:], self.pane_count):
            def activate(index=i):
                self.cfg["sources"].insert(self.pane_count, self.cfg["sources"].pop(index))
                self.pane_count += 1
                self.refresh_sources()
            menu.add_command(label="Nguồn đã lưu: "+spec["label"], command=activate)
        menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        menu.grab_release()

    def open_camera(self, new_frame=False):
        if not self.can_edit_sources():
            return
        from tkinter import simpledialog
        index = simpledialog.askinteger("Mở camera", "Chọn camera: 0 = camera mặc định, 1/2… = camera khác", initialvalue=0, minvalue=0, maxvalue=32, parent=self.root)
        if index is not None:
            if self.install_source({"label": f"Camera {index+1}", "source": index, "roi": None}, new_frame):
                self.start()

    def add_source(self, new_frame=False):
        if not self.can_edit_sources():
            return
        from tkinter import simpledialog
        label = simpledialog.askstring("Thêm camera", "Tên khu vực / camera:", parent=self.root)
        if not label:
            return
        source = simpledialog.askstring("Nguồn video", "Số webcam (0,1...), đường dẫn video hoặc URL RTSP/HTTP:", parent=self.root)
        if not source:
            return
        try:
            self.install_source({"label": label.strip(), "source": source.strip(), "roi": None}, new_frame)
        except ValueError as exc:
            self.messagebox.showerror("Nguồn chưa hợp lệ", str(exc))
            return
    def add_video(self, new_frame=False):
        if not self.can_edit_sources():
            return
        filename = self.dialog.askopenfilename(title="Chọn video giám sát", filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.webm"), ("Mọi tệp", "*.*")])
        if filename:
            self.install_source({"label": Path(filename).stem, "source": filename, "roi": None}, new_frame)

    def remove_source(self):
        if self.can_edit_sources():
            if len(self.cfg["sources"]) <= 1:
                self.footer.set("Giữ ít nhất một nguồn để giám sát.")
                return
            del self.cfg["sources"][self.selected_index()]
            self.pane_count = max(1, self.pane_count-1)
            self.refresh_sources()

    def browse_video_folder(self):
        if not self.can_edit_sources():
            return
        folder = self.dialog.askdirectory(title="Chọn thư mục chứa video cần soi")
        if not folder:
            return
        videos = sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        if not videos:
            self.messagebox.showinfo("Thư mục video", "Không tìm thấy video trong thư mục này.")
            return
        window = self.tk.Toplevel(self.root)
        window.title("Chọn video trong thư mục · hoặc soi toàn bộ")
        window.geometry("850x450")
        listing = self.tk.Listbox(window, selectmode="extended", bg="#172330", fg="white", font=("Segoe UI", 10))
        listing.pack(fill="both", expand=True, padx=12, pady=12)
        for video in videos:
            listing.insert("end", str(video.relative_to(folder)))
        actions = self.ttk.Frame(window)
        actions.pack(fill="x", padx=12, pady=10)
        def add_selected():
            selected = [videos[i] for i in listing.curselection()]
            if not selected or len(selected) > 8:
                self.footer.set("Chọn từ 1 đến 8 video để giám sát; soi hàng loạt không giới hạn 8 video.")
                return
            self.cfg["sources"] = [{"label": f"{i+1}. {v.stem}", "source": str(v), "roi": None} for i, v in enumerate(selected)]
            self.pane_count = len(selected)
            self.refresh_sources()
            window.destroy()
        def batch():
            window.destroy()
            self.analyze_directory(folder)
        self.ttk.Button(actions, text="Mở video đã chọn làm nguồn", command=add_selected).pack(side="left")
        self.ttk.Button(actions, text=f"Soi toàn bộ {len(videos)} video", command=batch).pack(side="right")

    def arm_roi(self):
        self.roi_armed = True
        self.zone_armed = False
        self.footer.set("Kéo một hình chữ nhật trên hình camera để chọn vùng giám sát; cần camera đang chạy.")

    def arm_hidden_zone(self):
        self.roi_armed=True;self.zone_armed=True
        self.footer.set('Kéo hình chữ nhật trên camera để đánh dấu góc khuất/cầu thang. Có thể chọn nhiều vùng; không cắt hình camera.')

    def update_hidden_zones(self,idx,zones):
        self.cfg['sources'][idx]['hidden_zones']=zones
        if self.monitor:
            worker=self.monitor.workers[idx]
            with worker.lock:worker.spec['hidden_zones']=copy.deepcopy(zones)
            worker.command('reset')
        if hasattr(self,'policy_editor'):
            try:advanced=json.loads(self.policy_editor.get('1.0','end'))
            except ValueError:return
            if not isinstance(advanced,dict) or not isinstance(advanced.get('hidden_zones'),dict):return
            advanced['hidden_zones'][self.cfg['sources'][idx]['label']]=zones
            self.policy_editor.delete('1.0','end');self.policy_editor.insert('1.0',json.dumps(advanced,ensure_ascii=False,indent=2))

    def clear_hidden_zones(self):
        self.update_hidden_zones(self.selected_index(),[])
        self.footer.set('Đã xóa vùng góc khuất của camera đang chọn. Lưu cài đặt để giữ cho lần sau.')

    def clear_roi(self):
        idx = self.selected_index()
        self.cfg["sources"][idx]["roi"] = None
        if self.monitor:
            self.monitor.workers[idx].set_roi(None)
        card = self.cards[idx]
        if card["rectangle"]:
            card["canvas"].delete(card["rectangle"])
            card["rectangle"] = None
        self.footer.set("Đã đặt giám sát toàn khung hình.")

    def roi_start(self, event, idx):
        self.source_tree.selection_set(str(idx))
        card = self.cards[idx]
        if self.roi_armed and card["mapping"]:
            card["drag"] = (event.x, event.y)
            if card["rectangle"]:
                card["canvas"].delete(card["rectangle"])
            card["rectangle"] = card["canvas"].create_rectangle(event.x, event.y, event.x, event.y, outline="#f2cf69", width=2)

    def roi_drag(self, event, idx):
        card = self.cards[idx]
        if card["drag"]:
            card["canvas"].coords(card["rectangle"], *card["drag"], event.x, event.y)

    def roi_end(self, event, idx):
        card = self.cards[idx]
        if not card["drag"] or not card["mapping"]:
            return
        ax, ay = card["drag"]
        ox, oy, dw, dh = card["mapping"]
        coords = [max(0, min(1, (min(ax, event.x)-ox)/dw)), max(0, min(1, (min(ay, event.y)-oy)/dh)),
                  max(0, min(1, (max(ax, event.x)-ox)/dw)), max(0, min(1, (max(ay, event.y)-oy)/dh))]
        card["drag"] = None
        if coords[2]-coords[0] < .05 or coords[3]-coords[1] < .05:
            self.footer.set("Vùng quá nhỏ; kéo lại vùng rộng hơn.")
            return
        if getattr(self,'zone_armed',False):
            zones=list(self.cfg['sources'][idx].get('hidden_zones',[]))+[coords]
            self.update_hidden_zones(idx,zones)
            self.roi_armed=False;self.zone_armed=False
            card['canvas'].delete(card['rectangle']);card['rectangle']=None
            self.footer.set('Đã đánh dấu góc khuất. Lưu cài đặt để giữ cho lần sau.')
            return
        self.cfg["sources"][idx]["roi"] = coords
        if self.monitor:
            self.monitor.workers[idx].set_roi(coords)
        self.roi_armed = False
        self.source_tree.item(str(idx), values=(self.cfg["sources"][idx]["label"], redact(self.cfg["sources"][idx]["source"]), "Đã chọn"))
        self.footer.set("Đã áp dụng vùng giám sát. Bấm Lưu cài đặt để giữ cho lần sau.")

    def update_display(self):
        old_privacy = self.cfg["display"]["privacy"]
        self.cfg["display"].update({key: variable.get() for key, variable in self.display_vars.items()})
        if self.cfg["display"]["privacy"] and not old_privacy:
            for idx in self.cards:
                self.blank_camera(idx)
        if self.monitor and old_privacy != self.cfg["display"]["privacy"]:
            for worker in self.monitor.workers:
                worker.command("reset")
        if self.cfg["display"]["privacy"]:
            self.footer.set("Che vùng đầu ước lượng từ hộp người; có thể bỏ sót. Ảnh/clip đã lưu trước đó giữ nguyên.")

    def preload_ai(self):
        if self.closing or self.smoke_seconds or self.monitor or self.starting or self.stopping:
            return
        if self.preloading or self.task_thread and self.task_thread.is_alive():
            self.preload_requested = True
            return
        self.preloading, self.preload_requested = True, False
        generation = self.model_generation
        snapshot = copy.deepcopy(self.cfg)
        self.health.set("Đang nạp AI nền · chưa mở camera")
        def load():
            try:
                bundle = self.model_cache.get(snapshot, lambda msg: self.bus.put(("ai_progress", (generation, msg))))
                self.bus.put(("ai_loaded", (generation, bundle.status())))
            except LoadingCancelled:
                self.bus.put(("ai_loaded", (generation, "Đã hủy lần nạp AI cũ")))
            except Exception as exc:
                self.bus.put(("ai_loaded", (generation, "Nạp AI thất bại: " + redact(exc))))
        self.preload_thread = threading.Thread(target=load, daemon=True, name="ai-preload")
        self.preload_thread.start()

    def invalidate_models(self):
        self.model_generation += 1
        self.model_cache.clear()
        self.preload_requested = True

    def start(self):
        if self.closing or self.monitor or self.starting or self.stopping or self.task_thread and self.task_thread.is_alive():
            self.footer.set("Đang có giám sát/tác vụ hoạt động; chờ hoặc dừng tác vụ trước.")
            return
        if not self.apply_settings():
            return
        self.starting = True
        self.boot_generation += 1
        generation = self.boot_generation
        self.start_cancel = cancel = threading.Event()
        self.start_button.configure(state="disabled")
        self.health.set("Đang nạp mô hình AI…")
        def boot():
            try:
                monitor = Monitor(self.cfg, self.store, self.offline, engine_cache=self.model_cache,
                                  progress=lambda msg: self.bus.put(("boot_progress", (generation, msg))), cancel=cancel)
                self.bus.put(("ready", (generation, monitor)))
            except LoadingCancelled:
                self.bus.put(("boot_cancelled", generation))
            except Exception as exc:
                self.bus.put(("error", (generation, redact(exc))))
        self.boot_thread = threading.Thread(target=boot, daemon=True, name="monitor-boot")
        self.boot_thread.start()

    def stop(self):
        if self.starting:
            self.start_cancel.set()
            self.health.set("Đã hủy khởi động camera — đang chờ tải mô hình kết thúc")
        if not self.monitor or self.stopping:
            return
        self.stopping = True
        self.start_button.configure(state="disabled")
        monitor = self.monitor
        monitor.request_stop()
        self.health.set("Đang đóng nguồn và hoàn tất clip…")
        def finish():
            try:
                monitor.close()
                self.bus.put(("stopped", monitor))
            except Exception as exc:
                # Không báo đã dừng nếu quyền sở hữu phiên chưa kết thúc.
                self.bus.put(("shutdown_error", redact(exc)))
        threading.Thread(target=finish, daemon=True, name="monitor-shutdown").start()

    def toggle_pause(self):
        if self.monitor:
            worker = self.monitor.workers[self.selected_index()]
            if worker.pause.is_set():
                worker.command("reset")
                worker.pause.clear()
                if worker.capture:
                    worker.capture.pause.clear()
            else:
                worker.pause.set()
                if worker.capture and not worker.capture.stream:
                    worker.capture.pause.set()

    def command_selected(self, command):
        if self.monitor and not self.stopping:
            self.monitor.workers[self.selected_index()].command(command)
        else:
            self.footer.set("Bắt đầu giám sát trước khi chụp ảnh hoặc ghi SOS.")

    def build_events_tab(self, parent):
        ttk, tk = self.ttk, self.tk
        tools = ttk.Frame(parent)
        tools.pack(fill="x", pady=(0, 10))
        self.search = tk.StringVar()
        self.filter_status = tk.StringVar(value="Tất cả")
        ttk.Label(tools, text="Tìm camera / nội dung:").pack(side="left")
        ttk.Entry(tools, textvariable=self.search, width=28).pack(side="left", padx=8)
        ttk.Combobox(tools, textvariable=self.filter_status, values=("Tất cả", *EventStore.STATUSES), state="readonly", width=17).pack(side="left", padx=8)
        ttk.Button(tools, text="Lọc / Làm mới", command=self.refresh_events).pack(side="left")
        ttk.Button(tools, text="Xuất CSV / JSON / HTML", command=self.export_events).pack(side="right")
        self.stats_var = tk.StringVar()
        ttk.Label(parent, textvariable=self.stats_var, foreground="#73d9b3").pack(fill="x", pady=(0, 10))
        cols = ("time", "camera", "kind", "score", "people", "status", "clip")
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        self.event_tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")
        for col, title, width in zip(cols, ("Thời gian", "Camera", "Sự kiện", "Điểm AI", "Người", "Trạng thái", "Clip bằng chứng"), (180, 140, 150, 80, 65, 120, 200)):
            self.event_tree.heading(col, text=title)
            self.event_tree.column(col, width=width)
        scroll = ttk.Scrollbar(frame, command=self.event_tree.yview)
        self.event_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.event_tree.pack(fill="both", expand=True)
        self.event_tree.bind("<<TreeviewSelect>>", self.event_selected)
        self.event_detail = tk.StringVar(value="Chọn sự kiện để xem lý do và bằng chứng.")
        ttk.Label(parent, textvariable=self.event_detail, wraplength=1180, foreground="#a9bfce").pack(fill="x", pady=10)
        actions = ttk.Frame(parent)
        actions.pack(fill="x")
        quick = ttk.Frame(parent)
        quick.pack(fill="x", pady=8)
        ttk.Button(quick, text="Đúng là xô xát", command=lambda: self.review_event("Đúng là xô xát"), style="Accent.TButton").pack(side="left")
        ttk.Button(quick, text="Báo nhầm", command=lambda: self.review_event("Báo nhầm")).pack(side="left", padx=8)
        ttk.Button(quick, text="Xuất mẫu báo nhầm để duyệt huấn luyện", command=self.export_false_alarms).pack(side="left", padx=8)
        ttk.Button(quick,text='Duyệt mẫu để AI học',command=self.approve_learning).pack(side='left',padx=8)
        ttk.Label(parent, text=HUMAN_GUIDANCE, wraplength=1150, foreground="#e8c481").pack(fill="x", pady=5)
        self.review_status = tk.StringVar(value="Chưa xem")
        ttk.Combobox(actions, textvariable=self.review_status, values=EventStore.STATUSES, state="readonly", width=17).pack(side="left")
        self.notes = tk.StringVar()
        ttk.Entry(actions, textvariable=self.notes, width=52).pack(side="left", padx=8)
        ttk.Button(actions, text="Lưu đánh giá / ghi chú", command=self.review_event).pack(side="left")
        ttk.Button(actions, text="Xem ảnh", command=lambda: self.open_evidence("snapshot")).pack(side="left", padx=8)
        ttk.Button(actions, text="Phát clip", command=lambda: self.open_evidence("clip")).pack(side="left")
        ttk.Button(actions,text='Clip 3 giây',command=lambda:self.open_evidence('quick_clip')).pack(side='left',padx=8)

    def refresh_events(self):
        selected = self.event_tree.selection()
        self.event_rows = {r["id"]: r for r in self.store.rows(self.search.get(), self.filter_status.get())}
        self.event_tree.delete(*self.event_tree.get_children())
        for eid, row in self.event_rows.items():
            self.event_tree.insert("", "end", iid=eid, values=(row["created"].replace("T", " ")[:19], row["camera"], row["kind"],
                                   f"{row['score']:.1%}" if row["score"] is not None else "—", row["people"], row["status"], row["clip_state"]))
        if selected and selected[0] in self.event_rows:
            self.event_tree.selection_set(selected[0])
        stats = self.store.stats()
        self.stats_var.set(" · ".join(f"{name}: {stats.get(name, 0)}" for name in EventStore.STATUSES) + " · Hiển thị tối đa 500 dòng; báo cáo xuất toàn bộ kết quả lọc")

    def event_selected(self, event=None):
        choice = self.event_tree.selection()
        if choice and choice[0] in self.event_rows:
            row = self.event_rows[choice[0]]
            self.review_status.set(row["status"])
            self.notes.set(row["notes"])
            self.event_detail.set(f"Mã {row['id']} · {row['reason']} · Vị trí video {row['position']:.1f}s · Mô hình {row['model']}")

    def review_event(self, status=None):
        choice = self.event_tree.selection()
        if choice:
            try:
                self.store.update(choice[0], status=status or self.review_status.get(), notes=self.notes.get())
                self.refresh_events()
                self.footer.set("Đã lưu đánh giá của người phụ trách và nhật ký thay đổi.")
            except Exception as exc:
                self.messagebox.showerror("Chưa lưu đánh giá", redact(exc))
        else:
            self.footer.set("Chọn sự kiện và xem bằng chứng trước khi đánh giá.")

    def export_false_alarms(self):
        from evaluation_workflow import export_false_alarms
        target = self.store.folder / "hard_negatives" / datetime.now(TZ).strftime("%Y%m%d_%H%M%S_%f")
        self.background_task("Xuất mẫu báo nhầm chờ duyệt", lambda emit: export_false_alarms(self.store, target))

    def approve_learning(self):
        from learning_feedback import approve,CATEGORIES,candidate_media
        chosen=self.event_tree.selection()
        if not chosen:return
        row=self.store.get(self.event_rows[chosen[0]]['id'])
        media=candidate_media(self.store,row)
        if row['status']!='Báo nhầm' or media is None:
            self.messagebox.showerror('Chưa duyệt được','Chọn sự kiện Báo nhầm đã có clip hoàn tất.');return
        window=self.tk.Toplevel(self.root);window.title('Duyệt mẫu học từ báo nhầm')
        self.ttk.Label(window,text='Xem toàn bộ clip này trước khi duyệt:\n'+str(media),wraplength=650).pack(padx=15,pady=12)
        reviewer=self.tk.StringVar();category=self.tk.StringVar(value='play')
        self.ttk.Label(window,text='Tên thầy/cô duyệt:').pack()
        self.ttk.Entry(window,textvariable=reviewer,width=40).pack(padx=15,pady=5)
        self.ttk.Label(window,text='Loại cảnh: play=đùa giỡn, sports=thể thao, crowd=chen lấn, fall=giúp người ngã').pack(padx=15)
        self.ttk.Combobox(window,textvariable=category,values=CATEGORIES,state='readonly',width=40).pack(padx=15,pady=5)
        checked=self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(window,text='Tôi đã xem toàn clip và xác nhận đoạn này không có xô xát',variable=checked).pack(padx=15,pady=12)
        def submit():
            if not checked.get():return
            try:
                result=approve(self.store,row['id'],reviewer.get(),category.get())
                self.footer.set(f"Đã duyệt mẫu học; {result['approved']} mẫu hợp lệ trong hộp học. Mô hình mới vẫn cần kiểm tra trước khi thay.")
                window.destroy()
            except ValueError as exc:self.messagebox.showerror('Chưa duyệt được',str(exc),parent=window)
        self.ttk.Button(window,text='Duyệt đưa vào bộ học',command=submit).pack(padx=15,pady=12)

    def export_events(self):
        filename = self.dialog.asksaveasfilename(title="Xuất lịch sử sự kiện", initialfile="bao_cao_schoolguard.csv", defaultextension=".csv",
                                                filetypes=[("CSV", "*.csv"), ("JSON", "*.json"), ("Báo cáo HTML", "*.html")])
        if filename:
            try:
                count = self.store.export(Path(filename), self.search.get(), self.filter_status.get())
                self.footer.set(f"Đã xuất {count} sự kiện: {filename}")
            except Exception as exc:
                self.messagebox.showerror("Không xuất được báo cáo", redact(exc))

    def open_evidence(self, key):
        chosen = self.event_tree.selection()
        if not chosen:
            return
        row = self.event_rows[chosen[0]]
        value = str(self.store.folder/'quick_clips'/(row['id']+'.mp4')) if key=='quick_clip' else row[key]
        if key == "clip" and row["clip_state"] == "Đang ghi":
            self.footer.set("Clip đang ghi; chờ hoàn tất trước khi phát.")
            return
        if not value or not Path(value).is_file():
            self.footer.set("Sự kiện chưa có bằng chứng này hoặc tệp đã bị di chuyển.")
            return
        target = Path(value).resolve()
        if not target.is_relative_to(self.store.folder):
            self.footer.set("Tệp bằng chứng nằm ngoài thư mục dữ liệu hiện tại.")
            return
        window = self.tk.Toplevel(self.root)
        window.title("Bằng chứng · " + row["camera"])
        label = self.ttk.Label(window)
        label.pack(fill="both", expand=True)
        if key == "snapshot":
            image = self.Image.open(target)
            image.thumbnail((1000, 680))
            photo = self.ImageTk.PhotoImage(image)
            label.configure(image=photo)
            label.image = photo
        else:
            runtime_imports()
            cap = cv2.VideoCapture(str(target))
            fps = cap.get(cv2.CAP_PROP_FPS) or 5
            alive = [True]
            def close_video():
                alive[0] = False
                cap.release()
                window.destroy()
            window.protocol("WM_DELETE_WINDOW", close_video)
            def advance():
                if not alive[0] or not window.winfo_exists():
                    cap.release()
                    return
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    return
                image = self.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((1000, 680))
                photo = self.ImageTk.PhotoImage(image)
                label.configure(image=photo)
                label.image = photo
                window.after(max(10, int(1000/fps)), advance)
            advance()

    def build_settings_tab(self, parent):
        # Nút Áp dụng/Lưu luôn thấy được; các trường dài cuộn trong phần còn lại.
        actions = self.ttk.Frame(parent)
        actions.pack(side="bottom", fill="x", pady=(8, 0))
        body = self.ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        self.settings_canvas = self.tk.Canvas(body, background="#172330", highlightthickness=0, yscrollincrement=24)
        scroll = self.ttk.Scrollbar(body, orient="vertical", command=self.settings_canvas.yview)
        scroll.pack(side="right", fill="y")
        self.settings_canvas.configure(yscrollcommand=scroll.set)
        self.settings_canvas.pack(side="left", fill="both", expand=True)
        parent = self.ttk.Frame(self.settings_canvas)
        content = self.settings_canvas.create_window(0, 0, anchor="nw", window=parent)
        parent.bind("<Configure>", lambda event: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.bind("<Configure>", lambda event: self.settings_canvas.itemconfigure(content, width=event.width))
        self.setting_vars = {}
        fields = [("model", "backend", "Bộ suy luận (auto / resnet / keras)"), ("model", "path", "Checkpoint ResNet (.pt)"),
                  ("model", "keras_path", "Mô hình Keras (.h5)"), ("detection", "threshold", "Ngưỡng điểm AI cảnh báo (0–1; không phải độ chính xác)"),
                  ("detection", "interaction_threshold", "Ngưỡng AI khi đã có tương tác mạnh lặp lại (0–1)"),
                      ("detection", "release_threshold", "Ngưỡng kết thúc nguy cơ (0–1; nhỏ hơn cảnh báo)"),
                  ("detection", "sustain_seconds", "Duy trì nguy cơ trước cảnh báo (giây)"),
                  ("detection", "cooldown_seconds", "Khoảng cách giữa các cảnh báo (giây)"),
                  ("detection", "analysis_fps", "Tần suất phân tích (khung/giây)"),
                  ("evidence", "pre_seconds", "Clip: thời gian trước cảnh báo (giây)"),
                  ("evidence", "post_seconds", "Clip: thời gian sau cảnh báo (giây)")]
        for i, (section, key, title) in enumerate(fields):
            self.ttk.Label(parent, text=title).grid(row=i, column=0, sticky="w", padx=8, pady=7)
            value = self.tk.StringVar(value=str(self.cfg[section][key]))
            self.setting_vars[(section, key)] = value
            self.ttk.Entry(parent, textvariable=value, width=58).grid(row=i, column=1, sticky="ew", padx=8)
        self.flag_vars = {}
        self.person_profile_var = self.tk.StringVar(value=person_profile_for(self.cfg["person"]))
        self.ttk.Label(parent, text="Chế độ khoanh vùng người").grid(row=len(fields), column=0, sticky="w", padx=8, pady=7)
        self.person_profile_box = self.ttk.Combobox(parent, textvariable=self.person_profile_var,
            values=(*PERSON_PROFILES, "Tùy chỉnh"), state="readonly", width=32)
        self.person_profile_box.grid(row=len(fields), column=1, sticky="w", padx=8)
        self.person_profile_box.bind("<<ComboboxSelected>>", self.person_profile_selected)
        for i, (section, key, title) in enumerate([
            ("detection", "require_interaction", "Cần chuỗi tương tác liên tiếp để cảnh báo tự động"),
            ("person", "body_occlusion", "Tăng cường nhận diện người bị che khuất"),
            ("person", "enhanced", "Dò bổ sung người mờ/che khuất (thêm tải CPU; FPS AI có thể giảm)"),
            ("evidence", "enabled", "Lưu clip trước / sau sự kiện"),
            ("evidence", "continuous", "Ghi video liên tục theo từng đoạn"),
            ("notifications", "enabled", "Gửi Discord / Telegram bằng thông tin đã cấu hình")]):
            value = self.tk.BooleanVar(value=self.cfg[section][key])
            self.flag_vars[(section, key)] = value
            self.ttk.Checkbutton(parent, text=title, variable=value).grid(row=len(fields)+i+1, column=0, columnspan=2, sticky="w", pady=6, padx=8)
        self.apply_settings_button = self.ttk.Button(actions, text="Áp dụng", command=self.apply_settings)
        self.apply_settings_button.pack(side="left", padx=8)
        self.save_settings_button = self.ttk.Button(actions, text="Lưu cài đặt", command=self.save_settings)
        self.save_settings_button.pack(side="left", padx=8)
        self.ttk.Button(actions, text="Kiểm tra hệ thống", command=self.show_doctor).pack(side="left", padx=8)
        self.ttk.Label(actions, text="AI có thể báo nhầm hoặc bỏ sót; cần người xác minh.", foreground="#a9bfce").pack(side="left", padx=8)
        self.ttk.Label(parent, text="Thông báo dùng SCHOOLGUARD_DISCORD_WEBHOOK, SCHOOLGUARD_TELEGRAM_TOKEN và SCHOOLGUARD_TELEGRAM_CHAT_ID.\n"
                       "Có thể dùng thông tin đã mã hóa cục bộ. Đổi mô hình / thông báo sau khi dừng giám sát.",
                       foreground="#a9bfce", wraplength=1050).grid(row=len(fields)+8, column=0, columnspan=2, sticky="w", padx=8)
        row=len(fields)+9
        self.policy_mode_var=self.tk.StringVar(value=MODE_LABELS[self.cfg['school_policy']['mode']])
        self.ttk.Label(parent,text='Chế độ theo giờ trường').grid(row=row,column=0,sticky='w',padx=8,pady=9)
        self.ttk.Combobox(parent,textvariable=self.policy_mode_var,values=list(MODE_LABELS.values()),state='readonly',width=35).grid(row=row,column=1,sticky='w',padx=8)
        self.quick_clip_var=self.tk.BooleanVar(value=self.cfg['quick_clip']['enabled'])
        self.ttk.Checkbutton(parent,text='Gửi thêm clip ngữ cảnh 3 giây qua Telegram / Discord',variable=self.quick_clip_var).grid(row=row+2,column=0,columnspan=2,sticky='w',padx=8,pady=6)
        self.ttk.Button(parent,text='Chỉnh lịch trường và tuần tra…',command=self.edit_school_schedule).grid(row=row+3,column=0,sticky='w',padx=8,pady=6)
        self.policy_editor=self.tk.Text(parent,height=17,width=90,wrap='none',font=('Consolas',10))
        advanced={'school_policy':self.cfg['school_policy'],'hidden_zones':{s['label']:s.get('hidden_zones',[]) for s in self.cfg['sources']},
                  'feedback':self.cfg['feedback']}
        self.policy_editor.insert('1.0',json.dumps(advanced,ensure_ascii=False,indent=2))
        self.policy_editor.grid(row=row+4,column=0,columnspan=2,sticky='ew',padx=8,pady=6)
        self.policy_editor.grid_remove()
        def toggle_advanced():
            if self.policy_editor.winfo_ismapped():self.policy_editor.grid_remove()
            else:self.policy_editor.grid()
        self.ttk.Button(parent,text='Hiện/ẩn cài đặt chuyên sâu',command=toggle_advanced).grid(row=row+3,column=1,sticky='w',padx=8)
        self.ttk.Label(parent,text='Bấm Vẽ góc khuất ở tab Giám sát để chọn trực tiếp trên hình camera.\n'
            'Chỉnh lịch mẫu đúng giờ trường trước khi bật. Tên thầy/cô là danh sách lựa chọn, không xác thực tài khoản.',
            wraplength=1000,foreground='#a9bfce').grid(row=row+5,column=0,columnspan=2,sticky='w',padx=8,pady=6)
        self.phone_base_var=self.tk.StringVar(value=self.cfg['feedback']['public_base_url'])
        self.ttk.Label(parent,text='Địa chỉ máy để điện thoại phản hồi (ví dụ http://192.168.1.20:8765)').grid(row=row+6,column=0,sticky='w',padx=8,pady=8)
        self.ttk.Entry(parent,textvariable=self.phone_base_var,width=58).grid(row=row+6,column=1,sticky='ew',padx=8)
        parent.columnconfigure(1, weight=1)
        def wheel(event):
            if event.delta:
                direction = -1 if event.delta > 0 else 1
                self.settings_canvas.yview_scroll(direction * max(1, abs(event.delta)//120), "units")
                return "break"
        def bind_wheel(widget):
            widget.bind("<MouseWheel>", wheel)
            for child in widget.winfo_children():
                bind_wheel(child)
        bind_wheel(parent)
        self.settings_canvas.bind("<MouseWheel>", wheel)

    def person_profile_selected(self, event=None):
        if self.person_profile_var.get() == "Camera mờ (ưu tiên chi tiết)":
            self.footer.set("Camera mờ: phân tích nhiều chi tiết hơn, CPU xử lý lâu hơn và FPS AI có thể giảm. Dừng giám sát rồi bấm Áp dụng.")
        else:
            self.footer.set("Bấm Áp dụng để dùng chế độ khoanh vùng đã chọn; Tùy chỉnh giữ trọng số và cỡ ảnh hiện tại.")

    def edit_school_schedule(self):
        try:advanced=json.loads(self.policy_editor.get('1.0','end'));policy=copy.deepcopy(advanced['school_policy'])
        except (ValueError,KeyError,TypeError):
            self.messagebox.showerror('Cài đặt chưa hợp lệ','Sửa cài đặt chuyên sâu về JSON hợp lệ trước.');return
        win=self.tk.Toplevel(self.root);win.title('Lịch trường và kiểm tra định kỳ')
        self.ttk.Label(win,text='Giờ theo Việt Nam, dạng HH:MM. Các khoảng không được chồng nhau.').grid(row=0,column=0,columnspan=5,padx=14,pady=10)
        for c,title in enumerate(('Dùng','Từ','Đến','Chế độ','Gửi ảnh tuần tra đầu giờ')):self.ttk.Label(win,text=title).grid(row=1,column=c,padx=10)
        rows=[]
        for i in range(max(8,len(policy['periods'])+2)):
            item=policy['periods'][i] if i<len(policy['periods']) else {}
            enabled=self.tk.BooleanVar(value=bool(item));start=self.tk.StringVar(value=item.get('start',''))
            end=self.tk.StringVar(value=item.get('end',''));mode=self.tk.StringVar(value=MODE_LABELS[item.get('mode','strict')]);patrol=self.tk.BooleanVar(value=item.get('patrol',False))
            self.ttk.Checkbutton(win,variable=enabled).grid(row=i+2,column=0)
            self.ttk.Entry(win,textvariable=start,width=8).grid(row=i+2,column=1,padx=8,pady=4)
            self.ttk.Entry(win,textvariable=end,width=8).grid(row=i+2,column=2,padx=8)
            self.ttk.Combobox(win,textvariable=mode,values=[MODE_LABELS[k] for k in ('standard','strict','anti_false')],state='readonly',width=24).grid(row=i+2,column=3,padx=8)
            self.ttk.Checkbutton(win,variable=patrol).grid(row=i+2,column=4)
            rows.append((enabled,start,end,mode,patrol))
        r=len(rows)+2;days=[];dayframe=self.ttk.Frame(win);dayframe.grid(row=r,column=0,columnspan=5,pady=8)
        for i,name in enumerate(('T2','T3','T4','T5','T6','T7','CN')):
            v=self.tk.BooleanVar(value=i in policy['days']);days.append(v);self.ttk.Checkbutton(dayframe,text=name,variable=v).pack(side='left',padx=6)
        fallback=self.tk.StringVar(value=MODE_LABELS[policy['default_mode']])
        self.ttk.Label(win,text='Ngoài các khoảng trên:').grid(row=r+1,column=0,columnspan=2,padx=8)
        self.ttk.Combobox(win,textvariable=fallback,values=[MODE_LABELS[k] for k in ('standard','strict','anti_false')],state='readonly',width=24).grid(row=r+1,column=2,columnspan=2,pady=6)
        patrol_enabled=self.tk.BooleanVar(value=policy['patrol_enabled']);health_enabled=self.tk.BooleanVar(value=policy['camera_health'])
        self.ttk.Checkbutton(win,text='Bật gửi ảnh kiểm tra định kỳ',variable=patrol_enabled).grid(row=r+2,column=0,columnspan=3,sticky='w',padx=12)
        self.ttk.Checkbutton(win,text='Báo camera bất thường',variable=health_enabled).grid(row=r+2,column=3,columnspan=2,sticky='w')
        times=self.tk.StringVar(value='; '.join(policy['patrol_times']))
        self.ttk.Label(win,text='Giờ tuần tra thêm/cuối ngày (ngăn bằng ;)').grid(row=r+3,column=0,columnspan=3,padx=8,pady=8)
        self.ttk.Entry(win,textvariable=times,width=25).grid(row=r+3,column=3,columnspan=2,padx=8)
        def save_schedule():
            try:
                reverse={v:k for k,v in MODE_LABELS.items()}
                policy['periods']=[{'start':a.get().strip(),'end':b.get().strip(),'mode':reverse[m.get()],'patrol':p.get()} for e,a,b,m,p in rows if e.get()]
                policy['days']=[i for i,v in enumerate(days) if v.get()];policy['default_mode']=reverse[fallback.get()]
                policy['patrol_enabled']=patrol_enabled.get();policy['camera_health']=health_enabled.get()
                policy['patrol_times']=[v.strip() for v in times.get().split(';') if v.strip()]
                validate_policy(policy,self.cfg['sources']);advanced['school_policy']=policy
                self.policy_editor.delete('1.0','end');self.policy_editor.insert('1.0',json.dumps(advanced,ensure_ascii=False,indent=2))
                self.footer.set('Đã chỉnh lịch. Bấm Áp dụng hoặc Lưu cài đặt để dùng.');win.destroy()
            except (ValueError,KeyError,TypeError) as exc:self.messagebox.showerror('Lịch chưa hợp lệ',str(exc),parent=win)
        self.ttk.Button(win,text='Dùng lịch này',command=save_schedule).grid(row=r+4,column=0,columnspan=5,pady=14)

    def apply_settings(self):
        candidate = copy.deepcopy(self.cfg)
        try:
            for (section, key), variable in self.setting_vars.items():
                original = DEFAULT_CONFIG[section][key]
                candidate[section][key] = float(variable.get()) if isinstance(original, (int, float)) else variable.get().strip()
            for (section, key), variable in self.flag_vars.items():
                candidate[section][key] = variable.get()
            profile = self.person_profile_var.get()
            if profile in PERSON_PROFILES:
                path, size = PERSON_PROFILES[profile]
                if resolve_path(candidate["person"]["path"]) != resolve_path(path):
                    candidate["person"]["path"] = path
                candidate["person"]["imgsz"] = size
            elif profile != "Tùy chỉnh":
                raise ValueError("Chọn một chế độ khoanh vùng người trong danh sách.")
            advanced=json.loads(self.policy_editor.get('1.0','end'))
            if not isinstance(advanced,dict) or not all(isinstance(advanced.get(k),dict) for k in ('school_policy','feedback','hidden_zones')):
                raise ValueError('JSON cần school_policy, feedback và hidden_zones là các đối tượng.')
            candidate['school_policy']=advanced['school_policy']
            candidate['school_policy']['mode']=next(k for k,v in MODE_LABELS.items() if v==self.policy_mode_var.get())
            candidate['feedback']=advanced['feedback']
            candidate['feedback']['public_base_url']=self.phone_base_var.get().strip()
            candidate['feedback']['host']='0.0.0.0' if candidate['feedback']['public_base_url'] else '127.0.0.1'
            candidate['quick_clip']['enabled']=self.quick_clip_var.get()
            for spec in candidate['sources']:
                zones=advanced['hidden_zones'].get(spec['label'],[])
                if zones or 'hidden_zones' in spec:
                    spec['hidden_zones']=zones
            validate_config(candidate)
            feedback_changed=candidate['feedback']!=self.cfg['feedback']
            if self.monitor or self.starting or self.stopping:
                if any(candidate[s] != self.cfg[s] for s in ("model", "person", "evidence", "notifications",'school_policy','quick_clip','sources','feedback')):
                    raise ValueError("Dừng giám sát trước khi đổi mô hình, lưu clip hoặc kênh thông báo.")
            models_changed = any(candidate[s] != self.cfg[s] for s in ("model", "person"))
            for section in ("model", "person", "detection", "evidence", "notifications",'school_policy','quick_clip','feedback'):
                self.cfg[section].update(candidate[section])
            self.cfg['sources']=candidate['sources']
            if models_changed:
                self.invalidate_models()
            self.footer.set("Đã bật dò bổ sung người; thêm tải CPU, FPS AI có thể giảm."
                            if candidate["person"].get("enhanced", False) else
                            ("Đã áp dụng chế độ Camera mờ; ưu tiên chi tiết, FPS AI có thể giảm."
                             if profile == "Camera mờ (ưu tiên chi tiết)" else "Đã áp dụng cài đặt."))
            if feedback_changed:self.footer.set('Đã áp dụng. Lưu cài đặt rồi mở lại ứng dụng để đổi máy chủ phản hồi trên điện thoại.')
            return True
        except (ValueError, TypeError, KeyError, StopIteration) as exc:
            self.messagebox.showerror("Cài đặt chưa hợp lệ", redact(exc))
            return False

    def save_settings(self):
        if self.apply_settings():
            try:
                atomic_json(ROOT / "main2_config.json", self.cfg)
                self.footer.set("Đã lưu main2_config.json; thông tin nguồn camera được giữ trong tệp cục bộ này.")
            except OSError as exc:
                self.messagebox.showerror("Không lưu được cấu hình", redact(exc))
    
    def show_doctor(self):
        report = doctor(self.cfg)
        lines = [f"Python {report['python']}", report["executable"], ""]
        lines.extend(f"{'✓' if ok else '—'} {name}" for name, ok in report["modules"].items())
        lines += ["", "— là thư viện chưa cài; chức năng liên quan sẽ không khả dụng."]
        self.messagebox.showinfo("Kiểm tra hệ thống", "\n".join(lines))


    def create_holdout(self):
        from evaluation_workflow import init_manifest
        folder = self.dialog.askdirectory(title="Chọn 20–30 video mới chưa dùng để chỉnh hệ thống")
        if not folder:
            return
        output = self.dialog.asksaveasfilename(title="Lưu danh sách nhãn để kiểm tra", initialfile="holdout.json", defaultextension=".json")
        if output:
            self.background_task("Tạo danh sách kiểm tra; cần điền nhãn trước khi khóa", lambda emit: init_manifest(folder, output))

    def lock_holdout(self):
        from evaluation_workflow import lock_manifest
        manifest = self.dialog.askopenfilename(title="Chọn danh sách đã điền nhãn/nguồn/người xem", filetypes=[("JSON", "*.json")])
        if manifest:
            config = copy.deepcopy(self.cfg)
            self.background_task("Kiểm tra nguồn và khóa tập đánh giá", lambda emit: lock_manifest(manifest, config))

    def run_holdout(self):
        from evaluation_workflow import run_evaluation
        manifest = self.dialog.askopenfilename(title="Chọn tập kiểm tra đã khóa (chỉ chạy một lần)", filetypes=[("JSON", "*.json")])
        if manifest:
            config = copy.deepcopy(self.cfg)
            output = self.store.folder / "reports" / ("holdout_" + datetime.now(TZ).strftime("%Y%m%d_%H%M%S_%f"))
            self.background_task("Đánh giá tập video riêng", lambda emit: run_evaluation(manifest, output, config, emit, self.task_cancel))

    def analyze_directory(self, folder=None):
        if not folder:
            folder = self.dialog.askdirectory(title="Soi tất cả video trong thư mục")
        if folder:
            config = copy.deepcopy(self.cfg)
            self.background_task("Phân tích thư mục", lambda emit: analyze_folder(Path(folder), config, emit, self.task_cancel))

    def run_fresh_eval(self):
        folder = Path(r"E:\.gemini\antigravity\scratch\violence_detection\fresh_eval_videos_20261017")
        if not folder.is_dir():
            folder = ROOT / "fresh_eval_videos_20261017"
        if not folder.is_dir():
            self.messagebox.showerror("Thư mục video", f"Không tìm thấy thư mục:\n{folder}")
            return
        if not self.task_available():
            return
        config = copy.deepcopy(self.cfg)
        def work(emit):
            result = analyze_folder(folder, config, emit, self.task_cancel)
            try:
                report_dir = Path(result["report_folder"])
                summary_html = report_dir / "summary.html"
                rows_html = []
                for idx, r in enumerate(result.get("results", []), 1):
                    v_name = html.escape(r.get("video", ""))
                    dec = html.escape(r.get("decision", ""))
                    risk = html.escape(r.get("risk", ""))
                    peak = f"{r.get('visual_peak', 0.0):.1%}" if r.get('visual_peak') is not None else "—"
                    alerts = r.get("automatic_alerts", 0)
                    err = html.escape(r.get("error", ""))
                    color = "#e57373" if risk == "high" or alerts > 0 else ("#81c784" if risk == "low" else "#ffb74d")
                    rows_html.append(f"<tr><td>{idx}</td><td style='font-weight:bold;'>{v_name}</td><td style='color:{color};font-weight:bold;'>{dec}</td><td>{risk}</td><td>{peak}</td><td>{alerts}</td><td style='color:#e57373;'>{err}</td></tr>")
                html_content = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Kết quả kiểm tra {result.get('videos', len(result.get('results', [])))} video (fresh_eval)</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #101720; color: #e5edf5; margin: 0; padding: 24px; }}
h1 {{ color: #ffffff; border-bottom: 2px solid #294156; padding-bottom: 12px; }}
.info {{ background: #1a2634; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px; line-height: 1.6; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: #172330; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #253849; font-size: 14px; }}
th {{ background: #223445; color: #73d9b3; text-transform: uppercase; font-size: 13px; letter-spacing: 0.5px; }}
tr:hover {{ background: #1e2f40; }}
.btn {{ display: inline-block; padding: 8px 16px; background: #17785f; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; margin-right: 10px; }}
</style></head><body>
<h1>SchoolGuard · Kết quả kiểm tra {result.get('videos', len(result.get('results', [])))} video</h1>
<div class="info">
<p><strong>Thư mục nguồn:</strong> {html.escape(str(folder))}</p>
<p><strong>Số video đã xử lý:</strong> {result.get('processed', 0)} / {result.get('videos', 0)}</p>
<p><strong>Thời gian:</strong> {now_iso()}</p>
<p><a class="btn" href="summary.csv">Mở file CSV</a></p>
</div>
<table><thead><tr><th>STT</th><th>Video</th><th>Kết luận AI</th><th>Mức rủi ro</th><th>Điểm AI cao nhất</th><th>Số lần cảnh báo</th><th>Lỗi (nếu có)</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody></table></body></html>"""
                summary_html.write_text(html_content, encoding="utf-8")
                if os.name == "nt":
                    os.startfile(str(summary_html))
            except Exception:
                pass
            return result
        self.background_task("Đánh giá 20-30 video fresh_eval", work)

    def task_available(self):
        if self.closing or self.preloading or self.monitor or self.starting or self.stopping or self.task_thread and self.task_thread.is_alive():
            self.footer.set("Dừng giám sát hoặc chờ tác vụ hiện tại trước khi chạy công cụ.")
            return False
        return self.apply_settings()


    def background_task(self, name, action):
        if not self.task_available():
            return
        self.task_cancel.clear()
        self.footer.set(name + "…")
        def emit(message):
            self.bus.put(("task_progress", name + ": " + str(message)))
        def work():
            try:
                result = action(emit)
                target = self.store.folder / "reports" / ("task_" + uuid.uuid4().hex + ".json")
                atomic_json(target, result)
                self.bus.put(("task_done", (name, str(target))))
            except Exception as exc:
                self.bus.put(("task_error", redact(exc)))
        self.task_thread = threading.Thread(target=work, daemon=True)
        self.task_thread.start()


    def stop_task(self):
        if self.task_thread and self.task_thread.is_alive():
            self.task_cancel.set()
            self.footer.set("Đã yêu cầu dừng; đang hoàn tất và lưu kết quả.")

    def build_help_tab(self, parent):
        evaluation_bar = self.ttk.Frame(parent)
        evaluation_bar.pack(fill="x", pady=(0, 10))
        self.ttk.Button(evaluation_bar, text="Tạo tập kiểm tra riêng", command=self.create_holdout).pack(side="left")
        self.ttk.Button(evaluation_bar, text="Khóa nhãn tập kiểm tra", command=self.lock_holdout).pack(side="left", padx=8)
        self.ttk.Button(evaluation_bar, text="Đánh giá tập đã khóa", command=self.run_holdout).pack(side="left", padx=8)
        self.ttk.Button(evaluation_bar, text="⚡ Chạy 20-30 video kiểm tra", style="Accent.TButton", command=self.run_fresh_eval).pack(side="left", padx=8)
        text = """1. CẢNH BÁO VÀ XÁC MINH
AI có thể báo nhầm hoặc bỏ sót. Câu cảnh báo: Nghi có xô xát - cần người đến kiểm tra.
Tin nhắn có hai nút: Đúng là xô xát / Báo nhầm. Chỉ phản hồi sau khi xem bằng chứng.
AI không dùng để kết luận hay kỷ luật học sinh. Người lớn vẫn nên xem camera định kỳ.
Nút trong tin nhắn cần địa chỉ máy phản hồi truy cập được từ điện thoại.

2. TẬP VIDEO KIỂM TRA RIÊNG
Tách 20–30 đoạn chưa dùng để huấn luyện hoặc chỉnh hệ thống, gồm xô xát, đùa giỡn,
chơi thể thao, chen lấn và đỡ bạn bị ngã. Dùng ba nút phía trên để tạo danh sách,
điền nhãn/nguồn, khóa mô hình và chạy một lượt; báo cáo đếm báo đúng, báo nhầm, bỏ sót.
Nhãn nguồn/AI còn cần người phụ trách xem toàn clip. Tập đã dùng để xem lỗi không còn là tập mới.

3. XÉT DIỄN BIẾN VÀ BỐI CẢNH
Chuyển động mạnh hoặc hai người đứng sát chưa đủ để kết luận xô xát.
Hệ thống xét chuỗi hình ảnh, tương tác tay/chân liên tiếp và bối cảnh đi ngang/đám đông.
Cần thử cả xô xát ít chuyển động hoặc bị che khuất; thiếu bằng chứng vẫn có thể bỏ sót.
Mỗi camera giữ lịch sử riêng và xóa trạng thái khi đổi cảnh, mất nguồn hoặc đổi vùng giám sát.

4. HỌC THÊM TỪ CẢNH HAY NHẦM
Trong Sự kiện & bằng chứng, bấm Báo nhầm sau khi kiểm tra, rồi Xuất mẫu báo nhầm để duyệt huấn luyện.
Ưu tiên tập võ, chơi bóng, đùa giỡn, chen lấn và giúp người ngã. Chỉ đưa clip vào train sau khi
người phụ trách xem và duyệt; không tự gán mọi phản hồi thành nhãn huấn luyện hoặc tự bật mô hình.
Giữ video kiểm tra ngoài dữ liệu học và so sánh cả báo nhầm lẫn bỏ sót trước khi sử dụng bản mới.
"""
        box = self.tk.Text(parent, wrap="word", bg="#172330", fg="#d1e0ec", relief="flat", font=("Segoe UI", 11), padx=12, pady=10)
        box.pack(fill="both", expand=True)
        box.insert("end", text)
        box.configure(state="disabled")

    def blank_camera(self, idx):
        card = self.cards[idx]
        card["canvas"].delete("feed")
        card["image"], card["mapping"], card["render_key"] = None, None, None
        self.images.pop(idx, None)

    def render_camera(self, idx, view):
        card = self.cards[idx]
        spec = self.cfg["sources"][idx]
        now = time.monotonic()
        stamps = card["rendered_at"]
        while stamps and stamps[0] <= now-5:
            stamps.popleft()
        began = card["display_started_at"]
        card["display_fps"] = len(stamps)/max(.5, min(5, now-began)) if began is not None else 0
        score = f"{view.score:.1%}" if view.score is not None else "—"
        title = f"{spec['label']} · {view.status} · AI {score}"
        label_key = title, view.reason, len(view.boxes)
        if label_key != card["label_key"] or now-card["label_at"] >= .2:
            card["title"].set(title)
            card["detail"].set(f"{len(view.boxes)} hộp người · {card['display_fps']:.1f} FPS hiển thị · {view.fps:.1f} FPS AI · xử lý {view.latency:.0f} ms · tuổi AI {view.prediction_age*1000:.0f} ms · {view.position:.1f}s\n{view.reason}")
            card["label_key"], card["label_at"] = label_key, now
        display = self.cfg["display"]
        if view.frame is None or display["privacy"] and not view.privacy_applied:
            self.blank_camera(idx)
            return False
        canvas = card["canvas"]
        cw, ch = max(200, canvas.winfo_width()), max(140, canvas.winfo_height())
        roi = spec.get("roi")
        source_key = (view.source_seq, view.source_stamp) if view.source_seq >= 0 else (id(view.frame), view.source_stamp)
        render_key = (source_key, view.analysis_seq, view.annotation_epoch, view.annotations_valid,
                      view.preview, view.privacy_applied, view.alert_active, view.alert_track_ids,
                      cw, ch, display["boxes"], display["heatmap"],
                      display["privacy"], tuple(roi) if roi else None, tuple(tuple(z) for z in spec.get('hidden_zones',[])))
        if render_key == card["render_key"]:
            return False
        h, w = view.frame.shape[:2]
        scale = min(1, cw/w, ch/h)
        dw, dh = max(1, int(w*scale)), max(1, int(h*scale))
        frame = cv2.resize(view.frame, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1 else view.frame.copy()
        if display["heatmap"] and view.heatmap is not None:
            heatmap = cv2.resize(view.heatmap, (dw, dh), interpolation=cv2.INTER_AREA)
            frame = cv2.addWeighted(frame, .8, heatmap, .2, 0)
        if display["boxes"]:
            frame = draw_tactical_boxes(frame, view.boxes, w, h, view.alert_active, (view.score or 0)*100,
                                        alert_track_ids=view.alert_track_ids, previous_frame=view.preview)
        image = self.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Kiểm tra lại ngay trước khi đưa ảnh lên Tk.
        if self.cfg["display"]["privacy"] and not view.privacy_applied:
            self.blank_camera(idx)
            return False
        ox, oy = (cw-dw)//2, (ch-dh)//2
        card["mapping"] = (ox, oy, dw, dh)
        photo = self.ImageTk.PhotoImage(image, master=canvas)
        canvas.delete("hint")
        if card["image"] is None:
            card["image"] = canvas.create_image(ox, oy, anchor="nw", image=photo, tags="feed")
        else:
            try:
                canvas.itemconfigure(card["image"], image=photo)
                canvas.coords(card["image"], ox, oy)
            except self.tk.TclError:
                card["image"] = canvas.create_image(ox, oy, anchor="nw", image=photo, tags="feed")
        self.images[idx] = photo
        card["render_key"] = render_key
        if source_key != card["last_source"]:
            card["last_source"] = source_key
            card["render_count"] += 1
            stamps.append(now)
            if card["display_started_at"] is None:
                card["display_started_at"] = now
        if roi and not card["drag"]:
            if card["rectangle"]:
                canvas.delete(card["rectangle"])
            card["rectangle"] = canvas.create_rectangle(ox+roi[0]*dw, oy+roi[1]*dh, ox+roi[2]*dw, oy+roi[3]*dh, outline="#f2cf69", width=2)
        if card["rectangle"]:
            canvas.tag_raise(card["rectangle"])
        canvas.delete('hidden-zone')
        for zone in spec.get('hidden_zones',[]):
            canvas.create_rectangle(ox+zone[0]*dw,oy+zone[1]*dh,ox+zone[2]*dw,oy+zone[3]*dh,
                                    outline='#f29a53',width=2,dash=(6,4),tags='hidden-zone')
        return True

    def render_views(self):
        workers = self.monitor.workers
        count = len(workers)
        began = time.monotonic()
        for offset in range(count):
            if offset and time.monotonic()-began >= .018:
                break  # Nhường Tk/CPU; lượt sau tiếp tục khung kế tiếp.
            index = (self.render_cursor+offset) % count
            self.render_camera(index, workers[index].snapshot_view(preview=True))
            next_cursor = (index+1) % count
        if count:
            self.render_cursor = next_cursor

    def poll(self):
        try:
            for _ in range(100):
                kind, payload = self.bus.get_nowait()
                if kind == "ai_progress":
                    generation, message = payload
                    if generation == self.model_generation and not (self.monitor or self.starting or self.stopping or self.closing):
                        self.health.set(message + " · chưa mở camera")
                elif kind == "ai_loaded":
                    generation, message = payload
                    self.preloading = False
                    if generation == self.model_generation and not (self.monitor or self.starting or self.stopping or self.closing):
                        self.health.set(message)
                    elif generation != self.model_generation and not self.closing:
                        self.preload_requested = True
                elif kind == "boot_progress":
                    generation, message = payload
                    if generation == self.boot_generation and self.starting and not self.start_cancel.is_set() and not self.closing:
                        self.health.set(message)
                elif kind == "ready":
                    generation, monitor = payload
                    self.starting = False
                    self.monitor = monitor
                    if self.closing or self.start_cancel.is_set() or generation != self.boot_generation:
                        self.stop()
                    else:
                        for idx, card in self.cards.items():
                            self.blank_camera(idx)
                            card["rendered_at"].clear()
                            card["display_started_at"], card["last_source"] = None, None
                            card["render_count"] = 0
                        self.monitor.start(self.cfg["sources"][:self.pane_count])
                        self.health.set("Đang giám sát" + (" · AI chưa đủ thành phần" if monitor.ai_notice else ""))
                        self.footer.set(monitor.ai_notice or "Mô hình " + monitor.visual.identity + " · FPS hiển thị là khung nguồn mới; FPS AI là phân tích hoàn tất. Hộp xám cần xác minh.")
                elif kind == "boot_cancelled":
                    self.starting = False
                    self.start_button.configure(state="normal")
                    self.health.set("Đã hủy khởi động · chưa mở camera")
                elif kind == "error":
                    generation, message = payload
                    self.starting = False
                    self.health.set("Không khởi động được AI")
                    self.start_button.configure(state="normal")
                    if not self.closing and not self.start_cancel.is_set():
                        self.messagebox.showerror("Lỗi khởi động", message)
                elif kind == "stopped":
                    self.monitor = None
                    self.stopping = False
                    self.start_button.configure(state="normal")
                    self.health.set("Đã dừng · bằng chứng đã được hoàn tất")
                    self.refresh_events()
                elif kind == "shutdown_error":
                    self.health.set("Chưa hoàn tất đóng nguồn: " + payload)
                elif kind == "task_progress":
                    self.footer.set(payload)
                elif kind == "task_done":
                    name, target = payload
                    self.footer.set(name + " hoàn tất. Báo cáo: " + target)
                    if not self.closing:
                        self.messagebox.showinfo(name, "Kết quả đã lưu tại:\n" + target)
                elif kind == "task_error":
                    self.footer.set("Tác vụ chưa hoàn tất: " + payload)
                    if not self.closing:
                        self.messagebox.showerror("Tác vụ chưa hoàn tất", payload)
        except queue.Empty:
            pass
        if self.monitor:
            self.render_views()
            if not self.stopping and self.monitor.workers and all(not w.thread.is_alive() for w in self.monitor.workers):
                self.stop()
        current = time.monotonic()
        if current-self.last_events > 2:
            self.last_events = current
            if not self.event_tree.selection() or self.book.index(self.book.select()) != 1:
                self.refresh_events()
            free_gb = shutil.disk_usage(self.store.folder).free/1024**3
            if self.monitor and not self.stopping:
                alive = sum(w.thread.is_alive() for w in self.monitor.workers)
                self.health.set(f"{alive}/{len(self.monitor.workers)} nguồn · trống {free_gb:.1f} GB" +
                                (" · AI chưa đủ thành phần" if self.monitor.ai_notice else ""))
        if self.closing and not self.monitor and not self.starting and not self.stopping:
            if self.preload_thread and self.preload_thread.is_alive() or self.boot_thread and self.boot_thread.is_alive():
                self.health.set("Đang chờ tải mô hình kết thúc…")
            elif self.task_thread and self.task_thread.is_alive():
                self.health.set("Đang hoàn tất báo cáo phân tích…")
            else:
                self.root.destroy()
                return
        if self.preload_enabled and self.preload_requested and not self.closing:
            self.preload_ai()
        self.root.after(33, self.poll)

    def close(self):
        self.closing = True
        self.model_cache.close()
        self.stop_task()
        self.stop()
        self.footer.set("Đang kết thúc ứng dụng và đóng các tệp.")

    def run(self):
        if not self.smoke_seconds:
            self.preload_enabled = True
            self.root.after(200, self.preload_ai)
        self.root.mainloop()


def self_test() -> int:
    """Kiểm tra trạng thái cảnh báo, cấu hình và lưu/xuất sự kiện, không dùng camera."""
    cfg = validate_config(copy.deepcopy(DEFAULT_CONFIG))
    gate = AlertGate()
    d = cfg["detection"]
    assert not gate.update(0, .95, True, d)[1]
    assert gate.update(1.3, .95, True, d)[1]
    assert not gate.update(2, .95, True, d)[1]
    assert gate.update(31.4, .95, True, d)[1]
    gate.update(32, .1, False, d)
    assert not gate.update(34, .1, False, d)[1] and not gate.active
    gate.update(35, None, False, d)
    assert not gate.active
    isolated = AlertGate()
    isolated.update(0, .99, False, d)
    assert not isolated.update(10, .99, False, d)[1]
    independent = AlertGate()
    assert not independent.update(2, .99, True, d)[1]
    bad = copy.deepcopy(cfg)
    bad["detection"]["threshold"] = float("nan")
    try:
        validate_config(bad)
        raise AssertionError("NaN accepted")
    except ValueError:
        pass
    bad = copy.deepcopy(cfg)
    bad["sources"][0]["roi"] = [1, 0, 0, 1]
    try:
        validate_config(bad)
        raise AssertionError("Invalid ROI accepted")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory(prefix="schoolguard-test-") as folder:
        store = EventStore(Path(folder))
        errors = []
        def add_events(n):
            try:
                for i in range(10):
                    store.add(f"Camera {n}", "Test", .8, .85, 2, "<script>alert(1)</script>")
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=add_events, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors and len(store.rows(limit=None)) == 40
        eid = store.add("=cmd", "Test", None, None, 0, "No model")
        store.update(eid, status="Báo giả", notes="Kiểm tra thủ công")
        assert store.stats()["Báo nhầm"] == 1
        for ext in ("csv", "json", "html"):
            target = Path(folder) / ("events."+ext)
            assert store.export(target) == 41
            assert target.stat().st_size > 0
        assert "<script>" not in (Path(folder)/"events.html").read_text(encoding="utf-8")
        assert "'=cmd" in (Path(folder)/"events.csv").read_text(encoding="utf-8-sig")
        store.close()
    print("PASS: thời gian/cooldown, trạng thái độc lập, thiếu bằng chứng, cấu hình, SQLite đa luồng, CSV/JSON/HTML.")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="SchoolGuard · hỗ trợ phát hiện nguy cơ bạo lực học đường")
    parser.add_argument("--config", help="Cấu hình JSON/YAML; mặc định main2_config.json, kế thừa main1 nếu chưa có")
    parser.add_argument("--source", action="append", help="Webcam, file hoặc RTSP; lặp lại để dùng nhiều nguồn")
    parser.add_argument("--label", help="Tên camera khi dùng một --source")
    parser.add_argument("--multi", action="store_true", help="Dùng tất cả nguồn đã cấu hình")
    parser.add_argument("--headless", action="store_true", help="Giám sát không mở giao diện")
    parser.add_argument("--duration", type=float, default=0, help="Dừng giám sát sau N giây; 0 = không giới hạn")
    parser.add_argument("--offline", action="store_true", help="Tắt thông báo ngoài ứng dụng")
    parser.add_argument("--notify", action="store_true", help="Bật thông báo Discord / Telegram đã cấu hình")
    parser.add_argument("--record", action="store_true", help="Ghi video liên tục theo từng đoạn")
    parser.add_argument("--no-save", action="store_true", help="Tắt ghi ảnh/clip; vẫn lưu sự kiện SQLite")
    parser.add_argument("--analyze", help="Phân tích video hình ảnh, xuất JSON")
    parser.add_argument("--folder", help="Phân tích tất cả video trong thư mục và thư mục con")
    parser.add_argument("--prepare-pose", action="store_true", help="Tải YOLO pose để hỗ trợ lọc người đi ngang/tương tác tay chân")
    parser.add_argument("--telegram-discover", action="store_true", help="Lấy ID nhóm Telegram từ cập nhật bot, không gửi tin")
    parser.add_argument("--export", help="Xuất toàn bộ lịch sử ra CSV/JSON/HTML")
    parser.add_argument("--output", help="Đường dẫn báo cáo khi --analyze")
    parser.add_argument("--doctor", action="store_true", help="Kiểm tra thư viện và checkpoint, không mở camera")
    parser.add_argument("--self-test", action="store_true", help="Kiểm tra logic cốt lõi không cần AI")
    parser.add_argument("--init-config", action="store_true", help="Tạo main2_config.json nếu chưa có")
    parser.add_argument("--job", choices=list(JOBS), help="Chạy công cụ dự án; tham số công cụ đặt sau --")
    parser.add_argument("--gui-smoke", type=float, default=0, help=argparse.SUPPRESS)
    parser.add_argument("job_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        cfg = load_config(args.config)
        if args.duration < 0 or not math.isfinite(args.duration):
            raise ValueError("--duration phải là số hữu hạn không âm.")
        if args.source:
            cfg["sources"] = [{"label": args.label if len(args.source) == 1 and args.label else f"Camera {i+1}", "source": value, "roi": None}
                              for i, value in enumerate(args.source)]
        if args.notify:
            cfg["notifications"]["enabled"] = True
        if args.offline:
            cfg["notifications"]["enabled"] = False
        if args.record:
            cfg["evidence"]["continuous"] = True
        if args.no_save:
            cfg["evidence"]["enabled"] = cfg["evidence"]["continuous"] = False
            cfg["evidence"]["save_snapshots"] = False
        validate_config(cfg)
        if args.doctor:
            print(json.dumps(doctor(cfg), ensure_ascii=False, indent=2))
            return 0
        if args.telegram_discover:
            result = discover_telegram(cfg["notifications"])
            if len(result["groups"]) == 1:
                save_credentials(telegram_chat_id=result["groups"][0]["id"])
                result["saved"] = True
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.prepare_pose:
            print(json.dumps(prepare_pose_model(), ensure_ascii=False))
            return 0
        if args.init_config:
            target = ROOT / "main2_config.json"
            if target.exists():
                print("Cấu hình đã tồn tại; giữ nguyên: " + str(target))
            else:
                atomic_json(target, cfg)
                print("Đã tạo " + str(target))
            return 0
        if args.job:
            extra = args.job_args[1:] if args.job_args[:1] == ["--"] else args.job_args
            return subprocess.call(job_command(args.job, extra), cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if args.job_args:
            parser.error("Tham số sau -- chỉ dùng cho --job.")
        folder = resolve_path(cfg["storage"]["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(folder / "system.log", maxBytes=2*1024**2, backupCount=3, encoding="utf-8")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(), handler])
        if args.analyze:
            result = analyze_clip(resolve_path(args.analyze), cfg)
            target = resolve_path(args.output) if args.output else folder / "reports" / f"analysis_{uuid.uuid4().hex}.json"
            atomic_json(target, result)
            print("Đã lưu phân tích: " + str(target))
            print(json.dumps({k: v for k, v in result.items() if k != "timeline"}, ensure_ascii=False, indent=2))
            return 0
        if args.folder:
            result = analyze_folder(resolve_path(args.folder), cfg)
            print(f"Đã soi {result['processed']}/{result['videos']} video. Báo cáo: {result['report_folder']}")
            return 0
        store = EventStore(folder)
        monitor = None
        try:
            feedback_cfg = cfg["feedback"]
            if feedback_cfg["enabled"] and not args.offline and not args.gui_smoke and not args.export:
                try:
                    store.feedback = ReviewServer(store,
                        **{k: feedback_cfg[k] for k in ("host", "port", "public_base_url")}).start()
                    LOG.info("Trang phản hồi sẵn sàng tại %s", store.feedback.base_url)
                except (OSError, ValueError) as exc:
                    LOG.warning("Chưa mở được trang phản hồi: %s; dùng hai nút trong tab Sự kiện.", redact(exc))
            if args.export:
                target = resolve_path(args.export)
                print(f"Đã xuất {store.export(target)} sự kiện: {target}")
            elif args.headless:
                monitor = Monitor(cfg, store, args.offline)
                sources = cfg["sources"] if args.multi or args.source else cfg["sources"][:1]
                monitor.start(sources)
                started = time.monotonic()
                while any(worker.thread.is_alive() for worker in monitor.workers):
                    if args.duration and time.monotonic()-started >= args.duration:
                        break
                    time.sleep(.2)
                monitor.close()
                failed = [w.failure for w in monitor.workers if w.failure]
                if failed:
                    print("Nguồn có lỗi: " + "; ".join(failed), file=sys.stderr)
                    return 1
            else:
                SchoolGuardApp(cfg, store, args.offline, args.gui_smoke).run()
        except KeyboardInterrupt:
            if monitor:
                monitor.close()
            print("Đã dừng và hoàn tất bằng chứng.")
        finally:
            if monitor:
                monitor.close()
            if store.feedback is not None:
                store.feedback.close()
            store.close()
        return 0
    except Exception as exc:
        print("SchoolGuard: " + redact(exc), file=sys.stderr)
        print("Dùng --doctor để kiểm tra thư viện và mô hình.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
