"""Signed human-review links. GET is read-only so chat previews cannot label events."""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

ALERT_TEXT = "Nghi có xô xát - cần người đến kiểm tra"
HUMAN_GUIDANCE = ("AI có thể báo nhầm hoặc bỏ sót. AI không dùng để kết luận hoặc kỷ luật học sinh. "
                  "Người phụ trách vẫn cần xem camera định kỳ.")
CHOICES = {"confirmed": "Đúng là xô xát", "false_alarm": "Báo nhầm"}


class ReviewServer:
    """Event-scoped bearer links, 48h TTL; no profile, face or identity data."""

    def __init__(self, store, host="127.0.0.1", port=8765, public_base_url="", ttl=172800):
        self.store, self.ttl = store, ttl
        self.key = secrets.token_bytes(32)
        self.thread = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # URLs contain review credentials; never log them.

            def reply(self, status, body):
                data = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(data)

            def handle_review(self, submit=False):
                parsed = urlsplit(self.path)
                if parsed.path != "/review":
                    self.reply(404, "Không tìm thấy trang.")
                    return
                args = parse_qs(parsed.query)
                token = args.get("token", [""])[0]
                choice = args.get("choice", [""])[0]
                verified = owner.verify(token)
                if verified is None or choice not in CHOICES:
                    self.reply(403, "Liên kết không hợp lệ hoặc đã hết hạn. Mở ứng dụng để đánh giá sự kiện.")
                    return
                row = owner.store.get(verified)
                if row is None:
                    self.reply(404, "Sự kiện không còn tồn tại.")
                    return
                label = CHOICES[choice]
                if submit:
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                    except ValueError:
                        length = 0
                    if not 0 < length <= 1024:
                        self.reply(400, "Yêu cầu không hợp lệ.")
                        return
                    form = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
                    if form.get("confirm") != ["yes"]:
                        self.reply(400, "Cần xác nhận sau khi xem bằng chứng.")
                        return
                    owner.store.update(verified, status=label)
                    self.reply(200, owner.page("Đã lưu: " + label, "<p>Đánh giá đã lưu cùng nhật ký thay đổi. Có thể sửa trong ứng dụng.</p>"))
                    return
                # A preview crawler or accidental open must not change training labels.
                action = "/review?" + urlencode({"token": token, "choice": choice})
                body = (f"<p>Camera: {html.escape(str(row['camera']))}<br>Thời gian: {html.escape(row['created'])}"
                        f"<br>Mã: {html.escape(row['id'])}<br>Đánh giá hiện tại: {html.escape(row['status'])}</p>"
                        "<p>Hãy xem ảnh/clip hoặc camera trong ứng dụng trước khi xác nhận.</p>"
                        f"<form method='post' action='{html.escape(action, quote=True)}'>"
                        f"<button name='confirm' value='yes'>Lưu đánh giá: {html.escape(label)}</button></form>")
                self.reply(200, owner.page(ALERT_TEXT, body))

            def do_GET(self):
                self.handle_review()

            def do_POST(self):
                self.handle_review(submit=True)

        if host not in ("127.0.0.1", "localhost", "0.0.0.0"):
            raise ValueError("Máy chủ phản hồi chỉ hỗ trợ 127.0.0.1, localhost hoặc 0.0.0.0.")
        parsed = urlsplit(public_base_url) if public_base_url else None
        if parsed and (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                       or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
            raise ValueError("feedback.public_base_url cần là URL HTTP/HTTPS gốc, không chứa tài khoản/query/path.")
        if host == "0.0.0.0" and not public_base_url:
            raise ValueError("Cần public_base_url để nhận phản hồi từ thiết bị khác.")
        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.httpd.daemon_threads = True
        actual_port = self.httpd.server_address[1]
        self.base_url = public_base_url.rstrip("/") or f"http://127.0.0.1:{actual_port}"

    @staticmethod
    def page(title, body):
        return ("<!doctype html><html lang='vi'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                "<title>SchoolGuard · Đánh giá cảnh báo</title><style>body{font:18px system-ui;max-width:680px;margin:40px auto;padding:20px;line-height:1.6}"
                "button{font:inherit;padding:14px;cursor:pointer}</style>"
                f"<h1>{html.escape(title)}</h1>{body}<p>{html.escape(HUMAN_GUIDANCE)}</p></html>")

    def start(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": .1}, daemon=True, name="human-review")
        self.thread.start()
        return self

    def links(self, event_id):
        if not re.fullmatch(r"[a-f0-9]{32}", event_id):
            raise ValueError("Mã sự kiện không hợp lệ.")
        expires = int(time.time() + self.ttl)
        message = f"{event_id}.{expires}"
        signature = hmac.new(self.key, message.encode(), hashlib.sha256).hexdigest()
        token = message + "." + signature
        links = {key: self.base_url + "/review?" + urlencode({"token": token, "choice": key}) for key in CHOICES}
        return links

    def verify(self, token):
        try:
            event_id, expires, signature = token.split(".")
            if not re.fullmatch(r"[a-f0-9]{32}", event_id) or int(expires) < time.time():
                return None
            expected = hmac.new(self.key, f"{event_id}.{expires}".encode(), hashlib.sha256).hexdigest()
            return event_id if hmac.compare_digest(signature, expected) else None
        except (ValueError, TypeError):
            return None

    def close(self):
        if self.thread is not None:
            self.httpd.shutdown()
            self.thread.join(timeout=2)
        self.httpd.server_close()


def bounded_notification_text(text, limit):
    """Keep adult-review guidance visible, counting UTF-16 units for chat limits."""
    suffix = "\n" + HUMAN_GUIDANCE
    body = str(text).replace(HUMAN_GUIDANCE, "").rstrip()
    budget = max(0, limit - len(suffix.encode("utf-16-le")) // 2)
    body = body.encode("utf-16-le")[:budget*2].decode("utf-16-le", errors="ignore").rstrip()
    return body + suffix


def notification_payload(channel, text, links=None):
    """Discord non-interactive URL components; Telegram URL inline keyboard."""
    links = links or {}
    buttons = [(CHOICES[key], links[key]) for key in CHOICES if links.get(key)]
    if channel == "Discord":
        result = {"content": bounded_notification_text(text, 1900), "allowed_mentions": {"parse": []}}
        if buttons:
            result["components"] = [{"type": 1, "components": [
                {"type": 2, "style": 5, "label": label, "url": url} for label, url in buttons]}]
        return result
    if not buttons:
        return None
    return {"inline_keyboard": [[{"text": label, "url": url} for label, url in buttons]]}
