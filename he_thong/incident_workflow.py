"""Persistent patrol and notification history plus periodic learning maintenance."""
from __future__ import annotations
import threading
from datetime import datetime


class IncidentMixin:
    def init_incidents(self):
        with self.lock, self.conn:
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS incident_claims (
              event_id TEXT PRIMARY KEY REFERENCES events(id), teacher TEXT NOT NULL, accepted_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS incident_dispatch (
              event_id TEXT PRIMARY KEY REFERENCES events(id), due_at TEXT NOT NULL, reminders INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS patrol_runs (slot TEXT PRIMARY KEY, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS notification_receipts (
              event_id TEXT, channel TEXT, message_id TEXT, media INTEGER, destination TEXT DEFAULT '', PRIMARY KEY(event_id,channel,message_id));
            CREATE TABLE IF NOT EXISTS claim_deliveries (event_id TEXT, channel TEXT, delivered_at TEXT, PRIMARY KEY(event_id,channel));
            """)
            columns={r[1] for r in self.conn.execute('PRAGMA table_info(notification_receipts)')}
            if 'destination' not in columns:
                self.conn.execute("ALTER TABLE notification_receipts ADD COLUMN destination TEXT DEFAULT ''")
        from learning_feedback import init_learning
        init_learning(self)

    def save_receipt(self, eid, channel, message_id, media=False, destination=''):
        with self.lock, self.conn:
            self.conn.execute('INSERT OR REPLACE INTO notification_receipts VALUES(?,?,?,?,?)', (eid,channel,str(message_id),int(media),destination))

    def receipt(self,eid,channel,destination=''):
        with self.lock:
            return [dict(row) for row in self.conn.execute('SELECT * FROM notification_receipts WHERE event_id=? AND channel=? AND destination=?',(eid,channel,destination)).fetchall()]

    def patrol_once(self, slot):
        with self.lock, self.conn:
            return self.conn.execute("INSERT OR IGNORE INTO patrol_runs VALUES(?,?)",
                                     (slot, datetime.now().astimezone().isoformat())).rowcount == 1


class MaintenanceLoop:
    """Only patrol and learning sync; legacy ownership tables stay inert."""

    def __init__(self, store, patrol=None):
        self.stop = threading.Event()
        self.store, self.patrol = store, patrol
        self.last_learning_sync = None
        self.thread = threading.Thread(target=self.run, daemon=True, name="schoolguard-maintenance")
        self.thread.start()

    def tick(self, now=None):
        now = now or datetime.now().astimezone()
        if self.patrol:
            self.patrol(now)
        if self.last_learning_sync is None or (now-self.last_learning_sync).total_seconds() >= 30:
            from learning_feedback import sync_inbox
            sync_inbox(self.store)
            self.last_learning_sync = now

    def run(self):
        import logging
        while not self.stop.wait(1):
            try:
                self.tick()
            except Exception:
                logging.getLogger('schoolguard').exception('Không hoàn tất tuần tra/đồng bộ mẫu học.')

    def close(self):
        self.stop.set()
        self.thread.join()
