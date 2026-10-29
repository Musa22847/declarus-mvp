"""Постоянные задания: SQLite хранит состояние, диск — оригиналы и результаты."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

STAGES = ("processing_ocr", "processing_qwen", "saving")
ACTIVE = {"queued", *STAGES}
TERMINAL = {"completed", "failed"}
ALLOWED_SUFFIXES = {".webp", ".png", ".jpg", ".jpeg", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024


def atomic_write(path: Path, content: bytes) -> None:
    """Готовый файл появляется целиком; незавершённая запись его не заменяет."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class JobStore:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).resolve()
        self.system_dir = self.output_dir / "system-folder"
        self.inputs_dir = self.system_dir / "jobs"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.database = self.system_dir / "jobs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_size INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'uploaded','queued','processing_ocr','processing_qwen',
                    'saving','completed','failed')),
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                stage_times TEXT NOT NULL DEFAULT '{}',
                qwen_response_id TEXT,
                result_path TEXT NOT NULL,
                excel_path TEXT NOT NULL,
                error_code TEXT,
                error_message TEXT,
                settings TEXT NOT NULL,
                retry_of TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at)")

    @contextmanager
    def connect(self):
        # Соединение открывается отдельно для каждой операции: UI и worker
        # работают в разных процессах; долгих транзакций во время API нет.
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        job = dict(row)
        job["stage_times"] = json.loads(job["stage_times"])
        job["settings"] = json.loads(job["settings"])
        return job

    def create(self, filename: str, data: bytes, settings: dict, *, retry_of=None) -> dict:
        safe_name = Path(filename.replace("\\", "/")).name
        safe_name = re.sub(r"[\x00-\x1f\x7f]", "", safe_name)[:200]
        if Path(safe_name).suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError("Поддерживаются PDF, JPG, JPEG, PNG и WebP")
        if not data or len(data) > MAX_FILE_SIZE:
            raise ValueError("Выберите непустой файл размером до 10 МБ")
        job_id = uuid.uuid4().hex
        created = time.time()  # Unix-время UTC сохраняется между процессами.
        source = self.inputs_dir / job_id / safe_name
        atomic_write(source, data)
        stem = re.sub(r"[^\w .()-]", "_", Path(safe_name).stem)[:100] or "document"
        result = self.output_dir / f"{stem}-{job_id}.json"
        with self.connect() as db:
            db.execute("""INSERT INTO jobs
                (job_id,original_name,source_path,source_size,status,created_at,
                 result_path,excel_path,settings,retry_of)
                VALUES (?,?,?,?, 'uploaded', ?,?,?,?,?)""",
                (job_id, safe_name, str(source), len(data), created, str(result),
                 str(result.with_suffix(".xlsx")), json.dumps(settings), retry_of))
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            return None
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    def list_jobs(self) -> list[dict]:
        with self.connect() as db:
            return [self.decode(row) for row in db.execute("SELECT * FROM jobs ORDER BY created_at DESC")]

    def enqueue(self, job_id: str) -> bool:
        # Только один вызов может перевести uploaded в queued, даже из двух вкладок.
        with self.connect() as db:
            return db.execute("UPDATE jobs SET status='queued' WHERE job_id=? AND status='uploaded'",
                              (job_id,)).rowcount == 1

    def claim_next(self) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT * FROM jobs
                WHERE status IN ('queued','processing_qwen','saving')
                ORDER BY CASE WHEN status='queued' THEN 1 ELSE 0 END, created_at
                LIMIT 1""").fetchone()
            if row is None:
                return None
            job = self.decode(row)
            if job["status"] == "queued":
                now = time.time()
                stage = "saving" if Path(job["result_path"]).is_file() else "processing_ocr"
                times = {stage: {"started_at": now, "finished_at": None}}
                db.execute("""UPDATE jobs SET status=?,started_at=?,stage_times=?
                    WHERE job_id=? AND status='queued'""", (stage, now, json.dumps(times), job["job_id"]))
                job.update(status=stage, started_at=now, stage_times=times)
            return job

    def transition(self, job_id: str, stage: str) -> None:
        if stage not in (*STAGES, "completed"):
            raise ValueError(f"Неизвестный этап: {stage}")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = self.decode(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
            if not job or job["status"] in TERMINAL or stage == job["status"]:
                return
            if job["status"] not in STAGES:
                raise ValueError("Задание ещё не взято worker")
            if stage != "completed" and STAGES.index(stage) < STAGES.index(job["status"]):
                return  # Повторное уведомление при восстановлении не сбрасывает часы.
            now = time.time()
            times = job["stage_times"]
            times[job["status"]]["finished_at"] = now
            if stage in STAGES:
                times[stage] = {"started_at": now, "finished_at": None}
            db.execute("UPDATE jobs SET status=?,stage_times=?,finished_at=? WHERE job_id=?",
                       (stage, json.dumps(times), now if stage == "completed" else None, job_id))

    def set_response_id(self, job_id: str, response_id: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET qwen_response_id=? WHERE job_id=?", (response_id, job_id))

    def fail(self, job_id: str, code: str, message: str) -> None:
        with self.connect() as db:
            db.execute("""UPDATE jobs SET status='failed',finished_at=?,error_code=?,error_message=?
                WHERE job_id=? AND status NOT IN ('completed','failed')""",
                       (time.time(), code, message[:1500], job_id))

    def recover(self) -> None:
        """Вызывается только при старте единственного worker с захваченным lock."""
        for job in self.list_jobs():
            status = job["status"]
            if status not in STAGES:
                continue
            if Path(job["result_path"]).is_file():
                self.transition(job["job_id"], "saving")
            elif status == "processing_ocr":
                self.fail(job["job_id"], "interrupted_ocr",
                          "Распознавание прервано перезапуском. Повторный запрос автоматически не отправлен.")
            elif status == "processing_qwen" and job["qwen_response_id"]:
                continue
            else:
                self.fail(job["job_id"], "interrupted_without_checkpoint",
                          "Обработка прервана, данных для безопасного продолжения нет. "
                          "Новый запрос автоматически не отправлен.")

    def retry(self, job_id: str) -> dict:
        previous = self.get(job_id)
        if not previous or previous["status"] != "failed":
            raise ValueError("Повтор доступен только для задания с ошибкой")
        repeated = self.create(previous["original_name"], Path(previous["source_path"]).read_bytes(),
                               previous["settings"], retry_of=job_id)
        if Path(previous["result_path"]).is_file():
            atomic_write(Path(repeated["result_path"]), Path(previous["result_path"]).read_bytes())
        return repeated
