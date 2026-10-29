"""Отдельный последовательный обработчик: python ui/worker.py."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from job_store import JobStore, atomic_write

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("declarus.worker")


class WorkerStopping(BaseException):
    """Остановка процесса сохраняет незавершённое задание для recover()."""


@contextmanager
def worker_lock(store):
    # Контейнеры работают под Linux и на Mac, и на Windows. Блокировка
    # автоматически освобождается ОС при аварийном завершении процесса.
    import fcntl

    with (store.system_dir / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Worker уже запущен для этой папки данных") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def process_job(store: JobStore, job: dict, *, pipeline=None, exporter=None, project_dir=None) -> None:
    """Один запуск или продолжение; UI здесь не импортируется."""
    job_id = job["job_id"]
    result_path = Path(job["result_path"])
    try:
        if result_path.is_file():
            # Ключи API не нужны, если осталось только сформировать Excel.
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            api_key = os.getenv("YANDEX_API_KEY", "").strip()
            folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
            if not api_key or not folder_id:
                raise RuntimeError("Не заданы YANDEX_API_KEY и/или YANDEX_FOLDER_ID")
            if pipeline is None:
                sys.path.insert(0, str(ROOT / "mvp"))
                from ocr_table_pipeline import run_documents_pipeline
                pipeline = run_documents_pipeline
            result = pipeline(
                api_key=api_key, folder_id=folder_id,
                project_dir=Path(project_dir or ROOT / "mvp"),
                document_paths=[Path(job["source_path"])],
                job_id=job_id, username="local-user",
                **job["settings"],
                resume_response_id=job["qwen_response_id"],
                stage_callback=lambda stage: store.transition(job_id, stage),
                response_id_callback=lambda response_id: store.set_response_id(job_id, response_id),
            )
            if not isinstance(result, dict):
                raise TypeError("Модель вернула результат не в формате JSON-объекта")
            atomic_write(result_path, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        store.transition(job_id, "saving")
        if exporter is None:
            sys.path.insert(0, str(ROOT / "mvp"))
            from xlsx_export import export_result_xlsx
            exporter = export_result_xlsx
        exporter(result, Path(job["excel_path"]))
        if not Path(job["excel_path"]).is_file():
            raise RuntimeError("Excel-файл не сохранён")
        store.transition(job_id, "completed")
        LOG.info("job=%s completed", job_id)
    except Exception as error:
        # Логи содержат ID и тип ошибки, без ключей и текста документов.
        message = str(error)
        for name in ("YANDEX_API_KEY", "YANDEX_FOLDER_ID"):
            secret = os.getenv(name, "")
            if secret:
                message = message.replace(secret, "[скрыто]")
        store.fail(job_id, type(error).__name__, message)
        LOG.error("job=%s failed type=%s", job_id, type(error).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Обработать одно задание и выйти")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    store = JobStore(os.getenv("DECLARUS_OUTPUT_DIR", str(ROOT / "ui" / "output")))

    def stop(signum, frame):
        raise WorkerStopping()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with worker_lock(store):
            store.recover()
            LOG.info("worker ready")
            while True:
                job = store.claim_next()
                if job:
                    process_job(store, job)
                elif not args.once:
                    time.sleep(1)
                if args.once:
                    break
    except WorkerStopping:
        LOG.info("worker stopped; unfinished jobs retained")


if __name__ == "__main__":
    main()
