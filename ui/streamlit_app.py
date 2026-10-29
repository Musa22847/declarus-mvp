"""Декларус: интерфейс постоянных заданий. Обработку выполняет worker."""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for module_dir in (ROOT / "ui", ROOT / "mvp"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from job_store import ACTIVE, STAGES, JobStore
from xlsx_export import export_result_xlsx

OUTPUT_DIR = Path(os.getenv("DECLARUS_OUTPUT_DIR", str(ROOT / "ui" / "output")))
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
STAGE_NAMES = {
    "processing_ocr": "Распознавание текста",
    "processing_qwen": "Извлечение данных",
    "saving": "Подготовка и сохранение Excel",
}

st.set_page_config(
    page_title="Декларус: ИИ-ассистент для заполнения таможенных деклараций",
    layout="centered",
)

# Локализация стандартного загрузчика Streamlit 1.50.
st.markdown("""
<style>
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {font-size:0;}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child::after {
    content:"Перетащите файл сюда"; font-size:1rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:last-child::after {
    content:"До 10 МБ на файл · WEBP, PNG, JPG, JPEG, PDF"; font-size:0.875rem;
}
[data-testid="stFileUploaderDropzone"] button {font-size:0;}
[data-testid="stFileUploaderDropzone"] button::after {content:"Выбрать файл"; font-size:0.875rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_store(output_dir):
    """Объект хранит только пути; SQLite-соединения не разделяются между сессиями."""
    return JobStore(output_dir)


store = get_store(str(OUTPUT_DIR.resolve()))


def duration_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60} мин {seconds % 60:02d} с"


def new_document():
    """Очистить выбор в этой вкладке, не удаляя предыдущее задание."""
    st.query_params.pop("job_id", None)
    st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1
    st.session_state.pop("upload_error", None)


def capture_upload(widget_key):
    """on_change срабатывает при выборе файла, но не при обычной перерисовке."""
    uploaded = st.session_state.get(widget_key)
    if uploaded is None:
        return
    try:
        # Сохраняем настройки модели, но никогда не API-ключ в базе задания.
        settings = {
            "model": os.getenv("YANDEX_MODEL", "qwen3.6-35b-a3b"),
            "max_output_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", "40000")),
            "background_timeout_seconds": int(os.getenv("BACKGROUND_TIMEOUT_SECONDS", "3600")),
            "max_pages": 10,
        }
        job = store.create(uploaded.name, uploaded.getvalue(), settings)
        st.query_params["job_id"] = job["job_id"]
        st.session_state.pop("upload_error", None)
    except Exception as error:
        st.session_state["upload_error"] = str(error)


def show_result_table(result):
    rows = []
    documents = result.get("documents", []) or []
    if isinstance(documents, dict):
        documents = [documents]
    for document in documents:
        for index, item in enumerate(document.get("items", []) or [], 1):
            row = {"Документ": document.get("document_id", ""), "Позиция": index}
            for label, field in item.items():
                row[label] = field.get("value") if isinstance(field, dict) else field
                if isinstance(field, dict) and field.get("unit"):
                    row[f"{label}: единица"] = field["unit"]
            rows.append(row)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("Реквизиты, товары и итоги доступны в Excel.")


def render_status(job):
    """И этапы, и время берутся из SQLite, независимо от памяти сессии."""
    status = job["status"]
    end = job["finished_at"] or time.time()
    total = duration_text(end - job["started_at"]) if job["started_at"] else None
    if status == "queued":
        label, state = "Ожидает обработки", "running"
    elif status == "completed":
        label, state = f"Обработка завершена · Всего: {total}", "complete"
    elif status == "failed":
        label = "Обработка завершилась ошибкой" + (f" · Прошло: {total}" if total else "")
        state = "error"
    else:
        label, state = f"{STAGE_NAMES[status]} · Прошло: {total}", "running"
    with st.status(label, state=state, expanded=True):
        for stage in STAGES:
            timing = job["stage_times"].get(stage)
            if not timing:
                continue
            if timing["finished_at"]:
                seconds = timing["finished_at"] - timing["started_at"]
                cumulative = timing["finished_at"] - job["started_at"]
                st.write(f"✓ {STAGE_NAMES[stage]}: {duration_text(seconds)}. "
                         f"Всего прошло: {duration_text(cumulative)}.")
            elif status == "failed":
                st.write(f"{STAGE_NAMES[stage]}: этап не завершён.")
            else:
                st.write(f"{STAGE_NAMES[stage]}… Прошло: "
                         f"{duration_text(time.time() - timing['started_at'])}.")
        if status == "queued":
            st.write("Задание сохранено. Worker обрабатывает документы по очереди.")
        if status == "failed":
            st.error(job["error_message"] or "Не удалось выполнить задание")


def show_job_panel(job_id, was_active):
    job = store.get(job_id)
    if not job:
        st.error("Задание не найдено")
        return
    # Полная перерисовка после завершения отключит таймер фрагмента.
    if was_active and job["status"] not in ACTIVE:
        st.rerun()
    if job["status"] == "uploaded":
        st.info("Файл сохранён. Нажмите «Начать обработку».")
        return
    render_status(job)
    if job["status"] == "completed":
        try:
            result = json.loads(Path(job["result_path"]).read_text(encoding="utf-8"))
            excel = Path(job["excel_path"]).read_bytes()
        except (OSError, ValueError):
            st.error("Не удалось прочитать сохранённый результат. Проверьте папку output.")
            return
        st.subheader("Результат")
        show_result_table(result)
        st.download_button("Скачать Excel (.xlsx)", data=excel,
                           file_name=Path(job["excel_path"]).name, mime=XLSX_MIME,
                           key=f"download-{job_id}")


@st.cache_data(show_spinner=False)
def legacy_excel(result, raw_text, details):
    """Старые JSON остаются доступными без повторного обращения к модели."""
    with tempfile.TemporaryDirectory(prefix="declarus-excel-") as tmp:
        path = export_result_xlsx(result, Path(tmp) / "result.xlsx",
                                  include_raw_text=raw_text, include_details=details)
        return path.read_bytes()


with st.sidebar:
    st.title("Декларус")
    selected_page = st.radio("Навигация", ["Новая обработка", "История"], key="navigation")
    st.divider()
    st.success("Сервис доступен")


def open_job(job_id):
    st.query_params["job_id"] = job_id
    st.session_state["navigation"] = "Новая обработка"


if selected_page == "Новая обработка":
    st.title("Распознавание документов")
    st.caption("Загрузите упаковочный лист — сервис подготовит Excel-файл с распознанными данными")
    job_id = st.query_params.get("job_id")
    if not job_id:
        key = f"document-upload-{st.session_state.get('upload_generation', 0)}"
        st.file_uploader("Перетащите файл сюда", type=["webp", "png", "jpg", "jpeg", "pdf"],
                         accept_multiple_files=False, key=key,
                         on_change=capture_upload, args=(key,))
        if st.session_state.get("upload_error"):
            st.error(st.session_state.upload_error)
        else:
            st.info("Выберите файл")
    else:
        job = store.get(job_id)
        st.button("Новый документ", on_click=new_document)
        if not job:
            st.error("Задание не найдено. Проверьте ссылку или откройте задание из истории.")
        else:
            st.success(f"Документ «{job['original_name']}» уже загружен")
            st.caption(f"Размер: {job['source_size'] / 1024:.1f} КБ")
            if job["status"] == "uploaded":
                if st.button("Начать обработку", type="primary"):
                    store.enqueue(job_id)
                    st.rerun()
            elif job["status"] == "failed":
                st.caption("Повтор создаст новое задание и может выполнить новый платный запрос к Yandex.")
                if st.button("Повторить обработку"):
                    try:
                        repeated = store.retry(job_id)
                        store.enqueue(repeated["job_id"])
                        st.query_params["job_id"] = repeated["job_id"]
                        st.rerun()
                    except (OSError, ValueError) as error:
                        st.error(f"Не удалось повторить обработку: {error}")
            was_active = job["status"] in ACTIVE
            panel = st.fragment(run_every=2 if was_active else None)(show_job_panel)
            panel(job_id, was_active)

else:
    st.title("История загрузок")
    jobs = store.list_jobs()
    labels = {"uploaded": "Файл загружен", "queued": "В очереди", "completed": "Готово", "failed": "Ошибка", **STAGE_NAMES}
    for job in jobs:
        with st.container(border=True):
            st.text(job["original_name"])
            st.caption(datetime.fromtimestamp(job["created_at"]).strftime("%d.%m.%Y %H:%M"))
            st.write(labels[job["status"]])
            st.button("Открыть задание", key=f"open-{job['job_id']}",
                      on_click=open_job, args=(job["job_id"],))
    known_results = {Path(job["result_path"]).resolve() for job in jobs}
    legacy_count = 0
    for path in sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.resolve() in known_results:
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                continue
        except (OSError, ValueError):
            continue
        legacy_count += 1
        with st.container(border=True):
            st.text(path.with_suffix(".xlsx").name)
            st.caption("Ранее сохранённый результат. Длительность не записана.")
            try:
                excel = legacy_excel(result,
                    os.getenv("XLSX_INCLUDE_RAW_TEXT", "false").lower() in {"1", "true", "yes"},
                    os.getenv("XLSX_INCLUDE_DETAILS", "false").lower() in {"1", "true", "yes"})
                st.download_button("Скачать Excel (.xlsx)", data=excel,
                                   file_name=path.with_suffix(".xlsx").name, mime=XLSX_MIME,
                                   key=f"legacy-{path.name}")
            except Exception as error:
                st.error(f"Не удалось подготовить Excel: {error}")
    if not jobs and not legacy_count:
        st.info("История пока пуста")
