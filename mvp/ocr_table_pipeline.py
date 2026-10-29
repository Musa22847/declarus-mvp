"""Image/JSON few-shot examples -> Qwen3.6 multimodal extraction."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from openai import APIConnectionError, OpenAI
from PIL import Image


OCR_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
AI_STUDIO_URL = "https://ai.api.cloud.yandex.net/v1"
OCR_MODEL = "table"
DEFAULT_VLM_MODEL = "qwen3.6-35b-a3b"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_OCR_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
DEFAULT_MAX_PAGES = 10


StageCallback = Callable[[str], None]
ResponseIdCallback = Callable[[str], None]


@dataclass(frozen=True)
class PreparedPage:
    source_name: str
    page_number: int
    path: Path


def _prepare_for_ocr(path: Path) -> tuple[str, str]:
    """Return Vision OCR MIME name and base64 content.

    Vision OCR Table accepts JPEG, PNG and PDF. WebP inputs are converted to
    PNG in memory because the source project already contains WebP documents.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Неподдерживаемый формат: {path}")

    if suffix != ".pdf":
        try:
            with Image.open(path) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        f"Изображение {path.name} больше 20 мегапикселей"
                    )
                image.verify()
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError(f"Не удалось прочитать изображение: {path.name}") from error

    if suffix == ".webp":
        with Image.open(path) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            binary = buffer.getvalue()
        mime_type = "PNG"
    else:
        binary = path.read_bytes()
        mime_type = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".pdf": "PDF",
        }[suffix]

    if len(binary) > MAX_OCR_FILE_SIZE:
        raise ValueError(
            f"Файл {path.name} после подготовки весит "
            f"{len(binary) / 1024 / 1024:.2f} МБ; лимит Vision OCR — 10 МБ."
        )

    return mime_type, base64.b64encode(binary).decode("ascii")


def _image_to_data_url(path: Path) -> str:
    """Prepare an original image for the LLM few-shot message.

    WebP is converted to PNG because the previous direct request failed with
    `image: unknown format`. PDF is accepted by Vision OCR but cannot be sent
    as `input_image`; PDF pages must be converted to PNG/JPEG first.
    """
    if path.suffix.lower() == ".pdf":
        raise ValueError(
            f"{path.name}: PDF нельзя передать как input_image. "
            "Преобразуйте страницы PDF в PNG или JPEG."
        )

    mime_name, encoded = _prepare_for_ocr(path)
    mime_type = {"JPEG": "image/jpeg", "PNG": "image/png"}[mime_name]
    return f"data:{mime_type};base64,{encoded}"


def _annotation_path(annotations_dir: Path, image_path: Path) -> Path:
    """Return the required same-named semantic annotation JSON."""
    path = annotations_dir / f"{image_path.stem}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Для {image_path.name} не найдена разметка {path.name} "
            f"в {annotations_dir}"
        )
    return path


def recognize_with_table_ocr(
    path: Path,
    *,
    api_key: str,
    folder_id: str,
) -> dict[str, Any]:
    """Recognize a document with the Yandex Vision OCR `table` model."""
    mime_type, content = _prepare_for_ocr(path)

    response = requests.post(
        OCR_URL,
        headers={
            "Authorization": f"Api-Key {api_key}",
            "x-folder-id": folder_id,
            "Content-Type": "application/json",
            "x-data-logging-enabled": "false",
        },
        json={
            "mimeType": mime_type,
            "languageCodes": ["*"],
            "model": OCR_MODEL,
            "content": content,
        },
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            f"Vision OCR не обработал {path.name}: "
            f"HTTP {response.status_code}: {response.text}"
        )

    payload = response.json()
    if not payload.get("result"):
        raise RuntimeError(
            f"Vision OCR вернул ответ без result для {path.name}: "
            f"{json.dumps(payload, ensure_ascii=False)[:1000]}"
        )
    return payload


def _bbox_percent(
    bounding_box: dict[str, Any] | None,
    *,
    page_width: int,
    page_height: int,
) -> dict[str, float] | None:
    """Convert an OCR pixel polygon into a compact percent rectangle."""
    vertices = (bounding_box or {}).get("vertices") or []
    if not vertices or page_width <= 0 or page_height <= 0:
        return None

    try:
        xs = [float(vertex["x"]) for vertex in vertices]
        ys = [float(vertex["y"]) for vertex in vertices]
    except (KeyError, TypeError, ValueError):
        return None

    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return {
        "x": round(left / page_width * 100, 4),
        "y": round(top / page_height * 100, 4),
        "width": round((right - left) / page_width * 100, 4),
        "height": round((bottom - top) / page_height * 100, 4),
    }


def compact_ocr_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep OCR evidence useful to the VLM and discard repeated internals.

    The full Vision response repeats text and pixel polygons at block, line,
    word and segment levels. For semantic extraction we keep page dimensions,
    full text, line-level evidence and the table cell grid.
    """
    annotation = result.get("textAnnotation") or {}
    page_width = int(annotation.get("width") or 0)
    page_height = int(annotation.get("height") or 0)

    lines: list[dict[str, Any]] = []
    for block in annotation.get("blocks") or []:
        for line in block.get("lines") or []:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            lines.append(
                {
                    "text": text,
                    "bbox_percent": _bbox_percent(
                        line.get("boundingBox"),
                        page_width=page_width,
                        page_height=page_height,
                    ),
                }
            )

    tables: list[dict[str, Any]] = []
    for table in annotation.get("tables") or []:
        cells: list[dict[str, Any]] = []
        for cell in table.get("cells") or []:
            cells.append(
                {
                    "row": int(cell.get("rowIndex") or 0),
                    "column": int(cell.get("columnIndex") or 0),
                    "row_span": int(cell.get("rowSpan") or 1),
                    "column_span": int(cell.get("columnSpan") or 1),
                    "text": (cell.get("text") or "").strip(),
                    "bbox_percent": _bbox_percent(
                        cell.get("boundingBox"),
                        page_width=page_width,
                        page_height=page_height,
                    ),
                }
            )
        tables.append(
            {
                "row_count": int(table.get("rowCount") or 0),
                "column_count": int(table.get("columnCount") or 0),
                "bbox_percent": _bbox_percent(
                    table.get("boundingBox"),
                    page_width=page_width,
                    page_height=page_height,
                ),
                "cells": cells,
            }
        )

    return {
        "page": 1,
        "page_size_px": {"width": page_width, "height": page_height},
        "full_text": annotation.get("fullText") or "",
        "lines": lines,
        "tables": tables,
    }


def _response_text(response: Any) -> str:
    """Extract text from OpenAI SDK and compatible Responses APIs."""
    text = response.output_text or ""
    if text:
        return text

    parts: list[str] = []
    for output_item in response.output or []:
        for content_item in (getattr(output_item, "content", None) or []):
            value = getattr(content_item, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts)


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse model JSON, tolerating an accidental Markdown code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def _render_pdf_pages(
    path: Path,
    output_dir: Path,
    *,
    remaining_pages: int,
) -> list[PreparedPage]:
    try:
        from pdf2image import convert_from_path
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "Для PDF нужны пакеты pdf2image, pypdf и poppler-utils"
        ) from error

    page_count = len(PdfReader(path).pages)
    if page_count > remaining_pages:
        raise ValueError(
            f"{path.name}: {page_count} страниц, а в задании осталось "
            f"место для {remaining_pages}"
        )
    images = convert_from_path(path, dpi=160, fmt="jpeg", thread_count=1)
    pages = []
    for page_number, image in enumerate(images, start=1):
        page_path = output_dir / f"{path.stem}-page-{page_number}.jpg"
        image.convert("RGB").save(page_path, format="JPEG", quality=90, optimize=True)
        if page_path.stat().st_size > MAX_OCR_FILE_SIZE:
            raise ValueError(
                f"{path.name}, страница {page_number}: после конвертации "
                "размер больше 10 МБ"
            )
        pages.append(PreparedPage(path.name, page_number, page_path))
    return pages


def _prepare_document_pages(
    document_paths: Iterable[Path],
    output_dir: Path,
    *,
    max_pages: int,
) -> list[PreparedPage]:
    pages: list[PreparedPage] = []
    for path in document_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Неподдерживаемый формат: {path.name}")
        if path.stat().st_size > MAX_OCR_FILE_SIZE:
            raise ValueError(f"{path.name}: размер больше 10 МБ")
        remaining = max_pages - len(pages)
        if remaining <= 0:
            raise ValueError(f"В задании больше {max_pages} страниц")
        if path.suffix.lower() == ".pdf":
            pages.extend(_render_pdf_pages(path, output_dir, remaining_pages=remaining))
        else:
            pages.append(PreparedPage(path.name, 1, path))
    if len(pages) > max_pages:
        raise ValueError(f"В задании больше {max_pages} страниц")
    return pages


def _build_model_input(
    *,
    few_shot_pairs: list[dict[str, Any]],
    target_pages: list[PreparedPage],
    ocr_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model_input: list[dict[str, Any]] = []
    for pair in few_shot_pairs:
        model_input.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Это независимый эталонный few-shot пример из "
                            "отдельной поставки. "
                            f"Документ: {pair['document_id']}. "
                            "Следующее сообщение assistant содержит "
                            "эталонную ручную разметку."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(pair["image_path"]),
                    },
                ],
            }
        )
        model_input.append(
            {
                "role": "assistant",
                "content": json.dumps(pair["expected_annotation"], ensure_ascii=False),
            }
        )

    target_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Выше приведены независимые эталонные примеры других "
                "поставок. Не включай их данные в результат. "
                "Обработай только новые страницы ниже. Они относятся "
                "к одной поставке. OCR JSON является вспомогательным; "
                "проверяй его по изображению. Верни только валидный JSON."
            ),
        }
    ]
    for page, ocr_result in zip(target_pages, ocr_results):
        target_content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"Источник: {page.source_name}; страница: "
                        f"{page.page_number}. OCR:\n"
                        + json.dumps(ocr_result, ensure_ascii=False, separators=(",", ":"))
                    ),
                },
                {"type": "input_image", "image_url": _image_to_data_url(page.path)},
            ]
        )
    model_input.append({"role": "user", "content": target_content})
    return model_input


def run_documents_pipeline(
    *,
    api_key: str,
    folder_id: str,
    project_dir: Path,
    document_paths: Iterable[Path],
    job_id: str,
    username: str,
    model: str = DEFAULT_VLM_MODEL,
    max_output_tokens: int = 40000,
    background_timeout_seconds: int = 3600,
    max_pages: int = DEFAULT_MAX_PAGES,
    resume_response_id: str | None = None,
    stage_callback: StageCallback | None = None,
    response_id_callback: ResponseIdCallback | None = None,
) -> dict[str, Any]:
    """Обработать конкретные документы и вернуть итоговый Python-словарь."""
    # job_id и username понадобятся при серверном запуске для аудита. В
    # локальном MVP они уже передаются интерфейсом, но пока не используются.
    del job_id, username  # Поля сохранены в интерфейсе для будущего аудита.

    # Проверяем входные параметры и наличие системного промпта/few-shot данных
    # до обращения к платным внешним API.
    if not api_key:
        raise RuntimeError("API-ключ Yandex не найден")
    document_paths = [Path(path) for path in document_paths]
    if not document_paths:
        raise ValueError("Не переданы документы")
    project_dir = Path(project_dir)
    prompt_path = project_dir / "system-prompt" / "customs-declaration-system-prompt.md"
    image_dir = project_dir / "documents" / "few-shot_raw"
    annotation_dir = project_dir / "documents" / "few-shot_labelled"
    for required in (prompt_path, image_dir, annotation_dir):
        if not required.exists():
            raise FileNotFoundError(f"Не найдено: {required}")

    # OpenAI SDK используется как клиент совместимого Responses API Yandex.
    client = OpenAI(base_url=AI_STUDIO_URL, api_key=api_key)
    model_uri = model if "://" in model else f"gpt://{folder_id}/{model}"
    if resume_response_id:
        # Этот путь позволяет продолжить уже созданный фоновый запрос после
        # временного обрыва соединения, не создавая новый платный запрос.
        if stage_callback:
            stage_callback("processing_qwen")
        response = client.responses.retrieve(resume_response_id)
    else:
        if stage_callback:
            stage_callback("processing_ocr")
        # Каждая пара «эталонное изображение + ручной JSON» становится
        # отдельным few-shot примером для основной модели.
        examples = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not examples:
            raise ValueError(f"В {image_dir} нет few-shot изображений")
        few_shot_pairs = []
        for path in examples:
            annotation_path = _annotation_path(annotation_dir, path)
            few_shot_pairs.append(
                {
                    "document_id": path.name,
                    "image_path": path,
                    "expected_annotation": json.loads(
                        annotation_path.read_text(encoding="utf-8")
                    ),
                }
            )

        # PDF временно преобразуется в изображения страниц. Обычная картинка
        # остаётся одной страницей. Временные файлы удалятся автоматически.
        with tempfile.TemporaryDirectory(prefix="customs-pages-") as temp_dir:
            pages = _prepare_document_pages(
                document_paths, Path(temp_dir), max_pages=max_pages
            )
            # Vision OCR извлекает текст и координаты каждой страницы. Полный
            # громоздкий ответ сокращается перед передачей в Qwen.
            ocr_results = []
            for page in pages:
                raw_ocr = recognize_with_table_ocr(
                    page.path,
                    api_key=api_key,
                    folder_id=folder_id,
                )
                compact = compact_ocr_result(raw_ocr["result"])
                compact["page"] = page.page_number
                compact["document_id"] = page.source_name
                ocr_results.append(compact)
            # Собираем единый запрос: few-shot примеры, OCR нового документа и
            # оригинальные изображения его страниц.
            model_input = _build_model_input(
                few_shot_pairs=few_shot_pairs,
                target_pages=pages,
                ocr_results=ocr_results,
            )
            if stage_callback:
                stage_callback("processing_qwen")
            # Запрос фоновый: он может выполняться дольше обычного HTTP-таймаута.
            response = client.responses.create(
                model=model_uri,
                instructions=prompt_path.read_text(encoding="utf-8"),
                input=model_input,
                reasoning={"effort": "none"},
                text={"format": {"type": "json_object"}, "verbosity": "low"},
                temperature=0.1,
                max_output_tokens=max_output_tokens,
                background=True,
            )
            if response_id_callback:
                response_id_callback(response.id)

    # Периодически запрашиваем состояние фоновой задачи. Ограниченное число
    # сетевых сбоев допускается, чтобы единичный обрыв не прерывал обработку.
    deadline = time.monotonic() + background_timeout_seconds
    terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
    connection_failures = 0
    while response.status not in terminal_statuses:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Qwen не завершила задачу {response.id} за "
                f"{background_timeout_seconds} секунд"
            )
        time.sleep(5)
        try:
            response = client.responses.retrieve(response.id)
            connection_failures = 0
        except APIConnectionError as error:
            connection_failures += 1
            if connection_failures >= 12:
                raise RuntimeError(
                    f"Не удалось проверить задачу {response.id} после 12 попыток"
                ) from error

    # Только completed считается успешным результатом. incomplete обычно
    # означает, что модели не хватило лимита выходных токенов.
    if response.status != "completed":
        raise RuntimeError(
            f"Qwen status={response.status}; incomplete={response.incomplete_details}"
        )
    result_text = _response_text(response)
    if not result_text:
        raise RuntimeError("Qwen не вернула текст")
    # Интерфейс работает со словарём, поэтому текст модели обязательно должен
    # быть валидным JSON. Markdown-ограждение удаляет _parse_json_response.
    try:
        return _parse_json_response(result_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("Qwen вернула невалидный JSON") from error


def _legacy_run_pipeline(
    *,
    api_key: str,
    folder_id: str,
    project_dir: Path,
    model: str = DEFAULT_VLM_MODEL,
    max_output_tokens: int = 40000,
    background_timeout_seconds: int = 3600,
    resume_response_id: str | None = None,
) -> dict[str, Any]:
    """Teach with image/JSON pairs and extract new unannotated documents."""
    project_dir = Path(project_dir)
    documents_dir = project_dir / "documents"
    few_shot_images_dir = documents_dir / "few-shot_raw"
    few_shot_annotations_dir = documents_dir / "few-shot_labelled"
    input_documents_dir = documents_dir / "input-documents"
    system_prompt_path = (
        project_dir / "system-prompt" / "customs-declaration-system-prompt.md"
    )
    output_dir = project_dir / "output"
    system_output_dir = output_dir / "system-folder"

    if not api_key:
        raise RuntimeError("API-ключ Yandex не найден")
    if not system_prompt_path.is_file():
        raise FileNotFoundError(f"Не найден системный промпт: {system_prompt_path}")
    if not few_shot_images_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка: {few_shot_images_dir}")
    if not few_shot_annotations_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка: {few_shot_annotations_dir}")
    if not input_documents_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка: {input_documents_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    system_output_dir.mkdir(parents=True, exist_ok=True)

    example_paths = sorted(
        path
        for path in few_shot_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not example_paths:
        raise ValueError(f"В {few_shot_images_dir} нет JPG, PNG или WebP")

    target_paths = sorted(
        path
        for path in input_documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not target_paths:
        raise ValueError(
            f"В {input_documents_dir} нет новых изображений JPG, PNG или WebP"
        )

    # Одиночные тесты сохраняем под именем исходного файла,
    # чтобы результат автоматически сопоставлялся с ground truth.
    run_name = (
        target_paths[0].stem
        if len(target_paths) == 1
        else "customs-declaration-result"
    )
    final_output_path = output_dir / f"{run_name}.json"

    few_shot_pairs: list[dict[str, Any]] = []

    print(f"Few-shot примеров: {len(example_paths)}")
    for index, path in enumerate(example_paths, start=1):
        annotation_path = _annotation_path(few_shot_annotations_dir, path)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))

        few_shot_pairs.append(
            {
                "document_id": path.name,
                "image_path": path,
                "expected_annotation": annotation,
            }
        )
        print(
            f"[{index}/{len(example_paths)}] {path.name} -> "
            f"{annotation_path.name}"
        )

    print(f"Новых документов: {len(target_paths)}")
    for path in target_paths:
        print(f"    {path.name}")

    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    # Each independent example is a real few-shot turn:
    # user = original image; assistant = manually verified semantic JSON.
    model_input: list[dict[str, Any]] = []
    for pair in few_shot_pairs:
        model_input.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Это независимый эталонный few-shot пример из "
                            "отдельной поставки. "
                            f"Документ: {pair['document_id']}. "
                            "Извлеки сведения из изображения. Следующее "
                            "сообщение assistant содержит эталонную ручную "
                            "разметку этого изображения."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(pair["image_path"]),
                    },
                ],
            }
        )
        model_input.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    pair["expected_annotation"],
                    ensure_ascii=False,
                ),
            }
        )

    target_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Выше приведены независимые эталонные примеры разных "
                "поставок. Не включай сведения из них в результат. Теперь "
                "обработай только новые документы, приложенные к этому "
                "сообщению. Новые документы относятся к одной поставке. "
                "Извлеки их данные по системным правилам и few-shot-примерам, "
                "сформируй documents и declaration_draft, сохрани evidence "
                "и bbox_percent. Верни только валидный JSON. Новые файлы: "
                + ", ".join(path.name for path in target_paths)
            ),
        }
    ]
    target_content.extend(
        {
            "type": "input_image",
            "image_url": _image_to_data_url(path),
        }
        for path in target_paths
    )
    model_input.append({"role": "user", "content": target_content})

    model_uri = model if "://" in model else f"gpt://{folder_id}/{model}"
    print(f"Qwen3.6: мультимодальная обработка ({model_uri})...")
    client = OpenAI(base_url=AI_STUDIO_URL, api_key=api_key)
    response_id_path = system_output_dir / f"{run_name}-qwen-response-id.txt"
    if resume_response_id:
        response = client.responses.retrieve(resume_response_id)
        print(f"Продолжаю существующую задачу: {response.id}")
    else:
        response = client.responses.create(
            model=model_uri,
            instructions=system_prompt,
            input=model_input,
            reasoning={"effort": "none"},
            text={
                "format": {"type": "json_object"},
                "verbosity": "low",
            },
            temperature=0.1,
            max_output_tokens=max_output_tokens,
            background=True,
        )
        response_id_path.write_text(response.id, encoding="utf-8")
        print(f"Фоновая задача создана: {response.id}")
        print(f"ID задачи сохранён: {response_id_path}")
    deadline = time.monotonic() + background_timeout_seconds
    previous_status: str | None = None
    poll_number = 0
    connection_failures = 0
    terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}

    while response.status not in terminal_statuses:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Qwen3.6 не завершила задачу {response.id} за "
                f"{background_timeout_seconds} секунд. Задача продолжает "
                "выполняться в Yandex AI Studio."
            )
        if response.status != previous_status or poll_number % 12 == 0:
            print(f"Статус задачи {response.id}: {response.status}")
            previous_status = response.status
        time.sleep(5)
        try:
            response = client.responses.retrieve(response.id)
            connection_failures = 0
        except APIConnectionError as error:
            connection_failures += 1
            print(
                "Временная ошибка соединения при проверке статуса "
                f"({connection_failures}/12): {error}"
            )
            if connection_failures >= 12:
                raise RuntimeError(
                    f"Не удалось проверить задачу {response.id}. "
                    f"Её ID сохранён в {response_id_path}; задача может "
                    "продолжать выполняться в Yandex AI Studio."
                ) from error
        poll_number += 1

    result_text = _response_text(response)
    if response.status != "completed":
        partial_path: Path | None = None
        if result_text:
            partial_path = final_output_path.with_suffix(".partial.txt")
            partial_path.write_text(result_text, encoding="utf-8")
        raise RuntimeError(
            f"Qwen3.6 status={response.status}; "
            f"error={response.error}; incomplete={response.incomplete_details}; "
            f"partial_output={partial_path}"
        )

    if not result_text:
        raise RuntimeError(
            "Qwen3.6 не вернула текст: "
            + response.model_dump_json(indent=2)
        )

    try:
        result = _parse_json_response(result_text)
    except json.JSONDecodeError as error:
        raw_path = final_output_path.with_suffix(".raw.txt")
        raw_path.write_text(result_text, encoding="utf-8")
        raise RuntimeError(f"Невалидный JSON; сырой ответ: {raw_path}") from error

    final_output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Готово: {final_output_path}")
    return result


def run_pipeline(
    *,
    api_key: str,
    folder_id: str,
    project_dir: Path,
    model: str = DEFAULT_VLM_MODEL,
    max_output_tokens: int = 40000,
    background_timeout_seconds: int = 3600,
    resume_response_id: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible notebook wrapper around the server pipeline."""
    project_dir = Path(project_dir)
    input_dir = project_dir / "documents" / "input-documents"
    target_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not target_paths:
        raise ValueError(f"В {input_dir} нет новых документов")
    run_name = target_paths[0].stem if len(target_paths) == 1 else "batch"
    output_dir = project_dir / "output"
    system_dir = output_dir / "system-folder"
    output_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)
    response_id_path = system_dir / f"{run_name}-qwen-response-id.txt"

    result = run_documents_pipeline(
        api_key=api_key,
        folder_id=folder_id,
        project_dir=project_dir,
        document_paths=target_paths,
        job_id=run_name,
        username="notebook",
        model=model,
        max_output_tokens=max_output_tokens,
        background_timeout_seconds=background_timeout_seconds,
        resume_response_id=resume_response_id,
        stage_callback=lambda stage: print(f"Статус: {stage}"),
        response_id_callback=lambda response_id: response_id_path.write_text(
            response_id, encoding="utf-8"
        ),
    )
    output_path = output_dir / f"{run_name}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Готово: {output_path}")
    return result
