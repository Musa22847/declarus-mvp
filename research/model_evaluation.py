"""Метрики качества извлечения данных из упаковочных листов."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


TEXT_FIELDS = {
    "sender",
    "consignor_address",
    "consignee",
    "consignee_address",
    "item_article",
    "item_description",
}

ITEM_ALIASES = {
    "gross_weight": "item_gross_weight",
    "net_weight": "item_net_weight",
    "item_packing": "item_boxes_count",
}

TOTAL_ALIASES = {
    "total_packages": "item_boxes_count_total",
}

PREDICTION_STEM_SUFFIXES = (
    " by model",
    "_by_model",
    "-by-model",
)


def normalize_text(value: Any) -> str:
    """Нормализация для сравнения OCR-текста без влияния регистра и пробелов."""
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return " ".join(text.casefold().split())


def normalize_business_text(value: Any) -> str:
    """Нормализация значения без влияния пробелов вокруг пунктуации."""
    text = normalize_text(value)
    return re.sub(r"\s*([,;:])\s*", r"\1", text)


def levenshtein_distance(reference: Iterable[Any], prediction: Iterable[Any]) -> int:
    """Расстояние Левенштейна для последовательностей символов или слов."""
    reference = list(reference)
    prediction = list(prediction)
    previous = list(range(len(prediction) + 1))
    for row, reference_token in enumerate(reference, start=1):
        current = [row]
        for column, prediction_token in enumerate(prediction, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_token != prediction_token),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: Any, prediction: Any, *, normalize: bool = True) -> float:
    """CER = (замены + удаления + вставки) / число символов эталона."""
    reference_text = normalize_text(reference) if normalize else str(reference or "")
    prediction_text = normalize_text(prediction) if normalize else str(prediction or "")
    if not reference_text:
        return 0.0 if not prediction_text else 1.0
    return levenshtein_distance(reference_text, prediction_text) / len(reference_text)


def word_error_rate(reference: Any, prediction: Any, *, normalize: bool = True) -> float:
    """WER = (замены + удаления + вставки) / число слов эталона."""
    reference_text = normalize_text(reference) if normalize else str(reference or "")
    prediction_text = normalize_text(prediction) if normalize else str(prediction or "")
    reference_words = reference_text.split()
    prediction_words = prediction_text.split()
    if not reference_words:
        return 0.0 if not prediction_words else 1.0
    return levenshtein_distance(reference_words, prediction_words) / len(reference_words)


def _unwrap(value: Any) -> Any:
    return value.get("value") if isinstance(value, dict) and "value" in value else value


def _is_missing_value(value: Any) -> bool:
    """Treat explicit JSON null wrappers as an absent extracted field."""
    if value is None:
        return True
    return (
        isinstance(value, dict)
        and "value" in value
        and value.get("value") is None
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        result = {"value": value.get("value")}
        if value.get("unit") not in (None, ""):
            result["unit"] = value["unit"]
        return result
    return value


def _normalize_unit(unit: Any) -> str:
    unit = normalize_text(unit).replace(".", "")
    aliases = {
        "kgs": "kg",
        "kilogram": "kg",
        "kilograms": "kg",
        "pcs": "pcs",
        "pc": "pcs",
        "pieces": "pcs",
        "piece": "pcs",
        "шт": "pcs",
        "sets": "set",
        "компл": "set",
        "packages": "package",
        "pack": "package",
        "box": "box",
        "boxes": "box",
        "carton": "box",
        "cartons": "box",
        "ctn": "box",
        "ctns": "box",
        "pcs/box": "pcs/box",
        "pcs/ctn": "pcs/box",
        "pcs per box": "pcs/box",
    }
    return aliases.get(unit, unit)


def values_equal(reference: Any, prediction: Any, *, numeric_tolerance: float = 1e-6) -> bool:
    """Точное бизнес-сравнение: числа с допуском, строки после нормализации."""
    reference = _canonical_value(reference)
    prediction = _canonical_value(prediction)

    if isinstance(reference, dict) and "value" in reference:
        prediction_value = prediction.get("value") if isinstance(prediction, dict) else prediction
        if not values_equal(reference["value"], prediction_value, numeric_tolerance=numeric_tolerance):
            return False
        if "unit" in reference:
            prediction_unit = prediction.get("unit") if isinstance(prediction, dict) else None
            return _normalize_unit(reference["unit"]) == _normalize_unit(prediction_unit)
        return True

    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        prediction = _unwrap(prediction)
        try:
            return math.isclose(float(reference), float(prediction), abs_tol=numeric_tolerance)
        except (TypeError, ValueError):
            return False
    return normalize_business_text(reference) == normalize_business_text(_unwrap(prediction))


def _party_fields(document: dict[str, Any]) -> dict[str, Any]:
    result = {}
    sender = document.get("sender")
    consignee = document.get("consignee")
    if isinstance(sender, dict):
        result["sender"] = _unwrap(sender.get("name"))
        result["consignor_address"] = _unwrap(sender.get("address"))
    elif sender is not None:
        result["sender"] = _unwrap(sender)
    if isinstance(consignee, dict):
        result["consignee"] = _unwrap(consignee.get("name"))
        result["consignee_address"] = _unwrap(consignee.get("address"))
    elif consignee is not None:
        result["consignee"] = _unwrap(consignee)
    for field in ("consignor_address", "consignee_address"):
        if field in document:
            result[field] = _unwrap(document[field])
    return {key: value for key, value in result.items() if value is not None}


def canonicalize_gold(gold: dict[str, Any]) -> dict[str, Any]:
    """Извлекает оцениваемую часть из компактного ground-truth JSON."""
    payload = gold.get("expected_output", gold)
    result = _party_fields(payload)
    for field, value in (payload.get("fields") or {}).items():
        if field in TEXT_FIELDS and not _is_missing_value(value):
            result[field] = _canonical_value(value)

    result["items"] = []
    for item in payload.get("items") or []:
        canonical_item = {}
        for field, value in item.items():
            canonical_field = ITEM_ALIASES.get(field, field)
            if canonical_field.startswith("item_") and not _is_missing_value(value):
                canonical_item[canonical_field] = _canonical_value(value)
        result["items"].append(canonical_item)

    result["totals"] = {
        TOTAL_ALIASES.get(field, field): _canonical_value(value)
        for field, value in (payload.get("totals") or {}).items()
        if not _is_missing_value(value)
    }
    return result


def canonicalize_prediction(prediction: dict[str, Any], *, document_index: int = 0) -> dict[str, Any]:
    """Приводит ответ MVP или уже компактный JSON к формату ground truth."""
    if "expected_output" in prediction:
        return canonicalize_gold(prediction)
    if "documents" in prediction:
        documents = prediction.get("documents") or []
        if document_index >= len(documents):
            return {"items": [], "totals": {}}
        prediction = documents[document_index]
    elif "declaration_draft" in prediction:
        prediction = prediction["declaration_draft"]

    result = _party_fields(prediction)
    result["items"] = []
    for item in prediction.get("items") or []:
        canonical_item = {}
        for field, value in item.items():
            canonical_field = ITEM_ALIASES.get(field, field)
            if canonical_field.startswith("item_") and not _is_missing_value(value):
                canonical_item[canonical_field] = _canonical_value(value)
        result["items"].append(canonical_item)
    result["totals"] = {
        TOTAL_ALIASES.get(key, key): _canonical_value(value)
        for key, value in (prediction.get("totals") or {}).items()
        if not _is_missing_value(value)
    }
    return result


def _bbox_from_value(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    bbox = value.get("bbox_percent")
    if isinstance(bbox, dict):
        return bbox
    for evidence in value.get("evidence") or []:
        bbox = evidence.get("bbox_percent")
        if isinstance(bbox, dict):
            return bbox
    return None


def _extract_gold_bboxes(gold: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Собирает эталонные bbox отдельно от сравнения значений."""
    payload = gold.get("expected_output", gold)
    evidence = gold.get("evidence") or {}
    result = {}

    for field, value in (payload.get("fields") or {}).items():
        bbox = _bbox_from_value(value) or _bbox_from_value((evidence.get("fields") or {}).get(field))
        if bbox:
            result[field] = bbox

    for item_index, item in enumerate(payload.get("items") or []):
        evidence_items = evidence.get("items") or []
        evidence_item = evidence_items[item_index] if item_index < len(evidence_items) else {}
        for field, value in item.items():
            canonical_field = ITEM_ALIASES.get(field, field)
            bbox = _bbox_from_value(value) or _bbox_from_value(evidence_item.get(field))
            if bbox:
                result[f"items[{item_index}].{canonical_field}"] = bbox

    for field, value in (payload.get("totals") or {}).items():
        canonical_field = TOTAL_ALIASES.get(field, field)
        bbox = _bbox_from_value(value) or _bbox_from_value((evidence.get("totals") or {}).get(field))
        if bbox:
            result[f"totals.{canonical_field}"] = bbox
    return result


def _prediction_document(prediction: dict[str, Any], document_index: int = 0) -> dict[str, Any]:
    if "documents" in prediction:
        documents = prediction.get("documents") or []
        return documents[document_index] if document_index < len(documents) else {}
    if "declaration_draft" in prediction:
        return prediction["declaration_draft"]
    return prediction


def _extract_prediction_bboxes(prediction: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Собирает bbox модели в тех же путях, что и ground truth."""
    prediction = _prediction_document(prediction)
    result = {}
    party_paths = {
        "sender": ("sender", "name"),
        "consignor_address": ("sender", "address"),
        "consignee": ("consignee", "name"),
        "consignee_address": ("consignee", "address"),
    }
    for field, (party, attribute) in party_paths.items():
        party_value = prediction.get(party)
        value = party_value.get(attribute) if isinstance(party_value, dict) else None
        bbox = _bbox_from_value(value)
        if bbox:
            result[field] = bbox

    for item_index, item in enumerate(prediction.get("items") or []):
        for field, value in item.items():
            canonical_field = ITEM_ALIASES.get(field, field)
            if not canonical_field.startswith("item_"):
                continue
            bbox = _bbox_from_value(value)
            if bbox:
                result[f"items[{item_index}].{canonical_field}"] = bbox

    for field, value in (prediction.get("totals") or {}).items():
        canonical_field = TOTAL_ALIASES.get(field, field)
        bbox = _bbox_from_value(value)
        if bbox:
            result[f"totals.{canonical_field}"] = bbox
    return result


def bbox_iou(reference: dict[str, float], prediction: dict[str, float]) -> float:
    """Intersection over Union для двух bbox в координатах 0–100%."""
    reference_right = reference["x"] + reference["width"]
    reference_bottom = reference["y"] + reference["height"]
    prediction_right = prediction["x"] + prediction["width"]
    prediction_bottom = prediction["y"] + prediction["height"]
    intersection_width = max(
        0.0,
        min(reference_right, prediction_right) - max(reference["x"], prediction["x"]),
    )
    intersection_height = max(
        0.0,
        min(reference_bottom, prediction_bottom) - max(reference["y"], prediction["y"]),
    )
    intersection = intersection_width * intersection_height
    union = (
        reference["width"] * reference["height"]
        + prediction["width"] * prediction["height"]
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _item_identity(item: dict[str, Any]) -> str:
    article = normalize_text(_unwrap(item.get("item_article")))
    description = normalize_text(_unwrap(item.get("item_description")))
    return f"{article} | {description}".strip(" |")


def match_items(gold_items: list[dict[str, Any]], predicted_items: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Сопоставляет строки по артикулу/описанию, а не только по индексу массива."""
    candidates = []
    for gold_index, gold_item in enumerate(gold_items):
        for predicted_index, predicted_item in enumerate(predicted_items):
            similarity = SequenceMatcher(
                None, _item_identity(gold_item), _item_identity(predicted_item)
            ).ratio()
            candidates.append((similarity, gold_index, predicted_index))
    matches = []
    used_gold = set()
    used_prediction = set()
    for _, gold_index, predicted_index in sorted(candidates, reverse=True):
        if gold_index not in used_gold and predicted_index not in used_prediction:
            matches.append((gold_index, predicted_index))
            used_gold.add(gold_index)
            used_prediction.add(predicted_index)
    return sorted(matches)


def _record_comparison(
    field: str,
    reference: Any,
    prediction: Any,
    counters: Counter,
    errors: list[dict[str, Any]],
    text_pairs: list[tuple[str, str, str]],
    path: str,
) -> bool:
    present = prediction is not None
    correct = present and values_equal(reference, prediction)
    counters["gold"] += 1
    if correct:
        counters["tp"] += 1
    else:
        counters["fn"] += 1
        if present:
            counters["fp"] += 1
        errors.append({"path": path, "expected": reference, "predicted": prediction})
    if field in TEXT_FIELDS and isinstance(_unwrap(reference), str):
        text_pairs.append((path, str(_unwrap(reference) or ""), str(_unwrap(prediction) or "")))
    return correct


def evaluate_document(gold: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Считает метрики одного документа."""
    gold_bboxes = _extract_gold_bboxes(gold)
    prediction_bboxes = _extract_prediction_bboxes(prediction)
    gold = canonicalize_gold(gold)
    prediction = canonicalize_prediction(prediction)
    counters = Counter()
    errors = []
    text_pairs = []

    for field, reference in gold.items():
        if field in {"items", "totals"}:
            continue
        _record_comparison(field, reference, prediction.get(field), counters, errors, text_pairs, field)

    for field, reference in gold.get("totals", {}).items():
        _record_comparison(
            field, reference, prediction.get("totals", {}).get(field), counters, errors,
            text_pairs, f"totals.{field}",
        )

    gold_items = gold.get("items", [])
    predicted_items = prediction.get("items", [])
    matches = match_items(gold_items, predicted_items)
    matched_gold = {gold_index for gold_index, _ in matches}
    matched_prediction = {predicted_index for _, predicted_index in matches}
    exact_items = 0
    item_details = []
    in_scope_item_fields = set().union(*(item.keys() for item in gold_items)) if gold_items else set()

    for gold_index, predicted_index in matches:
        gold_item = gold_items[gold_index]
        predicted_item = predicted_items[predicted_index]
        item_correct = True
        for field, reference in gold_item.items():
            item_correct &= _record_comparison(
                field, reference, predicted_item.get(field), counters, errors, text_pairs,
                f"items[{gold_index}].{field}",
            )
        extra_fields = (set(predicted_item) & in_scope_item_fields) - set(gold_item)
        counters["fp"] += len(extra_fields)
        item_correct &= not extra_fields
        exact_items += int(item_correct)
        item_details.append({
            "gold_index": gold_index,
            "prediction_index": predicted_index,
            "exact_match": bool(item_correct),
        })

    for gold_index in sorted(set(range(len(gold_items))) - matched_gold):
        for field, reference in gold_items[gold_index].items():
            _record_comparison(
                field, reference, None, counters, errors, text_pairs,
                f"items[{gold_index}].{field}",
            )
    for predicted_index in set(range(len(predicted_items))) - matched_prediction:
        counters["fp"] += len(set(predicted_items[predicted_index]) & in_scope_item_fields) or 1

    precision = counters["tp"] / (counters["tp"] + counters["fp"]) if counters["tp"] + counters["fp"] else 1.0
    recall = counters["tp"] / counters["gold"] if counters["gold"] else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    char_errors = sum(levenshtein_distance(normalize_text(r), normalize_text(p)) for _, r, p in text_pairs)
    reference_chars = sum(len(normalize_text(r)) for _, r, _ in text_pairs)
    word_errors = sum(levenshtein_distance(normalize_text(r).split(), normalize_text(p).split()) for _, r, p in text_pairs)
    reference_words = sum(len(normalize_text(r).split()) for _, r, _ in text_pairs)

    item_index_mapping = {gold_index: predicted_index for gold_index, predicted_index in matches}
    bbox_details = []
    bbox_iou_sum = 0.0
    bbox_present = 0
    bbox_matches_at_0_5 = 0
    for path, reference_bbox in gold_bboxes.items():
        prediction_path = path
        item_match = re.match(r"items\[(\d+)]\.(.+)", path)
        if item_match:
            gold_index = int(item_match.group(1))
            field = item_match.group(2)
            predicted_index = item_index_mapping.get(gold_index)
            prediction_path = (
                f"items[{predicted_index}].{field}"
                if predicted_index is not None
                else ""
            )
        predicted_bbox = prediction_bboxes.get(prediction_path)
        iou = bbox_iou(reference_bbox, predicted_bbox) if predicted_bbox else 0.0
        bbox_iou_sum += iou
        bbox_present += int(predicted_bbox is not None)
        bbox_matches_at_0_5 += int(iou >= 0.5)
        bbox_details.append({
            "path": path,
            "iou": iou,
            "expected": reference_bbox,
            "predicted": predicted_bbox,
        })

    bbox_gold = len(gold_bboxes)

    item_denominator = max(len(gold_items), len(predicted_items), 1)
    document_exact = not errors and counters["fp"] == 0 and len(gold_items) == len(predicted_items)
    return {
        "json_valid": True,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "field_exact_match_rate": counters["tp"] / counters["gold"] if counters["gold"] else 1.0,
        "cer": char_errors / reference_chars if reference_chars else 0.0,
        "wer": word_errors / reference_words if reference_words else 0.0,
        "bbox": {
            "mean_iou": bbox_iou_sum / bbox_gold if bbox_gold else None,
            "mean_iou_on_present": bbox_iou_sum / bbox_present if bbox_present else None,
            "match_rate_at_iou_0_5": bbox_matches_at_0_5 / bbox_gold if bbox_gold else None,
            "counts": {
                "gold": bbox_gold,
                "predicted_for_gold": bbox_present,
                "missing": bbox_gold - bbox_present,
                "matched_at_iou_0_5": bbox_matches_at_0_5,
            },
            "details": bbox_details,
        },
        "item_exact_match_rate": exact_items / item_denominator,
        "item_count_match": len(gold_items) == len(predicted_items),
        "document_exact_match": document_exact,
        "counts": {
            "true_positive": counters["tp"],
            "false_positive": counters["fp"],
            "false_negative": counters["fn"],
            "gold_fields": counters["gold"],
            "gold_items": len(gold_items),
            "predicted_items": len(predicted_items),
            "exact_items": exact_items,
        },
        "item_matches": item_details,
        "errors": errors,
    }


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def evaluate_files(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    *,
    document_index: int = 0,
) -> dict[str, Any]:
    """Загружает два JSON и оценивает выбранный документ ответа MVP."""
    gold = load_json(ground_truth_path)
    try:
        prediction = load_json(prediction_path)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as error:
        canonical_gold = canonicalize_gold(gold)
        gold_fields = (
            sum(1 for key in canonical_gold if key not in {"items", "totals"})
            + len(canonical_gold.get("totals", {}))
            + sum(len(item) for item in canonical_gold.get("items", []))
        )
        return {
            "json_valid": False,
            "field_precision": 0.0,
            "field_recall": 0.0,
            "field_f1": 0.0,
            "field_exact_match_rate": 0.0,
            "cer": 1.0,
            "wer": 1.0,
            "bbox": {
                "mean_iou": 0.0,
                "mean_iou_on_present": None,
                "match_rate_at_iou_0_5": 0.0,
                "counts": {
                    "gold": len(_extract_gold_bboxes(gold)),
                    "predicted_for_gold": 0,
                    "missing": len(_extract_gold_bboxes(gold)),
                    "matched_at_iou_0_5": 0,
                },
                "details": [],
            },
            "item_exact_match_rate": 0.0,
            "item_count_match": False,
            "document_exact_match": False,
            "counts": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": gold_fields,
                "gold_fields": gold_fields,
                "gold_items": len(canonical_gold.get("items", [])),
                "predicted_items": 0,
                "exact_items": 0,
            },
            "item_matches": [],
            "errors": [{"path": "$", "expected": "valid JSON", "predicted": str(error)}],
        }
    if "documents" in prediction:
        documents = prediction.get("documents") or []
        prediction = documents[document_index] if document_index < len(documents) else {}
    return evaluate_document(gold, prediction)


def evaluate_dataset(pairs: Iterable[tuple[str | Path, str | Path]]) -> dict[str, Any]:
    """Агрегирует результаты нескольких независимых тестовых документов."""
    documents = [evaluate_files(gold, prediction) for gold, prediction in pairs]
    totals = Counter()
    for document in documents:
        totals.update(document["counts"])
    precision = totals["true_positive"] / (totals["true_positive"] + totals["false_positive"]) if totals["true_positive"] + totals["false_positive"] else 1.0
    recall = totals["true_positive"] / totals["gold_fields"] if totals["gold_fields"] else 1.0
    bbox_gold = sum(document["bbox"]["counts"]["gold"] for document in documents)
    bbox_present = sum(
        document["bbox"]["counts"]["predicted_for_gold"] for document in documents
    )
    bbox_matches = sum(
        document["bbox"]["counts"]["matched_at_iou_0_5"] for document in documents
    )
    bbox_iou_sum = sum(
        (document["bbox"]["mean_iou"] or 0.0) * document["bbox"]["counts"]["gold"]
        for document in documents
    )
    return {
        "documents": len(documents),
        "json_success_rate": sum(document["json_valid"] for document in documents) / len(documents) if documents else 0.0,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_cer": sum(document["cer"] for document in documents) / len(documents) if documents else 0.0,
        "mean_wer": sum(document["wer"] for document in documents) / len(documents) if documents else 0.0,
        "bbox_mean_iou": bbox_iou_sum / bbox_gold if bbox_gold else None,
        "bbox_match_rate_at_iou_0_5": bbox_matches / bbox_gold if bbox_gold else None,
        "bbox_counts": {
            "gold": bbox_gold,
            "predicted_for_gold": bbox_present,
            "missing": bbox_gold - bbox_present,
            "matched_at_iou_0_5": bbox_matches,
        },
        "item_exact_match_rate": totals["exact_items"] / max(totals["gold_items"], totals["predicted_items"], 1),
        "document_exact_match_rate": sum(document["document_exact_match"] for document in documents) / len(documents) if documents else 0.0,
        "per_document": documents,
    }


def _normalized_document_key(path: Path, *, prediction: bool = False) -> str:
    """Return a Unicode-safe key used to pair gold and prediction files."""
    key = unicodedata.normalize("NFKC", path.stem).casefold().strip()
    if prediction:
        for suffix in PREDICTION_STEM_SUFFIXES:
            if key.endswith(suffix):
                key = key[: -len(suffix)].rstrip()
                break
    return key


def _index_json_files(
    directory: Path,
    *,
    prediction: bool = False,
    excluded_names: set[str] | None = None,
) -> dict[str, Path]:
    excluded_names = {
        unicodedata.normalize("NFKC", name).casefold()
        for name in (excluded_names or set())
    }
    result: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        normalized_name = unicodedata.normalize("NFKC", path.name).casefold()
        if path.name.startswith(".") or normalized_name in excluded_names:
            continue
        key = _normalized_document_key(path, prediction=prediction)
        if key in result:
            raise ValueError(
                f"Неоднозначное имя документа {key!r}: "
                f"{result[key].name}, {path.name}"
            )
        result[key] = path
    return result


def _compact_document_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the useful per-document metrics without large bbox dumps."""
    compact = {
        key: value
        for key, value in metrics.items()
        if key not in {"bbox", "item_matches"}
    }
    bbox = metrics.get("bbox") or {}
    compact["bbox"] = {
        key: value
        for key, value in bbox.items()
        if key != "details"
    }
    return compact


def evaluate_directory(
    ground_truth_dir: str | Path,
    predictions_dir: str | Path,
    *,
    report_name: str = "quality-report.json",
) -> dict[str, Any]:
    """Evaluate every same-named gold/prediction pair in two directories.

    Prediction names may equal the gold name or end with `` by model``.
    Unicode normalization makes macOS NFC/NFD filename variants equivalent.
    """
    ground_truth_dir = Path(ground_truth_dir)
    predictions_dir = Path(predictions_dir)
    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка ground truth: {ground_truth_dir}")
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка результатов: {predictions_dir}")

    gold_by_key = _index_json_files(ground_truth_dir)
    prediction_by_key = _index_json_files(
        predictions_dir,
        prediction=True,
        excluded_names={report_name},
    )
    matched_keys = sorted(set(gold_by_key) & set(prediction_by_key))
    pairs = [
        (gold_by_key[key], prediction_by_key[key])
        for key in matched_keys
    ]
    dataset_metrics = evaluate_dataset(pairs)
    per_document_metrics = dataset_metrics.pop("per_document")

    documents = []
    for key, metrics in zip(matched_keys, per_document_metrics):
        documents.append(
            {
                "document": gold_by_key[key].stem,
                "ground_truth_file": gold_by_key[key].name,
                "prediction_file": prediction_by_key[key].name,
                "metrics": _compact_document_metrics(metrics),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": dataset_metrics,
        "documents": documents,
        "unmatched_files": {
            "ground_truth_without_prediction": [
                gold_by_key[key].name
                for key in sorted(set(gold_by_key) - set(prediction_by_key))
            ],
            "prediction_without_ground_truth": [
                prediction_by_key[key].name
                for key in sorted(set(prediction_by_key) - set(gold_by_key))
            ],
        },
    }


def save_quality_report(report: dict[str, Any], path: str | Path) -> Path:
    """Atomically write the combined quality report as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path
