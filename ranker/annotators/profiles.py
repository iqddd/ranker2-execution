from __future__ import annotations

from pathlib import Path

from .core import AnnotationError, AnnotationProfile, LabelChoice, PassConfig, PROJECT_DIR, run_profile as _run


EMPTY_NUMERIC = {"", "+1", "-1", "0", "?"}


def _all_rows(_row: dict[str, str]) -> bool:
    return True


def _image_key(row: dict[str, str]) -> str:
    return row["image_id"]


def _validate_anchor(rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column in ANCHOR_COLUMNS[1:]:
            if row[column] not in EMPTY_NUMERIC:
                raise AnnotationError(
                    f"Строка {row_number}, {column}: допустимы '+1', '-1', '0', '?' или пусто."
                )


ANCHOR_COLUMNS = (
    "image_id",
    "buttock_size",
    "nudity",
    "nipple_visibility",
    "manual_breast_lift",
    "three_quarter_view",
)
NUMERIC_CHOICES = (
    LabelChoice("+1", "Ясный положительный", "1"),
    LabelChoice("-1", "Ясный отрицательный", "2"),
    LabelChoice("0", "Неприменимо / не видно", "0"),
    LabelChoice("?", "Неоднозначно", "?"),
)
ANCHOR_PASSES = (
    PassConfig(
        "buttock_size",
        "Размер ягодиц",
        "<b>+1</b> — ягодицы хорошо видны и явно очень большие.<br>"
        "<b>−1</b> — ягодицы хорошо видны и имеют небольшой или обычный размер.<br>"
        "<b>0</b> — ягодицы не видны либо недостаточно видны для оценки.<br>"
        "<b>?</b> — ягодицы видны, но размер уверенно определить нельзя.",
        "buttock_size", NUMERIC_CHOICES, _all_rows,
    ),
    PassConfig(
        "nudity",
        "Обнажённость груди",
        "<b>+1</b> — обнажённая грудь ясно видна, без бюстгальтера, купальника или непрозрачной одежды.<br>"
        "<b>−1</b> — область груди видна и закрыта одеждой, бельём или купальником.<br>"
        "<b>0</b> — грудь не попала в кадр либо показана только спина.<br>"
        "<b>?</b> — прикрытие или ракурс не позволяют решить уверенно.",
        "nudity", NUMERIC_CHOICES, _all_rows,
    ),
    PassConfig(
        "nipple_visibility",
        "Видимость сосков",
        "<b>+1</b> — хотя бы один сосок ясно виден.<br>"
        "<b>−1</b> — область груди видна, но оба соска закрыты.<br>"
        "<b>0</b> — область груди не видна либо показана только спина.<br>"
        "<b>?</b> — разрешение, волосы, руки или одежда не позволяют решить уверенно.",
        "nipple_visibility", NUMERIC_CHOICES, _all_rows,
    ),
    PassConfig(
        "manual_breast_lift",
        "Подъём груди руками",
        "Оценивайте только явное физическое воздействие рукой или руками.<br>"
        "<b>+1</b> — одна или обе груди явно приподнимаются, поддерживаются или сжимаются рукой либо руками.<br>"
        "<b>−1</b> — область груди хорошо видна, но руки не поддерживают и не сжимают грудь.<br>"
        "<b>0</b> — грудь не видна, руки и грудь не попали в кадр либо одежда полностью скрывает возможный контакт.<br>"
        "<b>?</b> — рука находится рядом с грудью, но контакт нельзя определить уверенно.",
        "manual_breast_lift", NUMERIC_CHOICES, _all_rows,
    ),
    PassConfig(
        "three_quarter_view",
        "Ракурс три четверти",
        "Оценивайте направление корпуса, а не головы.<br>"
        "<b>+1</b> — корпус повёрнут относительно камеры примерно на 20–70°.<br>"
        "<b>−1</b> — корпус почти фронтальный, отклонение меньше 20°.<br>"
        "<b>0</b> — чистый боковой ракурс, поворот больше 70° либо вид со спины.<br>"
        "<b>?</b> — положение корпуса нельзя определить уверенно.",
        "three_quarter_view", NUMERIC_CHOICES, _all_rows,
    ),
)
ANCHOR_PROFILE = AnnotationProfile(
    key="anchor-panel",
    title="Разметка image-anchored концептов",
    columns=ANCHOR_COLUMNS,
    expected_row_count=160,
    default_annotations=PROJECT_DIR / "artifacts" / "step31a_pooled_importance_anchor_panel" / "anchor_panel_annotations.csv",
    passes=ANCHOR_PASSES,
    validate_rows=_validate_anchor,
    unique_row_key=_image_key,
)


def _is_high(row: dict[str, str]) -> bool:
    return row["candidate_set"] == "high"


def _is_low(row: dict[str, str]) -> bool:
    return row["candidate_set"] == "low"


def _validate_back_view(rows: list[dict[str, str]]) -> None:
    counts = {"high": 0, "low": 0}
    for row_number, row in enumerate(rows, start=2):
        candidate_set = row["candidate_set"]
        if candidate_set not in counts:
            raise AnnotationError(f"Строка {row_number}: неизвестный набор кандидатов.")
        counts[candidate_set] += 1
        allowed = {"", "B", "Q", "N"} if candidate_set == "high" else {"", "F", "Q", "N"}
        if row["manual_label"] not in allowed:
            raise AnnotationError(f"Строка {row_number}: метка не подходит для этого набора кандидатов.")
    if counts != {"high": 16, "low": 16}:
        raise AnnotationError("Ожидаются ровно 16 кандидатов для каждого из двух проходов.")


BACK_VIEW_PROFILE = AnnotationProfile(
    key="back-view-candidates",
    title="Проверка ракурса кандидатов",
    columns=("image_id", "candidate_set", "manual_label"),
    expected_row_count=32,
    default_annotations=PROJECT_DIR / "artifacts" / "step32_back_view_candidate_validation" / "back_view_candidate_annotations.csv",
    passes=(
        PassConfig(
            "back", "Вид со спины",
            "Оценивайте направление корпуса, а не положение головы.<br><br>"
            "<b>B</b> — ясный полный или почти полный вид со спины: корпус развёрнут от камеры примерно на 110° или больше, а спина — основное направление.<br>"
            "<b>Q</b> — промежуточный или неуверенный ракурс.<br><b>N</b> — не является видом со спины.",
            "manual_label",
            (LabelChoice("B", "Ясный вид со спины", "B"), LabelChoice("Q", "Промежуточный / неуверенный", "Q"), LabelChoice("N", "Не является видом со спины", "N")),
            _is_high,
        ),
        PassConfig(
            "front", "Фронтальный вид",
            "Оценивайте направление корпуса, а не положение головы.<br><br>"
            "<b>F</b> — ясный фронтальный вид: корпус обращён к камере, отклонение от анфаса не превышает примерно 20°.<br>"
            "<b>Q</b> — промежуточный или неуверенный ракурс.<br><b>N</b> — не является фронтальным видом.",
            "manual_label",
            (LabelChoice("F", "Ясный фронтальный вид", "F"), LabelChoice("Q", "Промежуточный / неуверенный", "Q"), LabelChoice("N", "Не является фронтальным видом", "N")),
            _is_low,
        ),
    ),
    validate_rows=_validate_back_view,
    unique_row_key=_image_key,
)


def _validate_cliff(rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        if row["breast_contour_occluded_by_view"] not in EMPTY_NUMERIC:
            raise AnnotationError(f"Строка {row_number}: допустимы '+1', '-1', '0', '?' или пусто.")


CLIFF_PROFILE = AnnotationProfile(
    key="cliff-panel",
    title="Разметка cliff-концепта",
    columns=("image_id", "breast_contour_occluded_by_view"),
    expected_row_count=64,
    default_annotations=PROJECT_DIR / "artifacts" / "step32b_cliff_panel" / "cliff_annotations.csv",
    passes=(
        PassConfig(
            "cliff", "Размечено",
            "<b>+1</b> — корпус отвёрнут от камеры примерно на 110° или сильнее, спина — основное направление, а контуры груди полностью или почти полностью скрыты именно ракурсом.<br><br>"
            "<b>−1</b> — изображение достаточно информативно, но хотя бы одно условие <b>+1</b> явно не выполнено.<br><br>"
            "<b>0</b> — грудь или корпус обрезаны либо закрыты так, что влияние именно ракурса оценить нельзя.<br><br>"
            "<b>?</b> — изображение информативно, но граница между почти скрытыми и видимыми контурами неоднозначна.",
            "breast_contour_occluded_by_view", NUMERIC_CHOICES, _all_rows,
        ),
    ),
    validate_rows=_validate_cliff,
    unique_row_key=_image_key,
)


CLIFF_VALIDATION_PROFILE = AnnotationProfile(
    key="cliff-validation",
    title="Валидация cliff-концепта",
    columns=("image_id", "breast_contour_occluded_by_view"),
    expected_row_count=84,
    default_annotations=PROJECT_DIR / "artifacts" / "step33_cliff_validation" / "validation_annotations.csv",
    passes=CLIFF_PROFILE.passes,
    validate_rows=_validate_cliff,
    unique_row_key=_image_key,
)


CONTRASTIVE_COLUMNS = ("image_id", "strong_back_view", "breast_contour_state")
STRONG_BACK_CHOICES = (
    LabelChoice("+1", "Сильный вид со спины", "1"),
    LabelChoice("0", "Условие явно не выполнено", "0"),
    LabelChoice("?", "Угол или корпус неясны", "?"),
)
CONTOUR_CHOICES = (
    LabelChoice("S", "Контур скрыт / почти скрыт", "S"),
    LabelChoice("V", "Контур ясно остаётся виден", "V"),
    LabelChoice("?", "Граница S/V неоднозначна", "?"),
)


def _is_strong_back_candidate(_row: dict[str, str]) -> bool:
    return True


def _is_contour_candidate(row: dict[str, str]) -> bool:
    return row["strong_back_view"] == "+1"


def _clear_contour_when_not_strong(row: dict[str, str], value: str) -> None:
    if value != "+1":
        row["breast_contour_state"] = ""


def _validate_contrastive(rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        strong = row["strong_back_view"]
        contour = row["breast_contour_state"]
        if strong not in {"", "+1", "0", "?"}:
            raise AnnotationError(f"Строка {row_number}: strong_back_view допускает '+1', '0', '?' или пусто.")
        if contour not in {"", "S", "V", "?"}:
            raise AnnotationError(f"Строка {row_number}: breast_contour_state допускает 'S', 'V', '?' или пусто.")
        if strong != "+1" and contour:
            raise AnnotationError(
                f"Строка {row_number}: breast_contour_state можно заполнять только при strong_back_view='+1'."
            )


CONTRASTIVE_PROFILE = AnnotationProfile(
    key="contrastive-panel",
    title="Контрастивная разметка скрытого контура",
    columns=CONTRASTIVE_COLUMNS,
    expected_row_count=140,
    default_annotations=PROJECT_DIR / "artifacts" / "step34_contrastive_panel" / "contrastive_annotations.csv",
    passes=(
        PassConfig(
            "strong-back-view",
            "Сильный вид со спины",
            "<b>+1</b> — корпус развёрнут от камеры примерно на 110° или сильнее, а спина является основным видимым направлением корпуса.<br><br>"
            "<b>0</b> — условие сильного вида со спины явно не выполнено.<br><br>"
            "<b>?</b> — угол или направление корпуса определить нельзя уверенно.",
            "strong_back_view",
            STRONG_BACK_CHOICES,
            _is_strong_back_candidate,
            _clear_contour_when_not_strong,
        ),
        PassConfig(
            "breast-contour-state",
            "Контур груди среди сильных видов со спины",
            "Заполняйте только для сильного вида со спины.<br><br>"
            "<b>S</b> — контуры груди полностью или почти полностью не видны.<br>"
            "<b>V</b> — боковой контур хотя бы одной груди ясно остаётся виден.<br>"
            "<b>?</b> — граница между S и V неоднозначна.<br><br>"
            "Руки, волосы, одежду и предметы можно учитывать как вспомогательное скрытие при S, только когда основную невидимость создаёт разворот корпуса. Не ставьте S, если грудь скрыта преимущественно одеждой или предметом.",
            "breast_contour_state",
            CONTOUR_CHOICES,
            _is_contour_candidate,
        ),
    ),
    validate_rows=_validate_contrastive,
    unique_row_key=_image_key,
)


SEMANTIC_COLUMNS = ("probe", "sheet", "image_id", "manual_label")
SEMANTIC_PROBES = ("nudity", "manual_breast_lift", "three_quarter_view")
SEMANTIC_SHEETS = ("top", "bottom")
SEMANTIC_CHOICES = (
    LabelChoice("C", "Правильный полюс", "C"),
    LabelChoice("R", "Противоположный полюс", "R"),
    LabelChoice("0", "Неприменимо / не видно", "0"),
    LabelChoice("?", "Неоднозначно", "?"),
)


def _semantic_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["probe"], row["sheet"], row["image_id"]


def _semantic_matches(probe: str, sheet: str):
    return lambda row: row["probe"] == probe and row["sheet"] == sheet


def _semantic_guidance(concept_rule: str, sheet: str) -> str:
    pole = (
        "Это <b>положительный полюс</b>: <b>C</b> означает корректный положительный пример, "
        "<b>R</b> — корректный отрицательный."
        if sheet == "top"
        else "Это <b>отрицательный полюс</b>: <b>C</b> означает корректный отрицательный пример, "
        "<b>R</b> — корректный положительный."
    )
    return (
        f"{pole}<br><br>{concept_rule}<br><br>"
        "<b>0</b> — концепт неприменим или нужная область не видна.<br>"
        "<b>?</b> — решение неоднозначно."
    )


SEMANTIC_RULES = {
    "nudity": "<b>Обнажённость груди:</b> положительный пример — обнажённая грудь ясно видна без бюстгальтера, купальника или непрозрачной одежды; отрицательный — область груди видна и закрыта.",
    "manual_breast_lift": "<b>Подъём груди руками:</b> положительный пример — одна или обе груди явно приподнимаются, поддерживаются или сжимаются рукой либо руками; отрицательный — область груди видна, но руки не поддерживают и не сжимают её.",
    "three_quarter_view": "<b>Ракурс три четверти:</b> оценивайте корпус, не голову. Положительный пример — поворот корпуса примерно на 20–70°; отрицательный — почти фронтальный корпус с отклонением меньше 20°.",
}
SEMANTIC_TITLES = {
    "nudity": "Обнажённость груди",
    "manual_breast_lift": "Подъём груди руками",
    "three_quarter_view": "Ракурс три четверти",
}
SEMANTIC_PASSES = tuple(
    PassConfig(
        f"{probe}-{sheet}",
        f"{SEMANTIC_TITLES[probe]} — {'положительный' if sheet == 'top' else 'отрицательный'} полюс",
        _semantic_guidance(SEMANTIC_RULES[probe], sheet),
        "manual_label",
        SEMANTIC_CHOICES,
        _semantic_matches(probe, sheet),
    )
    for probe in SEMANTIC_PROBES
    for sheet in SEMANTIC_SHEETS
)


def _validate_semantic_gate(rows: list[dict[str, str]]) -> None:
    counts = {(probe, sheet): 0 for probe in SEMANTIC_PROBES for sheet in SEMANTIC_SHEETS}
    for row_number, row in enumerate(rows, start=2):
        group = row["probe"], row["sheet"]
        if group not in counts:
            raise AnnotationError(f"Строка {row_number}: неизвестный probe или полюс.")
        if row["manual_label"] not in {"", "C", "R", "0", "?"}:
            raise AnnotationError(f"Строка {row_number}: допустимы 'C', 'R', '0', '?' или пусто.")
        counts[group] += 1
    wrong_groups = [group for group, count in counts.items() if count != 12]
    if wrong_groups:
        raise AnnotationError("Ожидаются ровно 12 изображений для каждого concept-полюса.")


SEMANTIC_GATE_PROFILE = AnnotationProfile(
    key="probe-semantic-gate",
    title="Семантический гейт concept-probes",
    columns=SEMANTIC_COLUMNS,
    expected_row_count=72,
    default_annotations=PROJECT_DIR / "artifacts" / "step32_image_anchored_probes" / "probe_semantic_annotations.csv",
    passes=SEMANTIC_PASSES,
    validate_rows=_validate_semantic_gate,
    unique_row_key=_semantic_key,
)


PROFILE_REGISTRY = {
    profile.key: profile
    for profile in (
        ANCHOR_PROFILE,
        BACK_VIEW_PROFILE,
        CLIFF_PROFILE,
        CLIFF_VALIDATION_PROFILE,
        CONTRASTIVE_PROFILE,
        SEMANTIC_GATE_PROFILE,
    )
}


def run_profile(key: str, argv: list[str] | None = None) -> int:
    return _run(PROFILE_REGISTRY[key], argv)
