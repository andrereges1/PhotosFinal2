"""Modelos de dados da analise local de imagem."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ImageAnalysisReport:
    image_name: str
    width: int
    height: int
    orientation: str
    aspect_ratio: float
    target_format: str
    target_aspect_ratio: float
    required_crop_axis: str | None = None
    required_crop_percent: float = 0.0
    crop_amount_class: str = "none"
    faces_detected: int = 0
    face_boxes: list[dict[str, Any]] = field(default_factory=list)
    face_group_box: tuple[int, int, int, int] | None = None
    faces_near_edges: bool = False
    faces_safe_for_crop: bool = True
    persons_detected: int = 0
    person_boxes: list[dict[str, Any]] = field(default_factory=list)
    person_group_box: tuple[int, int, int, int] | None = None
    persons_near_edges: bool = False
    persons_safe_for_crop: bool = True
    text_detected: bool = False
    text_boxes: list[dict[str, Any]] = field(default_factory=list)
    text_near_edges: bool = False
    edge_importance_left: float = 0.0
    edge_importance_right: float = 0.0
    edge_importance_top: float = 0.0
    edge_importance_bottom: float = 0.0
    edge_importance_max: float = 0.0
    visual_complexity_score: float = 0.0
    center_importance_score: float = 0.0
    border_importance_score: float = 0.0
    suggested_strategy: str = "contain_with_borders"
    risk_level: str = "medium"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    objects_detected: int = 0
    object_boxes: list[dict[str, Any]] = field(default_factory=list)
    primary_subject_type: str = "unknown"
    primary_subject_box: tuple[int, int, int, int] | None = None
    primary_subject_expanded_box: tuple[int, int, int, int] | None = None
    primary_subject_confidence: float = 0.0
    subject_focus_score: float = 0.0
    background_waste_score: float = 0.0
    empty_area_top_score: float = 0.0
    empty_area_bottom_score: float = 0.0
    empty_area_left_score: float = 0.0
    empty_area_right_score: float = 0.0
    can_tighten_frame: bool = False
    recommended_crop_mode: str = "contain_with_borders"
    subject_crop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AIDecision:
    image_name: str
    decision: str
    confidence: float
    reason: str
    risk_level: str
    use_borders: bool
    allow_crop: bool
    max_crop_percent: float | None
    protect_faces: bool
    protect_people: bool
    protect_text: bool
    rotate_on_pdf: bool
    create_extra_page: bool
    warnings: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None
    validated: bool = False
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchReport:
    total_images: int
    vertical_count: int
    horizontal_count: int
    square_count: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    faces_total: int
    persons_total: int
    text_images_count: int
    images: list[dict[str, Any]] = field(default_factory=list)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    local_summary: dict[str, Any] = field(default_factory=dict)
    suggested_local_batch_strategy: str = "best_fit"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlannedSlot:
    position: str
    slot_type: str
    image_name: str | None
    rotate_on_pdf: bool
    fit_strategy: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlannedPage:
    page_number: int
    layout_type: str
    slots: list[PlannedSlot] = field(default_factory=list)
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FinalImageDecision:
    image_name: str
    final_strategy: str
    crop_allowed: bool
    use_borders: bool
    rotate_on_pdf: bool
    create_extra_page: bool
    pdf_page_number: int | None
    pdf_slot_position: str | None
    reason: str
    source: str
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchPlan:
    source: str
    strategy: str
    pages: list[PlannedPage] = field(default_factory=list)
    image_decisions: list[FinalImageDecision] = field(default_factory=list)
    global_warnings: list[str] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0
    validated: bool = False
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
