"""Geração dos PDFs A4 prontos para impressão."""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import logging

from PIL import Image, ImageOps
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.analysis_models import BatchPlan, PlannedSlot
from src.config import (
    A4_HEIGHT_CM,
    A4_WIDTH_CM,
    ORIENTATION_HORIZONTAL,
    ORIENTATION_SQUARE,
    ORIENTATION_VERTICAL,
    PDF_4UP_COLUMNS,
    PDF_4UP_HORIZONTAL_GAP_CM,
    PDF_4UP_ROWS,
    PDF_4UP_VERTICAL_GAP_CM,
    PDF_EDGE_MARGIN_CM,
    PDF_LAYOUT_4_REAL_PHOTOS,
    PDF_ORGANIZE_UPLOAD_ORDER,
    PDF_PHOTO_GAP_CM,
    PDF_LAYOUT_2_REAL,
    PDF_LAYOUT_3_PER_A4,
    PDF_LAYOUT_3_REAL_PHOTOS,
    PHOTO_4UP_HEIGHT_CM,
    PHOTO_4UP_WIDTH_CM,
    PHOTO_HORIZONTAL_HEIGHT_CM,
    PHOTO_HORIZONTAL_WIDTH_CM,
    PHOTO_10X15_HEIGHT_CM,
    PHOTO_10X15_WIDTH_CM,
    PHOTO_VERTICAL_HEIGHT_CM,
    PHOTO_VERTICAL_WIDTH_CM,
    TARGET_10X1452_PX,
    WHITE,
    cm_to_points,
)
from src.image_processing import save_jpg_bytes
from src.utils import PdfOptions, ProcessedImage, chunked

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PhotoSlot:
    processed: ProcessedImage
    width_cm: float
    height_cm: float


@dataclass(slots=True)
class _LayoutCandidate:
    page_width_cm: float
    page_height_cm: float
    arrangement: str
    total_width_cm: float
    total_height_cm: float


@dataclass(frozen=True, slots=True)
class _BoxCm:
    x_cm: float
    y_cm: float
    width_cm: float
    height_cm: float


@dataclass(frozen=True, slots=True)
class _ThreePhotoLayout:
    photo_width_cm: float
    photo_height_cm: float
    top_slot: _BoxCm
    left_slot: _BoxCm
    right_slot: _BoxCm
    bottom_center_slot: _BoxCm
    single_slot: _BoxCm
    pair_left_slot: _BoxCm
    pair_right_slot: _BoxCm
    horizontal_single_slot: _BoxCm
    horizontal_upper_slot: _BoxCm
    horizontal_lower_slot: _BoxCm


@dataclass(frozen=True, slots=True)
class _FourUpLayout:
    slots: tuple[_BoxCm, _BoxCm, _BoxCm, _BoxCm]
    horizontal_gap_cm: float
    vertical_gap_cm: float


@dataclass(frozen=True, slots=True)
class _Prepared4UpSlotImage:
    image: Image.Image
    fit_strategy: str
    rotated_on_pdf: bool
    rotation_degrees: int
    rotation_reason: str | None


@dataclass(frozen=True, slots=True)
class PdfPlacement:
    page_number: int
    output_name: str
    position_label: str
    slot_type: str
    rotated_on_pdf: bool
    fit_strategy: str


@dataclass(frozen=True, slots=True)
class PdfGenerationResult:
    pdf_bytes: bytes
    placements: list[PdfPlacement]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class _PlannedItem:
    processed: ProcessedImage
    position_label: str
    slot_type: str


def generate_pdf(processed_images: list[ProcessedImage], options: PdfOptions) -> bytes:
    return generate_pdf_with_summary(processed_images, options).pdf_bytes


def _ordered_images_from_batch_plan(
    processed_images: list[ProcessedImage],
    final_batch_plan: BatchPlan | None,
) -> list[ProcessedImage]:
    if not final_batch_plan:
        return processed_images

    items_by_name = {item.output_name: item for item in processed_images}
    ordered: list[ProcessedImage] = []
    for page in sorted(final_batch_plan.pages, key=lambda planned_page: planned_page.page_number):
        for planned_slot in page.slots:
            if planned_slot.image_name in items_by_name:
                ordered.append(items_by_name[planned_slot.image_name])

    seen = {item.output_name for item in ordered}
    ordered.extend(item for item in processed_images if item.output_name not in seen)
    return ordered


def generate_pdf_with_summary(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
    final_batch_plan: BatchPlan | None = None,
) -> PdfGenerationResult:
    if options.layout_mode == PDF_LAYOUT_3_REAL_PHOTOS:
        real_layout_images = _ordered_images_from_batch_plan(processed_images, final_batch_plan)
        result = generate_pdf_3_real_photos_a4_with_summary(real_layout_images, options)
        if final_batch_plan:
            result.warnings.append("Usei o plano final validado apenas para ordenar; o encaixe seguiu o layout 10x15 real.")
        return result
    if options.layout_mode == PDF_LAYOUT_4_REAL_PHOTOS:
        real_layout_images = _ordered_images_from_batch_plan(processed_images, final_batch_plan)
        result = generate_pdf_4_real_images_a4_with_summary(real_layout_images, options)
        if final_batch_plan:
            result.warnings.append("Usei o plano final validado apenas para ordenar; o encaixe seguiu o layout 4 imagens 10x14,52.")
        return result
    if options.layout_mode == PDF_LAYOUT_2_REAL:
        real_layout_images = _ordered_images_from_batch_plan(processed_images, final_batch_plan)
        pdf_bytes = generate_pdf_2_real_layout(real_layout_images, options)
        placements = [
            PdfPlacement(
                page_number=(index - 1) // 2 + 1,
                output_name=processed.output_name,
                position_label="10x15 real",
                slot_type=processed.final_orientation or _orientation_from_image(processed.image),
                rotated_on_pdf=False,
                fit_strategy="real_size",
            )
            for index, processed in enumerate(real_layout_images, start=1)
        ]
        warnings = []
        if final_batch_plan:
            warnings.append("Usei o plano final validado para ordenar o PDF 10x15 real.")
        return PdfGenerationResult(pdf_bytes=pdf_bytes, placements=placements, warnings=warnings)
    if final_batch_plan and final_batch_plan.pages:
        return generate_pdf_from_batch_plan(processed_images, final_batch_plan, options)
    return generate_pdf_3_photos_layout_with_summary(processed_images, options)


def _image_reader(image: Image.Image) -> ImageReader:
    return ImageReader(BytesIO(save_jpg_bytes(image).getvalue()))


def _page_size_points(width_cm: float, height_cm: float) -> tuple[float, float]:
    return cm_to_points(width_cm), cm_to_points(height_cm)


def draw_image_cm(
    pdf: canvas.Canvas,
    image: Image.Image,
    x_cm: float,
    y_cm: float,
    width_cm: float,
    height_cm: float,
    rotate_degrees: int = 0,
) -> None:
    """Desenha a imagem no tamanho fisico informado em centimetros."""
    rotation = rotate_degrees % 360
    x_pt = cm_to_points(x_cm)
    y_pt = cm_to_points(y_cm)
    width_pt = cm_to_points(width_cm)
    height_pt = cm_to_points(height_cm)
    reader = _image_reader(image)

    if rotation == 0:
        pdf.drawImage(
            reader,
            x_pt,
            y_pt,
            width=width_pt,
            height=height_pt,
            preserveAspectRatio=False,
            mask="auto",
        )
        return

    if rotation == 90:
        pdf.saveState()
        pdf.translate(x_pt + width_pt, y_pt)
        pdf.rotate(90)
        pdf.drawImage(
            reader,
            0,
            0,
            width=height_pt,
            height=width_pt,
            preserveAspectRatio=False,
            mask="auto",
        )
        pdf.restoreState()
        return

    if rotation == 180:
        pdf.saveState()
        pdf.translate(x_pt + width_pt, y_pt + height_pt)
        pdf.rotate(180)
        pdf.drawImage(
            reader,
            0,
            0,
            width=width_pt,
            height=height_pt,
            preserveAspectRatio=False,
            mask="auto",
        )
        pdf.restoreState()
        return

    logger.warning("Rotacao %s nao suportada no PDF; desenhei a foto sem rotacao.", rotate_degrees)
    draw_image_cm(pdf, image, x_cm, y_cm, width_cm, height_cm)


def _orientation_from_image(image: Image.Image) -> str:
    return ORIENTATION_HORIZONTAL if image.width >= image.height else ORIENTATION_VERTICAL


def _pdf_orientation(processed: ProcessedImage) -> str:
    return processed.final_orientation or _orientation_from_image(processed.image)


def _is_horizontal(processed: ProcessedImage) -> bool:
    return _pdf_orientation(processed) == ORIENTATION_HORIZONTAL


def _is_vertical_like(processed: ProcessedImage) -> bool:
    return not _is_horizontal(processed)


def _fit_image_in_box_cm(
    image: Image.Image,
    box_width_cm: float,
    box_height_cm: float,
) -> tuple[float, float, float, float]:
    """Retorna x/y internos e tamanho para desenhar sem distorcer."""
    image_aspect = image.width / image.height
    box_aspect = box_width_cm / box_height_cm

    if image_aspect > box_aspect:
        draw_width_cm = box_width_cm
        draw_height_cm = draw_width_cm / image_aspect
    else:
        draw_height_cm = box_height_cm
        draw_width_cm = draw_height_cm * image_aspect

    offset_x_cm = (box_width_cm - draw_width_cm) / 2
    offset_y_cm = (box_height_cm - draw_height_cm) / 2
    return offset_x_cm, offset_y_cm, draw_width_cm, draw_height_cm


def _draw_slot_image_cm(
    pdf: canvas.Canvas,
    processed: ProcessedImage,
    slot: _BoxCm,
    slot_type: str,
    page_number: int,
    position_label: str,
) -> PdfPlacement:
    image_orientation = _pdf_orientation(processed)
    rotated_on_pdf = False
    fit_strategy = "normal"

    if slot_type == ORIENTATION_HORIZONTAL and image_orientation == ORIENTATION_VERTICAL:
        draw_image_cm(
            pdf,
            processed.image,
            slot.x_cm,
            slot.y_cm,
            slot.width_cm,
            slot.height_cm,
            rotate_degrees=90,
        )
        rotated_on_pdf = True
        fit_strategy = "rotate_on_pdf"
    elif slot_type == ORIENTATION_VERTICAL and image_orientation == ORIENTATION_HORIZONTAL:
        _draw_image_cm(
            pdf,
            processed.image,
            slot.x_cm,
            slot.y_cm,
            slot.width_cm,
            slot.height_cm,
        )
        fit_strategy = "contain_with_borders"
    else:
        draw_image_cm(
            pdf,
            processed.image,
            slot.x_cm,
            slot.y_cm,
            slot.width_cm,
            slot.height_cm,
        )

    processed.pdf_slot_type = slot_type
    processed.rotated_on_pdf = rotated_on_pdf
    processed.pdf_page_number = page_number
    processed.pdf_position_label = position_label

    return PdfPlacement(
        page_number=page_number,
        output_name=processed.output_name,
        position_label=position_label,
        slot_type=slot_type,
        rotated_on_pdf=rotated_on_pdf,
        fit_strategy=fit_strategy,
    )


def _draw_image_cm(
    pdf: canvas.Canvas,
    image: Image.Image,
    x_cm: float,
    y_cm: float,
    width_cm: float,
    height_cm: float,
) -> None:
    offset_x_cm, offset_y_cm, draw_width_cm, draw_height_cm = _fit_image_in_box_cm(
        image,
        width_cm,
        height_cm,
    )
    pdf.drawImage(
        _image_reader(image),
        cm_to_points(x_cm + offset_x_cm),
        cm_to_points(y_cm + offset_y_cm),
        width=cm_to_points(draw_width_cm),
        height=cm_to_points(draw_height_cm),
        preserveAspectRatio=False,
        mask="auto",
    )


def _cover_resize_center(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    target_width, target_height = target_size
    target_aspect = target_width / target_height
    image_aspect = image.width / image.height

    if image_aspect > target_aspect:
        crop_height = image.height
        crop_width = int(round(crop_height * target_aspect))
    else:
        crop_width = image.width
        crop_height = int(round(crop_width / target_aspect))

    crop_width = max(1, min(crop_width, image.width))
    crop_height = max(1, min(crop_height, image.height))
    crop_x = int(round((image.width - crop_width) / 2))
    crop_y = int(round((image.height - crop_height) / 2))
    cropped = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    return cropped.resize(target_size, Image.Resampling.LANCZOS)


def _contain_resize_on_canvas(
    image: Image.Image,
    target_size: tuple[int, int],
    background_color: tuple[int, int, int] = WHITE,
) -> Image.Image:
    image = image.convert("RGB")
    target_width, target_height = target_size
    fitted = ImageOps.contain(image, target_size, method=Image.Resampling.LANCZOS)
    canvas_image = Image.new("RGB", target_size, background_color)
    offset_x = (target_width - fitted.width) // 2
    offset_y = (target_height - fitted.height) // 2
    canvas_image.paste(fitted, (offset_x, offset_y))
    return canvas_image


def _rgb_with_white_background(image: Image.Image, background_color: tuple[int, int, int] = WHITE) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, background_color + (255,))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image.copy()


def _detect_pdf_image_orientation(processed: ProcessedImage) -> str:
    if processed.original_orientation in {ORIENTATION_HORIZONTAL, ORIENTATION_VERTICAL, ORIENTATION_SQUARE}:
        return processed.original_orientation

    source_image = processed.source_image or processed.original_image or processed.image
    source_image = ImageOps.exif_transpose(source_image)
    width, height = source_image.size
    if width > height * 1.10:
        return ORIENTATION_HORIZONTAL
    if height > width * 1.10:
        return ORIENTATION_VERTICAL
    return ORIENTATION_SQUARE


def should_rotate_image_for_pdf_layout(
    processed_image: ProcessedImage,
    layout_name: str,
    slot_width_cm: float,
    slot_height_cm: float,
) -> dict[str, object]:
    image_orientation = _detect_pdf_image_orientation(processed_image)
    if (
        layout_name == PDF_LAYOUT_4_REAL_PHOTOS
        and slot_height_cm > slot_width_cm
        and image_orientation == ORIENTATION_HORIZONTAL
    ):
        return {
            "should_rotate": True,
            "rotation_degrees": 90,
            "reason": "Foto horizontal rotacionada 90 graus antes do encaixe no layout 4 imagens",
        }
    return {
        "should_rotate": False,
        "rotation_degrees": 0,
        "reason": None,
    }


def _fit_image_to_4up_slot_canvas(
    source_image: Image.Image,
    target_size_px: tuple[int, int] = TARGET_10X1452_PX,
    background_color: tuple[int, int, int] = WHITE,
) -> tuple[Image.Image, str]:
    del background_color
    image_aspect = source_image.width / source_image.height
    target_aspect = target_size_px[0] / target_size_px[1]
    if _close_enough(image_aspect, target_aspect, 0.01):
        return source_image.convert("RGB").resize(target_size_px, Image.Resampling.LANCZOS), "normal"
    return _cover_resize_center(source_image, target_size_px), "center_crop"


def _source_image_for_4up_pdf_slot(processed: ProcessedImage) -> Image.Image:
    if processed.source_image is not None:
        return _rgb_with_white_background(processed.source_image)
    if processed.original_image is not None:
        return _rgb_with_white_background(processed.original_image)
    return _rgb_with_white_background(processed.image)


def _rotate_image_copy_for_pdf(image: Image.Image, rotation_degrees: int) -> Image.Image:
    rotation = rotation_degrees % 360
    if rotation:
        return image.copy().rotate(rotation_degrees, expand=True)
    return image.copy()


def prepare_4up_slot_image(
    processed_image: ProcessedImage,
    slot_target_px: tuple[int, int] = TARGET_10X1452_PX,
    resize_mode: str = "auto",
    avoid_cutting_people: bool = True,
    background_color: tuple[int, int, int] = WHITE,
    slot_width_cm: float = PHOTO_4UP_WIDTH_CM,
    slot_height_cm: float = PHOTO_4UP_HEIGHT_CM,
) -> dict[str, object]:
    del resize_mode, avoid_cutting_people
    rotation_decision = should_rotate_image_for_pdf_layout(
        processed_image,
        PDF_LAYOUT_4_REAL_PHOTOS,
        slot_width_cm,
        slot_height_cm,
    )
    rotation_degrees = int(rotation_decision["rotation_degrees"])
    rotation_reason = rotation_decision["reason"]
    rotated_on_pdf = bool(rotation_decision["should_rotate"])

    image_for_pdf = _source_image_for_4up_pdf_slot(processed_image)
    if rotated_on_pdf:
        image_for_pdf = _rotate_image_copy_for_pdf(image_for_pdf, rotation_degrees)

    slot_image, fit_strategy = _fit_image_to_4up_slot_canvas(image_for_pdf, slot_target_px, background_color)
    if rotated_on_pdf:
        fit_strategy = "rotate_on_pdf"

    return {
        "image": slot_image,
        "fit_strategy": fit_strategy,
        "rotated_on_pdf": rotated_on_pdf,
        "rotation_degrees": rotation_degrees,
        "reason": str(rotation_reason) if rotation_reason else None,
    }


def prepare_image_for_4up_slot(
    processed_image: ProcessedImage,
    slot_target_px: tuple[int, int] = TARGET_10X1452_PX,
    resize_mode: str = "auto",
    avoid_cutting_people: bool = True,
    background_color: tuple[int, int, int] = WHITE,
    slot_width_cm: float = PHOTO_4UP_WIDTH_CM,
    slot_height_cm: float = PHOTO_4UP_HEIGHT_CM,
) -> dict[str, object]:
    return prepare_4up_slot_image(
        processed_image=processed_image,
        slot_target_px=slot_target_px,
        resize_mode=resize_mode,
        avoid_cutting_people=avoid_cutting_people,
        background_color=background_color,
        slot_width_cm=slot_width_cm,
        slot_height_cm=slot_height_cm,
    )


def prepare_image_for_4up_pdf_slot(
    processed_image: ProcessedImage,
    slot_width_cm: float = PHOTO_4UP_WIDTH_CM,
    slot_height_cm: float = PHOTO_4UP_HEIGHT_CM,
    resize_mode: str = "auto",
    background_color: tuple[int, int, int] = WHITE,
) -> _Prepared4UpSlotImage:
    prepared = prepare_4up_slot_image(
        processed_image=processed_image,
        slot_target_px=TARGET_10X1452_PX,
        resize_mode=resize_mode,
        avoid_cutting_people=True,
        background_color=background_color,
        slot_width_cm=slot_width_cm,
        slot_height_cm=slot_height_cm,
    )
    return _Prepared4UpSlotImage(
        image=prepared["image"],
        fit_strategy=str(prepared["fit_strategy"]),
        rotated_on_pdf=bool(prepared["rotated_on_pdf"]),
        rotation_degrees=int(prepared["rotation_degrees"]),
        rotation_reason=str(prepared["reason"]) if prepared["reason"] else None,
    )


def _draw_4up_slot_image_cm(
    pdf: canvas.Canvas,
    processed: ProcessedImage,
    slot: _BoxCm,
    page_number: int,
    position_label: str,
) -> PdfPlacement:
    prepared = prepare_image_for_4up_pdf_slot(processed, slot.width_cm, slot.height_cm)
    draw_image_cm(pdf, prepared.image, slot.x_cm, slot.y_cm, slot.width_cm, slot.height_cm)

    processed.pdf_slot_type = ORIENTATION_VERTICAL
    processed.rotated_on_pdf = prepared.rotated_on_pdf
    processed.pdf_rotation_degrees = prepared.rotation_degrees
    processed.pdf_rotation_reason = prepared.rotation_reason
    processed.pdf_layout_name = PDF_LAYOUT_4_REAL_PHOTOS
    processed.pdf_page_number = page_number
    processed.pdf_position_label = position_label

    return PdfPlacement(
        page_number=page_number,
        output_name=processed.output_name,
        position_label=position_label,
        slot_type=ORIENTATION_VERTICAL,
        rotated_on_pdf=prepared.rotated_on_pdf,
        fit_strategy=prepared.fit_strategy,
    )


def _draw_image_rotated_90_cm(
    pdf: canvas.Canvas,
    image: Image.Image,
    x_cm: float,
    y_cm: float,
    original_width_cm: float,
    original_height_cm: float,
) -> None:
    """Desenha uma foto vertical girada, ocupando altura x largura na página."""
    x_pt = cm_to_points(x_cm)
    y_pt = cm_to_points(y_cm)
    original_width_pt = cm_to_points(original_width_cm)
    original_height_pt = cm_to_points(original_height_cm)
    offset_x_cm, offset_y_cm, draw_width_cm, draw_height_cm = _fit_image_in_box_cm(
        image,
        original_width_cm,
        original_height_cm,
    )

    pdf.saveState()
    pdf.translate(x_pt + original_height_pt, y_pt)
    pdf.rotate(90)
    pdf.drawImage(
        _image_reader(image),
        cm_to_points(offset_x_cm),
        cm_to_points(offset_y_cm),
        width=cm_to_points(draw_width_cm),
        height=cm_to_points(draw_height_cm),
        preserveAspectRatio=False,
        mask="auto",
    )
    pdf.restoreState()


def _draw_cut_rect_cm(
    pdf: canvas.Canvas,
    x_cm: float,
    y_cm: float,
    width_cm: float,
    height_cm: float,
) -> None:
    pdf.saveState()
    pdf.setStrokeColor(colors.Color(0.72, 0.72, 0.72))
    pdf.setLineWidth(0.35)
    pdf.rect(
        cm_to_points(x_cm),
        cm_to_points(y_cm),
        cm_to_points(width_cm),
        cm_to_points(height_cm),
        stroke=1,
        fill=0,
    )
    pdf.restoreState()


def _calculate_three_photo_layout(options: PdfOptions) -> _ThreePhotoLayout:
    margin_cm = options.margin_cm
    gap_cm = options.gap_cm
    usable_width_cm = A4_WIDTH_CM - 2 * margin_cm
    usable_height_cm = A4_HEIGHT_CM - 2 * margin_cm

    if margin_cm < 0:
        raise ValueError("A margem do PDF nao pode ser negativa.")
    if gap_cm < 0:
        raise ValueError("O espaco entre fotos nao pode ser negativo.")
    if usable_width_cm <= gap_cm or usable_height_cm <= 0:
        raise ValueError("A margem do PDF deixou a area util pequena demais.")

    photo_width_cm = (usable_width_cm - gap_cm) / 2
    photo_height_cm = photo_width_cm * 1.5
    rotated_width_cm = photo_height_cm
    rotated_height_cm = photo_width_cm

    group_height_cm = rotated_height_cm + gap_cm + photo_height_cm
    if group_height_cm > usable_height_cm:
        raise ValueError("O layout de 3 fotos nao cabe na area util do A4.")

    bottom_width_cm = photo_width_cm * 2 + gap_cm
    bottom_y_cm = margin_cm + (usable_height_cm - group_height_cm) / 2
    top_y_cm = bottom_y_cm + photo_height_cm + gap_cm
    top_x_cm = margin_cm + (usable_width_cm - rotated_width_cm) / 2
    left_x_cm = margin_cm + (usable_width_cm - bottom_width_cm) / 2
    right_x_cm = left_x_cm + photo_width_cm + gap_cm
    bottom_center_x_cm = margin_cm + (usable_width_cm - photo_width_cm) / 2

    single_x_cm = margin_cm + (usable_width_cm - photo_width_cm) / 2
    single_y_cm = margin_cm + (usable_height_cm - photo_height_cm) / 2
    pair_y_cm = margin_cm + (usable_height_cm - photo_height_cm) / 2
    horizontal_x_cm = margin_cm + (usable_width_cm - rotated_width_cm) / 2
    horizontal_single_y_cm = margin_cm + (usable_height_cm - rotated_height_cm) / 2
    horizontal_group_height_cm = rotated_height_cm * 2 + gap_cm
    horizontal_lower_y_cm = margin_cm + (usable_height_cm - horizontal_group_height_cm) / 2
    horizontal_upper_y_cm = horizontal_lower_y_cm + rotated_height_cm + gap_cm

    return _ThreePhotoLayout(
        photo_width_cm=photo_width_cm,
        photo_height_cm=photo_height_cm,
        top_slot=_BoxCm(top_x_cm, top_y_cm, rotated_width_cm, rotated_height_cm),
        left_slot=_BoxCm(left_x_cm, bottom_y_cm, photo_width_cm, photo_height_cm),
        right_slot=_BoxCm(right_x_cm, bottom_y_cm, photo_width_cm, photo_height_cm),
        bottom_center_slot=_BoxCm(bottom_center_x_cm, bottom_y_cm, photo_width_cm, photo_height_cm),
        single_slot=_BoxCm(single_x_cm, single_y_cm, photo_width_cm, photo_height_cm),
        pair_left_slot=_BoxCm(left_x_cm, pair_y_cm, photo_width_cm, photo_height_cm),
        pair_right_slot=_BoxCm(right_x_cm, pair_y_cm, photo_width_cm, photo_height_cm),
        horizontal_single_slot=_BoxCm(horizontal_x_cm, horizontal_single_y_cm, rotated_width_cm, rotated_height_cm),
        horizontal_upper_slot=_BoxCm(horizontal_x_cm, horizontal_upper_y_cm, rotated_width_cm, rotated_height_cm),
        horizontal_lower_slot=_BoxCm(horizontal_x_cm, horizontal_lower_y_cm, rotated_width_cm, rotated_height_cm),
    )


def _calculate_three_real_photos_layout() -> _ThreePhotoLayout:
    margin_cm = PDF_EDGE_MARGIN_CM
    gap_cm = PDF_PHOTO_GAP_CM
    usable_width_cm = A4_WIDTH_CM - 2 * margin_cm
    usable_height_cm = A4_HEIGHT_CM - 2 * margin_cm

    top_photo_w_cm = PHOTO_HORIZONTAL_WIDTH_CM
    top_photo_h_cm = PHOTO_HORIZONTAL_HEIGHT_CM
    bottom_photo_w_cm = PHOTO_VERTICAL_WIDTH_CM
    bottom_photo_h_cm = PHOTO_VERTICAL_HEIGHT_CM

    bottom_group_w_cm = bottom_photo_w_cm * 2 + gap_cm
    group_w_cm = max(top_photo_w_cm, bottom_group_w_cm)
    group_h_cm = top_photo_h_cm + gap_cm + bottom_photo_h_cm

    group_x_cm = margin_cm + (usable_width_cm - group_w_cm) / 2
    group_y_cm = margin_cm + (usable_height_cm - group_h_cm) / 2

    top_x_cm = margin_cm + (usable_width_cm - top_photo_w_cm) / 2
    top_y_cm = group_y_cm + bottom_photo_h_cm + gap_cm
    left_x_cm = group_x_cm
    right_x_cm = left_x_cm + bottom_photo_w_cm + gap_cm
    bottom_center_x_cm = margin_cm + (usable_width_cm - bottom_photo_w_cm) / 2

    single_x_cm = margin_cm + (usable_width_cm - bottom_photo_w_cm) / 2
    single_y_cm = margin_cm + (usable_height_cm - bottom_photo_h_cm) / 2
    pair_y_cm = margin_cm + (usable_height_cm - bottom_photo_h_cm) / 2

    horizontal_x_cm = margin_cm + (usable_width_cm - top_photo_w_cm) / 2
    horizontal_single_y_cm = margin_cm + (usable_height_cm - top_photo_h_cm) / 2
    horizontal_group_height_cm = top_photo_h_cm * 2 + gap_cm
    horizontal_lower_y_cm = margin_cm + (usable_height_cm - horizontal_group_height_cm) / 2
    horizontal_upper_y_cm = horizontal_lower_y_cm + top_photo_h_cm + gap_cm

    return _ThreePhotoLayout(
        photo_width_cm=bottom_photo_w_cm,
        photo_height_cm=bottom_photo_h_cm,
        top_slot=_BoxCm(top_x_cm, top_y_cm, top_photo_w_cm, top_photo_h_cm),
        left_slot=_BoxCm(left_x_cm, group_y_cm, bottom_photo_w_cm, bottom_photo_h_cm),
        right_slot=_BoxCm(right_x_cm, group_y_cm, bottom_photo_w_cm, bottom_photo_h_cm),
        bottom_center_slot=_BoxCm(bottom_center_x_cm, group_y_cm, bottom_photo_w_cm, bottom_photo_h_cm),
        single_slot=_BoxCm(single_x_cm, single_y_cm, bottom_photo_w_cm, bottom_photo_h_cm),
        pair_left_slot=_BoxCm(left_x_cm, pair_y_cm, bottom_photo_w_cm, bottom_photo_h_cm),
        pair_right_slot=_BoxCm(right_x_cm, pair_y_cm, bottom_photo_w_cm, bottom_photo_h_cm),
        horizontal_single_slot=_BoxCm(horizontal_x_cm, horizontal_single_y_cm, top_photo_w_cm, top_photo_h_cm),
        horizontal_upper_slot=_BoxCm(horizontal_x_cm, horizontal_upper_y_cm, top_photo_w_cm, top_photo_h_cm),
        horizontal_lower_slot=_BoxCm(horizontal_x_cm, horizontal_lower_y_cm, top_photo_w_cm, top_photo_h_cm),
    )


def _calculate_4up_real_gaps() -> tuple[float, float]:
    usable_width_cm = A4_WIDTH_CM - 2 * PDF_EDGE_MARGIN_CM
    usable_height_cm = A4_HEIGHT_CM - 2 * PDF_EDGE_MARGIN_CM
    used_width_cm = PHOTO_4UP_WIDTH_CM * PDF_4UP_COLUMNS
    used_height_cm = PHOTO_4UP_HEIGHT_CM * PDF_4UP_ROWS

    horizontal_gap_cm = (usable_width_cm - used_width_cm) / max(1, PDF_4UP_COLUMNS - 1)
    vertical_gap_cm = (usable_height_cm - used_height_cm) / max(1, PDF_4UP_ROWS - 1)
    return horizontal_gap_cm, vertical_gap_cm


def _calculate_4up_real_layout() -> _FourUpLayout:
    horizontal_gap_cm, vertical_gap_cm = _calculate_4up_real_gaps()
    if horizontal_gap_cm < -0.001 or vertical_gap_cm < -0.001:
        raise ValueError("O layout 4 imagens 10x14,52 nao cabe na area util do A4.")

    left_x_cm = PDF_EDGE_MARGIN_CM
    right_x_cm = A4_WIDTH_CM - PDF_EDGE_MARGIN_CM - PHOTO_4UP_WIDTH_CM
    bottom_y_cm = PDF_EDGE_MARGIN_CM
    top_y_cm = A4_HEIGHT_CM - PDF_EDGE_MARGIN_CM - PHOTO_4UP_HEIGHT_CM

    return _FourUpLayout(
        slots=(
            _BoxCm(left_x_cm, top_y_cm, PHOTO_4UP_WIDTH_CM, PHOTO_4UP_HEIGHT_CM),
            _BoxCm(right_x_cm, top_y_cm, PHOTO_4UP_WIDTH_CM, PHOTO_4UP_HEIGHT_CM),
            _BoxCm(left_x_cm, bottom_y_cm, PHOTO_4UP_WIDTH_CM, PHOTO_4UP_HEIGHT_CM),
            _BoxCm(right_x_cm, bottom_y_cm, PHOTO_4UP_WIDTH_CM, PHOTO_4UP_HEIGHT_CM),
        ),
        horizontal_gap_cm=horizontal_gap_cm,
        vertical_gap_cm=vertical_gap_cm,
    )


def _close_enough(actual: float, expected: float, tolerance: float = 0.001) -> bool:
    return abs(actual - expected) <= tolerance


def validate_3_real_photos_layout() -> bool:
    """Valida tecnicamente o layout A4 de 3 fotos reais sem interromper o app."""
    errors: list[str] = []
    layout = _calculate_three_real_photos_layout()

    if not _close_enough(PDF_EDGE_MARGIN_CM, 0.3):
        errors.append("A margem externa nao e 0,3 cm.")
    if not _close_enough(PDF_PHOTO_GAP_CM, 0.2):
        errors.append("O espaco entre fotos nao e 0,2 cm.")

    expected_slots = [
        ("foto superior", layout.top_slot, PHOTO_HORIZONTAL_WIDTH_CM, PHOTO_HORIZONTAL_HEIGHT_CM),
        ("foto inferior esquerda", layout.left_slot, PHOTO_VERTICAL_WIDTH_CM, PHOTO_VERTICAL_HEIGHT_CM),
        ("foto inferior direita", layout.right_slot, PHOTO_VERTICAL_WIDTH_CM, PHOTO_VERTICAL_HEIGHT_CM),
    ]
    for label, slot, expected_width_cm, expected_height_cm in expected_slots:
        if not _close_enough(slot.width_cm, expected_width_cm) or not _close_enough(slot.height_cm, expected_height_cm):
            errors.append(f"{label} nao esta no tamanho fisico esperado.")
        if slot.x_cm < PDF_EDGE_MARGIN_CM - 0.001 or slot.y_cm < PDF_EDGE_MARGIN_CM - 0.001:
            errors.append(f"{label} ficou fora da margem minima.")
        if slot.x_cm + slot.width_cm > A4_WIDTH_CM - PDF_EDGE_MARGIN_CM + 0.001:
            errors.append(f"{label} ultrapassa a largura da pagina.")
        if slot.y_cm + slot.height_cm > A4_HEIGHT_CM - PDF_EDGE_MARGIN_CM + 0.001:
            errors.append(f"{label} ultrapassa a altura da pagina.")

    bottom_gap_cm = layout.right_slot.x_cm - (layout.left_slot.x_cm + layout.left_slot.width_cm)
    vertical_gap_cm = layout.top_slot.y_cm - (layout.left_slot.y_cm + layout.left_slot.height_cm)
    if not _close_enough(bottom_gap_cm, PDF_PHOTO_GAP_CM):
        errors.append("A distancia entre as fotos inferiores nao e 0,2 cm.")
    if not _close_enough(vertical_gap_cm, PDF_PHOTO_GAP_CM):
        errors.append("A distancia entre a foto superior e as inferiores nao e 0,2 cm.")

    group_width_cm = max(
        layout.top_slot.width_cm,
        layout.right_slot.x_cm + layout.right_slot.width_cm - layout.left_slot.x_cm,
    )
    group_height_cm = layout.top_slot.y_cm + layout.top_slot.height_cm - layout.left_slot.y_cm
    usable_width_cm = A4_WIDTH_CM - 2 * PDF_EDGE_MARGIN_CM
    usable_height_cm = A4_HEIGHT_CM - 2 * PDF_EDGE_MARGIN_CM
    if group_width_cm > usable_width_cm + 0.001 or group_height_cm > usable_height_cm + 0.001:
        errors.append("O conjunto de 3 fotos reais nao cabe na area util do A4.")

    if errors:
        logger.error("Validacao tecnica do layout 3 fotos reais falhou: %s", " ".join(errors))
        return False
    return True


def validate_4up_real_layout() -> bool:
    """Valida tecnicamente o layout A4 de 4 imagens 10x14,52."""
    errors: list[str] = []
    layout = _calculate_4up_real_layout()
    top_left, top_right, bottom_left, bottom_right = layout.slots
    expected_top_y_cm = A4_HEIGHT_CM - PDF_EDGE_MARGIN_CM - PHOTO_4UP_HEIGHT_CM
    expected_right_x_cm = A4_WIDTH_CM - PDF_EDGE_MARGIN_CM - PHOTO_4UP_WIDTH_CM

    if not _close_enough(PDF_EDGE_MARGIN_CM, 0.3):
        errors.append("A margem externa nao e 0,3 cm.")
    if not _close_enough(PHOTO_4UP_WIDTH_CM, 10.0):
        errors.append("A largura da imagem nao e 10 cm.")
    if not _close_enough(PHOTO_4UP_HEIGHT_CM, 14.52):
        errors.append("A altura da imagem nao e 14,52 cm.")
    if not _close_enough(layout.horizontal_gap_cm, PDF_4UP_HORIZONTAL_GAP_CM):
        errors.append("O gap horizontal nao e 0,4 cm.")
    if not _close_enough(layout.vertical_gap_cm, PDF_4UP_VERTICAL_GAP_CM):
        errors.append("O gap vertical nao e 0,06 cm.")

    expected_slots = [
        ("superior esquerda", top_left, PDF_EDGE_MARGIN_CM, expected_top_y_cm),
        ("superior direita", top_right, expected_right_x_cm, expected_top_y_cm),
        ("inferior esquerda", bottom_left, PDF_EDGE_MARGIN_CM, PDF_EDGE_MARGIN_CM),
        ("inferior direita", bottom_right, expected_right_x_cm, PDF_EDGE_MARGIN_CM),
    ]
    for label, slot, expected_x_cm, expected_y_cm in expected_slots:
        if not _close_enough(slot.x_cm, expected_x_cm) or not _close_enough(slot.y_cm, expected_y_cm):
            errors.append(f"A imagem {label} nao esta na posicao fisica esperada.")
        if not _close_enough(slot.width_cm, PHOTO_4UP_WIDTH_CM) or not _close_enough(slot.height_cm, PHOTO_4UP_HEIGHT_CM):
            errors.append(f"A imagem {label} nao esta com 10 x 14,52 cm.")
        if slot.x_cm < PDF_EDGE_MARGIN_CM - 0.001 or slot.y_cm < PDF_EDGE_MARGIN_CM - 0.001:
            errors.append(f"A imagem {label} ficou fora da margem minima.")
        if slot.x_cm + slot.width_cm > A4_WIDTH_CM - PDF_EDGE_MARGIN_CM + 0.001:
            errors.append(f"A imagem {label} ultrapassa a largura da pagina.")
        if slot.y_cm + slot.height_cm > A4_HEIGHT_CM - PDF_EDGE_MARGIN_CM + 0.001:
            errors.append(f"A imagem {label} ultrapassa a altura da pagina.")

    top_gap_cm = top_right.x_cm - (top_left.x_cm + top_left.width_cm)
    bottom_gap_cm = bottom_right.x_cm - (bottom_left.x_cm + bottom_left.width_cm)
    left_vertical_gap_cm = top_left.y_cm - (bottom_left.y_cm + bottom_left.height_cm)
    right_vertical_gap_cm = top_right.y_cm - (bottom_right.y_cm + bottom_right.height_cm)
    if not _close_enough(top_gap_cm, PDF_4UP_HORIZONTAL_GAP_CM):
        errors.append("O gap horizontal superior nao e 0,4 cm.")
    if not _close_enough(bottom_gap_cm, PDF_4UP_HORIZONTAL_GAP_CM):
        errors.append("O gap horizontal inferior nao e 0,4 cm.")
    if not _close_enough(left_vertical_gap_cm, PDF_4UP_VERTICAL_GAP_CM):
        errors.append("O gap vertical esquerdo nao e 0,06 cm.")
    if not _close_enough(right_vertical_gap_cm, PDF_4UP_VERTICAL_GAP_CM):
        errors.append("O gap vertical direito nao e 0,06 cm.")

    if errors:
        logger.error("Validacao tecnica do layout 4 imagens reais falhou: %s", " ".join(errors))
        return False
    return True


def validate_4up_rotation(processed_images: list[ProcessedImage], layout_name: str) -> list[str]:
    if layout_name != PDF_LAYOUT_4_REAL_PHOTOS:
        return []

    warnings: list[str] = []
    for processed in processed_images:
        orientation = _detect_pdf_image_orientation(processed)
        if orientation == ORIENTATION_HORIZONTAL:
            if not processed.rotated_on_pdf or processed.pdf_rotation_degrees not in {90, -90}:
                warnings.append(f"{processed.output_name}: foto horizontal nao foi rotacionada no PDF 4 imagens.")
        elif orientation in {ORIENTATION_VERTICAL, ORIENTATION_SQUARE}:
            if processed.rotated_on_pdf or processed.pdf_rotation_degrees != 0:
                warnings.append(f"{processed.output_name}: foto {orientation} foi rotacionada indevidamente no PDF 4 imagens.")
    return warnings


def validate_4up_rotation_results(page_images: list[ProcessedImage]) -> list[str]:
    return validate_4up_rotation(page_images, PDF_LAYOUT_4_REAL_PHOTOS)


def _pop_first_matching(
    items: list[ProcessedImage],
    predicate,
) -> ProcessedImage | None:
    for index, item in enumerate(items):
        if predicate(item):
            return items.pop(index)
    return None


def _pop_next_matching(
    items: list[ProcessedImage],
    predicate,
    limit: int,
) -> tuple[list[ProcessedImage], int]:
    selected: list[ProcessedImage] = []
    skipped = 0
    index = 0
    while index < len(items) and len(selected) < limit:
        if predicate(items[index]):
            selected.append(items.pop(index))
        else:
            skipped += 1
            index += 1
    return selected, skipped


def _items_to_vertical_slots(items: list[ProcessedImage], centered_when_single: bool = False) -> list[_PlannedItem]:
    if not items:
        return []
    if len(items) == 1:
        position = "inferior central" if centered_when_single else "centralizada"
        return [_PlannedItem(items[0], position, ORIENTATION_VERTICAL)]
    return [
        _PlannedItem(items[0], "inferior esquerda", ORIENTATION_VERTICAL),
        _PlannedItem(items[1], "inferior direita", ORIENTATION_VERTICAL),
    ]


def _horizontal_page_items(items: list[ProcessedImage]) -> list[_PlannedItem]:
    if len(items) == 1:
        return [_PlannedItem(items[0], "centralizada", ORIENTATION_HORIZONTAL)]
    return [
        _PlannedItem(items[0], "superior", ORIENTATION_HORIZONTAL),
        _PlannedItem(items[1], "inferior", ORIENTATION_HORIZONTAL),
    ]


def _real_vertical_page_items(items: list[ProcessedImage]) -> list[_PlannedItem]:
    if not items:
        return []
    if len(items) == 1:
        return [_PlannedItem(items[0], "centralizada", ORIENTATION_VERTICAL)]
    return [
        _PlannedItem(items[0], "par esquerda", ORIENTATION_VERTICAL),
        _PlannedItem(items[1], "par direita", ORIENTATION_VERTICAL),
    ]


def _plan_auto_pages(processed_images: list[ProcessedImage]) -> list[list[_PlannedItem]]:
    horizontals = [item for item in processed_images if _is_horizontal(item)]
    verticals = [item for item in processed_images if _is_vertical_like(item)]
    pages: list[list[_PlannedItem]] = []

    while horizontals or verticals:
        if horizontals and verticals:
            top = horizontals.pop(0)
            bottoms = [verticals.pop(0) for _ in range(min(2, len(verticals)))]
            pages.append([_PlannedItem(top, "topo", ORIENTATION_HORIZONTAL)] + _items_to_vertical_slots(bottoms, True))
        elif verticals:
            if len(verticals) >= 3:
                top = verticals.pop(0)
                bottoms = [verticals.pop(0), verticals.pop(0)]
                pages.append([_PlannedItem(top, "topo", ORIENTATION_HORIZONTAL)] + _items_to_vertical_slots(bottoms, True))
            elif len(verticals) == 2:
                pages.append(_items_to_vertical_slots([verticals.pop(0), verticals.pop(0)]))
            else:
                pages.append(_items_to_vertical_slots([verticals.pop(0)]))
        else:
            page_items = [horizontals.pop(0)]
            if horizontals:
                page_items.append(horizontals.pop(0))
            pages.append(_horizontal_page_items(page_items))

    return pages


def _plan_upload_order_pages(processed_images: list[ProcessedImage]) -> tuple[list[list[_PlannedItem]], list[str]]:
    remaining = list(processed_images)
    pages: list[list[_PlannedItem]] = []
    warnings: list[str] = []
    order_adjusted = False

    while remaining:
        first = remaining.pop(0)
        remaining_verticals = sum(1 for item in remaining if _is_vertical_like(item))

        if _is_horizontal(first):
            if remaining_verticals == 0:
                page_items = [first]
                second = _pop_first_matching(remaining, _is_horizontal)
                if second is not None:
                    page_items.append(second)
                pages.append(_horizontal_page_items(page_items))
                continue

            bottoms, skipped = _pop_next_matching(remaining, _is_vertical_like, 2)
            order_adjusted = order_adjusted or skipped > 0
            pages.append([_PlannedItem(first, "topo", ORIENTATION_HORIZONTAL)] + _items_to_vertical_slots(bottoms, True))
            continue

        total_verticals = remaining_verticals + 1
        if total_verticals >= 3:
            bottoms, skipped = _pop_next_matching(remaining, _is_vertical_like, 2)
            order_adjusted = order_adjusted or skipped > 0
            pages.append([_PlannedItem(first, "topo", ORIENTATION_HORIZONTAL)] + _items_to_vertical_slots(bottoms, True))
        elif total_verticals == 2:
            second, skipped = _pop_next_matching(remaining, _is_vertical_like, 1)
            order_adjusted = order_adjusted or skipped > 0
            page_items = [first]
            page_items.extend(second)
            pages.append(_items_to_vertical_slots(page_items))
        else:
            pages.append(_items_to_vertical_slots([first]))

    if order_adjusted:
        warnings.append("Preservei a ordem das fotos sempre que possível, mas ajustei alguns encaixes para evitar fotos na posição errada.")

    return pages, warnings


def _plan_three_photo_pages(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
) -> tuple[list[list[_PlannedItem]], list[str]]:
    extra_page_items = [item for item in processed_images if item.ai_create_extra_page_requested]
    regular_items = [item for item in processed_images if not item.ai_create_extra_page_requested]

    if options.organize_mode == PDF_ORGANIZE_UPLOAD_ORDER:
        pages, warnings = _plan_upload_order_pages(regular_items)
    else:
        pages = _plan_auto_pages(regular_items)
        warnings = ["Organizei as fotos automaticamente pela orientação."]

    for item in extra_page_items:
        pages.append(_horizontal_page_items([item]) if _is_horizontal(item) else _items_to_vertical_slots([item]))

    if extra_page_items:
        warnings.append("Criei página extra para preservar foto indicada pela decisão validada.")

    return pages, warnings


def _plan_three_real_auto_pages(processed_images: list[ProcessedImage]) -> list[list[_PlannedItem]]:
    horizontals = [item for item in processed_images if _is_horizontal(item)]
    verticals = [item for item in processed_images if _is_vertical_like(item)]
    pages: list[list[_PlannedItem]] = []

    while horizontals or verticals:
        if horizontals and verticals:
            top = horizontals.pop(0)
            bottoms = [verticals.pop(0) for _ in range(min(2, len(verticals)))]
            pages.append([_PlannedItem(top, "topo", ORIENTATION_HORIZONTAL)] + _items_to_vertical_slots(bottoms, True))
            continue

        if verticals:
            if len(verticals) >= 3:
                top = verticals.pop(0)
                bottoms = [verticals.pop(0), verticals.pop(0)]
                pages.append([_PlannedItem(top, "topo", ORIENTATION_HORIZONTAL)] + _items_to_vertical_slots(bottoms, True))
            elif len(verticals) == 2:
                pages.append(_real_vertical_page_items([verticals.pop(0), verticals.pop(0)]))
            else:
                pages.append(_real_vertical_page_items([verticals.pop(0)]))
            continue

        pages.append([_PlannedItem(horizontals.pop(0), "centralizada", ORIENTATION_HORIZONTAL)])

    return pages


def _plan_three_real_upload_order_pages(processed_images: list[ProcessedImage]) -> tuple[list[list[_PlannedItem]], list[str]]:
    remaining = list(processed_images)
    pages: list[list[_PlannedItem]] = []
    warnings: list[str] = []
    order_adjusted = False

    while remaining:
        first = remaining.pop(0)
        remaining_verticals = sum(1 for item in remaining if _is_vertical_like(item))

        if _is_horizontal(first):
            if remaining_verticals == 0:
                pages.append([_PlannedItem(first, "centralizada", ORIENTATION_HORIZONTAL)])
                continue

            bottoms, skipped = _pop_next_matching(remaining, _is_vertical_like, 2)
            order_adjusted = order_adjusted or skipped > 0
            pages.append([_PlannedItem(first, "topo", ORIENTATION_HORIZONTAL)] + _items_to_vertical_slots(bottoms, True))
            continue

        total_verticals = remaining_verticals + 1
        if total_verticals >= 3:
            bottoms, skipped = _pop_next_matching(remaining, _is_vertical_like, 2)
            order_adjusted = order_adjusted or skipped > 0
            pages.append([_PlannedItem(first, "topo", ORIENTATION_HORIZONTAL)] + _items_to_vertical_slots(bottoms, True))
        elif total_verticals == 2:
            second, skipped = _pop_next_matching(remaining, _is_vertical_like, 1)
            order_adjusted = order_adjusted or skipped > 0
            pages.append(_real_vertical_page_items([first] + second))
        else:
            pages.append(_real_vertical_page_items([first]))

    if order_adjusted:
        warnings.append("Preservei a ordem das fotos sempre que possível, ajustando apenas os encaixes do layout real.")

    return pages, warnings


def _plan_three_real_photo_pages(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
) -> tuple[list[list[_PlannedItem]], list[str]]:
    if options.organize_mode == PDF_ORGANIZE_UPLOAD_ORDER:
        pages, warnings = _plan_three_real_upload_order_pages(processed_images)
    else:
        pages = _plan_three_real_auto_pages(processed_images)
        warnings = ["Organizei as fotos automaticamente pela orientação no layout 10x15 real."]

    warnings.insert(0, "Layout 3 fotos 10x15 reais por A4: margem de 3 mm e 2 mm entre fotos.")
    return pages, warnings


def _slot_for_planned_item(layout: _ThreePhotoLayout, item: _PlannedItem) -> _BoxCm:
    if item.position_label == "topo":
        return layout.top_slot
    if item.position_label == "inferior esquerda":
        return layout.left_slot
    if item.position_label == "inferior direita":
        return layout.right_slot
    if item.position_label == "inferior central":
        return layout.bottom_center_slot
    if item.position_label == "par esquerda":
        return layout.pair_left_slot
    if item.position_label == "par direita":
        return layout.pair_right_slot
    if item.position_label == "centralizada":
        if item.slot_type == ORIENTATION_HORIZONTAL:
            return layout.horizontal_single_slot
        return layout.single_slot
    if item.position_label == "superior":
        return layout.horizontal_upper_slot
    if item.position_label == "inferior":
        return layout.horizontal_lower_slot
    return layout.single_slot


def _position_label_from_batch_slot(position: str) -> str:
    return {
        "top": "topo",
        "bottom_left": "inferior esquerda",
        "bottom_right": "inferior direita",
        "center": "centralizada",
        "top_1": "superior",
        "top_2": "inferior",
    }.get(position, "centralizada")


def _slot_for_batch_slot(layout: _ThreePhotoLayout, planned_slot: PlannedSlot) -> _BoxCm:
    if planned_slot.position == "top":
        return layout.top_slot
    if planned_slot.position == "bottom_left":
        return layout.left_slot
    if planned_slot.position == "bottom_right":
        return layout.right_slot
    if planned_slot.position == "top_1":
        return layout.horizontal_upper_slot
    if planned_slot.position == "top_2":
        return layout.horizontal_lower_slot
    if planned_slot.slot_type == ORIENTATION_HORIZONTAL:
        return layout.horizontal_single_slot
    return layout.single_slot


def generate_pdf_from_batch_plan(
    processed_images: list[ProcessedImage],
    final_batch_plan: BatchPlan,
    options: PdfOptions,
) -> PdfGenerationResult:
    """Gera PDF seguindo um BatchPlan ja validado localmente."""
    if options.layout_mode == PDF_LAYOUT_3_REAL_PHOTOS:
        ordered_images = _ordered_images_from_batch_plan(processed_images, final_batch_plan)
        return generate_pdf_3_real_photos_a4_with_summary(ordered_images, options)
    if options.layout_mode == PDF_LAYOUT_4_REAL_PHOTOS:
        ordered_images = _ordered_images_from_batch_plan(processed_images, final_batch_plan)
        return generate_pdf_4_real_images_a4_with_summary(ordered_images, options)

    if not final_batch_plan.pages:
        return generate_pdf_3_photos_layout_with_summary(processed_images, options)

    buffer = BytesIO()
    page_width_pt, page_height_pt = _page_size_points(A4_WIDTH_CM, A4_HEIGHT_CM)
    pdf = canvas.Canvas(buffer, pagesize=(page_width_pt, page_height_pt))
    layout = _calculate_three_photo_layout(options)
    items_by_name = {item.output_name: item for item in processed_images}
    placements: list[PdfPlacement] = []
    warnings = list(final_batch_plan.global_warnings)

    for page in final_batch_plan.pages:
        pdf.setPageSize((page_width_pt, page_height_pt))
        for planned_slot in page.slots:
            if not planned_slot.image_name:
                continue
            processed = items_by_name.get(planned_slot.image_name)
            if processed is None:
                warnings.append(f"Foto nao encontrada no plano final: {planned_slot.image_name}")
                continue

            slot = _slot_for_batch_slot(layout, planned_slot)
            placement = _draw_slot_image_cm(
                pdf,
                processed,
                slot,
                planned_slot.slot_type,
                page.page_number,
                _position_label_from_batch_slot(planned_slot.position),
            )
            placements.append(
                replace(
                    placement,
                    rotated_on_pdf=placement.rotated_on_pdf or planned_slot.rotate_on_pdf,
                    fit_strategy=planned_slot.fit_strategy or placement.fit_strategy,
                )
            )
            if options.show_cut_lines:
                _draw_cut_rect_cm(pdf, slot.x_cm, slot.y_cm, slot.width_cm, slot.height_cm)
        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    if final_batch_plan.source == "ai_validated":
        warnings.append("Usei o plano de lote validado pela IA e pelas regras locais.")
    elif final_batch_plan.source == "fallback":
        warnings.append("A IA nao respondeu ou foi descartada; usei o planejamento local.")
    else:
        warnings.append("Usei o planejamento local do lote.")
    return PdfGenerationResult(
        pdf_bytes=buffer.getvalue(),
        placements=placements,
        warnings=list(dict.fromkeys(warnings)),
    )


def generate_pdf_4_real_images_a4(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
) -> bytes:
    return generate_pdf_4_real_images_a4_with_summary(processed_images, options).pdf_bytes


def generate_pdf_4_real_images_a4_with_summary(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
) -> PdfGenerationResult:
    """Gera A4 retrato com ate 4 imagens 10x14,52 por folha."""
    layout_valid = validate_4up_real_layout()
    buffer = BytesIO()
    page_width_pt, page_height_pt = _page_size_points(A4_WIDTH_CM, A4_HEIGHT_CM)
    pdf = canvas.Canvas(buffer, pagesize=(page_width_pt, page_height_pt))

    layout = _calculate_4up_real_layout()
    warnings = [
        "Layout 4 imagens 10x14,52 por A4: margem de 3 mm, gap horizontal de 4 mm e gap vertical de 0,6 mm."
    ]
    if not layout_valid:
        warnings.append("Aviso tecnico: a validacao do layout 4 imagens falhou; detalhes foram registrados no log.")

    placements: list[PdfPlacement] = []
    position_labels = (
        "superior esquerda",
        "superior direita",
        "inferior esquerda",
        "inferior direita",
    )

    for page_number, page_images in enumerate(chunked(processed_images, 4), start=1):
        pdf.setPageSize((page_width_pt, page_height_pt))

        for processed, slot, position_label in zip(page_images, layout.slots, position_labels):
            placement = _draw_4up_slot_image_cm(
                pdf,
                processed,
                slot,
                page_number,
                position_label,
            )
            placements.append(placement)
            if options.show_cut_lines:
                _draw_cut_rect_cm(pdf, slot.x_cm, slot.y_cm, slot.width_cm, slot.height_cm)

        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    rotated_count = sum(1 for placement in placements if placement.rotated_on_pdf)
    if rotated_count:
        warnings.append(f"{rotated_count} foto(s) horizontais foram rotacionadas apenas no PDF para caber no layout 4 imagens.")
    rotation_warnings = validate_4up_rotation(processed_images, PDF_LAYOUT_4_REAL_PHOTOS)
    if rotation_warnings:
        logger.error("Validacao de rotacao do layout 4 imagens falhou: %s", " ".join(rotation_warnings))
        warnings.append("Aviso tecnico: a validacao de rotacao do layout 4 imagens falhou; detalhes foram registrados no log.")
    return PdfGenerationResult(
        pdf_bytes=buffer.getvalue(),
        placements=placements,
        warnings=list(dict.fromkeys(warnings)),
    )


def generate_pdf_3_real_photos_a4(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
) -> bytes:
    return generate_pdf_3_real_photos_a4_with_summary(processed_images, options).pdf_bytes


def generate_pdf_3_real_photos_a4_with_summary(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
) -> PdfGenerationResult:
    """Gera A4 retrato com ate 3 fotos 10x15 reais por folha."""
    layout_valid = validate_3_real_photos_layout()
    buffer = BytesIO()
    page_width_pt, page_height_pt = _page_size_points(A4_WIDTH_CM, A4_HEIGHT_CM)
    pdf = canvas.Canvas(buffer, pagesize=(page_width_pt, page_height_pt))

    layout = _calculate_three_real_photos_layout()
    planned_pages, warnings = _plan_three_real_photo_pages(processed_images, options)
    if not layout_valid:
        warnings.append("Aviso tecnico: a validacao do layout real falhou; detalhes foram registrados no log.")

    placements: list[PdfPlacement] = []

    for page_number, page_items in enumerate(planned_pages, start=1):
        pdf.setPageSize((page_width_pt, page_height_pt))

        for item in page_items:
            slot = _slot_for_planned_item(layout, item)
            placement = _draw_slot_image_cm(
                pdf,
                item.processed,
                slot,
                item.slot_type,
                page_number,
                item.position_label,
            )
            placements.append(placement)
            if options.show_cut_lines:
                _draw_cut_rect_cm(pdf, slot.x_cm, slot.y_cm, slot.width_cm, slot.height_cm)

        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return PdfGenerationResult(
        pdf_bytes=buffer.getvalue(),
        placements=placements,
        warnings=list(dict.fromkeys(warnings)),
    )


def generate_pdf_3_photos_layout(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
) -> bytes:
    return generate_pdf_3_photos_layout_with_summary(processed_images, options).pdf_bytes


def generate_pdf_3_photos_layout_with_summary(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
) -> PdfGenerationResult:
    """Gera A4 retrato com 3 fotos em proporção 10x15 por folha."""
    buffer = BytesIO()
    page_width_pt, page_height_pt = _page_size_points(A4_WIDTH_CM, A4_HEIGHT_CM)
    pdf = canvas.Canvas(buffer, pagesize=(page_width_pt, page_height_pt))

    layout = _calculate_three_photo_layout(options)
    planned_pages, warnings = _plan_three_photo_pages(processed_images, options)
    placements: list[PdfPlacement] = []

    for page_number, page_items in enumerate(planned_pages, start=1):
        pdf.setPageSize((page_width_pt, page_height_pt))

        for item in page_items:
            slot = _slot_for_planned_item(layout, item)
            placement = _draw_slot_image_cm(
                pdf,
                item.processed,
                slot,
                item.slot_type,
                page_number,
                item.position_label,
            )
            placements.append(placement)
            if options.show_cut_lines:
                _draw_cut_rect_cm(pdf, slot.x_cm, slot.y_cm, slot.width_cm, slot.height_cm)

        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return PdfGenerationResult(
        pdf_bytes=buffer.getvalue(),
        placements=placements,
        warnings=warnings,
    )


def _slot_for_real_photo(processed: ProcessedImage) -> _PhotoSlot:
    if processed.image.width >= processed.image.height:
        return _PhotoSlot(processed, PHOTO_10X15_HEIGHT_CM, PHOTO_10X15_WIDTH_CM)
    return _PhotoSlot(processed, PHOTO_10X15_WIDTH_CM, PHOTO_10X15_HEIGHT_CM)


def _fits(
    total_width_cm: float,
    total_height_cm: float,
    page_width_cm: float,
    page_height_cm: float,
    margin_cm: float,
) -> bool:
    return (
        total_width_cm <= page_width_cm - 2 * margin_cm + 0.001
        and total_height_cm <= page_height_cm - 2 * margin_cm + 0.001
    )


def _layout_for_slots(
    slots: list[_PhotoSlot],
    margin_cm: float,
    gap_cm: float,
) -> _LayoutCandidate | None:
    if len(slots) == 1:
        slot = slots[0]
        page_width_cm, page_height_cm = A4_WIDTH_CM, A4_HEIGHT_CM
        if slot.width_cm > slot.height_cm:
            page_width_cm, page_height_cm = A4_HEIGHT_CM, A4_WIDTH_CM
        return _LayoutCandidate(page_width_cm, page_height_cm, "single", slot.width_cm, slot.height_cm)

    arrangements = [
        (A4_WIDTH_CM, A4_HEIGHT_CM, "stack"),
        (A4_WIDTH_CM, A4_HEIGHT_CM, "side_by_side"),
        (A4_HEIGHT_CM, A4_WIDTH_CM, "side_by_side"),
        (A4_HEIGHT_CM, A4_WIDTH_CM, "stack"),
    ]

    candidates: list[_LayoutCandidate] = []
    for page_width_cm, page_height_cm, arrangement in arrangements:
        if arrangement == "side_by_side":
            total_width_cm = sum(slot.width_cm for slot in slots) + gap_cm
            total_height_cm = max(slot.height_cm for slot in slots)
        else:
            total_width_cm = max(slot.width_cm for slot in slots)
            total_height_cm = sum(slot.height_cm for slot in slots) + gap_cm

        if _fits(total_width_cm, total_height_cm, page_width_cm, page_height_cm, margin_cm):
            candidates.append(
                _LayoutCandidate(
                    page_width_cm,
                    page_height_cm,
                    arrangement,
                    total_width_cm,
                    total_height_cm,
                )
            )

    if not candidates:
        return None
    return candidates[0]


def _draw_real_page(
    pdf: canvas.Canvas,
    slots: list[_PhotoSlot],
    layout: _LayoutCandidate,
    options: PdfOptions,
) -> None:
    page_width_pt, page_height_pt = _page_size_points(layout.page_width_cm, layout.page_height_cm)
    pdf.setPageSize((page_width_pt, page_height_pt))

    usable_width_cm = layout.page_width_cm - 2 * options.margin_cm
    usable_height_cm = layout.page_height_cm - 2 * options.margin_cm
    start_x_cm = options.margin_cm + (usable_width_cm - layout.total_width_cm) / 2
    start_y_cm = options.margin_cm + (usable_height_cm - layout.total_height_cm) / 2

    if layout.arrangement == "single":
        slot = slots[0]
        x_cm = options.margin_cm + (usable_width_cm - slot.width_cm) / 2
        y_cm = options.margin_cm + (usable_height_cm - slot.height_cm) / 2
        draw_image_cm(pdf, slot.processed.image, x_cm, y_cm, slot.width_cm, slot.height_cm)
        if options.show_cut_lines:
            _draw_cut_rect_cm(pdf, x_cm, y_cm, slot.width_cm, slot.height_cm)
        return

    if layout.arrangement == "side_by_side":
        x_cm = start_x_cm
        for slot in slots:
            y_cm = start_y_cm + (layout.total_height_cm - slot.height_cm) / 2
            draw_image_cm(pdf, slot.processed.image, x_cm, y_cm, slot.width_cm, slot.height_cm)
            if options.show_cut_lines:
                _draw_cut_rect_cm(pdf, x_cm, y_cm, slot.width_cm, slot.height_cm)
            x_cm += slot.width_cm + options.gap_cm
        return

    current_top_cm = start_y_cm + layout.total_height_cm
    for slot in slots:
        y_cm = current_top_cm - slot.height_cm
        x_cm = start_x_cm + (layout.total_width_cm - slot.width_cm) / 2
        draw_image_cm(pdf, slot.processed.image, x_cm, y_cm, slot.width_cm, slot.height_cm)
        if options.show_cut_lines:
            _draw_cut_rect_cm(pdf, x_cm, y_cm, slot.width_cm, slot.height_cm)
        current_top_cm = y_cm - options.gap_cm


def generate_pdf_2_real_layout(
    processed_images: list[ProcessedImage],
    options: PdfOptions,
) -> bytes:
    """Gera PDF com fotos em tamanho físico real 10x15 ou 15x10."""
    buffer = BytesIO()
    initial_page = _page_size_points(A4_WIDTH_CM, A4_HEIGHT_CM)
    pdf = canvas.Canvas(buffer, pagesize=initial_page)

    for group in chunked(processed_images, 2):
        slots = [_slot_for_real_photo(processed) for processed in group]
        layout = _layout_for_slots(slots, options.margin_cm, options.gap_cm)

        if layout is None and len(slots) == 2:
            for slot in slots:
                single_layout = _layout_for_slots([slot], options.margin_cm, options.gap_cm)
                if single_layout is None:
                    logger.warning("Nao foi possivel posicionar a foto %s em tamanho real", slot.processed.output_name)
                    continue
                _draw_real_page(pdf, [slot], single_layout, options)
                pdf.showPage()
            continue

        if layout is not None:
            _draw_real_page(pdf, slots, layout, options)
            pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
