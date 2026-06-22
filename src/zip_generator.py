"""Geração do ZIP com as imagens processadas."""

from __future__ import annotations

from io import BytesIO
import zipfile

from src.image_processing import save_jpg_bytes
from src.utils import ProcessedImage


def generate_zip_bytes(processed_images: list[ProcessedImage]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for processed in processed_images:
            image_bytes = save_jpg_bytes(processed.image).getvalue()
            zip_file.writestr(processed.output_name, image_bytes)

    buffer.seek(0)
    return buffer.getvalue()

