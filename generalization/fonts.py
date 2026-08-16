"""Register the audited Liberation Sans assets used by generalization plots."""

from __future__ import annotations

import hashlib
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
FONT_FILES = tuple(
    FONT_DIR / name
    for name in (
        "LiberationSans-Regular.ttf",
        "LiberationSans-Italic.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationSans-BoldItalic.ttf",
    )
)
EXPECTED_SHA256 = {
    "LiberationSans-Regular.ttf": (
        "76d04c18ea243f426b7de1f3ad208e927008f961dc5945e5aad352d0dfde8ee8"
    ),
    "LiberationSans-Italic.ttf": (
        "e5bae5c4cde31f22142753855f4f8fb86da6ff39955ed3c0a11248b0d16948b0"
    ),
    "LiberationSans-Bold.ttf": ("788abee4c806d660e8aee46689dd8540cd4bb98da03dcc9d171ce3efd99a9173"),
    "LiberationSans-BoldItalic.ttf": (
        "698da70fc191cc5f33ad4d6d3fe830fe4624b898ea2e3169955928b7c491f1ee"
    ),
}


def register_fonts() -> None:
    """Verify and register only the bundled formal manuscript font files."""
    from matplotlib import font_manager

    missing = [path for path in FONT_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing bundled fonts: {missing}")
    allowed = {path.resolve() for path in FONT_FILES}
    font_manager.fontManager.ttflist = [
        entry
        for entry in font_manager.fontManager.ttflist
        if entry.name != "Liberation Sans" or Path(entry.fname).resolve() in allowed
    ]
    for path in FONT_FILES:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != EXPECTED_SHA256[path.name]:
            raise ValueError(f"Unexpected font checksum: {path}")
        font_manager.fontManager.addfont(path)
    resolved = Path(font_manager.findfont("Liberation Sans", fallback_to_default=False))
    if resolved.resolve() not in allowed:
        raise RuntimeError(f"Liberation Sans resolved outside bundled assets: {resolved}")
