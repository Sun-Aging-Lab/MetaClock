"""Register the audited Liberation Sans assets used by generalization plots."""

from __future__ import annotations

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
def register_fonts() -> None:
    """Register only the bundled formal manuscript font files."""
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
        font_manager.fontManager.addfont(path)
    resolved = Path(font_manager.findfont("Liberation Sans", fallback_to_default=False))
    if resolved.resolve() not in allowed:
        raise RuntimeError(f"Liberation Sans resolved outside bundled assets: {resolved}")
