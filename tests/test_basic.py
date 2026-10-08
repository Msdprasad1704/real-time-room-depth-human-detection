from pathlib import Path

import numpy as np


def test_project_layout_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "main.py").exists()
    assert (root / "src").is_dir()
    assert (root / "requirements.txt").exists()
    assert (root / "README.md").exists()


def test_main_module_is_importable():
    import main

    assert hasattr(main, "build_parser")


def test_video_writer_uses_reliable_windows_codec_and_creates_file():
    import main

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    writer, path = main.create_video_writer(frame.shape, fps=10.0)

    assert writer is not None
    assert writer.isOpened()
    writer.write(frame)
    writer.release()
    assert path is not None
    assert path.exists()
    assert path.parent == main.VIDEO_DIR
    assert path.suffix.lower() == ".avi"
