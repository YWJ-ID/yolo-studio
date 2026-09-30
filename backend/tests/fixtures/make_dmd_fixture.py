"""生成 DMD / OpenLABEL(VCD) 测试夹具。

用官方 `vcd` 库产出**规范一致**的 VCD 文件，避免手写 JSON 猜错 schema。

本脚本需要 `vcd` 包（6.x），仅在生成夹具时使用一次；
生成的 JSON + 帧图片会被提交，测试本身不依赖 vcd 库。

用法（用 DMD 仓库自带的环境执行）：
    & "<DMD>/py38/Scripts/python.exe" backend/tests/fixtures/make_dmd_fixture.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import vcd.core as core
import vcd.types as types
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent / "dmd_sample"
NUM_FRAMES = 60
FRAME_SIZE = (48, 36)

# driver_actions 时间轴：(帧区间, 标签)
DRIVER_ACTIONS = [
    ([0, 9], "safe_drive"),
    ([10, 29], "texting_right"),
    ([30, 34], "drinking"),
    ([35, 59], "safe_drive"),
]

# gaze_on_road 时间轴
GAZE = [
    ([0, 59], "looking_road"),
]

# objects_in_scene（object 类型，无 bbox）
OBJECTS = [
    ([10, 29], "cellphone"),
]


def make_frames(frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_FRAMES):
        # 用帧号生成可区分的纯色图，便于人眼确认映射是否正确
        color = ((i * 4) % 256, (i * 8) % 256, (i * 16) % 256)
        Image.new("RGB", FRAME_SIZE, color).save(frames_dir / f"{i:06d}.jpg", quality=60)


def make_vcd(path: Path) -> None:
    vcd = core.VCD()

    vcd.add_name("gA_s1_attm_2020-01-01_driver_actions")
    vcd.add_ontology("http://dmd.vicomtech.org/ontology")
    vcd.add_annotator("fixture")

    # 与 TaTo 一致：单路 general_camera 承载 mosaic 视频
    vcd.add_stream(
        "general_camera",
        "sessionA.mp4",
        "Unique general camera",
        core.StreamType.camera,
    )
    vcd.add_stream_properties(
        stream_name="general_camera",
        properties={"total_frames": NUM_FRAMES},
        stream_sync=types.StreamSync(frame_shift=0),
    )

    # action 类标注：semantic_type = "level/label"
    for (start, end), label in DRIVER_ACTIONS:
        vcd.add_action(
            "",
            semantic_type=f"driver_actions/{label}",
            frame_value=[[start, end]],
            ont_uid=0,
        )

    for (start, end), label in GAZE:
        vcd.add_action(
            "",
            semantic_type=f"gaze_on_road/{label}",
            frame_value=[[start, end]],
            ont_uid=0,
        )

    # object 类标注：semantic_type = label
    for (start, end), label in OBJECTS:
        vcd.add_object(
            "",
            semantic_type=label,
            frame_value=[[start, end]],
            ont_uid=0,
        )

    vcd.save(str(path), pretty=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_frames(OUT_DIR / "frames")
    make_vcd(OUT_DIR / "sessionA.json")
    print(f"夹具已生成: {OUT_DIR}")
    print(f"  帧数: {NUM_FRAMES}")
    print(f"  VCD : {(OUT_DIR / 'sessionA.json').stat().st_size} bytes")


if __name__ == "__main__":
    main()
