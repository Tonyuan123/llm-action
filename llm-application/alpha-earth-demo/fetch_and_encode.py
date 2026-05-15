"""从 Google Earth Engine 的 AlphaEarth Foundations 数据集拉取一块 ROI 影像，
保存为 PNG 并做 base64 编码 + 编码格式校验。

数据集: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
    - 每年 1 张 mosaic
    - 64 个嵌入波段 A00 ~ A63
    - 原生分辨率 10 m

前置:
    pip install earthengine-api requests
    earthengine authenticate          # 首次需登录
    # 或者 export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json 使用服务账号

用法示例:
    python fetch_and_encode.py --project my-gee-project \\
        --lon 116.397 --lat 39.908 --year 2024
"""

import argparse
import base64
import binascii
import sys
from pathlib import Path

import ee
import requests


ALPHA_EARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def init_ee(project: str | None) -> None:
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


def build_image(year: int, lon: float, lat: float, buffer_m: int):
    region = ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds()
    coll = (
        ee.ImageCollection(ALPHA_EARTH_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
    )
    return coll.mosaic().clip(region), region


def fetch_png(image, region, bands: list[str], scale: int) -> bytes:
    url = image.getThumbURL(
        {
            "region": region,
            "scale": scale,
            "bands": bands,
            "min": -0.3,
            "max": 0.3,
            "format": "png",
        }
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def validate_base64_png(b64: str) -> dict:
    """校验一段 base64 字符串：字符集、padding、能否解码、是否 PNG。"""
    report = {
        "length_chars": len(b64),
        "padding_ok": len(b64) % 4 == 0,
        "valid_base64": False,
        "decoded_bytes": 0,
        "is_png": False,
        "error": None,
    }
    try:
        raw = base64.b64decode(b64, validate=True)
    except binascii.Error as exc:
        report["error"] = str(exc)
        return report
    report["valid_base64"] = True
    report["decoded_bytes"] = len(raw)
    report["is_png"] = raw.startswith(PNG_MAGIC)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=None, help="GEE Cloud Project ID")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--lon", type=float, default=116.397)
    p.add_argument("--lat", type=float, default=39.908)
    p.add_argument("--buffer", type=int, default=2000, help="ROI 缓冲半径 (米)")
    p.add_argument("--scale", type=int, default=10, help="重采样分辨率 (米)")
    p.add_argument(
        "--bands",
        nargs=3,
        default=["A01", "A16", "A09"],
        help="从 A00~A63 里挑 3 个嵌入维度做 RGB 可视化",
    )
    p.add_argument("--out", default="alpha_earth.png")
    p.add_argument("--out-b64", default="alpha_earth.b64")
    args = p.parse_args()

    init_ee(args.project)
    image, region = build_image(args.year, args.lon, args.lat, args.buffer)
    png_bytes = fetch_png(image, region, args.bands, args.scale)
    Path(args.out).write_bytes(png_bytes)

    b64 = base64.b64encode(png_bytes).decode("ascii")
    Path(args.out_b64).write_text(b64)

    r = validate_base64_png(b64)
    print(f"PNG bytes      : {len(png_bytes)}  -> {args.out}")
    print(f"base64 chars   : {r['length_chars']}  -> {args.out_b64}")
    print(f"padding %4==0  : {r['padding_ok']}")
    print(f"base64 decodes : {r['valid_base64']}  ({r['decoded_bytes']} bytes back)")
    print(f"PNG magic ok   : {r['is_png']}")
    if r["error"]:
        print(f"error          : {r['error']}", file=sys.stderr)

    return 0 if r["valid_base64"] and r["is_png"] else 1


if __name__ == "__main__":
    sys.exit(main())
