import os
from pathlib import Path
from PIL import Image
import pillow_heif

# 注册 HEIC/HEIF 支持
pillow_heif.register_heif_opener()

INPUT_DIR = Path("/root/mqx/LSMRT/data/aligned-data/img/heic")
OUTPUT_DIR = Path("/root/mqx/LSMRT/data/aligned-data/img/")

def convert_heic_to_jpg(input_dir: Path, output_dir: Path, quality: int = 95):
    output_dir.mkdir(parents=True, exist_ok=True)

    heic_files = list(input_dir.glob("*.heic")) + list(input_dir.glob("*.HEIC"))

    if not heic_files:
        print("没有找到 HEIC 文件")
        return

    print(f"共找到 {len(heic_files)} 个 HEIC 文件")

    success = 0
    failed = 0

    for heic_file in heic_files:
        try:
            jpg_file = output_dir / (heic_file.stem + ".jpg")

            with Image.open(heic_file) as img:
                img = img.convert("RGB")
                img.save(jpg_file, "JPEG", quality=quality)

            print(f"[OK] {heic_file.name} -> {jpg_file.name}")
            success += 1

        except Exception as e:
            print(f"[FAIL] {heic_file.name}: {e}")
            failed += 1

    print("\n=== 转换完成 ===")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    convert_heic_to_jpg(INPUT_DIR, OUTPUT_DIR)