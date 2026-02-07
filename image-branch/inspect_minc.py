import os

# root = "..\data\minc-2500"

# 脚本所在目录：.../LSMRT/image-branch
HERE = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：.../LSMRT
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

# 数据目录：.../LSMRT/data/minc-2500
root = os.path.join(PROJECT_ROOT, "data", "minc-2500")

def walk_top(root, max_depth=2):
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = rel.count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        print(f"[DIR] {rel}")
        # 只显示少量文件名
        for fn in filenames[:8]:
            print("   -", fn)
        if len(filenames) > 8:
            print("   ...")
        print()

walk_top(root, max_depth=3)
