"""Build a distributable Windows version of PitchView.

    python build.py

Produces dist/PitchView/PitchView.exe (a folder build: starts fast, fewer antivirus false
positives than a single-file exe) and release/PitchView-<version>-win64.zip to hand out.
Needs: pip install pyinstaller
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION = re.search(r'^VERSION = "([^"]+)"', open(os.path.join(ROOT, "pitchview.py"), encoding="utf-8").read(), re.M)[1]
DIST = os.path.join(ROOT, "dist", "PitchView")
RELEASE = os.path.join(ROOT, "release")
ZIP_BASE = os.path.join(RELEASE, f"PitchView-{VERSION}-win64")


def main():
    os.chdir(ROOT)
    sep = os.pathsep  # PyInstaller's --add-data separator (";" on Windows)
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
        "--name", "PitchView", "--icon", os.path.join("assets", "pitchview.ico"),
        "--add-data", f"fonts{sep}fonts", "--add-data", f"assets{sep}assets",
        # installed here but not used by the app; keeps the build small
        "--exclude-module", "scipy", "--exclude-module", "matplotlib", "--exclude-module", "tkinter",
        "--exclude-module", "PIL", "--exclude-module", "pyqtgraph", "--exclude-module", "fontTools",
        "pitchview.py",
    ]
    print("building PitchView", VERSION)
    subprocess.run(cmd, check=True)

    for name in ("README.md", "THIRD_PARTY_NOTICES.txt"):
        shutil.copy2(os.path.join(ROOT, name), DIST)
    os.makedirs(RELEASE, exist_ok=True)
    zip_path = shutil.make_archive(ZIP_BASE, "zip", os.path.dirname(DIST), "PitchView")
    print("done:", os.path.join(DIST, "PitchView.exe"))
    print("zip: ", zip_path, f"({os.path.getsize(zip_path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
