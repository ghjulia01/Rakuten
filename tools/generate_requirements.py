#!/usr/bin/env python3
"""
Scan du projet pour générer un requirements.txt minimal :
- Parcourir .py et .ipynb
- Extraire les imports réels
- Mapper les modules vers les paquets PyPI
- Lire la version installée si dispo
- Écrire requirements.txt trié

Usage:
  python tools/generate_requirements.py
  pip install -r requirements.txt
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Iterable, Set, Dict

try:
    from importlib import metadata as importlib_metadata  # Py≥3.8
except Exception:
    import importlib_metadata  # type: ignore

# --- Modules stdlib les plus courants (heuristique) ---
STDLIB_HINT = {
    "os","sys","re","json","math","pathlib","itertools","collections","functools",
    "datetime","time","typing","argparse","string","subprocess","shutil","tempfile",
    "logging","random","statistics","textwrap","dataclasses","enum","fractions",
    "hashlib","heapq","inspect","io","ipaddress","pprint","queue","signal","site",
    "sqlite3","tarfile","threading","tkinter","traceback","types","uuid","zipfile",
    "importlib","ast","glob","contextlib","copy","weakref","tokenize","warnings",
    "unicodedata","pickle","config","__future__"
}

# --- Mapping module → package PyPI (compléter au besoin) ---
MODULE_TO_PYPI = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow", "Pillow": "pillow",
    "tomli": "tomli", "tomllib": "tomli",  # Py<3.11
    "imblearn": "imbalanced-learn",
    "nltk": "nltk",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "toml": "toml",
    "seaborn": "seaborn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "click": "click",
    "colorama": "colorama",
    "pytest": "pytest",
    "pygments": "Pygments",
    # --- Ajouts pour la branche CNN ---
    "torch": "torch",
    "torchvision": "torchvision",
}
# Ajouts pour la branche 
MODULE_TO_PYPI.update({
    "shap": "shap",
    "lime": "lime",
    "umap": "umap-learn",
    "plotly": "plotly",
})
# --- Regex pour extraire les imports ---
RE_IMPORT = re.compile(
    r'^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))',
    flags=re.MULTILINE
)

def is_stdlib_module(name: str) -> bool:
    base = name.split(".")[0]
    return base in STDLIB_HINT or base.startswith("_")

def normalize_to_pypi(modname: str) -> str | None:
    base = modname.split(".")[0]
    if base in MODULE_TO_PYPI:
        return MODULE_TO_PYPI[base]
    if not is_stdlib_module(base):
        return base
    return None

def extract_imports_from_py(path: Path) -> Set[str]:
    try:
        code = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()
    return { (m.group(1) or m.group(2)).strip()
             for m in RE_IMPORT.finditer(code) if (m.group(1) or m.group(2)) }

def extract_imports_from_ipynb(path: Path) -> Set[str]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return set()
    mods: Set[str] = set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for m in RE_IMPORT.finditer(src):
            g = m.group(1) or m.group(2)
            if g:
                mods.add(g.strip())
    return mods

def walk_imports(root: Path) -> Set[str]:
    imports: Set[str] = set()
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix == ".py":
            imports |= extract_imports_from_py(p)
        elif p.suffix == ".ipynb":
            imports |= extract_imports_from_ipynb(p)
    return imports

def modules_to_packages(mods: Iterable[str]) -> Dict[str, str]:
    pkgs: Dict[str, str] = {}
    for mod in mods:
        pkg = normalize_to_pypi(mod)
        if pkg:
            pkgs[pkg] = mod
    return pkgs

def get_installed_version(pkg: str) -> str | None:
    try:
        return importlib_metadata.version(pkg)
    except Exception:
        try:
            return importlib_metadata.version(pkg.replace("_", "-"))
        except Exception:
            return None

DEFAULT_PIN = {
    # Exemple: "tomli": "2.0.1",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Racine du projet à scanner")
    ap.add_argument("--out", default="requirements.txt", help="Chemin de sortie")
    ap.add_argument("--no_version", action="store_true", help="N’écrire que les noms (sans versions)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pkgs = modules_to_packages(walk_imports(root))

    # Ne pas considérer comme paquets PyPI les modules internes du projet
    # (mais on garde leurs FICHIERS pour détecter les dépendances tierces)
    internal_roots = {"features", "models", "main", "utils", "tools"}
    filtered = {}
    for pkg, mod in pkgs.items():
        base = mod.split(".")[0]
        if base in internal_roots:
            continue
        filtered[pkg] = mod
    pkgs = filtered

    lines = []
    for pkg in sorted(pkgs.keys()):
        if args.no_version:
            lines.append(pkg)
        else:
            ver = get_installed_version(pkg)
            if ver:
                lines.append(f"{pkg}=={ver}")
            else:
                lines.append(f"{pkg}=={DEFAULT_PIN[pkg]}" if pkg in DEFAULT_PIN else pkg)

    out_path = Path(args.out).resolve()
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Requirements écrit : {out_path}")
    print("Paquets détectés :", ", ".join(sorted(pkgs.keys())))

if __name__ == "__main__":
    main()