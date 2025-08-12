#!/usr/bin/env python3
"""
Scan du projet pour générer un requirements.txt minimal :
- Parcourt .py et .ipynb
- Extrait les imports réels
- Mappe les modules vers les paquets PyPI (avec règles connues)
- Lit la version installée (importlib.metadata) si dispo
- Écrit requirements.txt trié

Usage:
  python tools/generate_requirements.py            # génère requirements.txt à la racine
  pip install -r requirements.txt          # pour installer les paquets
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Set, Dict

try:
    # Py≥3.8
    from importlib import metadata as importlib_metadata
except Exception:
    import importlib_metadata  # type: ignore

# === 1) Modules stdlib à ignorer (approx: on filtre via une liste + heuristique) ===
# NB: ce set n'est pas exhaustif, mais couvre l'essentiel pour éviter de lister du stdlib
STDLIB_HINT = {
    "os","sys","re","json","math","pathlib","itertools","collections","functools",
    "datetime","time","typing","argparse","string","subprocess","shutil","tempfile",
    "logging","random","statistics","textwrap","dataclasses","enum","fractions",
    "hashlib","heapq","inspect","io","ipaddress","pprint","queue","signal","site",
    "sqlite3","tarfile","threading","tkinter","traceback","types","uuid","zipfile",
    "importlib","ast","glob","contextlib","copy","weakref","tokenize"
}

# === 2) Mapping module -> package PyPI ===
# (ajoute ici si ton projet utilise d’autres libs)
MODULE_TO_PYPI = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "Pillow": "pillow",
    "tomli": "tomli",
    "tomllib": "tomli",           # pour Py<3.11
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
}

# === 3) Regex d’import ===
RE_IMPORT = re.compile(
    r'^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))',
    flags=re.MULTILINE
)

def is_stdlib_module(name: str) -> bool:
    base = name.split(".")[0]
    if base in STDLIB_HINT:
        return True
    # heuristique: si importlib_metadata trouve rien ET pas dans mapping, on décidera plus tard
    return False

def normalize_to_pypi(modname: str) -> str | None:
    base = modname.split(".")[0]
    # mapping explicite
    if base in MODULE_TO_PYPI:
        return MODULE_TO_PYPI[base]
    # heuristique: si pas stdlib, on suppose que le nom du package == module racine
    if not is_stdlib_module(base):
        return base
    return None

def extract_imports_from_py(path: Path) -> Set[str]:
    try:
        code = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()
    mods = set()
    for m in RE_IMPORT.finditer(code):
        g = m.group(1) or m.group(2)
        if g:
            mods.add(g.strip())
    return mods

def extract_imports_from_ipynb(path: Path) -> Set[str]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return set()
    mods = set()
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
            pkgs[pkg] = mod  # conserve un lien (debug)
    return pkgs

def get_installed_version(pkg: str) -> str | None:
    try:
        return importlib_metadata.version(pkg)
    except Exception:
        # parfois nom du dist diffère : ex scikit-learn distribue "scikit-learn"
        # si échec: tenter sur l’alternative sans tirets (rarement utile)
        try:
            return importlib_metadata.version(pkg.replace("_", "-"))
        except Exception:
            return None

DEFAULT_PIN = {
    # si version introuvable, on peut fixer une version "safe" (optionnel)
    # "tomli": "2.0.1",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Racine du projet à scanner")
    ap.add_argument("--out", default="requirements.txt", help="Chemin de sortie")
    ap.add_argument("--no-version", action="store_true", help="N’écrit pas les versions (juste les noms)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pkgs = modules_to_packages(walk_imports(root))

    # filtre quelques faux positifs potentiels (ex: modules internes de ton projet)
    internal_roots = {"features", "models", "main", "utils"}
    filtered = {}
    for pkg, mod in pkgs.items():
        base = mod.split(".")[0]
        if base in internal_roots:
            continue
        filtered[pkg] = mod
    pkgs = filtered

    lines = []
    for pkg in sorted(pkgs.keys()):
        if args.no-version:
            lines.append(pkg)
        else:
            ver = get_installed_version(pkg)
            if ver:
                lines.append(f"{pkg}=={ver}")
            else:
                # fallback : version non trouvée -> nom seul ou pin par défaut
                if pkg in DEFAULT_PIN:
                    lines.append(f"{pkg}=={DEFAULT_PIN[pkg]}")
                else:
                    lines.append(pkg)

    out_path = Path(args.out).resolve()
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] requirements écrit : {out_path}")
    print("[i] Paquets détectés :", ", ".join(sorted(pkgs.keys())))

if __name__ == "__main__":
    main()