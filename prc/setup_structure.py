"""
setup_structure.py

Génération arborescence prc/ — Étape 0 refactor.

Usage:
    python setup_structure.py
    python setup_structure.py --dry-run
    python setup_structure.py --core-src ../prc_framework/core

Crée:
    prc/                    Nouveau pipeline (vide, prêt à implémenter)
    prc/DIVERGENCES.md      Écarts documentés au Charter 7.0
    requirements.txt        Dépendances POC
"""

import argparse
import shutil
from pathlib import Path


# =============================================================================
# CONSTANTES
# =============================================================================

ROOT = Path("prc")


# =============================================================================
# CONTENU PLACEHOLDERS
# =============================================================================

def placeholder_py(module_name: str, responsibility: str, step: str) -> str:
    """
    Génère contenu placeholder pour fichier .py.

    Args:
        module_name     : ex "prc.featuring.hub_lite"
        responsibility  : ex "Orchestration extraction features"
        step            : ex "Étape 2"

    Returns:
        str : Contenu fichier avec docstring uniquement
    """
    return f'''"""
{module_name}

Responsabilité : {responsibility}

À implémenter : {step}
"""
'''


def placeholder_test(test_name: str, module_tested: str) -> str:
    """
    Génère contenu placeholder pour fichier test.

    Args:
        test_name    : ex "test_featuring_lite"
        module_tested: ex "prc.featuring"

    Returns:
        str : Fichier test avec test_placeholder() → assert True
    """
    return f'''"""
{test_name}.py

Tests : {module_tested}

À implémenter progressivement selon étapes charter.
"""


def test_placeholder():
    """Placeholder — passe immédiatement. Remplacer par vrais tests."""
    assert True
'''


def placeholder_yaml(description: str, step: str) -> str:
    """
    Génère contenu placeholder pour fichier .yaml.

    Args:
        description : ex "Config POC minimale"
        step        : ex "Étape 4"

    Returns:
        str : YAML commenté vide
    """
    return f"""# {description}
# À compléter : {step}
# Voir Charter 7.0 Section 4.4 (YAML partout)
"""


# =============================================================================
# CONTENU FICHIERS SPÉCIAUX
# =============================================================================

def create_divergences_md() -> str:
    """
    Retourne le contenu de DIVERGENCES.md.

    Returns:
        str : Contenu markdown avec D1/D2/D3 formalisées
    """
    return """# DIVERGENCES — Charter 7.0 vs Nouveau Pipeline

> Tout écart intentionnel au Charter 7.0 doit être documenté ici
> **AVANT** d'être implémenté.
>
> Date création : 2026-02-17
> Version charter de référence : 7.0

---

## Règle de gestion

1. Identifier l'écart avec le charter
2. Le documenter dans ce fichier (ID, description, justification, impact)
3. Faire valider par l'utilisateur
4. Seulement ensuite implémenter

Aucun écart silencieux. Aucun "quick fix" non documenté.

---

## Divergences actives

### D1 — Format stockage résultats

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 4.6 — DB unique SQLite `db_results.db` |
| **Nouveau pipeline** | Fichiers Parquet dans `prc/data/results/` (1 fichier par phase) |
| **Justification** | Volumétrie -71% (339 MB vs 1 171 MB), RAM verdict ×22 réduit (89 MB vs 2 GB), I/O 2-5× plus rapide, pas de migrations schema |
| **Impact** | `utils/database.py` utilise pandas/pyarrow au lieu de sqlite3. Queries via pandas filtering au lieu de SQL. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

### D2 — Organisation orchestration batch

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 2.3 — `batch_runner.py` et `verdict.py` à la racine |
| **Nouveau pipeline** | Module `prc/running/` contenant hub, compositions, discovery, runner, verdict |
| **Justification** | Séparation des responsabilités, testabilité unitaire, extensibilité (nouveaux axes sans modifier point d'entrée) |
| **Impact** | `batch.py` racine devient façade légère appelant `running/hub.py`. Charter Section 2.1 organigramme flux reste valide conceptuellement. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

### D3 — Format colonnes features

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 4.6 — `features TEXT NOT NULL` (JSON colonne unique) |
| **Nouveau pipeline** | Une colonne Parquet par feature (ex: `frobenius_norm_final`, `mean_value_final`, ...) |
| **Justification** | Charge partielle native (lire 3 colonnes sur 150 sans désérialiser JSON), RAM critique pour verdict, cohérent avec D1 (Parquet) |
| **Impact** | Pas de migration schema au sens SQL — ajout feature = nouvelle colonne Parquet. Profiling/Analysing lisent colonnes directement via pandas. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

## Divergences archivées

_(aucune pour l'instant)_
"""


def create_requirements_txt() -> str:
    """
    Retourne le contenu de requirements.txt.

    Returns:
        str : Dépendances POC une par ligne
    """
    return """# requirements.txt — PRC Pipeline (POC minimal)
# Installation : pip install -r requirements.txt

numpy>=1.24
scipy>=1.10
pandas>=2.0
pyarrow>=12.0
scikit-learn>=1.3
pyyaml>=6.0
pytest>=7.0
"""


# =============================================================================
# DÉFINITION ARBORESCENCE
# =============================================================================

def get_structure() -> dict:
    """
    Retourne la définition complète de l'arborescence.

    Returns:
        dict avec deux clés :
            'dirs'  : List[str]  — dossiers à créer
            'files' : List[tuple(str, str)] — (chemin relatif à ROOT, contenu)
    """
    dirs = [
        "core",
        "atomics/operators",
        "atomics/D_encodings",
        "atomics/modifiers",
        "featuring/registries",
        "running",
        "profiling",
        "analysing",
        "utils",
        "configs/phases/poc",
        "configs/features/minimal",
        "data/results",
        "tests",
    ]

    files = [
        # --- featuring ---
        (
            "featuring/hub_lite.py",
            placeholder_py(
                "prc.featuring.hub_lite",
                "Orchestration extraction features minimales (3-5 features scalaires)",
                "Étape 2",
            ),
        ),
        (
            "featuring/extractor_lite.py",
            placeholder_py(
                "prc.featuring.extractor_lite",
                "Extraction features scalaires depuis history np.ndarray",
                "Étape 2",
            ),
        ),
        (
            "featuring/layers_lite.py",
            placeholder_py(
                "prc.featuring.layers_lite",
                "Inspection shape/rank history — routing layer universal",
                "Étape 2",
            ),
        ),
        (
            "featuring/registries/universal_lite.py",
            placeholder_py(
                "prc.featuring.registries.universal_lite",
                "Fonctions extraction universelles (tout tenseur) — 3-5 features",
                "Étape 2",
            ),
        ),
        # --- running ---
        (
            "running/hub.py",
            placeholder_py(
                "prc.running.hub",
                "Façade orchestration : discovery → compositions → runs → Parquet",
                "Étapes 3-4",
            ),
        ),
        (
            "running/compositions.py",
            placeholder_py(
                "prc.running.compositions",
                "Génération compositions (produit cartésien axes YAML)",
                "Étape 3",
            ),
        ),
        (
            "running/discovery.py",
            placeholder_py(
                "prc.running.discovery",
                "Scan atomics/ → résolution IDs en callables (gamma, encoding, modifier)",
                "Étape 3",
            ),
        ),
        (
            "running/runner.py",
            placeholder_py(
                "prc.running.runner",
                "Exécution run unique : prepare_state → run_kernel → history np.ndarray",
                "Étape 3",
            ),
        ),
        (
            "running/verdict.py",
            placeholder_py(
                "prc.running.verdict",
                "Analyses post-batch : profiling + analysing → rapport",
                "Étape 5",
            ),
        ),
        # --- profiling ---
        (
            "profiling/hub_lite.py",
            placeholder_py(
                "prc.profiling.hub_lite",
                "Orchestration aggregation cross-runs (inter-run)",
                "Étape 5",
            ),
        ),
        (
            "profiling/aggregation_lite.py",
            placeholder_py(
                "prc.profiling.aggregation_lite",
                "Aggregation basique : median, IQR par gamma/encoding",
                "Étape 5",
            ),
        ),
        # --- analysing ---
        (
            "analysing/hub_lite.py",
            placeholder_py(
                "prc.analysing.hub_lite",
                "Orchestration patterns ML (inter-run)",
                "Étape 5",
            ),
        ),
        (
            "analysing/clustering_lite.py",
            placeholder_py(
                "prc.analysing.clustering_lite",
                "Clustering simple (KMeans) sur features cross-runs",
                "Étape 5",
            ),
        ),
        # --- utils ---
        (
            "utils/database.py",
            placeholder_py(
                "prc.utils.database",
                "Helpers lecture/écriture Parquet (divergence D1/D3 vs Charter 7.0)",
                "Étape 3",
            ),
        ),
        # --- racine prc/ ---
        (
            "batch.py",
            placeholder_py(
                "prc.batch",
                "Façade racine — point d'entrée utilisateur → running/hub.py (divergence D2)",
                "Étape 4",
            ),
        ),
        # --- configs ---
        (
            "configs/phases/poc/poc.yaml",
            placeholder_yaml("Config phases POC minimale", "Étape 4"),
        ),
        (
            "configs/features/minimal/universal.yaml",
            placeholder_yaml("Params features universelles minimales (3-5 features)", "Étape 2"),
        ),
        # --- tests ---
        (
            "tests/test_core.py",
            placeholder_test("test_core", "prc.core (kernel + state_preparation)"),
        ),
        (
            "tests/test_featuring_lite.py",
            placeholder_test("test_featuring_lite", "prc.featuring (hub_lite, extractor_lite)"),
        ),
        (
            "tests/test_running_lite.py",
            placeholder_test("test_running_lite", "prc.running (discovery, compositions, runner)"),
        ),
        (
            "tests/test_batch_lite.py",
            placeholder_test("test_batch_lite", "prc.batch (intégration bout-en-bout POC)"),
        ),
        # --- gitkeep ---
        ("data/results/.gitkeep", ""),
    ]

    return {"dirs": dirs, "files": files}


# =============================================================================
# CRÉATION
# =============================================================================

def create_dirs(dirs: list, dry_run: bool = False) -> None:
    """
    Crée les dossiers.

    Args:
        dirs     : Liste chemins relatifs à ROOT (strings)
        dry_run  : Si True, affiche sans créer
    """
    for d in dirs:
        path = ROOT / d
        if dry_run:
            print(f"  [DRY] mkdir {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)


def create_files(files: list, dry_run: bool = False) -> None:
    """
    Crée les fichiers avec leur contenu placeholder.

    Args:
        files    : Liste tuples (chemin_relatif_à_ROOT, contenu_str)
        dry_run  : Si True, affiche sans créer
    """
    for rel_path, content in files:
        path = ROOT / rel_path
        if dry_run:
            print(f"  [DRY] write {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def copy_core(core_src: str, dry_run: bool = False) -> None:
    """
    Copie kernel.py et state_preparation.py depuis prc_framework.

    Args:
        core_src : Chemin dossier core/ source (ex: ../prc_framework/core)
        dry_run  : Si True, affiche sans copier
    """
    src = Path(core_src)
    dst = ROOT / "core"

    files_to_copy = ["kernel.py", "state_preparation.py", "core_catalog.md"]

    for filename in files_to_copy:
        src_file = src / filename
        dst_file = dst / filename

        if not src_file.exists():
            print(f"  [WARN] Source introuvable : {src_file} — skipped")
            continue

        if dry_run:
            print(f"  [DRY] copy {src_file} → {dst_file}")
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            print(f"  [OK]  copy {filename}")


def create_root_files(dry_run: bool = False) -> None:
    """
    Crée DIVERGENCES.md et requirements.txt à la racine (hors prc/).

    Args:
        dry_run : Si True, affiche sans créer
    """
    root_files = [
        (Path("prc/DIVERGENCES.md"), create_divergences_md()),
        (Path("requirements.txt"), create_requirements_txt()),
    ]

    for path, content in root_files:
        if dry_run:
            print(f"  [DRY] write {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


# =============================================================================
# RÉCAPITULATIF
# =============================================================================

def print_summary(dirs: list, files: list, dry_run: bool) -> None:
    """
    Affiche récapitulatif après création.

    Args:
        dirs    : Dossiers définis
        files   : Fichiers définis
        dry_run : Indique si simulation
    """
    mode = "DRY-RUN" if dry_run else "CRÉÉ"
    print(f"\n{'='*55}")
    print(f"  PRC — Arborescence {mode}")
    print(f"{'='*55}")
    print(f"  Racine         : prc/")
    print(f"  Dossiers       : {len(dirs)}")
    print(f"  Fichiers       : {len(files)} (+ DIVERGENCES.md + requirements.txt)")
    print(f"  Core           : copié depuis --core-src (si fourni)")
    print(f"{'='*55}")
    print(f"\n  Prochaine étape : pytest prc/tests/  →  4 tests doivent passer")
    print(f"  Puis            : Étape 1 — audit core\n")


# =============================================================================
# MAIN
# =============================================================================

def main(dry_run: bool = False, core_src: str = None) -> None:
    """
    Point d'entrée principal.

    Args:
        dry_run  : Si True, simulation sans écriture disque
        core_src : Chemin dossier core/ legacy (optionnel)
    """
    structure = get_structure()

    print(f"\n  Génération arborescence prc/ ({'dry-run' if dry_run else 'réel'})...\n")

    # Dossiers
    print("  [1/4] Dossiers...")
    create_dirs(structure["dirs"], dry_run=dry_run)

    # Fichiers placeholders
    print("  [2/4] Fichiers placeholders...")
    create_files(structure["files"], dry_run=dry_run)

    # Fichiers racine (DIVERGENCES.md, requirements.txt)
    print("  [3/4] Fichiers racine...")
    create_root_files(dry_run=dry_run)

    # Core (copie si source fournie)
    print("  [4/4] Core...")
    if core_src:
        copy_core(core_src, dry_run=dry_run)
    else:
        print("  [SKIP] --core-src non fourni — core/ restera vide")
        print("         Fournir plus tard : python setup_structure.py --core-src <chemin>")

    print_summary(structure["dirs"], structure["files"], dry_run)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Génère arborescence prc/ — Étape 0 refactor PRC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python setup_structure.py
  python setup_structure.py --dry-run
  python setup_structure.py --core-src ../prc_framework/core
  python setup_structure.py --core-src ../prc_framework/core --dry-run
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche ce qui serait créé sans toucher le disque",
    )
    parser.add_argument(
        "--core-src",
        type=str,
        default=None,
        help="Chemin dossier core/ legacy (ex: ../prc_framework/core)",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, core_src=args.core_src)
