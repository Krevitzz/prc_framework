"""
tests/test_data_loading_jax.py

Validation data_loading_jax : discovery atomics + I/O parquet.
Exécution : python -m tests.test_data_loading_jax  (depuis prc_framework/)

Cinq validations :
  T1 — discover_gammas_jax   : GAM-011 présent, callable valide
  T2 — discover_encodings_jax : SYM-001 présent, callable valide
  T3 — discover_modifiers_jax : M0 présent, callable valide
  T4 — load_yaml             : charge YAML existant, erreur si absent
  T5 — write_parquet         : écrit + relit, colonnes correctes
"""

import sys
import warnings
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper affichage
# ---------------------------------------------------------------------------

def _fmt(label, ok, details):
    status = "✓" if ok else "✗"
    return f"  {status}  {label:<35} {details}"


# ---------------------------------------------------------------------------
# T1 — discover_gammas_jax
# ---------------------------------------------------------------------------

def t1_discover_gammas():
    try:
        from utils.data_loading_jax import discover_gammas_jax

        registry = discover_gammas_jax()

        assert len(registry) > 0, "registre gammas vide"
        assert 'GAM-011' in registry, f"GAM-011 absent — trouvés: {list(registry.keys())}"

        entry = registry['GAM-011']
        assert callable(entry['callable']), "callable non callable"
        assert 'id' in entry['metadata'], "metadata sans id"
        assert entry['metadata']['id'] == 'GAM-011'

        return True, (
            f"{len(registry)} gamma(s) découvert(s) | "
            f"GAM-011 ✓ | "
            f"callable: {entry['callable'].__name__}"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T2 — discover_encodings_jax
# ---------------------------------------------------------------------------

def t2_discover_encodings():
    try:
        from utils.data_loading_jax import discover_encodings_jax

        registry = discover_encodings_jax()

        assert len(registry) > 0, "registre encodings vide"
        assert 'SYM-001' in registry, f"SYM-001 absent — trouvés: {list(registry.keys())}"

        entry = registry['SYM-001']
        assert callable(entry['callable']), "callable non callable"
        assert entry['metadata']['id'] == 'SYM-001'

        return True, (
            f"{len(registry)} encoding(s) découvert(s) | "
            f"SYM-001 ✓ | "
            f"callable: {entry['callable'].__name__}"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T3 — discover_modifiers_jax
# ---------------------------------------------------------------------------

def t3_discover_modifiers():
    try:
        from utils.data_loading_jax import discover_modifiers_jax

        registry = discover_modifiers_jax()

        assert len(registry) > 0, "registre modifiers vide"
        assert 'M0' in registry, f"M0 absent — trouvés: {list(registry.keys())}"

        entry = registry['M0']
        assert callable(entry['callable']), "callable non callable"
        assert entry['metadata']['id'] == 'M0'

        return True, (
            f"{len(registry)} modifier(s) découvert(s) | "
            f"M0 ✓ | "
            f"callable: {entry['callable'].__name__}"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T4 — load_yaml
# ---------------------------------------------------------------------------

def t4_load_yaml():
    try:
        from utils.data_loading_jax import load_yaml

        # Créer un YAML temporaire pour le test
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, encoding='utf-8'
        ) as f:
            f.write("phase: test_v7\nn_dof: [10, 50]\nmax_it: 100\n")
            tmp_path = Path(f.name)

        try:
            config = load_yaml(tmp_path)
            assert isinstance(config, dict), "résultat non dict"
            assert config.get('phase') == 'test_v7', f"phase: {config.get('phase')}"
            assert config.get('n_dof') == [10, 50], f"n_dof: {config.get('n_dof')}"
        finally:
            tmp_path.unlink(missing_ok=True)

        # FileNotFoundError si path inexistant
        raised = False
        try:
            load_yaml(Path('nonexistent_xyz.yaml'))
        except FileNotFoundError:
            raised = True
        assert raised, "FileNotFoundError non levée sur path inexistant"

        return True, "load ✓ | FileNotFoundError sur path inexistant ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T5 — write_parquet
# ---------------------------------------------------------------------------

def t5_write_parquet():
    try:
        import pandas as pd
        import math
        from utils.data_loading_jax import write_parquet
        from featuring.hub_featuring import FEATURE_NAMES

        # Construire 2 rows minimales
        rows = [
            {
                'composition': {
                    'gamma_id'       : 'GAM-011',
                    'encoding_id'    : 'SYM-001',
                    'modifier_id'    : 'M0',
                    'n_dof'          : 10,
                    'rank_eff'       : 2,
                    'max_it'         : 50,
                    'gamma_params'   : {'scale': 1.0},
                    'encoding_params': {},
                    'modifier_params': {},
                    'seed_CI'        : 0,
                    'seed_run'       : None,
                },
                'features': {name: float(i * 0.1 + 1.0) for i, name in enumerate(FEATURE_NAMES)},
            },
            {
                'composition': {
                    'gamma_id'       : 'GAM-011',
                    'encoding_id'    : 'SYM-001',
                    'modifier_id'    : 'M0',
                    'n_dof'          : 10,
                    'rank_eff'       : 2,
                    'max_it'         : 50,
                    'gamma_params'   : {'scale': 0.5},
                    'encoding_params': {},
                    'modifier_params': {},
                    'seed_CI'        : 1,
                    'seed_run'       : None,
                },
                'features': {name: float(i * 0.2 + 2.0) for i, name in enumerate(FEATURE_NAMES)},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = write_parquet(rows, 'test_v7', Path(tmp_dir))

            assert filepath.exists(), f"fichier non créé : {filepath}"

            df = pd.read_parquet(filepath)

            # Colonnes composition
            comp_cols = [
                'phase', 'gamma_id', 'encoding_id', 'modifier_id',
                'n_dof', 'rank_eff', 'max_it',
                'gamma_params', 'encoding_params', 'modifier_params',
                'seed_CI', 'seed_run',
            ]
            for col in comp_cols:
                assert col in df.columns, f"colonne manquante: {col}"

            # Colonnes features
            for name in FEATURE_NAMES:
                assert name in df.columns, f"feature manquante: {name}"

            # Dimensions
            assert len(df) == 2, f"attendu 2 lignes, reçu {len(df)}"

            n_cols = len(df.columns)

        return True, (
            f"2 lignes ✓ | "
            f"{n_cols} colonnes ✓ | "
            f"comp + features présentes ✓"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 70)
    print("  PRC v7 — test_data_loading_jax : discovery + parquet")
    print("=" * 70)
    print()

    validations = [
        ("T1  discover_gammas_jax     ", t1_discover_gammas),
        ("T2  discover_encodings_jax  ", t2_discover_encodings),
        ("T3  discover_modifiers_jax  ", t3_discover_modifiers),
        ("T4  load_yaml               ", t4_load_yaml),
        ("T5  write_parquet           ", t5_write_parquet),
    ]

    results = []
    for label, fn in validations:
        ok, details = fn()
        print(_fmt(label, ok, details))
        results.append(ok)

    print()
    print("=" * 70)
    n_ok   = sum(results)
    n_fail = len(results) - n_ok
    if n_fail == 0:
        print(f"  Résultat : {n_ok}/{len(results)} ✓ — data_loading_jax opérationnel")
    else:
        print(f"  Résultat : {n_ok}/{len(results)} ✓  {n_fail} ✗ — corriger avant de continuer")
    print("=" * 70)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
