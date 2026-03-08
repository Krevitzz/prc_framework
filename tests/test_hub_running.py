"""
tests/test_hub_running.py

Validation hub_running : orchestration batch JAX end-to-end.
Exécution : python -m tests.test_hub_running  (depuis prc_framework/)

Cinq validations :
  T1 — run_batch_jax minimal  → parquet créé, n_success=1
  T2 — colonnes parquet       → FEATURE_NAMES + colonnes composition
  T3 — multi runs             → scale=[0.5,1.0,2.0] → 3 lignes parquet
  T4 — flush_every            → flush_every=2, 3 runs → 3 lignes complètes
  T5 — NaN no crash           → gamma NaN → ligne parquet présente avec NaN
"""

import sys
import tempfile
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper affichage
# ---------------------------------------------------------------------------

def _fmt(label, ok, details):
    status = "✓" if ok else "✗"
    return f"  {status}  {label:<35} {details}"


# ---------------------------------------------------------------------------
# Helper YAML temporaire
# ---------------------------------------------------------------------------

def _write_yaml(tmp_dir, config: dict) -> Path:
    path = Path(tmp_dir) / 'run.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path


def _base_config(extra_axes=None, **kwargs):
    config = {
        'phase'         : 'test_v7',
        'n_dof'         : 10,
        'max_iterations': 50,
        'axes': {
            'gamma'   : [{'id': 'GAM-011'}],
            'encoding': [{'id': 'SYM-001'}],
            'modifier': [{'id': 'M0'}],
        },
    }
    config.update(kwargs)
    if extra_axes:
        config['axes'].update(extra_axes)
    return config


# ---------------------------------------------------------------------------
# T1 — run_batch_jax minimal
# ---------------------------------------------------------------------------

def t1_run_batch_minimal():
    try:
        import pandas as pd
        from running.hub_running import run_batch_jax

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path  = _write_yaml(tmp, _base_config())
            output_dir = Path(tmp) / 'results'

            result = run_batch_jax(
                yaml_path    = yaml_path,
                output_dir   = output_dir,
                auto_confirm = True,
                chunk_size   = 256,
                flush_every  = 1000,
                verbose      = False,
            )

            assert result['n_success'] == 1, (
                f"attendu n_success=1, reçu {result['n_success']}"
            )
            assert result['n_skipped'] == 0
            assert result['parquet'] is not None
            assert Path(result['parquet']).exists()

            df = pd.read_parquet(result['parquet'])
            assert len(df) == 1, f"attendu 1 ligne, reçu {len(df)}"

        return True, f"n_success=1 ✓ | parquet créé ✓ | 1 ligne ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T2 — colonnes parquet
# ---------------------------------------------------------------------------

def t2_parquet_colonnes():
    try:
        import pandas as pd
        from running.hub_running import run_batch_jax
        from featuring.hub_featuring import FEATURE_NAMES

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path  = _write_yaml(tmp, _base_config())
            output_dir = Path(tmp) / 'results'

            result = run_batch_jax(
                yaml_path=yaml_path, output_dir=output_dir,
                auto_confirm=True, verbose=False,
            )

            df = pd.read_parquet(result['parquet'])

            comp_cols = [
                'phase', 'gamma_id', 'encoding_id', 'modifier_id',
                'n_dof', 'rank_eff', 'max_it',
                'gamma_params', 'encoding_params', 'modifier_params',
                'seed_CI', 'seed_run',
            ]
            for col in comp_cols:
                assert col in df.columns, f"colonne manquante: {col}"

            for feat in FEATURE_NAMES:
                assert feat in df.columns, f"feature manquante: {feat}"

        return True, (
            f"{len(df.columns)} colonnes ✓ | "
            f"composition ✓ | features ✓"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T3 — multi runs (3 scales)
# ---------------------------------------------------------------------------

def t3_multi_runs():
    try:
        import pandas as pd
        from running.hub_running import run_batch_jax

        config = _base_config(
            extra_axes={
                'gamma': [{
                    'id'    : 'GAM-011',
                    'params': {'scale': [0.5, 1.0, 2.0]},
                }]
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path  = _write_yaml(tmp, config)
            output_dir = Path(tmp) / 'results'

            result = run_batch_jax(
                yaml_path=yaml_path, output_dir=output_dir,
                auto_confirm=True, verbose=False,
            )

            assert result['n_success'] == 3, (
                f"attendu 3, reçu {result['n_success']}"
            )

            df = pd.read_parquet(result['parquet'])
            assert len(df) == 3, f"attendu 3 lignes, reçu {len(df)}"

        return True, f"n_success=3 ✓ | 3 lignes parquet ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T4 — flush_every
# ---------------------------------------------------------------------------

def t4_flush_every():
    try:
        import pandas as pd
        from running.hub_running import run_batch_jax

        config = _base_config(
            extra_axes={
                'gamma': [{
                    'id'    : 'GAM-011',
                    'params': {'scale': [0.5, 1.0, 2.0]},
                }]
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path  = _write_yaml(tmp, config)
            output_dir = Path(tmp) / 'results'

            # flush_every=2 avec 3 runs → 2 flushes (2 puis 1)
            result = run_batch_jax(
                yaml_path   = yaml_path,
                output_dir  = output_dir,
                auto_confirm= True,
                flush_every = 2,
                verbose     = False,
            )

            assert result['n_success'] == 3
            df = pd.read_parquet(result['parquet'])
            assert len(df) == 3, f"attendu 3 lignes, reçu {len(df)}"

        return True, (
            f"flush_every=2 | 3 runs → 2 flushes | "
            f"parquet complet 3 lignes ✓"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T5 — NaN no crash
# ---------------------------------------------------------------------------

def t5_nan_no_crash():
    try:
        import math
        import pandas as pd
        import jax
        import jax.numpy as jnp
        from running.hub_running import run_batch_jax

        # Injecter un gamma NaN dans le registry temporairement
        # via un fichier gamma temporaire dans atomics/operators/
        # → trop intrusif pour un test. Alternative : tester via
        # hub_running avec scale=2.5 × max_it=200 → explosion → inf

        config = _base_config(
            max_iterations=200,
            extra_axes={
                'gamma': [{
                    'id'    : 'GAM-011',
                    'params': {'scale': [2.5]},  # explosion → inf
                }]
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path  = _write_yaml(tmp, config)
            output_dir = Path(tmp) / 'results'

            result = run_batch_jax(
                yaml_path   = yaml_path,
                output_dir  = output_dir,
                auto_confirm= True,
                verbose     = False,
            )

            # Pas de crash — parquet présent
            assert result['parquet'] is not None
            df = pd.read_parquet(result['parquet'])
            assert len(df) == 1

            # frob_norm_final = inf (pas NaN, mais valeur aberrante tolérée)
            final = df['frob_norm_final'].iloc[0]
            assert math.isinf(final) or math.isnan(final), (
                f"attendu inf ou NaN pour scale=2.5×200it, reçu {final}"
            )

        return True, (
            f"scale=2.5 × 200it → final={final:.2e} | "
            f"pas de crash ✓ | ligne parquet présente ✓"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 70)
    print("  PRC v7 — test_hub_running : orchestration batch end-to-end")
    print("=" * 70)
    print()

    validations = [
        ("T1  run_batch minimal        ", t1_run_batch_minimal),
        ("T2  colonnes parquet         ", t2_parquet_colonnes),
        ("T3  multi runs               ", t3_multi_runs),
        ("T4  flush_every              ", t4_flush_every),
        ("T5  NaN / inf no crash       ", t5_nan_no_crash),
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
        print(f"  Résultat : {n_ok}/{len(results)} ✓ — hub_running opérationnel")
    else:
        print(f"  Résultat : {n_ok}/{len(results)} ✓  {n_fail} ✗ — corriger avant de continuer")
    print("=" * 70)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
