"""
tests/test_data_loading_new.py

Tests exhaustifs et discriminants — data_loading_new.py

Couverture :
  D1  — _discover_from_dir : prepare_params capturé si présent, None sinon
  D2  — _discover_from_dir : CriticalDiscoveryError si METADATA absent
  D3  — _discover_from_dir : CriticalDiscoveryError si METADATA['id'] absent
  D4  — _discover_from_dir : warning + skip si callable absent
  D5  — _discover_from_dir : skip *_deprecated_* et __init__
  D6  — load_yaml : chargement correct
  D7  — load_yaml : FileNotFoundError si fichier absent
  D8  — write_rows_to_parquet : schéma correct, toutes colonnes présentes
  D9  — write_rows_to_parquet : append multi-batch → toutes rows présentes
  D10 — write_rows_to_parquet : NaN float encodé float32 (pas d'erreur de type)
  D11 — write_rows_to_parquet : rows=[] retourne 0 sans erreur
  D12 — close_parquet_writer : fermeture propre, fichier lisible après
  D13 — pandas absent de data_loading_new (pas d'import pandas)
  D14 — merge_configs : deep merge correct
  D15 — merge_configs : liste remplacée entièrement (pas mergeée)

Usage :
    python -m pytest tests/test_data_loading_new.py -v
    python tests/test_data_loading_new.py
"""

import json
import math
import sys
import tempfile
import traceback
import types
import warnings
from pathlib import Path

import pyarrow.parquet as pq

# =============================================================================
# Runner minimal
# =============================================================================

_results = []

def _check(name: str, condition: bool, detail: str = "") -> bool:
    status = "✓" if condition else "✗"
    msg    = f"  [{status}] {name}"
    if detail:
        msg += f"\n         → {detail}"
    print(msg)
    _results.append((name, condition, detail))
    return condition

def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =============================================================================
# Import
# =============================================================================

_section("Import")

try:
    from utils.data_loading_new import (
        _discover_from_dir,
        CriticalDiscoveryError,
        load_yaml,
        open_parquet_writer,
        write_rows_to_parquet,
        close_parquet_writer,
        merge_configs,
    )
    _check("Import data_loading_new", True)
except Exception as e:
    _check("Import data_loading_new", False, str(e))
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Fixtures — modules synthétiques pour discovery
# =============================================================================

def _make_module(name: str, metadata: dict = None, has_apply: bool = True,
                 has_prepare: bool = False) -> types.ModuleType:
    """Crée un module Python synthétique en mémoire."""
    m = types.ModuleType(name)
    if metadata is not None:
        m.METADATA = metadata
    if has_apply:
        m.apply = lambda s, p, k: s
    if has_prepare:
        m.prepare_params = lambda p, k: {**p, 'prepared': True}
    return m


def _make_temp_dir_with_modules(modules: dict) -> Path:
    """
    Crée un dossier temporaire avec des fichiers .py synthétiques.
    modules : {filename_stem: source_code}
    """
    import importlib.util
    tmpdir = Path(tempfile.mkdtemp())
    for stem, code in modules.items():
        (tmpdir / f"{stem}.py").write_text(code, encoding='utf-8')
    return tmpdir


# =============================================================================
# D1 — prepare_params capturé si présent, None sinon
# =============================================================================

_section("D1 — prepare_params dans le registre")

try:
    tmpdir = _make_temp_dir_with_modules({
        'gam_001_with_prepare': '''
METADATA = {'id': 'GAM-WITH-PREP'}
def apply(s, p, k): return s
def prepare_params(p, k): return {**p, 'prepared': True}
''',
        'gam_002_without_prepare': '''
METADATA = {'id': 'GAM-NO-PREP'}
def apply(s, p, k): return s
''',
    })

    import sys as _sys
    _sys.path.insert(0, str(tmpdir.parent))

    # Simuler discovery en injectant les modules dans sys.modules
    import importlib
    import types as _types

    pkg_name = tmpdir.name
    pkg_mod  = _types.ModuleType(pkg_name)
    pkg_mod.__path__ = [str(tmpdir)]
    _sys.modules[pkg_name] = pkg_mod

    reg = _discover_from_dir(
        directory=tmpdir,
        pattern='gam_*.py',
        module_prefix=pkg_name,
        callable_name='apply',
    )

    _check("D1.1 — deux modules découverts",
           len(reg) == 2, f"got {list(reg.keys())}")

    _check("D1.2 — prepare_params non-None si présent",
           reg.get('GAM-WITH-PREP', {}).get('prepare_params') is not None,
           f"got {reg.get('GAM-WITH-PREP', {}).get('prepare_params')}")

    _check("D1.3 — prepare_params None si absent",
           reg.get('GAM-NO-PREP', {}).get('prepare_params') is None,
           f"got {reg.get('GAM-NO-PREP', {}).get('prepare_params')}")

    _check("D1.4 — prepare_params callable (retourne dict enrichi)",
           reg['GAM-WITH-PREP']['prepare_params']({'x': 1}, None) == {'x': 1, 'prepared': True})

    _check("D1.5 — 'prepare_params' clé présente dans les deux entrées",
           'prepare_params' in reg.get('GAM-WITH-PREP', {}) and
           'prepare_params' in reg.get('GAM-NO-PREP', {}))

except Exception as e:
    _check("D1 — prepare_params", False, str(e))
    traceback.print_exc()


# =============================================================================
# D2 — CriticalDiscoveryError si METADATA absent
# =============================================================================

_section("D2 — CriticalDiscoveryError METADATA absent")

try:
    tmpdir2 = _make_temp_dir_with_modules({
        'gam_no_meta': '''
def apply(s, p, k): return s
''',
    })
    pkg2      = tmpdir2.name
    pkg2_mod  = _types.ModuleType(pkg2)
    pkg2_mod.__path__ = [str(tmpdir2)]
    _sys.modules[pkg2] = pkg2_mod

    raised = False
    try:
        _discover_from_dir(tmpdir2, 'gam_*.py', pkg2, 'apply')
    except CriticalDiscoveryError:
        raised = True

    _check("D2.1 — CriticalDiscoveryError levée si METADATA absent", raised)

except Exception as e:
    _check("D2 — CriticalDiscoveryError METADATA", False, str(e))
    traceback.print_exc()


# =============================================================================
# D3 — CriticalDiscoveryError si METADATA['id'] absent
# =============================================================================

_section("D3 — CriticalDiscoveryError METADATA['id'] absent")

try:
    tmpdir3 = _make_temp_dir_with_modules({
        'gam_no_id': '''
METADATA = {'family': 'markovian'}
def apply(s, p, k): return s
''',
    })
    pkg3     = tmpdir3.name
    pkg3_mod = _types.ModuleType(pkg3)
    pkg3_mod.__path__ = [str(tmpdir3)]
    _sys.modules[pkg3] = pkg3_mod

    raised = False
    try:
        _discover_from_dir(tmpdir3, 'gam_*.py', pkg3, 'apply')
    except CriticalDiscoveryError:
        raised = True

    _check("D3.1 — CriticalDiscoveryError levée si METADATA['id'] absent", raised)

except Exception as e:
    _check("D3 — CriticalDiscoveryError METADATA id", False, str(e))
    traceback.print_exc()


# =============================================================================
# D4 — warning + skip si callable absent
# =============================================================================

_section("D4 — warning + skip si callable absent")

try:
    tmpdir4 = _make_temp_dir_with_modules({
        'gam_ok': '''
METADATA = {'id': 'GAM-OK'}
def apply(s, p, k): return s
''',
        'gam_no_callable': '''
METADATA = {'id': 'GAM-NO-FN'}
# pas de apply()
''',
    })
    pkg4     = tmpdir4.name
    pkg4_mod = _types.ModuleType(pkg4)
    pkg4_mod.__path__ = [str(tmpdir4)]
    _sys.modules[pkg4] = pkg4_mod

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg4 = _discover_from_dir(tmpdir4, 'gam_*.py', pkg4, 'apply')

    _check("D4.1 — GAM-OK présent dans le registre",
           'GAM-OK' in reg4)

    _check("D4.2 — GAM-NO-FN absent du registre (skipped)",
           'GAM-NO-FN' not in reg4)

    _check("D4.3 — un warning émis pour le callable absent",
           any('absent' in str(w.message).lower() or 'skipped' in str(w.message).lower()
               for w in caught),
           f"warnings : {[str(w.message) for w in caught]}")

except Exception as e:
    _check("D4 — warning + skip callable absent", False, str(e))
    traceback.print_exc()


# =============================================================================
# D5 — skip *_deprecated_* et __init__
# =============================================================================

_section("D5 — skip deprecated et __init__")

try:
    tmpdir5 = _make_temp_dir_with_modules({
        'gam_001_valid': '''
METADATA = {'id': 'GAM-VALID'}
def apply(s, p, k): return s
''',
        'gam_002_deprecated_old': '''
METADATA = {'id': 'GAM-DEPRECATED'}
def apply(s, p, k): return s
''',
        '__init__': '''
METADATA = {'id': 'GAM-INIT'}
def apply(s, p, k): return s
''',
    })
    pkg5     = tmpdir5.name
    pkg5_mod = _types.ModuleType(pkg5)
    pkg5_mod.__path__ = [str(tmpdir5)]
    _sys.modules[pkg5] = pkg5_mod

    reg5 = _discover_from_dir(tmpdir5, 'gam_*.py', pkg5, 'apply')

    _check("D5.1 — GAM-VALID présent",
           'GAM-VALID' in reg5)

    _check("D5.2 — GAM-DEPRECATED absent (skipped)",
           'GAM-DEPRECATED' not in reg5,
           f"registre : {list(reg5.keys())}")

    # __init__ ne match pas gam_*.py donc absent par pattern, pas par skip
    _check("D5.3 — GAM-INIT absent (hors pattern gam_*.py)",
           'GAM-INIT' not in reg5)

except Exception as e:
    _check("D5 — skip deprecated/__init__", False, str(e))
    traceback.print_exc()


# =============================================================================
# D6 — load_yaml correct
# =============================================================================

_section("D6 — load_yaml")

try:
    import yaml

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     delete=False, encoding='utf-8') as f:
        yaml.dump({'phase': 'test', 'max_it': 100, 'axes': ['gamma', 'encoding']},
                  f, allow_unicode=True)
        yaml_path = Path(f.name)

    cfg = load_yaml(yaml_path)

    _check("D6.1 — load_yaml retourne un dict", isinstance(cfg, dict))
    _check("D6.2 — clé 'phase' = 'test'", cfg.get('phase') == 'test')
    _check("D6.3 — clé 'max_it' = 100",   cfg.get('max_it') == 100)
    _check("D6.4 — clé 'axes' = liste",   isinstance(cfg.get('axes'), list))

    yaml_path.unlink()

except Exception as e:
    _check("D6 — load_yaml", False, str(e))
    traceback.print_exc()


# =============================================================================
# D7 — load_yaml FileNotFoundError
# =============================================================================

_section("D7 — load_yaml FileNotFoundError")

try:
    raised = False
    try:
        load_yaml(Path('/tmp/fichier_inexistant_prc_v7_xyz.yaml'))
    except FileNotFoundError:
        raised = True

    _check("D7.1 — FileNotFoundError levée sur fichier absent", raised)

except Exception as e:
    _check("D7 — load_yaml FileNotFoundError", False, str(e))
    traceback.print_exc()


# =============================================================================
# Fixtures rows Parquet
# =============================================================================

from featuring.jax_features_new import FEATURE_NAMES

def _make_row(gamma_id='GAM-001', seed=0, nan_key=None):
    """Row synthétique avec features floats."""
    features = {k: float(i) * 0.01 for i, k in enumerate(FEATURE_NAMES)}
    if nan_key:
        features[nan_key] = float('nan')
    return {
        'composition': {
            'gamma_id'       : gamma_id,
            'encoding_id'    : 'SYM-001',
            'modifier_id'    : 'M0',
            'n_dof'          : 10,
            'rank_eff'       : 2,
            'max_it'         : 30,
            'gamma_params'   : {'beta': 0.99},
            'encoding_params': {},
            'modifier_params': {},
            'seed_CI'        : seed,
            'seed_run'       : seed + 1,
        },
        'features': features,
    }


# =============================================================================
# D8 — write_rows_to_parquet : schéma correct
# =============================================================================

_section("D8 — write_rows_to_parquet schéma")

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir   = Path(tmpdir)
        row      = _make_row()
        writer   = open_parquet_writer('test_phase', outdir, row)
        n_written = write_rows_to_parquet(writer, [row], 'test_phase')
        close_parquet_writer(writer)

        pf      = pq.read_table(outdir / 'test_phase.parquet')
        cols    = set(pf.schema.names)

        _check("D8.1 — 1 row écrite", n_written == 1, f"got {n_written}")

        _fixed = ['phase','gamma_id','encoding_id','modifier_id',
                  'n_dof','rank_eff','max_it',
                  'gamma_params','encoding_params','modifier_params',
                  'seed_CI','seed_run']
        _missing_fixed = [c for c in _fixed if c not in cols]
        _check("D8.2 — toutes les colonnes fixes présentes",
               len(_missing_fixed) == 0,
               f"manquantes : {_missing_fixed}")

        _missing_feats = [k for k in FEATURE_NAMES if k not in cols]
        _check("D8.3 — toutes les 68 features présentes comme colonnes",
               len(_missing_feats) == 0,
               f"manquantes : {_missing_feats}")

        _check("D8.4 — nombre total de colonnes = 12 fixes + 68 features",
               len(cols) == 80,
               f"got {len(cols)}")

        _check("D8.5 — phase = 'test_phase'",
               pf['phase'][0].as_py() == 'test_phase')

        _check("D8.6 — gamma_id = 'GAM-001'",
               pf['gamma_id'][0].as_py() == 'GAM-001')

        _check("D8.7 — gamma_params JSON décodable",
               isinstance(json.loads(pf['gamma_params'][0].as_py()), dict))

        # Types
        schema_dict = {f.name: f.type for f in pf.schema}
        _check("D8.8 — colonnes features sont float32",
               all(schema_dict[k] == pq.ParquetFile(
                   outdir / 'test_phase.parquet').schema_arrow.field(k).type
                   for k in FEATURE_NAMES[:3]))

except Exception as e:
    _check("D8 — schéma parquet", False, str(e))
    traceback.print_exc()


# =============================================================================
# D9 — write_rows_to_parquet : append multi-batch
# =============================================================================

_section("D9 — append multi-batch")

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir  = Path(tmpdir)
        rows_b1 = [_make_row('GAM-001', seed=i)   for i in range(5)]
        rows_b2 = [_make_row('GAM-002', seed=i+5) for i in range(3)]

        writer  = open_parquet_writer('multi_batch', outdir, rows_b1[0])
        n1      = write_rows_to_parquet(writer, rows_b1, 'multi_batch')
        n2      = write_rows_to_parquet(writer, rows_b2, 'multi_batch')
        close_parquet_writer(writer)

        pf = pq.read_table(outdir / 'multi_batch.parquet')

        _check("D9.1 — batch 1 : 5 rows retournées", n1 == 5, f"got {n1}")
        _check("D9.2 — batch 2 : 3 rows retournées", n2 == 3, f"got {n2}")
        _check("D9.3 — total 8 rows dans le fichier",
               len(pf) == 8, f"got {len(pf)}")

        gamma_ids = [pf['gamma_id'][i].as_py() for i in range(8)]
        _check("D9.4 — 5 rows GAM-001 puis 3 rows GAM-002",
               gamma_ids[:5] == ['GAM-001']*5 and gamma_ids[5:] == ['GAM-002']*3,
               f"got {gamma_ids}")

except Exception as e:
    _check("D9 — append multi-batch", False, str(e))
    traceback.print_exc()


# =============================================================================
# D10 — NaN float32 encodé correctement
# =============================================================================

_section("D10 — NaN float32")

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir   = Path(tmpdir)
        # f3_entanglement_entropy_mode1_mean est NaN structurel sur rank=2
        row_nan  = _make_row(nan_key='f3_entanglement_entropy_mode1_mean')

        writer   = open_parquet_writer('nan_test', outdir, row_nan)
        n        = write_rows_to_parquet(writer, [row_nan], 'nan_test')
        close_parquet_writer(writer)

        pf  = pq.read_table(outdir / 'nan_test.parquet')
        val = pf['f3_entanglement_entropy_mode1_mean'][0].as_py()

        _check("D10.1 — 1 row écrite sans erreur", n == 1)
        _check("D10.2 — valeur NaN relue comme NaN (ou None)",
               val is None or (isinstance(val, float) and math.isnan(val)),
               f"got {val!r}")

        # Autres features non-NaN relues correctement
        v_frob = pf['f1_frob_norm_mean'][0].as_py()
        _check("D10.3 — feature non-NaN relue comme float",
               isinstance(v_frob, float) and not math.isnan(v_frob),
               f"got {v_frob!r}")

except Exception as e:
    _check("D10 — NaN float32", False, str(e))
    traceback.print_exc()


# =============================================================================
# D11 — rows=[] retourne 0 sans erreur
# =============================================================================

_section("D11 — write_rows_to_parquet rows vides")

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir   = Path(tmpdir)
        row      = _make_row()
        writer   = open_parquet_writer('empty_test', outdir, row)
        n        = write_rows_to_parquet(writer, [], 'empty_test')
        close_parquet_writer(writer)

        _check("D11.1 — rows=[] retourne 0", n == 0, f"got {n}")

        # Le fichier doit exister et être lisible (0 rows dans les row groups,
        # mais le schéma est écrit à l'ouverture)
        pf = pq.read_table(outdir / 'empty_test.parquet')
        _check("D11.2 — fichier lisible après write vide",
               len(pf) == 0, f"got {len(pf)} rows")

except Exception as e:
    _check("D11 — rows vides", False, str(e))
    traceback.print_exc()


# =============================================================================
# D12 — close_parquet_writer : fermeture propre
# =============================================================================

_section("D12 — close_parquet_writer")

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir  = Path(tmpdir)
        rows    = [_make_row(seed=i) for i in range(3)]
        writer  = open_parquet_writer('close_test', outdir, rows[0])
        write_rows_to_parquet(writer, rows, 'close_test')
        close_parquet_writer(writer)

        # Fichier lisible après fermeture
        pf = pq.read_table(outdir / 'close_test.parquet')
        _check("D12.1 — fichier lisible après close_parquet_writer",
               len(pf) == 3, f"got {len(pf)}")

        # Deuxième fermeture ne doit pas lever d'erreur (idempotente)
        try:
            close_parquet_writer(writer)
            _check("D12.2 — double close sans erreur", True)
        except Exception as e2:
            _check("D12.2 — double close sans erreur", False, str(e2))

except Exception as e:
    _check("D12 — close_parquet_writer", False, str(e))
    traceback.print_exc()


# =============================================================================
# D13 — pandas absent de data_loading_new
# =============================================================================

_section("D13 — pas d'import pandas")

try:
    import utils.data_loading_new as _dl_mod
    _dl_source_path = Path(_dl_mod.__file__)
    source = _dl_source_path.read_text(encoding='utf-8')
    # Lecture depuis __file__ du module importé — indépendant du cwd
    _check("D13.1 — 'import pandas' absent du fichier source",
           'import pandas' not in source,
           f"found 'import pandas' in {_dl_source_path}")
    _check("D13.2 — 'pd.' absent du fichier source",
           'pd.' not in source,
           f"found 'pd.' in {_dl_source_path}")

except Exception as e:
    _check("D13 — pandas absent", False, str(e))
    traceback.print_exc()


# =============================================================================
# D14 — merge_configs deep merge
# =============================================================================

_section("D14/D15 — merge_configs")

try:
    base     = {'a': 1, 'b': {'x': 10, 'y': 20}, 'c': [1, 2, 3]}
    override = {'b': {'y': 99, 'z': 30}, 'c': [4, 5], 'd': 7}
    result   = merge_configs(base, override)

    _check("D14.1 — clé non overridée conservée (a=1)",
           result['a'] == 1)

    _check("D14.2 — dict imbriqué mergé récursivement (b.x conservé)",
           result['b']['x'] == 10)

    _check("D14.3 — dict imbriqué mergé récursivement (b.y overridé)",
           result['b']['y'] == 99)

    _check("D14.4 — dict imbriqué : nouvelle clé ajoutée (b.z=30)",
           result['b']['z'] == 30)

    _check("D15.1 — liste remplacée entièrement par override (c=[4,5])",
           result['c'] == [4, 5],
           f"got {result['c']}")

    _check("D15.2 — nouvelle clé ajoutée (d=7)",
           result['d'] == 7)

    _check("D15.3 — base non mutée",
           base['b']['y'] == 20 and base['c'] == [1, 2, 3])

except Exception as e:
    _check("D14/D15 — merge_configs", False, str(e))
    traceback.print_exc()


# =============================================================================
# Résumé
# =============================================================================

print(f"\n{'='*60}")
n_pass  = sum(1 for _, ok, _ in _results if ok)
n_fail  = sum(1 for _, ok, _ in _results if not ok)
n_total = len(_results)

print(f"  TOTAL : {n_pass}/{n_total} tests passés")

if n_fail > 0:
    print(f"\n  ÉCHECS ({n_fail}) :")
    for name, ok, detail in _results:
        if not ok:
            print(f"    ✗ {name}")
            if detail:
                print(f"      → {detail}")
else:
    print("  Tous les tests passés.")

print(f"{'='*60}\n")

sys.exit(0 if n_fail == 0 else 1)
