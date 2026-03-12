"""
analysing/hub_analysing_new.py

Responsabilité : Routing pur — charge config + données, route vers pipeline.

Zéro logique métier ici.

Interface publique :
    run_verdict_from_parquet(parquet_path, cfg_path, ...)
    run_verdict_cross_phases(results_dir, cfg_path)
    scan_major_phases(results_dir)
"""

from pathlib import Path
from typing import Dict, List, Optional

from utils.data_loading_new import load_yaml
from analysing.parquet_filter import load_analysing_data
from analysing.pipeline import run_analysing_pipeline
from analysing.verdict_new import write_verdict_report, write_verdict_report_txt
from analysing.concordance_lite import run_concordance_cross_phases


def scan_major_phases(results_dir: Optional[Path] = None) -> List[Path]:
    """Scanne les parquets phases principales (r0.parquet, r1.parquet, ...)."""
    if results_dir is None:
        results_dir = Path('data/results')
    if not results_dir.exists():
        return []
    return sorted([
        p for p in Path(results_dir).glob('*.parquet')
        if p.stem.startswith('r') and len(p.stem) == 2 and p.stem[1].isdigit()
    ])


# =============================================================================
# RUN VERDICT SINGLE-PHASE
# =============================================================================

def run_verdict_from_parquet(
    parquet_path : Path,
    cfg_path     : Optional[Path] = None,
    output_dir   : Optional[Path] = None,
    label        : Optional[str]  = None,
    plot         : bool = True,
    save_debug   : bool = False,
) -> Dict:
    """
    Verdict complet depuis un fichier parquet v7.

    Flux :
        charge cfg YAML
        → load_rows(parquet_path, scope, apply_pool)
        → run_analysing_pipeline(rows, cfg, ...)
        → write_verdict_report[_txt]
        → retourne dict résultat
    """
    parquet_path = Path(parquet_path)
    cfg          = load_yaml(Path(cfg_path).resolve())
    _label       = label or parquet_path.stem

    print(f"\n=== Verdict depuis parquet : {parquet_path.name} ===")

    scope      = cfg.get('scope', {})
    apply_pool = cfg.get('pool_requirements', {}).get('apply', False)
    data = load_analysing_data(parquet_path, scope=scope, apply_pool=apply_pool)

    if data.n == 0:
        print("⚠️  Aucune row après filtrage scope")
        return {'metadata': {'n_observations': 0, 'label': _label}}

    # Si plot demandé sans output_dir explicite → défaut data/results/reports/<label>
    _output_dir = Path(output_dir) if output_dir else (
        Path('data/results/reports') / _label if plot else None
    )
    if _output_dir:
        _output_dir.mkdir(parents=True, exist_ok=True)

    result = run_analysing_pipeline(
        data       = data,
        cfg        = cfg,
        output_dir = _output_dir,
        label      = _label,
        plot       = plot,
        save_debug = save_debug,
    )

    if _output_dir:
        write_verdict_report    (result, _output_dir / f'verdict_{_label}.json')
        write_verdict_report_txt(result, _output_dir / f'verdict_{_label}.txt')

    return result


# =============================================================================
# RUN VERDICT CROSS-PHASES
# =============================================================================

def run_verdict_cross_phases(
    results_dir : Optional[Path] = None,
    cfg_path    : Optional[Path] = None,
    output_dir  : Optional[Path] = None,
    plot        : bool = False,
) -> Dict:
    """Verdict inter-phases — même scope sur toutes les phases principales."""
    cfg           = load_yaml(Path(cfg_path).resolve())
    parquet_paths = scan_major_phases(results_dir)

    print(f"\n=== Verdict cross-phases ===")
    if not parquet_paths:
        print("⚠️  Aucune phase principale trouvée (r0.parquet, r1.parquet, ...)")
        return {'phases': {}, 'concordance': {}, 'metadata': {'n_phases': 0}}

    print(f"Phases trouvées : {[p.stem for p in parquet_paths]}")

    scope      = cfg.get('scope', {})
    apply_pool = cfg.get('pool_requirements', {}).get('apply', False)

    phases_results : Dict[str, Dict]       = {}
    phases_rows    : Dict[str, List[Dict]] = {}

    for parquet_path in parquet_paths:
        phase_name = parquet_path.stem
        print(f"\n--- Phase {phase_name} ---")

        rows = load_rows(parquet_path, scope=scope, apply_pool=apply_pool)
        phases_rows[phase_name] = rows

        if not rows:
            print(f"  ⚠️  Aucune row pour {phase_name}")
            continue

        out_phase = Path(output_dir) / phase_name if output_dir else None

        phases_results[phase_name] = run_analysing_pipeline(
            rows       = rows,
            cfg        = cfg,
            output_dir = out_phase,
            label      = phase_name,
            plot       = plot,
        )

    print(f"\n--- Concordance cross-phases ---")
    concordance = run_concordance_cross_phases(phases_rows)
    print(f"  Kappa pairs : {len(concordance.get('kappa', {}))}")

    result = {
        'phases'     : phases_results,
        'concordance': concordance,
        'metadata'   : {
            'n_phases'       : len(parquet_paths),
            'phases_analyzed': [p.stem for p in parquet_paths],
        },
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_verdict_report(result, out / 'verdict_cross_phases.json')
        write_verdict_report_txt(result, out / 'verdict_cross_phases.txt')

    return result
