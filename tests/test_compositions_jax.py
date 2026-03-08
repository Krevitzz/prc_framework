"""
tests/test_compositions_jax.py

Validation compositions_jax : génération groupes vmappables.
Exécution : python -m tests.test_compositions_jax  (depuis prc_framework/)

T1 — generate_groups minimal : 1 gamma × 1 enc × 1 mod → 1 groupe, 1 run
T2 — params cartésien        : scale=[0.5,1.0,2.0] → 3 runs dans le groupe
T3 — rank_eff                : SYM→2, RN-001(rank=4)→4, défaut→3
T4 — seed_to_key             : reproductible, None→PRNGKey(0)
T5 — chunk_group             : 10 runs / chunk_size=3 → 4 chunks [3,3,3,1]
T6 — count_stats             : n_groups × runs_par_groupe = n_runs
T7 — XLA cache structure     : homogénéité intra-groupe, amortissement, ordre
"""

import sys
import warnings

warnings.filterwarnings("ignore")


def _fmt(label, ok, details):
    status = "✓" if ok else "✗"
    return f"  {status}  {label:<35} {details}"


def _build_registries():
    from utils.data_loading_jax import (
        discover_gammas_jax,
        discover_encodings_jax,
        discover_modifiers_jax,
    )
    return {
        'gamma'   : discover_gammas_jax(),
        'encoding': discover_encodings_jax(),
        'modifier': discover_modifiers_jax(),
    }


# ---------------------------------------------------------------------------
# T1 — generate_groups minimal
# ---------------------------------------------------------------------------

def t1_generate_groups_minimal():
    try:
        from compositions.compositions_jax import generate_groups

        run_config = {
            'phase'         : 'test',
            'n_dof'         : 10,
            'max_iterations': 50,
            'axes': {
                'gamma'   : [{'id': 'GAM-011'}],
                'encoding': [{'id': 'SYM-001'}],
                'modifier': [{'id': 'M0'}],
            }
        }

        registries = _build_registries()
        groups     = generate_groups(run_config, registries)

        assert len(groups) == 1, f"attendu 1 groupe, reçu {len(groups)}"
        grp = groups[0]
        assert grp['gamma_id']  == 'GAM-011'
        assert grp['enc_id']    == 'SYM-001'
        assert grp['mod_id']    == 'M0'
        assert grp['n_dof']     == 10
        assert grp['max_it']    == 50
        assert grp['rank_eff']  == 2
        assert len(grp['runs']) == 1

        run = grp['runs'][0]
        assert 'key_CI' in run and 'key_run' in run

        return True, (
            f"1 groupe ✓ | 1 run ✓ | "
            f"rank_eff={grp['rank_eff']} | "
            f"non_markovian={grp['non_markovian']}"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T2 — params cartésien
# ---------------------------------------------------------------------------

def t2_params_cartesian():
    try:
        from compositions.compositions_jax import generate_groups

        run_config = {
            'phase'         : 'test',
            'n_dof'         : 10,
            'max_iterations': 50,
            'axes': {
                'gamma': [{'id': 'GAM-011', 'params': {'scale': [0.5, 1.0, 2.0]}}],
                'encoding': [{'id': 'SYM-001'}],
                'modifier': [{'id': 'M0'}],
            }
        }

        registries = _build_registries()
        groups     = generate_groups(run_config, registries)

        assert len(groups) == 1, f"attendu 1 groupe, reçu {len(groups)}"
        grp = groups[0]
        assert len(grp['runs']) == 3, f"attendu 3 runs, reçu {len(grp['runs'])}"

        scales = [r['gamma_params'].get('scale') for r in grp['runs']]
        assert set(scales) == {0.5, 1.0, 2.0}

        return True, f"1 groupe ✓ | 3 runs ✓ | scales={sorted(scales)}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T3 — rank_eff
# ---------------------------------------------------------------------------

def t3_rank_eff():
    try:
        from compositions.compositions_jax import _get_rank_eff

        enc_sym = {'metadata': {'id': 'SYM-001', 'rank': 2, 'stochastic': False}, 'params': {}}
        assert _get_rank_eff(enc_sym) == 2

        enc_rn = {'metadata': {'id': 'RN-001', 'rank': None, 'stochastic': True}, 'params': {'rank': 4}}
        assert _get_rank_eff(enc_rn) == 4

        enc_def = {'metadata': {'id': 'RN-001'}, 'params': {}}
        assert _get_rank_eff(enc_def) == 3

        return True, "SYM-001→2 ✓ | RN-001(params=4)→4 ✓ | défaut→3 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T4 — seed_to_key
# ---------------------------------------------------------------------------

def t4_seed_to_key():
    try:
        import jax, jax.numpy as jnp
        from compositions.compositions_jax import _seed_to_key, _make_run_key

        k42 = _seed_to_key(42)
        assert k42.shape == (2,)

        k0 = _seed_to_key(None)
        assert jnp.allclose(k0, jax.random.PRNGKey(0)).item()

        assert jnp.allclose(k42, _seed_to_key(42)).item()
        assert not jnp.allclose(k42, _seed_to_key(99)).item()

        base = _seed_to_key(0)
        kr1  = _make_run_key(base, 'GAM-011', 'SYM-001')
        kr2  = _make_run_key(base, 'GAM-011', 'SYM-001')
        assert jnp.allclose(kr1, kr2).item()
        assert not jnp.allclose(kr1, _make_run_key(base, 'GAM-001', 'SYM-002')).item()

        return True, "PRNGKey(2,) ✓ | None→PRNGKey(0) ✓ | reproductible ✓ | fold_in distinct ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T5 — chunk_group
# ---------------------------------------------------------------------------

def t5_chunk_group():
    try:
        from compositions.compositions_jax import chunk_group

        fake_runs  = [{'run_id': i} for i in range(10)]
        fake_group = {'gamma_id': 'GAM-011', 'runs': fake_runs}
        chunks     = chunk_group(fake_group, chunk_size=3)

        assert len(chunks) == 4
        sizes = [len(c['runs']) for c in chunks]
        assert sizes == [3, 3, 3, 1]
        assert [r for c in chunks for r in c['runs']] == fake_runs
        assert chunks[0]['gamma_id'] == 'GAM-011'

        return True, f"4 chunks ✓ | tailles={sizes} ✓ | union complète ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T6 — count_stats
# ---------------------------------------------------------------------------

def t6_count_stats():
    try:
        from compositions.compositions_jax import generate_groups, count_stats

        run_config = {
            'phase'         : 'test',
            'n_dof'         : [10, 20],
            'max_iterations': [50, 100],
            'axes': {
                'gamma'   : [{'id': 'GAM-011'}],
                'encoding': [{'id': 'SYM-001'}],
                'modifier': [{'id': 'M0'}],
            }
        }

        registries = _build_registries()
        groups     = generate_groups(run_config, registries)
        stats      = count_stats(groups, chunk_size=256)

        assert stats['n_groups'] == 4
        assert stats['n_runs']   == 4
        assert sum(len(g['runs']) for g in groups) == stats['n_runs']

        return True, (
            f"n_groups={stats['n_groups']} ✓ | "
            f"n_runs={stats['n_runs']} ✓ | "
            f"n_chunks={stats['n_chunks']} ✓"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T7 — Structure XLA cache : homogénéité, ordre séquentiel, amortissement
# ---------------------------------------------------------------------------
#
# Propriétés vérifiées :
#
#   P1 — Homogénéité intra-groupe
#        Chaque groupe a une clé XLA unique (gamma_id, enc_id, mod_id, n_dof, rank_eff, max_it)
#        → 1 compilation par groupe, zéro recompilation intra-chunk
#
#   P2 — Isolation inter-groupes
#        chunk_group ne sort jamais du groupe (ordre préservé, runs complets)
#        → le cache XLA ne flush JAMAIS au sein d'un groupe
#
#   P3 — Ordre séquentiel = super-chunk implicite
#        Simulation de l'itération hub_running : for group → for chunk_of_group
#        → tous les chunks d'un groupe s'exécutent avant passage au suivant
#        → n_compilations_réelles = n_groups (pas n_chunks, pas n_runs)
#
#   P4 — Amortissement vmap
#        n_runs >> n_groups → chaque compilation est amortie sur plusieurs calls
#
#   P5 — rank_eff sépare les groupes correctement
#        RN-001 rank=2 et RN-001 rank=3 → deux groupes distincts
#        (shapes différentes → recompilation correctement anticipée)
#
# Config test :
#   3 gammas × {1,2,4} params scalaires
#   3 encodings (SYM-001, SYM-002, RN-001 rank=[2,3])
#   2 modifiers × {1,2} params scalaires
#   n_dof=[10,20], max_it=[50,100], seed_CI=[0,1], seed_run=[null,0]
# ---------------------------------------------------------------------------

def t7_xla_cache_structure():
    try:
        from compositions.compositions_jax import generate_groups, chunk_group, count_stats

        run_config = {
            'phase'         : 'test_xla',
            'n_dof'         : [10, 20],
            'max_iterations': [50, 100],
            'seed_CI'       : [0, 1],
            'seed_run'      : [None, 0],
            'axes': {
                'gamma': [
                    # 2 entrées (2 betas)
                    {'id': 'GAM-001', 'params': {'beta': [1.0, 3.0]}},
                    # 2 entrées (2 gammas)
                    {'id': 'GAM-004', 'params': {'gamma': [0.05, 0.1]}},
                    # 4 entrées (2 betas × 2 sigmas)
                    {'id': 'GAM-009', 'params': {'beta': [1.0, 2.0], 'sigma': [0.01, 0.1]}},
                ],
                'encoding': [
                    # 1 entrée — rank_eff=2
                    {'id': 'SYM-001'},
                    # 1 entrée — rank_eff=2
                    {'id': 'SYM-002'},
                    # 2 entrées — rank_eff=2 ET rank_eff=3 → groupes séparés
                    {'id': 'RN-001', 'params': {'rank': [2, 3]}},
                ],
                'modifier': [
                    # 1 entrée
                    {'id': 'M0'},
                    # 2 entrées (2 sigmas)
                    {'id': 'M1', 'params': {'sigma': [0.01, 0.1]}},
                ],
            }
        }

        registries = _build_registries()
        groups     = generate_groups(run_config, registries)
        stats      = count_stats(groups, chunk_size=32)

        n_groups = stats['n_groups']
        n_runs   = stats['n_runs']
        n_chunks = stats['n_chunks']

        # --- P1 : Clés XLA uniques par groupe --------------------------------
        seen_keys = {}
        for grp in groups:
            xla_key = (
                grp['gamma_id'], grp['enc_id'], grp['mod_id'],
                grp['n_dof'], grp['rank_eff'], grp['max_it'],
            )
            assert xla_key not in seen_keys, (
                f"Clé XLA dupliquée : {xla_key}\n"
                f"  groupe 1 : {seen_keys[xla_key]}\n"
                f"  groupe 2 : {grp}"
            )
            seen_keys[xla_key] = grp

        assert len(seen_keys) == n_groups

        # --- P2 : chunk_group préserve l'ordre et l'isolation ----------------
        for grp in groups:
            chunks = chunk_group(grp, chunk_size=32)
            original_ids = [id(r) for r in grp['runs']]
            chunk_ids    = [id(r) for c in chunks for r in c['runs']]
            assert original_ids == chunk_ids, (
                f"chunk_group a perdu/réordonné des runs dans {grp['gamma_id']}"
            )
            # Chaque chunk ne sort pas du groupe
            for c in chunks:
                assert c['gamma_id'] == grp['gamma_id']
                assert c['enc_id']   == grp['enc_id']

        # --- P3 : Simulation itération hub_running → n_compilations = n_groups
        compilation_events = []
        current_xla_key    = None

        for grp in groups:
            xla_key = (
                grp['gamma_id'], grp['enc_id'], grp['mod_id'],
                grp['n_dof'], grp['rank_eff'], grp['max_it'],
            )
            chunks = chunk_group(grp, chunk_size=32)
            for _chunk in chunks:
                if xla_key != current_xla_key:
                    # Nouvelle compilation XLA
                    compilation_events.append(xla_key)
                    current_xla_key = xla_key
                # else : même groupe → cache hit, zéro recompilation

        n_compilations = len(compilation_events)
        assert n_compilations == n_groups, (
            f"n_compilations={n_compilations} ≠ n_groups={n_groups} "
            f"→ recompilations inattendues dans l'ordre d'itération"
        )

        # --- P4 : Amortissement ----------------------------------------------
        ratio = n_runs / n_groups if n_groups > 0 else 0.0
        assert ratio >= 1.0

        runs_per_group = [len(g['runs']) for g in groups]
        min_r = min(runs_per_group)
        max_r = max(runs_per_group)
        avg_r = sum(runs_per_group) / len(runs_per_group)

        # --- P5 : rank_eff sépare RN-001 rank=2 vs rank=3 -------------------
        rn001_groups = [g for g in groups if g['enc_id'] == 'RN-001']
        if rn001_groups:
            rank_effs = {g['rank_eff'] for g in rn001_groups}
            assert len(rank_effs) >= 2, (
                f"RN-001 rank=2 et rank=3 non séparés : rank_effs={rank_effs}"
            )

        return True, (
            f"\n"
            f"    n_groupes     = {n_groups}  (compilations XLA théoriques)\n"
            f"    n_runs        = {n_runs}\n"
            f"    n_chunks      = {n_chunks}  (chunk_size=32)\n"
            f"    runs/groupe   : min={min_r}  max={max_r}  moy={avg_r:.1f}\n"
            f"    amortissement : {ratio:.1f}×  (runs par compilation)\n"
            f"    P1 clés XLA uniques par groupe      ✓\n"
            f"    P2 isolation inter-groupes          ✓\n"
            f"    P3 ordre séquentiel = super-chunk   ✓  ({n_compilations} compiles = n_groupes)\n"
            f"    P4 amortissement vmap               ✓\n"
            f"    P5 rank_eff sépare RN-001 rank=2/3  ✓  rank_effs={sorted(rank_effs) if rn001_groups else 'N/A'}"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 70)
    print("  PRC v7 — test_compositions_jax : groupes vmappables")
    print("=" * 70)
    print()

    validations = [
        ("T1  generate_groups minimal  ", t1_generate_groups_minimal),
        ("T2  params cartésien         ", t2_params_cartesian),
        ("T3  rank_eff                 ", t3_rank_eff),
        ("T4  seed_to_key              ", t4_seed_to_key),
        ("T5  chunk_group              ", t5_chunk_group),
        ("T6  count_stats              ", t6_count_stats),
        ("T7  XLA cache structure      ", t7_xla_cache_structure),
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
        print(f"  Résultat : {n_ok}/{len(results)} ✓ — compositions_jax opérationnel")
    else:
        print(f"  Résultat : {n_ok}/{len(results)} ✓  {n_fail} ✗ — corriger avant de continuer")
    print("=" * 70)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
