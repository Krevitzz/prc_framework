"""
Script analyse biais R3 dans outliers

Vérifie si R3 surreprésenté outliers par biais features ou vraie différence
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_verdict_and_parquet(phase: str = 'poc'):
    """Charge verdict JSON + parquet données."""
    verdict_path = Path(f'reports/verdict_{phase}.json')
    parquet_path = Path(f'data/results/{phase}.parquet')
    
    if not verdict_path.exists():
        print(f"❌ Verdict introuvable: {verdict_path}")
        return None, None
    
    if not parquet_path.exists():
        print(f"❌ Parquet introuvable: {parquet_path}")
        return None, None
    
    with open(verdict_path, 'r') as f:
        verdict = json.load(f)
    
    df = pd.read_parquet(parquet_path)
    
    return verdict, df


def analyze_r3_outliers(verdict, df):
    """
    Analyse surreprésentation R3 dans outliers.
    
    Hypothèses testées :
    H1 : R3 outlier par manque features (biais méthode)
    H2 : R3 outlier par vraie différence distributions
    """
    print("\n" + "="*60)
    print("ANALYSE BIAIS R3 OUTLIERS")
    print("="*60)
    
    # Outliers indices
    outlier_indices = verdict['outliers']['outlier_indices']
    
    # Identifier rank encodings
    df['rank'] = df['encoding_id'].apply(lambda x: 
        'R3' if x.startswith('R3-') else 
        'ASY' if x.startswith('ASY-') else 
        'SYM'
    )
    
    # Stats pool global
    n_total = len(df)
    n_r3_total = (df['rank'] == 'R3').sum()
    n_rank2_total = (df['rank'] != 'R3').sum()
    
    print(f"\n=== Pool global ===")
    print(f"Total runs: {n_total}")
    print(f"R3: {n_r3_total} ({n_r3_total/n_total*100:.1f}%)")
    print(f"Rank 2 (SYM+ASY): {n_rank2_total} ({n_rank2_total/n_total*100:.1f}%)")
    
    # Stats outliers
    df_outliers = df.iloc[outlier_indices]
    n_outliers = len(df_outliers)
    n_r3_outliers = (df_outliers['rank'] == 'R3').sum()
    n_rank2_outliers = (df_outliers['rank'] != 'R3').sum()
    
    print(f"\n=== Outliers ===")
    print(f"Total outliers: {n_outliers}")
    print(f"R3: {n_r3_outliers} ({n_r3_outliers/n_outliers*100:.1f}%)")
    print(f"Rank 2: {n_rank2_outliers} ({n_rank2_outliers/n_outliers*100:.1f}%)")
    
    # Surreprésentation
    r3_pool_frac = n_r3_total / n_total
    r3_outlier_frac = n_r3_outliers / n_outliers
    overrep = r3_outlier_frac / r3_pool_frac
    
    print(f"\n=== Surreprésentation ===")
    print(f"R3 pool: {r3_pool_frac*100:.1f}%")
    print(f"R3 outliers: {r3_outlier_frac*100:.1f}%")
    print(f"Overrepresentation: ×{overrep:.2f}")
    
    if overrep > 1.5:
        print("⚠️  R3 fortement surreprésenté outliers")
    
    # Analyse distributions features communes
    print(f"\n=== Distributions features communes ===")
    
    # Features communes (universal)
    common_features = [
        'euclidean_norm_initial',
        'euclidean_norm_final', 
        'euclidean_norm_mean',
        'entropy_initial',
        'entropy_final'
    ]
    
    for feature in common_features:
        if feature not in df.columns:
            continue
        
        # R3 vs Rank 2
        r3_vals = df[df['rank'] == 'R3'][feature].dropna()
        rank2_vals = df[df['rank'] != 'R3'][feature].dropna()
        
        # Stats
        r3_mean = r3_vals.mean()
        rank2_mean = rank2_vals.mean()
        r3_std = r3_vals.std()
        rank2_std = rank2_vals.std()
        
        # Différence relative
        if rank2_mean != 0:
            rel_diff = abs(r3_mean - rank2_mean) / abs(rank2_mean)
        else:
            rel_diff = 0
        
        print(f"\n{feature}:")
        print(f"  R3    : mean={r3_mean:.3f}, std={r3_std:.3f}")
        print(f"  Rank 2: mean={rank2_mean:.3f}, std={rank2_std:.3f}")
        print(f"  Diff relative: {rel_diff*100:.1f}%")
        
        if rel_diff > 0.5:
            print(f"  → Différence significative (>50%)")
    
    # Conclusion
    print(f"\n=== Conclusion ===")
    
    if overrep < 1.2:
        print("✓ R3 pas surreprésenté — distribution normale")
    elif overrep > 2.0:
        print("⚠️  R3 très surreprésenté (×{:.1f})".format(overrep))
        print("   Causes possibles:")
        print("   1. Vraie différence distributions features")
        print("   2. Biais méthode (moins de features → profil différent)")
        print("\n   Action: Comparer distributions features ci-dessus")
        print("   Si différences >50% → vraie outlier")
        print("   Si différences <20% → biais features manquantes")
    else:
        print("○ R3 légèrement surreprésenté (×{:.1f})".format(overrep))
        print("   Probablement normal (variance échantillonnage)")


def main():
    """Point d'entrée."""
    import sys
    
    phase = sys.argv[1] if len(sys.argv) > 1 else 'poc'
    
    print(f"Chargement phase: {phase}")
    verdict, df = load_verdict_and_parquet(phase)
    
    if verdict is None or df is None:
        return
    
    analyze_r3_outliers(verdict, df)


if __name__ == '__main__':
    main()
