# PLAN IMPLÉMENTATION REFACTOR - PAR ÉTAPES

**Date** : 17 février 2026  
**Version** : 1.0  
**Approche** : Proof-of-concept minimal → Enrichissement progressif  
**Méthodologie** : Algo → Structure → Code avec validation utilisateur obligatoire

---

## 🎯 PRINCIPE GÉNÉRAL

### Stratégie implémentation

**Phase 1 : SQUELETTE (Proof-of-Concept)**
- Version_lite chaque module (fonctionnalités minimales)
- Test intégration bout-en-bout
- Validation architecture avant enrichissement

**Phase 2 : ENRICHISSEMENT**
- Ajout fonctionnalités progressif
- Tests après chaque ajout majeur
- Migration données legacy

**Règles** :
- ✅ Toujours commencer par version_lite
- ✅ Tester avant d'enrichir
- ✅ Validation utilisateur chaque étape majeure
- ✅ Méthodologie Algo → Structure → Code respectée

---

## 📊 VUE D'ENSEMBLE ÉTAPES

```
ÉTAPE 0 : Préparation environnement
  ├─ Structure dossiers
  ├─ Dépendances
  └─ Tests vides

ÉTAPE 1 : Core minimal (inchangé)
  └─ Validation core existant fonctionne

ÉTAPE 2 : Featuring_lite
  ├─ Hub minimal
  ├─ Extractor basique (3-5 features)
  ├─ Layers universal uniquement
  └─ Tests featuring_lite

ÉTAPE 3 : Running_lite
  ├─ Compositions génération basique
  ├─ Run single composition
  ├─ Parquet write minimal
  └─ Tests running_lite

ÉTAPE 4 : Batch_lite (intégration bout-en-bout)
  ├─ batch.py façade
  ├─ YAML config minimal
  ├─ Dry-run estimation
  └─ Tests batch_lite (10 runs)

ÉTAPE 5 : Verdict_lite
  ├─ Profiling aggregation basique
  ├─ Analysing clustering simple
  ├─ Rapport minimal
  └─ Tests verdict_lite

---

ÉTAPE 6 : Enrichissement featuring
  ├─ Registries complets
  ├─ Layers multiples
  ├─ Régimes/Timelines
  └─ Tests featuring_full

ÉTAPE 7 : Enrichissement running
  ├─ Discovery axes
  ├─ Dry-run avancé
  ├─ Axes temporaires
  └─ Tests running_full

ÉTAPE 8 : Enrichissement verdict
  ├─ Profiling régimes/timelines
  ├─ Analysing concordance
  ├─ Rapports détaillés
  └─ Tests verdict_full

ÉTAPE 9 : Migration legacy
  ├─ Script migration DBs
  ├─ Tests comparaison pipelines
  └─ Validation résultats

ÉTAPE 10 : Optimisation
  ├─ Benchmarks performance
  ├─ Optimisations critiques
  └─ Documentation finale
```

**Durée estimée** :
- Étapes 0-5 (POC) : 2-3 semaines
- Étapes 6-8 (enrichissement) : 3-4 semaines
- Étapes 9-10 (migration/opti) : 1-2 semaines
- **Total** : 6-9 semaines

---

## ÉTAPE 0 : PRÉPARATION ENVIRONNEMENT

### Objectif
Créer structure dossiers vide + dépendances + tests boilerplate

### Actions

**0.1 Structure dossiers**
```
prc/
├── core/                      # Inchangé (validation)
│   ├── kernel.py
│   ├── state_preparation.py
│   └── core_catalog.md
│
├── atomics/                   # Inchangé (validation)
│   ├── operators/
│   ├── D_encodings/
│   └── modifiers/
│
├── featuring/                 # Nouveau (vide, préparation)
│   ├── __init__.py
│   ├── hub_lite.py           # Version_lite
│   ├── extractor_lite.py
│   ├── layers_lite.py
│   └── registries/
│       ├── __init__.py
│       └── universal_lite.py
│
├── running/                   # Nouveau (dossier principal)
│   ├── __init__.py
│   ├── hub.py                # Orchestration
│   ├── compositions.py       # Génération compositions
│   ├── run.py                # Exécution runs (ancien batch_runner)
│   └── verdict.py            # Analyses post-batch
│
├── profiling/                 # Nouveau (vide)
│   ├── __init__.py
│   ├── hub_lite.py
│   └── aggregation_lite.py
│
├── analysing/                 # Nouveau (vide)
│   ├── __init__.py
│   ├── hub_lite.py
│   └── clustering_lite.py
│
├── utils/                     # Nouveau
│   ├── __init__.py
│   └── database.py           # Helpers Parquet
│
├── configs/                   # Existant (minimal YAML)
│   ├── phases/
│   │   └── test/
│   │       └── poc.yaml      # Config POC minimal
│   └── features/
│       └── minimal/
│           └── universal.yaml
│
├── data/                      # Nouveau (vs prc_databases/)
│   └── results/
│       └── .gitkeep
│
├── tests/                     # Tests boilerplate
│   ├── test_core.py
│   ├── test_featuring_lite.py
│   ├── test_running_lite.py
│   ├── test_batch_lite.py
│   └── test_verdict_lite.py
│
├── batch.py                   # Fichier racine façade
└── README.md
```

**0.2 Dépendances**
```bash
# requirements.txt (minimal POC)
numpy>=1.24
scipy>=1.10
pandas>=2.0
pyarrow>=12.0  # Parquet
scikit-learn>=1.3  # Clustering
pyyaml>=6.0
pytest>=7.0
```

**0.3 Tests boilerplate**
- Créer tests vides structure
- Vérifier imports fonctionnent
- Placeholder assertions

**Validation étape 0** :
- [ ] Structure dossiers créée
- [ ] requirements.txt installé
- [ ] `pytest tests/` passe (tests vides)
- [ ] Imports `from prc.featuring import hub_lite` fonctionnent

**Durée estimée** : 1 jour

---

## ÉTAPE 1 : CORE MINIMAL (VALIDATION EXISTANT)

### Objectif
Vérifier core existant fonctionne sans modification

### Actions

**1.1 Audit core/**
- Lire `core/kernel.py`, `state_preparation.py`
- Vérifier conformité Charter Section 1.2 (Core aveugle K1-K5)
- Documenter signature exacte fonctions

**1.2 Tests core**
```python
# tests/test_core.py
def test_prepare_state():
    """Valider prepare_state retourne np.ndarray."""
    # TODO: implémenter

def test_run_kernel():
    """Valider run_kernel retourne history (T, *dims)."""
    # TODO: implémenter
```

**1.3 Validation**
- Core reste aveugle (pas validation contenu)
- Fonctions pures (state → state)
- Aucune modification nécessaire

**Validation étape 1** :
- [ ] Core fonctionne sans modification
- [ ] Tests core passent
- [ ] Signature fonctions documentée

**Durée estimée** : 2 jours

---

## ÉTAPE 2 : FEATURING_LITE

### Objectif
Version minimale featuring (3-5 features, layer universal uniquement)

### Philosophie version_lite

**Featuring_lite inclut** :
- Hub minimal (orchestration)
- Extractor basique (3-5 features scalaires)
- Layer universal uniquement (tous tenseurs)
- Pas de projections multiples (final uniquement)
- Pas de statistics complexes (mean, std uniquement)
- Pas de dynamic events
- Pas de régimes/timelines

**Featuring_lite exclut** :
- Layers matrix_2d, tensor_3d, spatial_2d
- Registries multiples (1 seul : universal)
- Projections temporelles (initial, max, min, ...)
- Events timeline
- ML intra-run

### Actions

**2.1 ALGO featuring_lite**

**Objectif** : Extraire 3-5 features scalaires depuis history

**Algo** :
1. Recevoir history (T, *dims) depuis kernel
2. Inspecter shape/rank history
3. Calculer 3-5 features universelles :
   - frobenius_norm_final (norme Frobenius snapshot final)
   - mean_value_final (moyenne valeurs snapshot final)
   - std_value_final (écart-type valeurs snapshot final)
   - frobenius_norm_mean (norme Frobenius moyenne timeline)
   - frobenius_norm_std (norme Frobenius écart-type timeline)
4. Retourner dict features scalaires

**2.2 STRUCTURE featuring_lite**

**hub_lite.py** :
```python
def extract_features_lite(history: np.ndarray, config: dict) -> dict:
    """
    Extrait features minimales depuis history.
    
    Args:
        history : np.ndarray (T, *dims) - Timeline états
        config : dict - Config YAML (minimal)
    
    Returns:
        dict : Features scalaires
            {
                'frobenius_norm_final': float,
                'mean_value_final': float,
                'std_value_final': float,
                'frobenius_norm_mean': float,
                'frobenius_norm_std': float
            }
    
    Utilise :
        - extractor_lite.py (calculs)
        - layers_lite.py (inspection)
    """
    pass
```

**extractor_lite.py** :
```python
class FeatureExtractorLite:
    """Extraction features minimales."""
    
    def __init__(self, config: dict):
        """Config minimal."""
        pass
    
    def extract(self, history: np.ndarray) -> dict:
        """
        Extrait 3-5 features.
        
        Returns:
            dict : Features scalaires
        """
        pass
    
    def _compute_frobenius_norm(self, state: np.ndarray) -> float:
        """Wrapper np.linalg.norm."""
        pass
    
    def _compute_mean_std(self, state: np.ndarray) -> tuple:
        """Wrapper np.mean, np.std."""
        pass
```

**layers_lite.py** :
```python
def get_layer(history: np.ndarray) -> str:
    """
    Inspection directe history (rank, dims).
    
    Version_lite : Retourne toujours 'universal'.
    Version_full : Retournera matrix_2d, tensor_3d, etc.
    
    Returns:
        str : 'universal' (lite) ou 'matrix_2d', 'tensor_3d' (full)
    """
    return 'universal'
```

**registries/universal_lite.py** :
```python
class UniversalRegistryLite:
    """Registry features universelles (wrappers numpy)."""
    
    def frobenius_norm(self, state: np.ndarray) -> float:
        """Wrapper np.linalg.norm('fro')."""
        pass
    
    def mean_value(self, state: np.ndarray) -> float:
        """Wrapper np.mean."""
        pass
    
    def std_value(self, state: np.ndarray) -> float:
        """Wrapper np.std."""
        pass
```

**2.3 VALIDATION featuring_lite**

**Tests** :
```python
# tests/test_featuring_lite.py
def test_extract_features_lite():
    """Test extraction 5 features."""
    history = np.random.rand(201, 10, 10)
    config = {}
    features = extract_features_lite(history, config)
    
    assert len(features) == 5
    assert 'frobenius_norm_final' in features
    assert isinstance(features['frobenius_norm_final'], float)
    assert np.isfinite(features['frobenius_norm_final'])

def test_universal_registry():
    """Test wrappers numpy."""
    registry = UniversalRegistryLite()
    state = np.random.rand(10, 10)
    
    norm = registry.frobenius_norm(state)
    assert np.isclose(norm, np.linalg.norm(state, 'fro'))
```

**Validation étape 2** :
- [ ] `extract_features_lite()` retourne 5 features scalaires
- [ ] Tests featuring_lite passent
- [ ] Pas de dépendances circulaires
- [ ] Wrappers numpy robustes (try/except)

**Durée estimée** : 3-4 jours

---

## ÉTAPE 3 : RUNNING_LITE

### Objectif
Orchestration minimale : compositions basiques + run single + write Parquet

### Philosophie version_lite

**Running_lite inclut** :
- Génération compositions simples (1-2 axes)
- Run single composition (kernel + featuring)
- Write Parquet 1 fichier phase
- Pas de dry-run
- Pas de discovery
- Pas de batch multiple compositions

**Running_lite exclut** :
- Discovery axes (all, random)
- Dry-run estimation
- Confirmation utilisateur
- Batch itération multiples compositions
- Axes temporaires (seed, DOF, ...)

### Actions

**3.1 ALGO running_lite**

**Objectif** : Exécuter 1 composition (gamma, d_encoding) → Parquet

**Algo** :
1. Définir composition manuelle :
   ```python
   composition = {
       'gamma_id': 'GAM-001',
       'd_encoding_id': 'SYM-001',
       'phase': 'test_poc'
   }
   ```
2. Charger atomics (gamma, d_encoding) depuis fichiers
3. Appeler `run_kernel(composition, config)` → history
4. Appeler `extract_features_lite(history, config)` → features
5. Créer observation = composition | features
6. Write Parquet `data/results/test_poc.parquet`

**3.2 STRUCTURE running_lite**

**compositions.py** :
```python
def create_composition_manual(gamma_id: str, d_encoding_id: str, 
                               phase: str) -> dict:
    """
    Crée composition manuelle (pas discovery).
    
    Args:
        gamma_id : str - ID gamma (ex: 'GAM-001')
        d_encoding_id : str - ID encoding (ex: 'SYM-001')
        phase : str - Phase (ex: 'test_poc')
    
    Returns:
        dict : Composition
            {
                'gamma_id': 'GAM-001',
                'd_encoding_id': 'SYM-001',
                'phase': 'test_poc',
                'exec_id': 'POC_GAM-001_SYM-001_001'
            }
    """
    pass
```

**run.py** :
```python
def run_single_composition(composition: dict, config: dict) -> dict:
    """
    Exécute 1 composition (kernel + featuring).
    
    Args:
        composition : dict - Composition (gamma_id, d_encoding_id, phase)
        config : dict - Config kernel
    
    Returns:
        dict : Observation (composition | features)
            {
                'exec_id': 'POC_001',
                'phase': 'test_poc',
                'gamma_id': 'GAM-001',
                'd_encoding_id': 'SYM-001',
                'frobenius_norm_final': 8.2,
                ...
            }
    
    Process :
        1. Load atomics (gamma, d_encoding)
        2. run_kernel(composition, config) → history
        3. extract_features_lite(history, config) → features
        4. Merge composition | features → observation
    """
    pass
```

**utils/database.py** :
```python
def write_parquet(observations: list, phase: str, 
                  base_path: str = 'data/results/') -> str:
    """
    Écrit observations Parquet.
    
    Args:
        observations : list[dict] - Liste observations
        phase : str - Phase (nom fichier)
        base_path : str - Chemin base
    
    Returns:
        str : Chemin fichier créé
    
    Example:
        >>> obs = [{'exec_id': 'POC_001', 'phase': 'test_poc', ...}]
        >>> write_parquet(obs, 'test_poc')
        'data/results/test_poc.parquet'
    """
    pass

def read_parquet(phase: str, base_path: str = 'data/results/') -> pd.DataFrame:
    """
    Lit observations Parquet.
    
    Args:
        phase : str - Phase (nom fichier)
        base_path : str - Chemin base
    
    Returns:
        pd.DataFrame : Observations
    """
    pass
```

**3.3 VALIDATION running_lite**

**Tests** :
```python
# tests/test_running_lite.py
def test_create_composition_manual():
    """Test création composition manuelle."""
    comp = create_composition_manual('GAM-001', 'SYM-001', 'test_poc')
    assert comp['gamma_id'] == 'GAM-001'
    assert comp['d_encoding_id'] == 'SYM-001'
    assert 'exec_id' in comp

def test_run_single_composition():
    """Test run 1 composition."""
    comp = create_composition_manual('GAM-001', 'SYM-001', 'test_poc')
    config = {}  # Config minimal
    observation = run_single_composition(comp, config)
    
    assert 'exec_id' in observation
    assert 'frobenius_norm_final' in observation
    assert len(observation) > 5  # 3 axes + 5 features

def test_write_read_parquet():
    """Test write/read Parquet."""
    obs = [{'exec_id': 'POC_001', 'phase': 'test_poc', 'value': 1.0}]
    path = write_parquet(obs, 'test_poc')
    
    df = read_parquet('test_poc')
    assert len(df) == 1
    assert df.iloc[0]['exec_id'] == 'POC_001'
```

**Validation étape 3** :
- [ ] `run_single_composition()` fonctionne bout-en-bout
- [ ] Parquet write/read OK
- [ ] Tests running_lite passent
- [ ] Fichier `data/results/test_poc.parquet` créé

**Durée estimée** : 3-4 jours

---

## ÉTAPE 4 : BATCH_LITE (INTÉGRATION BOUT-EN-BOUT)

### Objectif
Façade `batch.py` + config YAML minimal + exécution 10 compositions

### Philosophie version_lite

**Batch_lite inclut** :
- Façade `batch.py` (point entrée)
- Config YAML minimal (2 axes : gamma, d_encoding)
- Itération 10 compositions (2 gammas × 5 d_encodings)
- Dry-run estimation basique
- Confirmation utilisateur (o/n)

**Batch_lite exclut** :
- Discovery automatique (liste explicite IDs)
- Axes temporaires (seed, DOF, ...)
- Axes multiples (>3)
- Estimation avancée (RAM, temps précis)

### Actions

**4.1 ALGO batch_lite**

**Objectif** : Exécuter 10 compositions depuis YAML → Parquet

**Algo** :
1. Charger config YAML :
   ```yaml
   # configs/phases/test/poc.yaml
   phase: test_poc
   iteration_axes:
     gamma_id:
       - GAM-001
       - GAM-002
     d_encoding_id:
       - SYM-001
       - SYM-002
       - SYM-003
       - SYM-004
       - SYM-005
   ```
2. Générer compositions (produit cartésien axes)
3. Dry-run : Estimer nb runs, temps approximatif
4. Confirmation utilisateur (o/n)
5. Itérer compositions :
   - Run composition
   - Append observation liste
6. Write Parquet batch complet

**4.2 STRUCTURE batch_lite**

**batch.py** (racine) :
```python
"""
Point entrée batch runner.

Usage:
    python batch.py --config configs/phases/test/poc.yaml
"""
import argparse
from prc.running import hub

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    hub.run_batch(args.config)

if __name__ == '__main__':
    main()
```

**running/hub.py** :
```python
def run_batch(config_path: str):
    """
    Exécute batch complet depuis config YAML.
    
    Args:
        config_path : str - Chemin config YAML
    
    Process:
        1. Load config YAML
        2. Generate compositions
        3. Dry-run estimation
        4. Confirmation utilisateur
        5. Run compositions
        6. Write Parquet
    """
    pass

def dry_run_estimate(compositions: list, config: dict) -> dict:
    """
    Estimation basique batch.
    
    Returns:
        dict : {
            'n_compositions': int,
            'estimated_time_seconds': float,
            'estimated_size_mb': float
        }
    """
    pass

def confirm_execution(estimate: dict) -> bool:
    """
    Demande confirmation utilisateur.
    
    Args:
        estimate : dict - Estimation dry-run
    
    Returns:
        bool : True si utilisateur confirme (o), False sinon (n)
    """
    pass
```

**running/compositions.py** (enrichissement) :
```python
def generate_compositions_from_config(config: dict) -> list:
    """
    Génère compositions depuis config YAML.
    
    Args:
        config : dict - Config chargée YAML
            {
                'phase': 'test_poc',
                'iteration_axes': {
                    'gamma_id': ['GAM-001', 'GAM-002'],
                    'd_encoding_id': ['SYM-001', ...]
                }
            }
    
    Returns:
        list[dict] : Compositions (produit cartésien)
            [
                {'gamma_id': 'GAM-001', 'd_encoding_id': 'SYM-001', ...},
                {'gamma_id': 'GAM-001', 'd_encoding_id': 'SYM-002', ...},
                ...
            ]
    """
    pass
```

**4.3 VALIDATION batch_lite**

**Tests** :
```python
# tests/test_batch_lite.py
def test_generate_compositions():
    """Test génération compositions depuis config."""
    config = {
        'phase': 'test_poc',
        'iteration_axes': {
            'gamma_id': ['GAM-001', 'GAM-002'],
            'd_encoding_id': ['SYM-001', 'SYM-002']
        }
    }
    compositions = generate_compositions_from_config(config)
    assert len(compositions) == 4  # 2 × 2

def test_dry_run_estimate():
    """Test estimation dry-run."""
    compositions = [{'gamma_id': 'GAM-001', 'd_encoding_id': 'SYM-001'}] * 10
    config = {}
    estimate = dry_run_estimate(compositions, config)
    
    assert estimate['n_compositions'] == 10
    assert estimate['estimated_time_seconds'] > 0

def test_batch_integration():
    """Test batch 10 runs complet."""
    # Config minimal 2×5=10 compositions
    config_path = 'configs/phases/test/poc.yaml'
    
    # Mock confirmation utilisateur (skip interactive)
    # run_batch(config_path)  # TODO: implémenter
    
    # Vérifier Parquet créé
    df = read_parquet('test_poc')
    assert len(df) == 10
    assert 'frobenius_norm_final' in df.columns
```

**Validation étape 4** :
- [ ] `python batch.py --config ...` fonctionne
- [ ] 10 compositions générées
- [ ] Dry-run affiche estimation
- [ ] Confirmation utilisateur OK
- [ ] Parquet `test_poc.parquet` contient 10 observations
- [ ] Tests batch_lite passent

**Durée estimée** : 4-5 jours

---

## ÉTAPE 5 : VERDICT_LITE

### Objectif
Analyses post-batch minimales : aggregation + clustering simple + rapport

### Philosophie version_lite

**Verdict_lite inclut** :
- Profiling aggregation basique (median features par gamma)
- Analysing clustering simple (HDBSCAN)
- Rapport texte minimal

**Verdict_lite exclut** :
- Régimes/Timelines (pas calculés featuring_lite)
- Concordance cross-phases
- Outliers détection
- Variance analysis
- Rapports visuels (plots)

### Actions

**5.1 ALGO verdict_lite**

**Objectif** : Analyser 10 observations → rapport texte

**Algo** :
1. Charger observations Parquet phase
2. **Profiling** : Aggregation median features par gamma
   - Grouper par gamma_id
   - Calculer median frobenius_norm_final
   - Identifier gamma dominant (median max)
3. **Analysing** : Clustering HDBSCAN
   - Features : frobenius_norm_final, mean_value_final
   - HDBSCAN (min_cluster_size=2)
   - Identifier clusters
4. **Rapport** : Texte synthesis
   - Profils gammas (median features)
   - Clusters détectés
   - Recommandations basiques

**5.2 STRUCTURE verdict_lite**

**running/verdict.py** :
```python
"""
Analyses post-batch (verdict).

Usage:
    python -m prc.running.verdict --phase test_poc
"""
import argparse
from prc.profiling import hub_lite as profiling_hub
from prc.analysing import hub_lite as analysing_hub

def run_verdict(phase: str):
    """
    Exécute verdict complet.
    
    Args:
        phase : str - Phase à analyser
    
    Process:
        1. Load observations Parquet
        2. Profiling aggregation
        3. Analysing clustering
        4. Generate rapport
    """
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True)
    args = parser.parse_args()
    
    run_verdict(args.phase)

if __name__ == '__main__':
    main()
```

**profiling/hub_lite.py** :
```python
def aggregate_profiles(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregation features par gamma.
    
    Args:
        observations : pd.DataFrame - Observations phase
    
    Returns:
        pd.DataFrame : Profils gammas
            gamma_id | frobenius_norm_median | mean_value_median | ...
    """
    pass
```

**profiling/aggregation_lite.py** :
```python
def compute_gamma_profiles(observations: pd.DataFrame, 
                            features: list) -> pd.DataFrame:
    """
    Calcule profils gammas (median features).
    
    Args:
        observations : pd.DataFrame
        features : list[str] - Features à agréger
    
    Returns:
        pd.DataFrame : Profils
    """
    pass
```

**analysing/hub_lite.py** :
```python
def run_clustering(observations: pd.DataFrame, 
                   features: list) -> dict:
    """
    Clustering HDBSCAN.
    
    Args:
        observations : pd.DataFrame
        features : list[str] - Features clustering
    
    Returns:
        dict : {
            'labels': np.ndarray,
            'n_clusters': int,
            'cluster_sizes': dict
        }
    """
    pass
```

**analysing/clustering_lite.py** :
```python
from sklearn.cluster import HDBSCAN

def hdbscan_clustering(features_matrix: np.ndarray, 
                       min_cluster_size: int = 2) -> np.ndarray:
    """
    Wrapper HDBSCAN sklearn.
    
    Args:
        features_matrix : np.ndarray (n_obs, n_features)
        min_cluster_size : int
    
    Returns:
        np.ndarray : Labels clusters (-1 = noise)
    """
    pass
```

**5.3 VALIDATION verdict_lite**

**Tests** :
```python
# tests/test_verdict_lite.py
def test_aggregate_profiles():
    """Test aggregation profils gammas."""
    observations = pd.DataFrame({
        'gamma_id': ['GAM-001', 'GAM-001', 'GAM-002'],
        'frobenius_norm_final': [8.0, 9.0, 12.0]
    })
    profiles = aggregate_profiles(observations)
    
    assert len(profiles) == 2  # 2 gammas
    assert profiles.loc[profiles['gamma_id'] == 'GAM-001', 
                        'frobenius_norm_median'].values[0] == 8.5

def test_clustering():
    """Test clustering HDBSCAN."""
    observations = pd.DataFrame({
        'frobenius_norm_final': [1, 1.1, 1.2, 10, 10.1],
        'mean_value_final': [0.5, 0.51, 0.52, 0.9, 0.91]
    })
    result = run_clustering(observations, 
                            ['frobenius_norm_final', 'mean_value_final'])
    
    assert result['n_clusters'] >= 2  # 2 clusters attendus
```

**Validation étape 5** :
- [ ] `python -m prc.running.verdict --phase test_poc` fonctionne
- [ ] Profils gammas calculés
- [ ] Clustering détecté
- [ ] Rapport texte généré
- [ ] Tests verdict_lite passent

**Durée estimée** : 3-4 jours

---

## 🎉 CHECKPOINT POC (ÉTAPES 0-5)

**Durée totale** : 2-3 semaines

**Livrables** :
- ✅ Structure dossiers running/ + batch.py
- ✅ Featuring_lite (5 features)
- ✅ Running_lite (1 composition)
- ✅ Batch_lite (10 compositions YAML)
- ✅ Verdict_lite (aggregation + clustering)
- ✅ Tests intégration passent
- ✅ Parquet write/read fonctionne

**Validation utilisateur** :
- [ ] Architecture running/ validée
- [ ] Flux bout-en-bout fonctionne (batch → verdict)
- [ ] Parquet format OK
- [ ] Performance acceptable (10 runs < 1min)

**Décision GO/NO-GO** :
- ✅ GO → Étapes 6-8 enrichissement
- ❌ NO-GO → Refactor architecture

---

## ÉTAPE 6 : ENRICHISSEMENT FEATURING

### Objectif
Passer de featuring_lite (5 features) → featuring_full (~150 features)

### Actions

**6.1 Registries complets**

**Nouveaux registries** :
- algebra_registry.py (trace, determinant, condition_number, ...)
- graph_registry.py (density, clustering_coeff, ...)
- spectral_registry.py (eigenvalues, spectral_gap, ...)
- pattern_registry.py (symmetry, sparsity, ...)
- statistical_registry.py (entropy, kurtosis, ...)
- tensor_registry.py (tucker_energy, cp_rank, ...) [si rank 3]
- spatial_registry.py (gradient, connected_components, ...) [si 2D]

**Processus** : Chaque registry ajouté individuellement
1. Lire refactor.md annexes (référence)
2. Algo → Structure → Code
3. Tests unitaires
4. Validation utilisateur

**6.2 Layers multiples**

**Enrichissement layers.py** :
```python
def get_layer(history: np.ndarray) -> str:
    """
    Inspection directe history.
    
    Returns:
        - 'universal' : Tous tenseurs
        - 'matrix_2d' : Rank 2
        - 'matrix_square' : Rank 2 carrée
        - 'tensor_3d' : Rank ≥3
        - 'spatial_2d' : Analyses spatiales 2D
    """
    rank = len(history.shape) - 1  # -1 pour dimension temps
    dims = history.shape[1:]
    
    if rank == 2 and dims[0] == dims[1]:
        return 'matrix_square'
    elif rank == 2:
        return 'matrix_2d'
    elif rank >= 3:
        return 'tensor_3d'
    else:
        return 'universal'
```

**6.3 Projections temporelles**

**Enrichissement projections.py** :
```python
def apply_projections(history: np.ndarray, 
                      projections: list = ['initial', 'final', 'mean', 
                                           'max', 'min', 'max_deviation']
                     ) -> dict:
    """
    Applique projections temporelles.
    
    Args:
        history : np.ndarray (T, *dims)
        projections : list[str] - Projections à calculer
    
    Returns:
        dict : {
            'initial': state_t0,
            'final': state_tT,
            'mean': mean_timeline,
            ...
        }
    """
    pass
```

**6.4 Régimes/Timelines intra-run**

**Nouveaux modules** :
- featuring/regimes.py
- featuring/timelines.py

**regimes.py** :
```python
def classify_regime(features: dict, thresholds: dict) -> str:
    """
    Classifie régime observation.
    
    Args:
        features : dict - Features calculées
        thresholds : dict - Seuils YAML
    
    Returns:
        str : Régime
            - 'CONSERVES_NORM'
            - 'PATHOLOGICAL'
            - 'NUMERIC_INSTABILITY'
            - 'UNKNOWN'
    
    Logic:
        if frobenius_norm_final / frobenius_norm_initial < threshold:
            return 'CONSERVES_NORM'
        elif any NaN in features:
            return 'NUMERIC_INSTABILITY'
        ...
    """
    pass
```

**timelines.py** :
```python
def build_timeline_descriptor(events: dict) -> str:
    """
    Construit descriptor timeline depuis events.
    
    Args:
        events : dict - Dynamic events
            {
                'deviation_detected': 1,
                'deviation_time': 0.2,
                'saturation_detected': 1,
                'saturation_time': 0.8,
                ...
            }
    
    Returns:
        str : Timeline descriptor
            - 'stable'
            - 'early_deviation_then_saturation'
            - 'collapse_immediate'
            - ...
    """
    pass
```

**6.5 Dynamic events**

**Nouveau module** : featuring/dynamic_events.py

```python
def detect_dynamic_events(history: np.ndarray, 
                          thresholds: dict) -> dict:
    """
    Détecte events timeline.
    
    Args:
        history : np.ndarray (T, *dims)
        thresholds : dict - Seuils YAML
    
    Returns:
        dict : {
            'deviation_detected': int (0/1),
            'deviation_time': float (temps normalisé [0,1]),
            'saturation_detected': int,
            'saturation_time': float,
            'collapse_detected': int,
            'collapse_time': float,
            'instability_detected': int,
            'oscillation_detected': int
        }
    """
    pass
```

**6.6 Intégration hub**

**Enrichissement hub.py** :
```python
def extract_features_full(history: np.ndarray, config: dict) -> dict:
    """
    Extraction complète features (~150).
    
    Process:
        1. Get layer (universal, matrix_2d, ...)
        2. Load registries applicables
        3. Apply projections (initial, final, mean, ...)
        4. Compute features layer × projection × registry
        5. Detect dynamic events
        6. Classify regime
        7. Build timeline descriptor
        8. Return dict ~150 features
    """
    pass
```

**Validation étape 6** :
- [ ] ~150 features calculées
- [ ] Registries multiples fonctionnent
- [ ] Layers inspection correcte
- [ ] Régimes/Timelines calculés
- [ ] Tests featuring_full passent
- [ ] Pas de régression performance (<5s par run)

**Durée estimée** : 2-3 semaines

---

## ÉTAPE 7 : ENRICHISSEMENT RUNNING

### Objectif
Discovery axes + dry-run avancé + axes temporaires

### Actions

**7.1 Discovery axes**

**Enrichissement compositions.py** :
```python
def discover_atomics(config: dict) -> dict:
    """
    Discovery automatique atomics disponibles.
    
    Args:
        config : dict - Config YAML
            {
                'iteration_axes': {
                    'gamma_id': 'all',  # Discovery auto
                    'd_encoding_id': ['SYM-001', 'SYM-002'],  # Liste explicite
                    'modifier_id': {'random': 3}  # Échantillon 3
                }
            }
    
    Returns:
        dict : Atomics découverts
            {
                'gamma_id': ['GAM-001', 'GAM-002', ...],
                'd_encoding_id': ['SYM-001', 'SYM-002'],
                'modifier_id': ['M1', 'M2', 'M3']
            }
    """
    pass
```

**7.2 Dry-run avancé**

**Enrichissement hub.py dry-run** :
```python
def dry_run_estimate_advanced(compositions: list, 
                               config: dict) -> dict:
    """
    Estimation avancée batch.
    
    Process:
        1. Test run 1 composition (benchmark)
        2. Estimer temps = n_compositions × temps_moyen
        3. Estimer RAM = max(history_size, features_size)
        4. Estimer Parquet size = n_compositions × observation_size
    
    Returns:
        dict : {
            'n_compositions': int,
            'estimated_time_seconds': float,
            'estimated_time_human': str,  # "2h 30min"
            'estimated_ram_mb': float,
            'estimated_parquet_mb': float,
            'benchmark_run_seconds': float
        }
    """
    pass
```

**7.3 Axes temporaires**

**Config YAML enrichi** :
```yaml
# configs/phases/test/exploration.yaml
phase: test_exploration
iteration_axes:
  gamma_id: all
  d_encoding_id: all
  modifier_id: all
  seed:  # Axe temporaire
    - 42
    - 123
    - 456
  DOF:  # Axe temporaire (exploration)
    - 10
    - 20
```

**Gestion axes temporaires** :
- Colonnes Parquet normales
- Marqueur config "axes_temporaires" (documentation)
- Suppression manuelle après calibration

**7.4 Validation compositions invalides**

**Enrichissement compositions.py** :
```python
def validate_composition(composition: dict, 
                         discovery: dict) -> bool:
    """
    Valide composition (atomics existent).
    
    Args:
        composition : dict
        discovery : dict - Atomics disponibles
    
    Returns:
        bool : True si valide, False sinon
    
    Logic:
        - Vérifier gamma_id existe fichier
        - Vérifier d_encoding_id existe
        - Vérifier modifier_id existe (si présent)
        - Ignorer compositions invalides (pas raise)
    """
    pass
```

**Validation étape 7** :
- [ ] Discovery `all` fonctionne
- [ ] Discovery `random: N` fonctionne
- [ ] Dry-run estimation avancée
- [ ] Axes temporaires gérés
- [ ] Compositions invalides ignorées naturellement
- [ ] Tests running_full passent

**Durée estimée** : 1-2 semaines

---

## ÉTAPE 8 : ENRICHISSEMENT VERDICT

### Objectif
Profiling régimes/timelines + Analysing concordance + Rapports détaillés

### Actions

**8.1 Profiling régimes**

**Enrichissement profiling/regimes.py** :
```python
def analyze_regime_distribution(observations: pd.DataFrame) -> dict:
    """
    Analyse distribution régimes par gamma.
    
    Returns:
        dict : {
            'GAM-001': {
                'CONSERVES_NORM': 0.8,
                'PATHOLOGICAL': 0.2
            },
            ...
        }
    """
    pass
```

**8.2 Profiling timelines**

**Enrichissement profiling/timelines.py** :
```python
def analyze_timeline_frequencies(observations: pd.DataFrame) -> dict:
    """
    Analyse fréquences timelines par gamma.
    
    Returns:
        dict : {
            'GAM-001': {
                'stable': 0.6,
                'early_deviation_then_saturation': 0.3,
                ...
            }
        }
    """
    pass
```

**8.3 Analysing concordance**

**Nouveau module** : analysing/concordance.py

```python
def compute_regime_concordance(observations_r0: pd.DataFrame,
                                observations_r1: pd.DataFrame) -> dict:
    """
    Concordance régimes R0 ↔ R1.
    
    Returns:
        dict : {
            'kappa': float,  # Cohen's kappa
            'agreement_rate': float,
            'confusion_matrix': np.ndarray
        }
    """
    pass

def compute_timeline_concordance(observations_r0: pd.DataFrame,
                                  observations_r1: pd.DataFrame) -> dict:
    """
    Concordance timelines R0 ↔ R1 (DTW).
    
    Returns:
        dict : {
            'dtw_distance_mean': float,
            'dtw_distance_std': float
        }
    """
    pass
```

**8.4 Analysing outliers/variance**

**Nouveaux modules** :
- analysing/outliers.py
- analysing/variance.py

**outliers.py** :
```python
from sklearn.ensemble import IsolationForest

def detect_outliers(observations: pd.DataFrame, 
                    features: list) -> np.ndarray:
    """
    Détection outliers IsolationForest.
    
    Returns:
        np.ndarray : Labels (1 = normal, -1 = outlier)
    """
    pass
```

**variance.py** :
```python
def compute_eta_squared(observations: pd.DataFrame, 
                        feature: str, 
                        factor: str) -> float:
    """
    Variance expliquée η² (ANOVA).
    
    Args:
        feature : str - Feature dépendante
        factor : str - Facteur (gamma_id, d_encoding_id, ...)
    
    Returns:
        float : η² [0, 1]
    """
    pass
```

**8.5 Rapports détaillés**

**Enrichissement verdict.py** :
```python
def generate_report(phase: str, 
                    profiles: pd.DataFrame,
                    clustering: dict,
                    outliers: dict,
                    variance: dict) -> str:
    """
    Génère rapport synthesis détaillé.
    
    Returns:
        str : Chemin fichier rapport Markdown
    """
    pass
```

**Validation étape 8** :
- [ ] Régimes/Timelines profiling calculés
- [ ] Concordance cross-phases fonctionne
- [ ] Outliers détectés
- [ ] Variance η² calculée
- [ ] Rapport détaillé généré
- [ ] Tests verdict_full passent

**Durée estimée** : 2-3 semaines

---

## 🎉 CHECKPOINT ENRICHISSEMENT (ÉTAPES 6-8)

**Durée totale** : 5-8 semaines (depuis POC)

**Livrables** :
- ✅ Featuring_full (~150 features)
- ✅ Running_full (discovery, dry-run avancé)
- ✅ Verdict_full (profiling, analysing complet)
- ✅ Régimes/Timelines intra-run
- ✅ Concordance cross-phases
- ✅ Rapports détaillés

**Validation utilisateur** :
- [ ] Features complètes validées
- [ ] Discovery axes fonctionne
- [ ] Concordance cross-phases OK
- [ ] Performance acceptable (150 features <10s par run)

**Décision GO/NO-GO** :
- ✅ GO → Étapes 9-10 migration/optimisation
- ❌ NO-GO → Debug/refactor modules

---

## ÉTAPE 9 : MIGRATION LEGACY

### Objectif
Migrer DBs existantes (SQLite R0_results.db, R1_results.db) → Parquet

### Actions

**9.1 Script migration**

**Nouveau fichier** : scripts/migrate_legacy_to_parquet.py

```python
"""
Migration legacy SQLite → Parquet.

Usage:
    python scripts/migrate_legacy_to_parquet.py \
        --legacy prc_databases/R0_results.db \
        --output data/results/R0.parquet \
        --phase R0
"""
import sqlite3
import pandas as pd
import argparse

def migrate_db(legacy_path: str, output_path: str, phase: str):
    """
    Migre SQLite → Parquet.
    
    Process:
        1. Load observations SQLite
        2. Rename colonnes (si mapping nécessaire)
        3. Add colonne phase
        4. Write Parquet
    """
    conn = sqlite3.connect(legacy_path)
    df = pd.read_sql('SELECT * FROM observations', conn)
    conn.close()
    
    # Add phase
    df['phase'] = phase
    
    # Write Parquet
    df.to_parquet(output_path, compression='snappy', index=False)
    
    print(f"Migrated {len(df)} observations → {output_path}")
```

**9.2 Tests comparaison pipelines**

**Nouveau fichier** : tests/test_compare_pipelines.py

```python
def compare_features_correlation(legacy_df: pd.DataFrame,
                                  new_df: pd.DataFrame,
                                  features: list) -> dict:
    """
    Compare features legacy vs nouveau pipeline.
    
    Returns:
        dict : {
            'feature_name': correlation,
            ...
        }
    """
    pass

def compare_regimes_agreement(legacy_df: pd.DataFrame,
                               new_df: pd.DataFrame) -> float:
    """
    Compare régimes legacy vs nouveau.
    
    Returns:
        float : Agreement rate [0, 1]
    """
    pass
```

**9.3 Validation résultats**

**Process** :
1. Migrer R0_results.db → R0.parquet
2. Run nouveau pipeline phase R0 (mêmes compositions)
3. Comparer features correlation (attendu >0.9)
4. Comparer régimes agreement (attendu >0.8)
5. Documenter différences

**Validation étape 9** :
- [ ] Script migration fonctionne
- [ ] R0, R1, R2 migrés Parquet
- [ ] Tests comparaison passent
- [ ] Features correlation >0.9
- [ ] Régimes agreement >0.8
- [ ] Différences documentées

**Durée estimée** : 1 semaine

---

## ÉTAPE 10 : OPTIMISATION

### Objectif
Benchmarks performance + optimisations critiques + documentation finale

### Actions

**10.1 Benchmarks performance**

**Nouveau fichier** : tests/benchmarks/bench_full_pipeline.py

```python
def benchmark_featuring(n_runs: int = 100):
    """Benchmark extraction features."""
    pass

def benchmark_batch_runner(n_compositions: int = 1000):
    """Benchmark batch complet."""
    pass

def benchmark_verdict(phase: str = 'R0'):
    """Benchmark verdict analyses."""
    pass

def benchmark_parquet_io():
    """Benchmark write/read Parquet."""
    pass
```

**Métriques cibles** :
- Featuring : <10s par run (150 features)
- Batch 1000 runs : <3h
- Verdict R0 (10k obs) : <5min
- Parquet write : <2s
- Parquet read (charge partielle) : <0.5s

**10.2 Optimisations critiques**

**Candidats optimisation** (si benchmarks révèlent) :
- Featuring : Parallélisation calculs registries
- Batch : Multiprocessing compositions
- Verdict : Cache aggregations
- Parquet : Compression optimale (snappy vs gzip)

**Process** :
1. Identifier bottlenecks (profiling)
2. Algo → Structure → Code optimisation
3. Benchmarks avant/après
4. Validation utilisateur gains

**10.3 Documentation finale**

**Mise à jour documents** :
- Charter 7.0 Sections 2.1-2.3, 4.6
- README.md (usage batch.py, verdict)
- Catalogues (featuring, profiling, analysing)

**Nouveaux documents** :
- GUIDE_UTILISATEUR.md
- TROUBLESHOOTING.md
- CHANGELOG.md

**Validation étape 10** :
- [ ] Benchmarks exécutés
- [ ] Optimisations validées
- [ ] Documentation mise à jour
- [ ] Charter 7.0 synchronisé

**Durée estimée** : 1-2 semaines

---

## 🎉 LIVRAISON FINALE

**Durée totale projet** : 8-13 semaines

**Livrables complets** :
- ✅ Architecture running/ + batch.py
- ✅ Featuring_full (~150 features)
- ✅ Régimes/Timelines intra-run
- ✅ Parquet migration complète
- ✅ Discovery axes YAML
- ✅ Dry-run avancé
- ✅ Verdict concordance cross-phases
- ✅ Migration legacy validée
- ✅ Benchmarks performance
- ✅ Documentation complète

**Métriques succès** :
- ✅ Features correlation legacy/new >0.9
- ✅ Régimes agreement >0.8
- ✅ Performance <10s par run
- ✅ Volumétrie -71% (Parquet vs SQL)
- ✅ Tests 100% passent
- ✅ Zéro dépendances circulaires

---

## 📋 CHECKLIST VALIDATION UTILISATEUR

### Après POC (Étape 5)
- [ ] Architecture running/ validée
- [ ] Flux batch → verdict fonctionne
- [ ] Parquet format acceptable
- [ ] Performance POC OK (<1min 10 runs)
- [ ] **DÉCISION** : GO enrichissement

### Après enrichissement (Étape 8)
- [ ] 150 features validées
- [ ] Discovery axes fonctionne
- [ ] Régimes/Timelines cohérents
- [ ] Concordance cross-phases OK
- [ ] Performance full OK (<10s par run)
- [ ] **DÉCISION** : GO migration

### Après migration (Étape 9)
- [ ] Migration legacy réussie
- [ ] Comparaison pipelines acceptable
- [ ] Différences documentées
- [ ] **DÉCISION** : GO optimisation

### Livraison finale (Étape 10)
- [ ] Benchmarks validés
- [ ] Documentation complète
- [ ] Charter 7.0 synchronisé
- [ ] Tests 100% passent
- [ ] **DÉCISION** : Production ready

---

## 📝 NOTES MÉTHODOLOGIE

### Principe versions_lite

**Objectif** : Tester architecture AVANT enrichir fonctionnalités

**Règles** :
1. Toujours commencer version_lite (fonctionnalités minimales)
2. Tester intégration bout-en-bout
3. Valider architecture avec utilisateur
4. Enrichir progressivement (pas refactor massif)

**Exemple featuring** :
- featuring_lite : 5 features, 1 registry, 1 layer
- featuring_full : 150 features, 10 registries, 5 layers

**Avantages** :
- Tests rapides (<1min vs 10min)
- Bugs détectés tôt (architecture, pas détails)
- Validation utilisateur fréquente
- Réduction risque refactor massif raté

### Principe Algo → Structure → Code

**Rappel Charter Section 3.1** :

1. **ALGO** : Langage courant, logique métier, zéro code
2. **STRUCTURE** : Squelette fonction, passages I/O ancrés existant
3. **CODE** : Implémentation après validation structure

**Validation utilisateur obligatoire** chaque étape

**Application refactor** :
- Chaque module : Algo → Structure → Code
- Chaque registry : Algo → Structure → Code
- Validation utilisateur après chaque module majeur

### Principe tests après chaque étape

**Règles tests** :
- Tests unitaires chaque module
- Tests intégration chaque étape majeure
- Benchmarks après enrichissement
- Comparaison pipelines après migration

**Pas de code sans tests** : Test-Driven Development (TDD) recommandé

---

**FIN PLAN IMPLÉMENTATION**

Ce plan structure implémentation refactor combiné en 10 étapes progressives.  
Approche proof-of-concept → enrichissement minimise risque.  
Validation utilisateur fréquente garantit alignement philos