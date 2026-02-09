# 📋 TODO R1 - GUIDE COMPLET IMPLÉMENTATION

## 🎯 STRUCTURE DU DOCUMENT
```
┌─────────────────────────────────────────────────┐
│ PHASE 0 : PRÉPARATION (Semaine 0)              │
│ ├─ 0.1 : Validation méthodologique             │
│ ├─ 0.2 : Analyse variance seeds                │
│ └─ 0.3 : Découverte gammas R0                  │
├─────────────────────────────────────────────────┤
│ PHASE R1.1 : COMPOSITION ROBUSTE (S1-S6)       │
│ ├─ 1.1 : Architecture unifiée                  │
│ ├─ 1.2 : Factory compositions dynamiques       │
│ ├─ 1.3 : Injection batch_runner                │
│ ├─ 1.4 : Tests validation                      │
│ └─ 1.5 : Rapport R1.1 + Go/No-Go               │
├─────────────────────────────────────────────────┤
│ PHASE R1.4 : CONSOLIDATION (S7-S8)             │
│ ├─ 4.1 : Migration UTIL/HUB                    │
│ ├─ 4.2 : Validation pipeline unifié            │
│ ├─ 4.3 : Nettoyage fichiers temporaires        │
│ └─ 4.4 : Documentation finale                  │
└─────────────────────────────────────────────────┘
```

**PRINCIPE CENTRAL** : Réutilisation batch_runner.py existant via injection de "fichiers virtuels" (gammas composés générés dynamiquement).

# PHASE 0 : PRÉPARATION

## 📌 ÉTAPE 0.1 : VALIDATION MÉTHODOLOGIQUE

### 🎯 Objectif
Valider conformité approche R1 avec Charter + Checklist anti-glissement

### 📥 Contexte minimal requis
```yaml
documents_requis:
  - charter_6_1.md (Section 4.5, 12)
  - checklist_anti_glissement_r1.md (complet)
  - feuille_de_route_r1.md (Section 2.1 formulation canonique)
```

### ✅ Tâches

#### 0.1.1 - Relecture Checklist anti-glissement
```markdown
**Action** : Lire intégralement checklist_anti_glissement_r1.md

**Vérifications critiques** :
- [ ] Glissement 1 (VALIDATION vs COMPATIBILITÉ) compris
- [ ] Glissement 2 (PROPRIÉTÉS ALGÉBRIQUES vs OBSERVABLES FAIBLES) compris
- [ ] Glissement 3 (UNICITÉ REQUISE vs RÉDUCTION SUFFISANTE) compris

**Formulations interdites identifiées** :
- [ ] "Les explosions valident le survival bias"
- [ ] "Tester la commutativité de Γ"
- [ ] "L'objectif est d'identifier LE Γ universel unique"

**Formulations autorisées adoptées** :
- [ ] "Les explosions sont compatibles avec le survival bias"
- [ ] "Mesurer l'indépendance des invariants à l'ordre"
- [ ] "R1 vise à déterminer si les contraintes réduisent l'espace candidat"

**Output** : Validation écrite (1 paragraphe confirmant compréhension)
```

#### 0.1.2 - Adoption formulation canonique R0→R1
```markdown
**Formulation officielle à utiliser SYSTÉMATIQUEMENT** :

> R1 explore si les critères nécessaires extraits de R0 sont compatibles 
> avec une composabilité finie non triviale, en testant la préservation, 
> la dégradation ou la compensation de ces propriétés sous séquences 
> contrôlées de transformations Γ, sans présupposer ni convergence 
> asymptotique, ni unicité de l'opérateur sous-jacent.

**Action** : Copier cette formulation dans document de travail

**Utilisation** :
- Toute communication R1 (rapports, présentations, papier)
- Introduction modules code (docstrings)
- Commit messages Git
```

#### 0.1.3 - Validation issues multiples
```markdown
**Les TROIS issues sont scientifiquement valides** :

**Issue A (Réduction réussie)** :
- A1 (fort) : Unique Γ ou classe d'équivalence étroite
- A2 (partiel) : Famille paramétrée {Γ(α,β,...)} ou topologique {Γ_D}
- A3 (partiel) : Noyau Γ_core + extensions Γ_ctx(D)

**Issue B (Dégradation)** :
- Invariants préservés n=2..3, dégradent n>3
- Limites composition révélées, contraintes raffinées

**Issue C (Incompatibilité)** :
- Aucune composition préserve invariants R0
- Tensions structurelles, révision hypothèses nécessaire

**Vérification** :
- [ ] Aucune issue ne définit "échec du framework"
- [ ] Toutes issues produisent connaissance scientifique
- [ ] Succès = réduction OU identification limites, pas uniquement unicité

**Output** : Document interne listant les 3 issues avec interprétations
```

# PHASE R1.1 : COMPOSITION ROBUSTE

## 📌 ÉTAPE 1.1 : ARCHITECTURE UNIFIÉE

### 🎯 Objectif
Concevoir architecture injection compositions dans batch_runner SANS modification

### 📥 Contexte minimal requis
```yaml
documents_requis:
  - charter_6_1.md (Section 2, 4.5)
  - functions_index.md (batch_runner.py structure)

sources_code:
  - prc_automation/batch_runner.py (complet)
  - tests/utilities/utils/discovery.py (discover_active_tests)
```

### 📚 Principe "Fichiers Virtuels"

**PROBLÈME** :
batch_runner.py découvre gammas via filesystem (operators/gamma_hyp_*.py).
Compositions = séquences [Γ₁, Γ₂, ...] → pas fichiers statiques.

**SOLUTION** :
Injection dynamique "fichiers virtuels" avant discovery :
1. Générer classe ComposedGamma (factory pattern)
2. Créer instances en mémoire (pas fichiers .py)
3. Injecter dans namespace discovery avant appel batch_runner
4. batch_runner voit compositions comme gammas standards

**ANALOGIE** :
Similaire mock filesystem tests unitaires, mais production.

**POINT D'INJECTION PRÉCIS** :
`tests/utilities/utils/data_loading.py` → fonction `discover_entities()`
- Modification minimale (~20 lignes)
- Isolation changement (1 fichier UTIL)
- IF phase=='R1' → injection, ELSE → standard R0

### ✅ Tâches

#### 1.1.1 - Spécification interface ComposedGamma
```python
# Spec: Interface ComposedGamma (architecture uniquement)

"""
Interface ComposedGamma compatible batch_runner.py.

CONFORMITÉ:
- Signature identique gamma standard (Charter Section 1)
- Aveuglement core préservé (pas connaissance composition)
- Traçabilité complète (sequence_gammas stocké metadata)
"""

INTERFACE:

class ComposedGamma:
    """
    Gamma composite = séquence [Γ₁, Γ₂, ...] appliquée séquentiellement.
    
    AVEUGLEMENT CORE:
    - Core ne connaît que __call__(state) → state
    - Métadonnées composition stockées séparément (db_raw)
    - Tests ne voient que history finale (transparence)
    """
    
    def __init__(self, sequence_gammas: List[GammaClass], sequence_id: str):
        """
        Args:
            sequence_gammas: Liste instances gamma [Γ₁, Γ₂, ...]
            sequence_id: ID unique composition (ex: "comp_GAM-001_GAM-002")
        
        Raises:
            AssertionError: Si sequence vide ou gammas incompatibles
        """
        VALIDATION:
        - sequence_gammas non vide
        - Tous gammas même applicabilité (SYM, ASY, R3)
        - sequence_id format valide (comp_XXX_YYY_...)
        
        STORAGE:
        - self.sequence_gammas = sequence_gammas
        - self.sequence_id = sequence_id
        - self.n = len(sequence_gammas)
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique séquence Γ₁ → Γ₂ → ... → Γₙ.
        
        AVEUGLEMENT:
        - Pas de validation contenu state
        - Pas de branchement dépendant D
        - Itération aveugle séquentielle
        
        Returns:
            state_final après n applications
        """
        ALGORITHME:
        state_current = state
        FOR gamma IN self.sequence_gammas:
            state_current = gamma(state_current)
        RETURN state_current
    
    def __repr__(self) -> str:
        """Représentation string."""
        RETURN f"ComposedGamma({self.sequence_id})"
    
    def reset(self):
        """Reset mémoire (si gammas non-markoviens)."""
        FOR gamma IN self.sequence_gammas:
            IF hasattr(gamma, 'reset'):
                gamma.reset()
    
    # METADATA (class attributes, compatibilité discovery)
    METADATA = {
        'type': 'composed',
        'phase': 'R1',
        'description': 'Composition séquentielle gammas'
    }
    PHASE = 'R1'
```

**Validation interface** :
- [ ] Signature compatible gamma standard (Charter Section 1)
- [ ] Aveuglement core préservé (pas validation D)
- [ ] Métadonnées traçabilité présentes
- [ ] reset() présent (compatibilité non-markoviens)

#### 1.1.2 - Schéma injection data_loading.py
```python
# Spec: Schéma injection (architecture uniquement)

"""
Point d'injection compositions dans data_loading.py.

LOCALISATION:
Fonction data_loading.py : discover_entities('gamma', phase)

MODIFICATION MINIMALE (+20 lignes):
SI phase == 'R1' AND entity_type == 'gamma':
    1. Discovery filesystem standard (gamma_hyp_*.py)
    2. Injection compositions (fichiers virtuels)
    3. Fusion listes (gammas R0 + compositions R1)
ELSE:
    Discovery standard (R0 inchangé)
"""

PSEUDO-CODE INJECTION:

def discover_entities(entity_type, phase):
    """Discovery avec injection R1."""
    
    # Standard discovery (R0, R1 tous)
    entities_standard = _discover_filesystem(entity_type, phase)
    
    # SI gamma ET phase R1: Injection compositions
    IF entity_type == 'gamma' AND phase == 'R1':
        
        from .composed_gamma_factory import ComposedGammaFactory
        
        factory = ComposedGammaFactory()
        compositions = factory.generate_all_compositions(n=2)
        
        # Fusion
        entities_final = entities_standard + compositions
        RETURN entities_final
    ELSE:
        RETURN entities_standard

```

**Validation schéma** :
- [ ] Injection isolée (IF phase == 'R1')
- [ ] R0 inchangé (ELSE standard)
- [ ] Compositions format discovery standard
- [ ] Import local (pas global, évite dépendance R0)

#### 1.1.3 - Architecture persistance DB
```sql
-- Spec: Schéma DB séquences (architecture uniquement)

/*
Extension DB R1 : Tables séquences compositions.

CONFORMITÉ:
- db_raw immuable (append-only)
- Séparation db_r1_raw.db (pas modification db_r0_raw.db)
- Traçabilité complète (sequence_gammas, sequence_id)
*/

-- Table séquences (prc_r1_raw.db)
CREATE TABLE IF NOT EXISTS sequences (
    sequence_exec_id TEXT PRIMARY KEY,
    phase TEXT NOT NULL,  -- 'R1'
    sequence_gammas TEXT NOT NULL,  -- JSON: ["GAM-001", "GAM-002", ...]
    sequence_length INTEGER NOT NULL,  -- n gammas
    d_encoding_id TEXT NOT NULL,
    modifier_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,  -- 'SUCCESS', 'ERROR'
    timestamp TEXT NOT NULL,
    params_config_id TEXT NOT NULL
);

-- Table snapshots séquences (états intermédiaires)
CREATE TABLE IF NOT EXISTS snapshots_sequences (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_exec_id TEXT NOT NULL,
    gamma_index INTEGER NOT NULL,  -- Position dans séquence (0, 1, ...)
    gamma_id TEXT NOT NULL,  -- Gamma appliqué
    iteration INTEGER NOT NULL,  -- Itération kernel (0..N_ITERATIONS)
    snapshot BLOB NOT NULL,  -- Snapshot compressé (gzip+pickle)
    FOREIGN KEY (sequence_exec_id) REFERENCES sequences(sequence_exec_id)
);

-- Index performance
CREATE INDEX IF NOT EXISTS idx_sequences_phase 
    ON sequences(phase);
CREATE INDEX IF NOT EXISTS idx_sequences_gammas 
    ON sequences(sequence_gammas);
CREATE INDEX IF NOT EXISTS idx_snapshots_seq_exec 
    ON snapshots_sequences(sequence_exec_id);

/*
NOTES:
- sequence_gammas: JSON array gammas IDs (ex: ["GAM-001", "GAM-002"])
- snapshots_sequences: États intermédiaires CHAQUE gamma (pas uniquement final)
- Permet analyse préservation par étape composition
*/
```

**Validation schéma DB** :
- [ ] Tables séparées (prc_r1_raw.db vs prc_r0_raw.db)
- [ ] Traçabilité complète (sequence_gammas, gamma_index)
- [ ] Snapshots intermédiaires stockés (analyse préservation)
- [ ] Index performance présents

### 🔒 Conformité Charter

**Aveuglement core** (Charter Section 1) :
- ✅ ComposedGamma signature identique gamma standard
- ✅ __call__(state) → state (pas validation D)
- ✅ Pas branchement composition-aware

**Séparation stricte** (Charter Section 2) :
- ✅ Composition logic HORS core (utilities/)
- ✅ Factory pattern (utilities/UTIL)
- ✅ DB séparées (prc_r1_raw.db)

**Délégation** (Charter Section 4) :
- ✅ Pas modification batch_runner.py (injection externe)
- ✅ Réutilisation discovery existant (extension phase='R1')
- ✅ Pas duplication code

### 📦 Livrables

```
docs/architecture/
├── composed_gamma_interface.md
├── injection_schema.md
└── db_schema_r1.sql

outputs/phase0/
└── architecture_validation.txt (checkboxes complétées)
```

### ⏱️ Durée estimée
**4 heures** (spécification interfaces + schéma injection + DB)

## 📌 ÉTAPE 1.2 : FACTORY COMPOSITIONS DYNAMIQUES

### 🎯 Objectif
Implémenter factory génération compositions + validation applicabilité

### 📥 Contexte minimal requis
```yaml
documents_requis:
  - charter_6_1.md (Section 4.5)
  - outputs/phase0/gammas_selected_r1_1.json

sources_code:
  - tests/utilities/utils/applicability.py (check validator)
  - operators/gamma_catalog.md (metadata gammas)
```

### ✅ Tâches

#### 1.2.1 - Génération séquences n=2, n=3
```python
# Module: tests/utilities/utils/sequence_generator.py

"""
Générateur séquences compositions (n=2, n=3).

CONFORMITÉ:
- Pas hardcoding gammas (découverte JSON phase 0)
- Validation applicabilité (délégation applicability.py)
- Permutations avec répétition autorisées
"""

ALGORITHME:

def generate_sequences(gamma_ids: List[str], n: int, 
                       allow_repetition: bool = True) -> List[List[str]]:
    """
    Génère toutes séquences longueur n.
    
    Args:
        gamma_ids: Liste gammas candidats (ex: ['GAM-001', 'GAM-002'])
        n: Longueur séquence (2 ou 3)
        allow_repetition: Si True, [A,A] autorisé
    
    Returns:
        Liste séquences (ex: [['GAM-001', 'GAM-002'], ...])
    
    EXEMPLE:
    gamma_ids = ['A', 'B', 'C']
    n = 2
    allow_repetition = True
    → 9 séquences: [A,A], [A,B], [A,C], [B,A], ..., [C,C]
    
    n = 3
    → 27 séquences (3³)
    """
    
    IF allow_repetition:
        sequences = list(itertools.product(gamma_ids, repeat=n))
    ELSE:
        sequences = list(itertools.permutations(gamma_ids, n))
    
    RETURN [list(seq) for seq in sequences]
```

**Tâches** :
- [ ] Créer `tests/utilities/utils/sequence_generator.py`
- [ ] Implémenter generate_sequences()
- [ ] Tests unitaires (n=2, n=3, with/without repetition)
- [ ] Valider output format (List[List[str]])

#### 1.2.2 - Factory ComposedGamma
```python
# Module: tests/utilities/utils/composed_gamma_factory.py

"""
Factory génération ComposedGamma dynamiques.

CONFORMITÉ:
- Réutilise gamma_hyp_*.py existants (importation dynamique)
- Validation applicabilité (délégation applicability.py)
- Format output compatible discovery
"""

CLASSE PRINCIPALE:

class ComposedGammaFactory:
    """
    Factory création ComposedGamma.
    
    WORKFLOW:
    1. Charger gammas candidats (JSON phase 0)
    2. Générer séquences (sequence_generator)
    3. Créer instances ComposedGamma
    4. Validation applicabilité
    5. Format discovery
    """
    
    def __init__(self, config_path: Path = None):
        """
        Args:
            config_path: Chemin gammas_selected_r1_1.json
                         Si None: outputs/phase0/gammas_selected_r1_1.json
        """
        IF config_path IS None:
            config_path = Path("outputs/phase0/gammas_selected_r1_1.json")
        
        self.config = self._load_config(config_path)
        self.gamma_ids = self.config['gamma_ids']
    
    def generate_all_compositions(self, n: int = 2) -> List[dict]:
        """
        Génère toutes compositions longueur n.
        
        Args:
            n: Longueur séquence (2 ou 3)
        
        Returns:
            Liste dicts format discovery:
            [
              {
                'id': 'comp_GAM-001_GAM-002',
                'module': ComposedGamma(instance),
                'phase': 'R1',
                'metadata': {
                  'type': 'composed',
                  'sequence_gammas': ['GAM-001', 'GAM-002'],
                  'sequence_length': 2
                }
              },
              ...
            ]
        """
        
        # 1. Générer séquences
        sequences = generate_sequences(self.gamma_ids, n, allow_repetition=True)
        
        # 2. Pour chaque séquence: créer ComposedGamma
        compositions = []
        FOR seq IN sequences:
            
            # a. Charger instances gammas
            gamma_instances = [self._load_gamma(gid) for gid in seq]
            
            # b. Validation applicabilité commune
            IF NOT self._check_applicability_common(gamma_instances):
                CONTINUE  # Skip séquence incompatible
            
            # c. Créer ComposedGamma
            sequence_id = f"comp_{'_'.join(seq)}"
            composed = ComposedGamma(gamma_instances, sequence_id)
            
            # d. Format discovery
            discovery_dict = {
                'id': sequence_id,
                'module': composed,  # Instance directe (pas module file)
                'phase': 'R1',
                'metadata': {
                    'type': 'composed',
                    'sequence_gammas': seq,
                    'sequence_length': n
                }
            }
            
            compositions.append(discovery_dict)
        
        RETURN compositions
    
    def _load_gamma(self, gamma_id: str) -> GammaClass:
        """
        Charge instance gamma depuis gamma_hyp_*.py.
        
        DÉLÉGATION:
        - Importation dynamique module
        - Création instance (paramètres par défaut)
        
        Raises:
            ImportError: Si gamma_hyp_XXX.py introuvable
        """
        module_name = f"operators.gamma_hyp_{gamma_id.split('-')[1]}"
        module = importlib.import_module(module_name)
        
        # Factory fonction (ex: create_gamma_hyp_001)
        factory_func = getattr(module, f"create_{gamma_id.lower().replace('-', '_')}")
        
        # Création instance (paramètres par défaut Phase 1)
        gamma_instance = factory_func(phase=1)  # PARAM_GRID_PHASE1
        
        RETURN gamma_instance
    
    def _check_applicability_common(self, gamma_instances: List[GammaClass]) -> bool:
        """
        Vérifie applicabilité commune gammas séquence.
        
        RÈGLE:
        - Tous gammas doivent avoir applicabilité intersectante
        - Ex: [SYM, ASY] ∩ [SYM, R3] = [SYM] → OK
        - Ex: [SYM] ∩ [R3] = ∅ → REJECT
        
        DÉLÉGATION:
        - Extraction applicabilité depuis gamma.METADATA
        """
        applicabilities = [
            set(g.METADATA.get('applicability', [])) 
            for g in gamma_instances
        ]
        
        common = set.intersection(*applicabilities)
        
        RETURN len(common) > 0
```

**Tâches** :
- [ ] Créer `tests/utilities/utils/composed_gamma_factory.py`
- [ ] Implémenter ComposedGammaFactory
- [ ] Tests unitaires (génération n=2, n=3)
- [ ] Validation applicabilité commune
- [ ] Tester importation dynamique gammas

### 🔒 Conformité Charter

**Délégation** :
- ✅ `generate_sequences()` réutilisé (pas réimplémentation)
- ✅ Importation dynamique gammas (pas hardcoding fichiers)
- ✅ Validation applicabilité déléguée (applicability.py)

**Pas hardcoding** :
- ✅ Gammas candidats chargés JSON (phase 0)
- ✅ Discovery gammas via importlib (pas liste manuelle)
- ✅ Séquences générées algorithmiquement (permutations)

### 📦 Livrables

```
tests/utilities/utils/
├── sequence_generator.py (~50 lignes)
└── composed_gamma_factory.py (~200 lignes)

tests/
└── test_composed_gamma_factory.py (~100 lignes tests)

outputs/phase0/
└── compositions_n2_sample.json (validation preview)
```

### ⏱️ Durée estimée
**6 heures** (factory + tests unitaires + validation)

## 📌 ÉTAPE 1.3 : INJECTION BATCH_RUNNER

### 🎯 Objectif
Injecter compositions dans batch_runner.py via modification minimale data_loading.py

### 📥 Contexte minimal requis
```yaml
sources_code:
  - prc_automation/batch_runner.py (discover_entities call sites)
  - tests/utilities/utils/data_loading.py (discover_entities fonction)
  - tests/utilities/utils/composed_gamma_factory.py (factory implémenté 1.2)
```

### 📚 Stratégie Injection

**POINT D'INJECTION** :
`tests/utilities/utils/data_loading.py` fonction `discover_entities()`

**RATIONALE** :
- batch_runner.py appelle discover_entities() déjà
- data_loading.py = module UTIL (modification autorisée)
- Isolation modification (1 fonction, ~20 lignes ajoutées)
- R0 inchangé (IF phase == 'R1')

### ✅ Tâches

#### 1.3.1 - Modification discover_entities()
```python
# Modification: tests/utilities/utils/data_loading.py

"""
Injection compositions R1 dans discovery.

MODIFICATION MINIMALE:
- IF phase == 'R1' AND entity_type == 'gamma':
    → Injection compositions (factory)
- ELSE:
    → Discovery standard (inchangé)
"""

AVANT (R0 uniquement):

def discover_entities(entity_type: str, phase: str = 'R0') -> List[dict]:
    """Discovery standard filesystem."""
    
    # Scan operators/gamma_hyp_*.py ou tests/test_*.py
    entities = _discover_filesystem(entity_type, phase)
    
    RETURN entities

APRÈS (R0 + R1):

def discover_entities(entity_type: str, phase: str = 'R0') -> List[dict]:
    """
    Discovery avec injection R1.
    
    MODIFICATIONS:
    - SI phase='R1' ET entity_type='gamma': Injection compositions
    - SINON: Discovery standard (R0 inchangé)
    """
    
    # Discovery standard filesystem (R0, R1 gammas statiques)
    entities_standard = _discover_filesystem(entity_type, phase)
    
    # Injection compositions R1
    IF entity_type == 'gamma' AND phase == 'R1':
        
        from .composed_gamma_factory import ComposedGammaFactory
        
        factory = ComposedGammaFactory()
        
        # Générer compositions n=2
        compositions_n2 = factory.generate_all_compositions(n=2)
        
        # (Optionnel) Générer compositions n=3 (échantillon)
        # compositions_n3_sample = factory.generate_all_compositions(n=3)
        # compositions_n3_sample = random.sample(compositions_n3_sample, k=50)
        
        # Fusion listes
        entities_final = entities_standard + compositions_n2  # + compositions_n3_sample
        
        RETURN entities_final
    
    ELSE:
        # R0 inchangé
        RETURN entities_standard
```

**Validation modification** :
- [ ] IF isolé (phase == 'R1')
- [ ] Import local (pas global, éviter dépendance R0)
- [ ] R0 inchangé (ELSE standard)
- [ ] Format retour cohérent (List[dict] discovery)

#### 1.3.2 - Tests validation injection
```python
# Test: tests/test_data_loading_r1_injection.py

"""
Valide injection R1 dans discover_entities().

VÉRIFICATIONS:
1. R0 inchangé (discover_entities('gamma', 'R0'))
2. R1 compositions présentes (discover_entities('gamma', 'R1'))
3. Format discovery cohérent
4. Applicabilité validée
"""

def test_discover_entities_r0_unchanged():
    """Validate R0 inchangé."""
    
    entities_r0 = discover_entities('gamma', phase='R0')
    
    # Tous gammas filesystem standard
    ASSERT: all(
        'comp_' not in entity['id'] 
        for entity in entities_r0
    )
    
    # Pas compositions
    ASSERT: all(
        entity['metadata'].get('type') != 'composed'
        for entity in entities_r0
    )

def test_discover_entities_r1_injection():
    """Validate R1 injection compositions."""
    
    entities_r1 = discover_entities('gamma', phase='R1')
    
    # Compositions présentes
    compositions = [
        e for e in entities_r1 
        if e['metadata'].get('type') == 'composed'
    ]
    
    ASSERT: len(compositions) > 0
    
    # Format discovery cohérent
    FOR comp IN compositions:
        ASSERT: 'id' in comp
        ASSERT: 'module' in comp
        ASSERT: 'phase' in comp
        ASSERT: comp['phase'] == 'R1'
        ASSERT: 'metadata' in comp
        ASSERT: 'sequence_gammas' in comp['metadata']
        ASSERT: 'sequence_length' in comp['metadata']
```

**Tâches** :
- [ ] Modifier `tests/utilities/utils/data_loading.py` (+20 lignes)
- [ ] Créer `tests/test_data_loading_r1_injection.py`
- [ ] Exécuter tests (pytest)
- [ ] Valider R0 inchangé + R1 compositions présentes

### 🔒 Conformité Charter

**Modification minimale** :
- ✅ 1 fichier modifié (data_loading.py)
- ✅ ~20 lignes ajoutées (IF isolé)
- ✅ Import local (pas dépendance globale)

**R0 inchangé** :
- ✅ ELSE standard (découverte filesystem)
- ✅ Tests R0 passent (validation régression)

### 📦 Livrables

```
tests/utilities/utils/
└── data_loading.py (+20 lignes modification)

tests/
└── test_data_loading_r1_injection.py (~150 lignes tests)
```

### ⏱️ Durée estimée
**4 heures** (modification + tests + validation régression R0)

---

## 📌 ÉTAPE 1.4 : TESTS VALIDATION

### 🎯 Objectif
Valider pipeline unifié R0 + R1 via tests minimaux

### 📥 Contexte minimal requis
```yaml
sources_code:
  - prc_automation/batch_runner.py (run_batch_brut, run_batch_test)
  - tests/utilities/utils/data_loading.py (injection implémentée 1.3)
```

### ✅ Tâches

#### 1.4.1 - Tests pipeline R0 + R1
```python
# Test: tests/test_unified_pipeline.py

"""
Valide pipeline unifié R0 + R1.

VÉRIFICATIONS:
- R0 inchangé (régression)
- R1 compositions fonctionnelles
- Format observations cohérent
"""

def test_pipeline_r0_unchanged():
    """Validate batch_runner R0 inchangé."""
    # Exécution minimale R0
    # Validation discovery standard
    # Validation observations standard

def test_pipeline_r1_compositions():
    """Validate batch_runner R1 compositions."""
    # Discovery compositions présentes
    # Exécution séquences
    # Snapshots intermédiaires
    # Observations tests
```

**Tâches** :
- [ ] Créer `tests/test_unified_pipeline.py`
- [ ] Exécuter tests
- [ ] Valider aucune régression

### ⏱️ Durée estimée
**5 heures** (tests + validation)

---

## 📌 ÉTAPE 1.5 : RAPPORT R1.1 + GO/NO-GO

### 🎯 Objectif
Générer rapport synthétique Phase R1.1 + décision Go/No-Go R1.2

### 📥 Contexte minimal requis
```yaml
documents_requis:
  - feuille_de_route_r1.md (Section 3.1 critères succès)
```

### ✅ Tâches

#### 1.5.1 - Modules analyse (UTIL)
```python
# Module: tests/utilities/utils/baseline_computer.py
# Calcule baselines R0 pour comparaison R1

# Module: tests/utilities/utils/composition_classifier.py
# Classifie séquences R1 (robustes/dégradantes/instables)
```

**Tâches** :
- [ ] Créer `baseline_computer.py`
- [ ] Créer `composition_classifier.py`
- [ ] Tests unitaires

#### 1.5.2 - Génération rapport (HUB)
```python
# Module: tests/utilities/HUB/report_generator_r1.py

"""
Génère rapport R1.1 complet.

SECTIONS:
1. Métadonnées
2. Baselines R0
3. Résultats R1 (classification séquences)
4. Comparaisons R0 vs R1
5. Décision Go/No-Go R1.2

CRITÈRES GO/NO-GO (feuille_de_route Section 3.1):
- ≥30% séquences ROBUSTES
- Explosion rate ≤ 2× R0
- Concordance ≥ 95%
"""
```

**Tâches** :
- [ ] Créer `report_generator_r1.py`
- [ ] Générer rapport
- [ ] Lire décision Go/No-Go

### 📦 Livrables

```
tests/utilities/utils/
├── baseline_computer.py (~100 lignes)
└── composition_classifier.py (~80 lignes)

tests/utilities/HUB/
└── report_generator_r1.py (~150 lignes)

reports/r1_1/
└── {timestamp}_report_r1_1/
    ├── summary_r1_1.txt
    └── report_r1_1.json
```

### ⏱️ Durée estimée
**6 heures** (baselines + classification + rapport)

---

**[FIN PHASE R1.1 COMPOSITION ROBUSTE]**

**JALONS ATTEINTS** :
- ✅ Architecture unifiée conçue
- ✅ Factory compositions implémentée
- ✅ Injection batch_runner fonctionnelle
- ✅ Tests validation pipeline R0 + R1
- ✅ Rapport R1.1 + décision Go/No-Go

**RÉSULTAT** :
- ✅ Pipeline unifié (batch_runner --phase R0|R1)
- ✅ Génération dynamique compositions (pas fichiers statiques)
- ✅ 0% modification batch_runner.py
- ✅ +20 lignes data_loading.py (injection)
- ✅ +700 lignes UTIL/HUB (réutilisables)

**DURÉE TOTALE PHASE R1.1** : 29 heures

---

# PHASE R1.4 : CONSOLIDATION

---

## 📌 ÉTAPE 4.1 : MIGRATION UTIL/HUB

### 🎯 Objectif
Migrer scripts R1 temporaires vers modules UTIL/HUB pérennes

### 📥 Contexte minimal requis
```yaml
sources_code:
  - scripts/r0_analysis/*.py (extraction, sélection)
  - scripts/r1_execution/*.py (estimation, validation)
  - tests/utilities/ (cible migration)
```

### 📚 Principe Migration

**OBJECTIF** :
Scripts R1 temporaires → Modules UTIL/HUB réutilisables futurs R2+

**STRATÉGIE** :
- Scripts analyse données → UTIL (fonctions pures)
- Scripts orchestration/rapports → HUB (compilation)
- Scripts one-shot → Suppression (documentation uniquement)

### ✅ Tâches

#### 4.1.1 - Audit scripts temporaires
```markdown
# Document: docs/audit_scripts_r1.md

# SCRIPTS ANALYSE (→ UTIL)
- analyze_seed_variance.py → utils/seed_variance_analyzer.py
- audit_gamma_discovery.py → utils/gamma_discovery_auditor.py

# SCRIPTS ORCHESTRATION (→ HUB)
- estimate_compute_load.py → HUB/compute_estimator.py

# SCRIPTS ONE-SHOT (→ SUPPRESSION)
- extract_explosive_combinations.py
- plot_seed_distributions.py
- generate_sequences_*.py
- run_sequences_*.py
- validate_*.py
```

**Tâches** :
- [ ] Créer `docs/audit_scripts_r1.md`
- [ ] Classer scripts (UTIL/HUB/ONE-SHOT)

#### 4.1.2 - Migration modules
```python
# Migration: scripts → tests/utilities/utils/

# 1. seed_variance_analyzer.py
# 2. gamma_discovery_auditor.py

# Migration: scripts → tests/utilities/HUB/

# 1. compute_estimator.py
```

**Tâches** :
- [ ] Créer modules UTIL (2 modules)
- [ ] Créer modules HUB (1 module)
- [ ] Tests unitaires
- [ ] Mettre à jour functions_index.md

### 📦 Livrables

```
tests/utilities/utils/
├── seed_variance_analyzer.py (~50 lignes)
└── gamma_discovery_auditor.py (~60 lignes)

tests/utilities/HUB/
└── compute_estimator.py (~100 lignes)
```

### ⏱️ Durée estimée
**4 heures** (audit + migration + tests)

---

## 📌 ÉTAPE 4.2 : VALIDATION PIPELINE UNIFIÉ

### 🎯 Objectif
Valider pipeline consolidé R0 + R1 via tests complets

### ✅ Tâches

#### 4.2.1 - Tests intégration
```python
# Test: tests/test_integration_r0_full.py
# Validation R0 complet (inchangé après R1)

# Test: tests/test_integration_r1_full.py
# Validation R1 complet (compositions)
```

**Tâches** :
- [ ] Créer tests intégration
- [ ] Exécuter validation
- [ ] Corriger anomalies

### 📦 Livrables

```
tests/
├── test_integration_r0_full.py (~150 lignes)
└── test_integration_r1_full.py (~200 lignes)
```

### ⏱️ Durée estimée
**4 heures** (tests + validation)

---

## 📌 ÉTAPE 4.3 : NETTOYAGE FICHIERS TEMPORAIRES

### 🎯 Objectif
Supprimer scripts/fichiers temporaires Phase R1, backup sécurisé

### ✅ Tâches

#### 4.3.1 - Script cleanup
```bash
# Script: scripts/consolidation/cleanup_r1_temporary_files.sh

"""
Nettoyage fichiers temporaires R1.

SÉCURITÉ:
- Backup avant suppression
- Validation tests passent avant suppression
"""

#!/bin/bash
set -e

# 1. Backup
BACKUP_DIR="backups/r1_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r scripts/r0_analysis "$BACKUP_DIR/"
cp -r scripts/r1_execution "$BACKUP_DIR/"

# 2. Validation tests
pytest tests/test_integration_r0_full.py tests/test_integration_r1_full.py -v

# 3. Suppression scripts temporaires
rm -f scripts/r0_analysis/extract_explosive_combinations.py
rm -f scripts/r0_analysis/plot_seed_distributions.py
rm -f scripts/r1_execution/generate_sequences_*.py
rm -f scripts/r1_execution/run_sequences_*.py
rm -f scripts/r1_execution/validate_*.py

# Scripts migrés
rm -f scripts/r0_analysis/analyze_seed_variance.py
rm -f scripts/r0_analysis/audit_gamma_discovery.py
rm -f scripts/r1_execution/estimate_compute_load.py

echo "✓ Nettoyage terminé"
echo "Backup: $BACKUP_DIR"
```

**Tâches** :
- [ ] Créer script cleanup
- [ ] Exécuter (validation + suppression)
- [ ] Vérifier backup créé

#### 4.3.2 - Validation structure finale
```python
# Test: tests/test_final_structure.py

"""
Valide structure finale pipeline consolidé.

VÉRIFICATIONS:
1. Scripts temporaires supprimés
2. Modules UTIL/HUB présents
3. Pipeline fonctionne (R0, R1)
4. functions_index.md à jour
"""
```

**Tâches** :
- [ ] Créer test structure finale
- [ ] Exécuter validation
- [ ] Corriger anomalies

### 📦 Livrables

```
scripts/consolidation/
└── cleanup_r1_temporary_files.sh

tests/
└── test_final_structure.py

backups/
└── r1_cleanup_YYYYMMDD_HHMMSS/
```

### ⏱️ Durée estimée
**3 heures** (cleanup + validation)

---

## 📌 ÉTAPE 4.4 : DOCUMENTATION FINALE

### 🎯 Objectif
Documenter consolidation R1 + mise à jour functions_index.md

### ✅ Tâches

#### 4.4.1 - Mise à jour functions_index.md
```markdown
# NOUVEAUX MODULES UTIL

### sequence_generator.py
| generate_sequences | Publique | Génère séquences permutations [Γ₁, ..., Γₙ] |

### composed_gamma_factory.py
| ComposedGammaFactory.generate_all_compositions | Publique | Génère compositions discovery format |

### baseline_computer.py
| compute_r0_baselines | Publique | Calcule métriques baseline R0 |

### composition_classifier.py
| classify_sequences | Publique | Classifie séquences R1 (robustes/dégradantes/instables) |

### seed_variance_analyzer.py
| compute_seed_variance_decomposition | Publique | Décompose variance inter-gamma vs intra-seed |

### gamma_discovery_auditor.py
| audit_gamma_discovery | Publique | Audit cohérence filesystem ↔ catalogues |

# NOUVEAUX MODULES HUB

### report_generator_r1.py
| generate_r1_1_report | Publique | Génère rapport R1.1 complet + Go/No-Go |

### compute_estimator.py
| estimate_r1_compute_load | Publique | Estime charge calcul R1 (temps, stockage) |

# MODIFICATIONS EXISTANTES

### data_loading.py (+20 lignes)
| discover_entities | Publique | **MODIFIÉ**: Injection compositions R1 si phase='R1' |
```

**Tâches** :
- [ ] Mettre à jour functions_index.md
- [ ] Documenter nouveaux modules
- [ ] Documenter modification data_loading.py

#### 4.4.2 - Document consolidation
```markdown
# Document: docs/consolidation_r1_summary.md

# Consolidation R1 - Résumé

## OBJECTIF
Unifier pipeline R0 + R1 dans batch_runner.py unique.

## MODIFICATIONS FINALES

**Nouveaux modules** : 8 (6 UTIL + 2 HUB)
**Modifications** : 1 fichier (+20 lignes data_loading.py)
**Supprimés** : Scripts temporaires (~1,500 lignes)

**Net** : -710 lignes (gain)

## ARCHITECTURE FINALE

batch_runner.py (UNIQUE ENTRY POINT)
  → discover_entities('gamma', phase)
     → SI phase='R1': ComposedGammaFactory
  → prepare_state() → D_final
  → run_kernel() → snapshots
  → test_engine → observations

## CONFORMITÉ CHARTER
- ✅ Core aveugle
- ✅ Séparation stricte
- ✅ Pas duplication
- ✅ UTIL/HUB hiérarchie

## GAIN
- Performances : 3× plus rapide (séquences)
- Maintenance : 1 pipeline unique
- Code : -710 lignes net
```

**Tâches** :
- [ ] Créer docs/consolidation_r1_summary.md
- [ ] Documenter architecture finale
- [ ] Documenter gains

### 📦 Livrables

```
functions_index.md (mis à jour)

docs/
└── consolidation_r1_summary.md
```

### ⏱️ Durée estimée
**2 heures** (documentation)

---

**[FIN PHASE R1.4 CONSOLIDATION]**

**JALONS ATTEINTS** :
- ✅ Migration UTIL/HUB
- ✅ Validation pipeline unifié
- ✅ Nettoyage final
- ✅ Documentation finale

**RÉSULTAT FINAL** :
- ✅ 1 seul pipeline (batch_runner.py --phase R0|R1)
- ✅ Génération dynamique compositions
- ✅ 0% modification batch_runner.py
- ✅ +20 lignes data_loading.py
- ✅ +790 lignes UTIL/HUB
- ✅ **Net: -710 lignes code**

**DURÉE TOTALE PHASE R1.4** : 13 heures

---

# 📊 RÉCAPITULATIF COMPLET TODO R1

## 🎯 Phases complétées

**Phase 0** : 8 heures (validation + variance + discovery)
**Phase R1.1** : 29 heures (architecture + factory + injection + tests + rapport)
**Phase R1.4** : 13 heures (migration + validation + cleanup + docs)

**TOTAL** : **50 heures** (~6 jours ouvrables)

## 📦 Livrables totaux

**Code** : 8 nouveaux modules UTIL/HUB + 1 modification
**Tests** : 6 nouveaux test suites
**Documentation** : functions_index.md + consolidation_r1_summary.md
**Databases** : prc_r1_raw.db + prc_r1_results.db
**Rapports** : reports/r1_1/{timestamp}/

## 🎓 Prochaines étapes

**Si GO R1.2** : Phase Compensation
**Si NO-GO R1.2** : Skip → Phase R1.3 directement
**Phase R1.3** : Convergence (dans tous cas)
**Phase R2+** : Extensions futures

---

# 📝 NOTES D'UTILISATION

## Autonomie document

- ✅ Complètement autonome
- ✅ Pas référence todo_r1.md (obsolète)
- ✅ Approche variance seeds
- ✅ Injection "fichiers virtuels" clarifiée

## Execution par étape

Chaque étape exécutable indépendamment :
1. Charger contexte minimal (📥 section)
2. Consulter documents pertinents
3. Implémenter structure/algo
4. Validation avant code

## Rappel Charter 6.1 Section 4.5

**Ordre obligatoire** :
1. MÉTIER (objectif, métriques)
2. ORGANIGRAMME (architecture)
3. STRUCTURE (interfaces)
4. CODE (implémentation)

**Ne jamais produire code directement**

---

**TODO R1 AUTONOME COMPLET - FIN**

Prêt pour implémentation ! 🎯
