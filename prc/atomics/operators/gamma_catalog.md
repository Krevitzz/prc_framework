# GAMMA CATALOG

> Opérateurs Δ (gamma) définissant mécanismes itératifs  
> Forme générale : state_{n+1} = gamma(state_n)  

**FAMILLES** :
- `markovian` : État futur dépend uniquement état présent
- `non_markovian` : Mémoire explicite (stocke états précédents)
- `stochastic` : Composante aléatoire
- `structural` : Préserve/force propriétés structurelles

---

## GAMMAS MARKOVIENS

### GAM-001 - Saturation Pure Pointwise
**Fichier** : `gamma_hyp_001.py`  
**Forme** : T_{n+1}[i,j] = tanh(β · T_n[i,j])  
**Paramètres** : β=2.0 (force saturation)  
**Famille** : markovian  
**Applicabilité** : SYM, ASY, R3

**Comportement attendu** :
- Convergence : Rapide (saturation borne [-1,1])
- Attracteurs : Possiblement triviaux (selon β)
- Diversité : Possible perte

**Notes** :
- Opération pointwise (indépendant par élément)
- Pas de couplage spatial

---

### GAM-002 - Diffusion Pure
**Fichier** : `gamma_hyp_002.py`  
**Forme** : T_{n+1}[i,j] = T_n[i,j] + α·(∑_voisins - 4·T_n[i,j])  
**Paramètres** : α=0.05 (stabilité : α < 0.25)  
**Famille** : markovian  
**Applicabilité** : SYM, ASY (rang 2 uniquement)

**Comportement attendu** :
- Convergence : Rapide (<500 iterations)
- Attracteurs : Uniformes (homogénéisation totale)
- Trivialité : Oui

**Notes** :
- Laplacien discret, voisinage 4-connexe
- Conditions périodiques
- Stabilité Von Neumann : α < 0.25
- Lisse toute structure initiale

---

### GAM-003 - Croissance Exponentielle
**Fichier** : `gamma_hyp_003.py`  
**Forme** : T_{n+1}[i,j] = T_n[i,j] · exp(γ)  
**Paramètres** : γ=0.05  
**Famille** : markovian  
**Applicabilité** : SYM, ASY, R3

**Comportement attendu** :
- Convergence : Jamais (divergence)
- Explosion : <100 iterations typiquement
- **CONÇU POUR ÉCHOUER** (validation détection explosions)

**Notes** :
- Test robustesse framework
- Devrait obtenir REJECTED[GLOBAL]
- Explosion rapide garantie

---

### GAM-004 - Décroissance Exponentielle
**Fichier** : `gamma_hyp_004.py`  
**Forme** : T_{n+1}[i,j] = T_n[i,j] · exp(-γ)  
**Paramètres** : γ=0.05  
**Famille** : markovian  
**Applicabilité** : SYM, ASY, R3

**Comportement attendu** :
- Convergence : Rapide vers zéro (<500 iterations)
- Attracteurs : Zéro (trivial)
- Temps caractéristique : 1/γ iterations

**Notes** :
- Perte systématique information
- Attendu : REJECTED[R0] pour trivialité

---

### GAM-005 - Oscillateur Harmonique
**Fichier** : `gamma_hyp_005.py`  
**Forme** : T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}  
**Paramètres** : ω=π/4 (fréquence angulaire)  
**Famille** : markovian (bien que non-markovien techniquement)  
**Applicabilité** : SYM, ASY, R3

**Comportement attendu** :
- Convergence : Jamais (oscillations périodiques)
- Période : 2π/ω iterations
- Conservation : Énergie (norme théoriquement constante)

**Notes** :
- Non-markovien ordre 1 (stocke T_{n-1})
- Pas de complexité émergente attendue
- Test détection périodicité
- Appeler reset() entre runs

---

## GAMMAS NON-MARKOVIENS

### GAM-006 - Saturation + Mémoire Ordre-1
**Fichier** : `gamma_hyp_006.py`  
**Forme** : T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))  
**Paramètres** : β=1.0, α=0.3  
**Famille** : non_markovian  
**Applicabilité** : SYM, ASY, R3

**Comportement attendu** :
- Convergence : Plus lent que markovien (inertie)
- Attracteurs : Non-triviaux possibles
- Diversité : Possible préservation avec α adéquat

**Notes** :
- Stocke T_{n-1} en interne
- Première itération : comportement markovien
- Inertie peut éviter attracteurs triviaux
- Appeler reset() entre runs

---

### GAM-007 - Régulation Moyenne Glissante
**Fichier** : `gamma_hyp_007.py`  
**Forme** : T_{n+1}[i,j] = (1-ε)·T_n[i,j] + ε·mean(voisins_8)  
**Paramètres** : ε=0.1  
**Famille** : non_markovian  
**Applicabilité** : SYM, ASY (rang 2 uniquement)

**Comportement attendu** :
- Convergence : Moyenne (500-1000 iterations)
- Attracteurs : Uniformes (plus lent que GAM-002)
- Trivialité : Oui

**Notes** :
- Voisinage 8-connexe (diagonales incluses)
- Plus doux que diffusion Laplacienne
- Implémentation O(N²) (peut être lent)
- Conditions périodiques

---

### GAM-008 - Mémoire Différentielle
**Fichier** : `gamma_hyp_008.py`  
**Forme** : T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)  
**Paramètres** : γ=0.3, β=1.0  
**Famille** : non_markovian  
**Applicabilité** : SYM, ASY, R3

**Comportement attendu** :
- Convergence : Oscillations amorties possibles
- Attracteurs : Non-triviaux possibles
- Diversité : Maintien possible avec γ adéquat

**Notes** :
- Combine inertie + saturation + friction
- Balance γ (inertie) vs β (saturation)
- Similaire GAM-006 avec terme β additionnel
- Oscillations amorties si bien paramétré
- Appeler reset() entre runs

---

## GAMMAS STOCHASTIQUES

### GAM-009 - Saturation + Bruit Additif
**Fichier** : `gamma_hyp_009.py`  
**Forme** : T_{n+1} = tanh(β·T_n) + σ·ε, ε ~ N(0,1)  
**Paramètres** : β=1.0, σ=0.01, seed=42  
**Famille** : stochastic  
**Applicabilité** : SYM, ASY, R3

**Comportement attendu** :
- Convergence : Équilibre stochastique possible
- Attracteurs : Distribution stationnaire
- Diversité : Maintien possible avec σ adéquat

**Notes** :
- Processus non-déterministe
- Fixer seed pour reproductibilité
- Balance déterminisme (β) / exploration (σ)
- TEST-UNIV-004 (sensibilité CI) particulièrement pertinent
- Moyenner sur plusieurs seeds pour analyses

---

### GAM-010 - Bruit Multiplicatif
**Fichier** : `gamma_hyp_010.py`  
**Forme** : T_{n+1} = tanh(T_n · (1 + σ·ε)), ε ~ N(0,1)  
**Paramètres** : σ=0.05, seed=42  
**Famille** : stochastic  
**Applicabilité** : SYM, ASY, R3

**Comportement attendu** :
- Convergence : Variable (dépend σ)
- Attracteurs : Structures amplifiées ou chaos
- Diversité : Possible augmentation (amplification)

**Notes** :
- Bruit multiplicatif : amplifie proportionnellement
- Différent de GAM-009 (bruit additif)
- Saturation nécessaire pour stabilité
- Risque avalanche si σ > 0.2
- Peut créer hétérogénéité (riches plus riches)
- Intéressant : comparer GAM-009 vs GAM-010

---

## GAMMAS STRUCTURAUX

### GAM-012 - Préservation Symétrie Forcée
**Fichier** : `gamma_hyp_012.py`  
**Forme** : T_{n+1} = (F(T_n) + F(T_n)^T) / 2, F = tanh(β·)  
**Paramètres** : β=2.0  
**Famille** : structural  
**Applicabilité** : SYM, ASY (rang 2 uniquement)

**Comportement attendu** :
- Convergence : Similaire GAM-001 mais symétrique
- Attracteurs : Symétriques garantis
- Trivialité : Possible (comme GAM-001)

**Notes** :
- Force symétrie artificiellement
- TEST-SYM-001 devrait toujours PASS
- TEST-SYM-002 : peut créer symétrie depuis ASY
- Robuste au bruit asymétrique (M1, M2)
- Question : forçage aide-t-il non-trivialité ?

---

### GAM-013 - Renforcement Hebbien Local
**Fichier** : `gamma_hyp_013.py`  
**Forme** : T_{n+1}[i,j] = T_n[i,j] + η·Σ_k T_n[i,k]·T_n[k,j]  
**Paramètres** : η=0.01  
**Famille** : structural  
**Applicabilité** : SYM, ASY (rang 2 carré uniquement)

**Comportement attendu** :
- Convergence : Instable (risque explosion)
- Attracteurs : Structures émergentes ou explosion
- Diversité : Augmentation (clusters)

**Notes** :
- **INSTABLE sans régulation additionnelle**
- Produit matriciel T @ T (coûteux : O(N³))
- Nécessite matrices CARRÉES
- Risque explosion si η trop grand ou D mal conditionné
- Peut créer structures hiérarchiques
- TEST-UNIV-001 (norme) critique pour détecter explosions
- Intéressant si combiné avec saturation (voir GAM-103[R1])

---

## PATTERN IMPLEMENTATION

**Tous les gammas suivent cette structure** :
```python
class SomeGamma:
    """Docstring avec mécanisme et comportement attendu."""
    
    def __init__(self, param1, param2, ...):
        # Validation paramètres
        assert param1 > 0, "param1 doit être > 0"
        self.param1 = param1
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Applique transformation Δ."""
        # Validation applicabilité (rang, dimension)
        if state.ndim != expected_rank:
            raise ValueError(...)
        
        # Algorithme
        result = ...
        
        return result
    
    def __repr__(self):
        return f"SomeGamma(param1={self.param1})"
    
    def reset(self):  # Si non-markovien ou stochastique
        """Réinitialise mémoire/RNG."""
        pass

def create_gamma_hyp_NNN(param1, ...) -> Callable:
    """Factory pour GAM-NNN."""
    return SomeGamma(param1=param1, ...)

METADATA = {
    'id': 'GAM-NNN',
    'name': '...',
    'family': 'markovian|non_markovian|stochastic|structural',
    'form': '...',
    'parameters': {...},
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {...},
    'notes': [...]
}
```

---

## GRILLES DE PARAMÈTRES

**PHASE1** : Exploration nominale (1 paramset par gamma)
**PHASE2** : Exploration paramétrique (3-6 paramsets par gamma)

**Exemples** :
```python
PARAM_GRID_PHASE1 = {
    'nominal': {'beta': 2.0}
}

PARAM_GRID_PHASE2 = {
    'beta_low': {'beta': 0.5},
    'beta_nominal': {'beta': 1.0},
    'beta_high': {'beta': 2.0},
    'beta_very_high': {'beta': 5.0},
}
```

---

## VALIDATION APPLICABILITÉ

**Gammas DOIVENT** :
- Vérifier `state.ndim` si rang spécifique requis
- Lever `ValueError` si non applicable
- Documenter applicabilité dans METADATA['d_applicability']

**Exemples** :
- Rang 2 uniquement : GAM-002, GAM-007, GAM-012, GAM-013
- Tous rangs : GAM-001, GAM-003, GAM-004, GAM-006, GAM-008, GAM-009, GAM-010
- Rang 2 carré : GAM-013 (produit matriciel)

---

## GESTION MÉMOIRE (NON-MARKOVIENS)

**Pattern standard** :
```python
class NonMarkovianGamma:
    def __init__(self, ...):
        self._previous_state: Optional[np.ndarray] = None
    
    def __call__(self, state):
        if self._previous_state is None:
            # Première itération
            result = markovian_fallback(state)
        else:
            # Utilise mémoire
            result = compute_with_memory(state, self._previous_state)
        
        self._previous_state = state.copy()
        return result
    
    def reset(self):
        """OBLIGATOIRE : réinitialise mémoire entre runs."""
        self._previous_state = None
```

---

## GESTION STOCHASTICITÉ

**Pattern standard** :
```python
class StochasticGamma:
    def __init__(self, ..., seed=None):
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def __call__(self, state):
        noise = self.rng.randn(*state.shape)  # Utilise self.rng
        # ...
```

**Règles** :
- Toujours accepter paramètre `seed` optionnel
- Utiliser `RandomState` si seed fourni (reproductibilité)
- Sinon utiliser `np.random` (comportement par défaut)

---

## DÉPENDANCES

**Autorisées** : NumPy uniquement  
**Interdites** : core/, D_encodings/, modifiers/, tests/, utilities/

---

## STATUTS AUTORISÉS

- `WIP[R0-open]` : Exploration en cours
- `WIP[R0-closed]` : R0 terminé, ambigu → mise en attente
- `SURVIVES[R0]` : Non éliminé à R0
- `REJECTED[R0]` : Éliminé comme autonome à R0
- `REJECTED[GLOBAL]` : Éliminé définitif tous rangs

---

## EXTENSIONS FUTURES

**Checklist ajout nouveau gamma** :
- [ ] Définir mécanisme mathématique clair
- [ ] Identifier famille (markovian, non_markovian, stochastic, structural)
- [ ] Documenter comportement attendu (convergence, attracteurs, trivialité)
- [ ] Spécifier applicabilité D (SYM, ASY, R3)
- [ ] Implémenter validation applicabilité (rang, dimension)
- [ ] Fournir grilles paramètres PHASE1 et PHASE2
- [ ] Compléter METADATA
- [ ] Ajouter à ce catalogue avec ID séquentiel (GAM-014, ...)

---

**FIN GAMMA CATALOG**