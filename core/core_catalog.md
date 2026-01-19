# CORE CATALOG

> Exécution aveugle du système PRC  
> Deux fonctions pures : prepare_state() et run_kernel()  

## MODULES

### state_preparation.py
**Responsabilité** : Appliquer séquentiellement modifiers sur tenseur base  
**Fonction principale** : `prepare_state(base, modifiers) → np.ndarray`  
**Aveuglement** : Ne connaît ni dimension, ni structure, ni interprétation du tenseur  
**Usage typique** : Composition D^(base) + M1 + M2 → D^(final)

**Comportement** :
- Si `modifiers=None` ou `[]` → retourne `base.copy()`
- Applique séquentiellement : `state → modifier_1 → modifier_2 → ... → state_final`
- Chaque modifier reçoit résultat du précédent

### kernel.py
**Responsabilité** : Générer états successifs via gamma itératif  
**Fonction principale** : `run_kernel(initial_state, gamma, ...) → Generator`  
**Aveuglement** : Ne connaît ni gamma, ni state, ni leur dimension

**Modes opératoires** :
1. **Sans historique** (défaut) : `yield (iteration, state)` — Économie mémoire
2. **Avec historique** : `yield (iteration, state, history)` — Stockage complet O(n×size)

**Contrôle arrêt** (délégué au TM) :
- `max_iterations` atteint
- `convergence_check(state_n, state_{n+1})` retourne `True`
- TM break la boucle

**Paramètres clés** :
- `record_history=False` : Mode économe (défaut)
- `convergence_check=None` : Pas de vérification automatique
- `max_iterations=10000` : Limite sécurité

## RÈGLES AXIOMATIQUES (K1-K5)

**K1** : Core ne valide JAMAIS contenu (symétrie, bornes, conservation, etc.)  
**K2** : Core ne connaît PAS les classes State/Operator  
**K3** : Core ne branche PAS selon D ou Δ  
**K4** : Arrêt kernel contrôlé exclusivement par TM (via break ou convergence_check)  
**K5** : prepare_state() applique modifiers sans inspection

## DÉPENDANCES

**Autorisées** : 
- `numpy` uniquement

**Interdites** : 
- Tout module PRC (operators/, D_encodings/, modifiers/, tests/, utilities/)
- Toute bibliothèque domaine-spécifique

## HIÉRARCHIE DES NIVEAUX (RAPPEL)

```
L0 (ontologique) → L1 (épistémique) → L2 (théorique) → L3 (opérationnel) → L4 (documentaire)
```

**Core = L3** (opérationnel pur)
- Ne référence QUE NumPy (externe)
- Consommé par operators/, tests/, prc_automation/
- Exception explicite : Peut être documenté par L4 (ce catalogue) sans en dériver règles

## Principe fondamental :
Le Core est le socle immuable du système PRC.  
Toute intelligence (validation, classification, décision) réside HORS du Core.

---

**FIN CORE CATALOG**