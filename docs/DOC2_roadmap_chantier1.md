# DOCUMENT 2 — ROADMAP CHANTIER 1
## Refonte du layer running — Deux couches, deux grammaires

> Plan d'action opérationnel pour le chantier 1 du pipeline PRC.
> Ancré sur CHARTER (philosophie), ancrage.md (exécution + protocole d'audit), et DOC1_features_running.md (cadre scientifique des features).
>
> Ce document est la **boussole de travail** du chantier. À consulter en début de session pour se réaligner.
> Il est mis à jour après chaque sous-chantier terminé (état, décisions, bifurcations éventuelles).

---

## TABLE DES MATIÈRES

1. Philosophie en une page
2. Décisions actées (résumé exécutif)
3. État actuel vs état cible
4. Sous-chantiers ordonnés
5. Expériences standalone recommandées
6. Critères de succès
7. Risques identifiés et mitigations
8. Hooks vers les chantiers suivants
9. Glossaire opérationnel du chantier

---

## 1. PHILOSOPHIE EN UNE PAGE

Le chantier 1 corrige trois défauts architecturaux du layer running actuel :

**Défaut 1 — Incohérence de grammaire.** Les features post-process (DMD, F6, autocorr, PNN, temporal) sont appliquées à des signaux extraits **aux points du masque**, c'est-à-dire à un échantillonnage temporel non-uniforme. Ces méthodes supposent pourtant un pas uniforme. La grammaire signal est appliquée à une grammaire virages — résultat scientifiquement douteux, même si numériquement défini.

**Défaut 2 — Calcul gaspillé.** Le JVP de F4 est calculé dans P1 à **chaque itération** pour tous les runs, même ceux qui ne nécessitent pas de mesure de Jacobien à cet instant. C'est la source principale de pression VRAM et de compute inutile. P1 en est alourdi, P2 re-propage sans bénéficier.

**Défaut 3 — Doublons et gaspillages.** `frob_norm` et `shannon_entropy` sont calculés deux fois (dyn P1 + spec P2). `mode_asymmetry` est placé en couche spectrale alors qu'il est O(n²). `_f3_rank3` effectue deux SVD là où une suffit. Les 10 features dynamiques sont compressées en 3 scalaires (`dyn_acc`) alors que leur timeline complète serait peu coûteuse.

**La refonte corrige ces trois défauts simultanément** en réorganisant les mesures en deux couches :

- **Couche A** : 11 observables canoniques O(n²) calculés à chaque itération de P2, en timeline complète uniforme, qui nourrissent la grammaire signal du post-process (DMD, F6, temporal, autocorr, PNN).
- **Couche B** : 15 features coûteuses (O(n³) spectrales + JVP F4) calculées uniquement aux points du masque, en timelines tronquées, qui nourrissent la grammaire virages (agrégats + phasic features).

Le **masque** devient ce qu'il a toujours voulu être : la schématisation des virages de la trajectoire, pas un sous-échantillonnage temporel. Les deux grammaires coexistent sans se contaminer.

Le **JVP sort de P1** et est désormais dans la couche B sous `lax.cond` : payé uniquement aux virages pour les gammas différentiables. Gain VRAM attendu dominant. Gain compute proportionnel à K_i / max_it.

**Ce que le chantier 1 ne fait pas** :
- Il ne modifie pas classify_and_mask (la détection des virages reste identique)
- Il ne modifie pas le pool d'atomics gamma/encoding/modifier
- Il ne touche pas aux phases suivantes de l'analysing (chantier 2+)
- Il n'introduit pas de chunking P2 dans la première version (différé à mesure de nécessité)

---

## 2. DÉCISIONS ACTÉES (RÉSUMÉ EXÉCUTIF)

Les décisions suivantes ont été validées dans la discussion préparatoire au chantier 1. Elles sont **immuables** pour la durée du chantier sauf remise en cause explicite documentée.

| # | Décision | Justification courte |
|---|---|---|
| D1 | **Deux couches** : A (observables O(n²) continus) + B (features coûteuses aux virages) | Cohérence grammaires signal/virages |
| D2 | **P1 minimal** : propagate + screening uniquement, aucune différentiation | Réduction VRAM dominante |
| D3 | **JVP F4 migré en couche B sous `lax.cond`** | Payé seulement aux virages, vrai conditionnel XLA |
| D4 | **Masque inchangé** dans son principe : calculé par `classify_and_mask` à partir des timelines screening | Stabilité du pipeline, pas de régression |
| D5 | **Toutes les features de couche B sous `lax.map + lax.cond`** (pas seulement SVD) | Extension du mécanisme déjà validé à 15× vs vmap |
| D6 | **Pool couche A enrichi à 11 observables** (au lieu du minimum à 8) | Politique "une en trop plutôt qu'une en moins" (Q-sci-4) |
| D7 | **Pool couche A contient IPR, volume_proxy, mode_asymmetry_o2** comme nouvelles features O(n²) | Q-sci-1, Q-sci-2 validés |
| D8 | **F6 redéfini sur paires d'observables couche A** (pas sur spectrales) | Grammaire signal cohérente (Q-sci-3) |
| D9 | **DMD redéfini sur observables couche A** (pas sur sigmas_buf) | Universalité par observables canoniques, grammaire signal cohérente |
| D10 | **Stationnarité ajoutée en post-scan** (Q-sci-5) | Q3 formalisée explicitement |
| D11 | **Doublons éliminés** : frob_norm et shannon_entropy définis une seule fois (couche A), SVD unique par step en couche B | Nettoyage et économie compute |
| D12 | **Chunking P2 différé** | Probablement non nécessaire après D2+D3, à mesurer |
| D13 | **Pathologiques préservés** : couche A calculée jusqu'à t_effective même pour EXPLOSION/COLLAPSED | Matière première du chantier 3 |
| D14 | **Parquet à 3 niveaux** : timelines complètes (11) + timelines tronquées (15) + scalaires (~90) | Analysing n'a pas besoin de charger les timelines pour les scalaires |
| D15 | **Renommages explicites** des features post-process pour refléter le changement de signal source (`von_neumann → shannon_comp`, etc.) | Sémantique honnête |

---

## 3. ÉTAT ACTUEL vs ÉTAT CIBLE

### 3.1 Avant / Après

| Élément | État actuel | État cible |
|---|---|---|
| **P1 scan** | propagate + screening + dynamic (dont JVP) | propagate + screening (4 scalaires) |
| **Timelines P1** | 5 (delta_D, frob, cos_dissim, is_finite, lyap) | 4 (delta_D, frob, cos_dissim, is_finite) |
| **Dyn accumulator** | 10 clés × 5 champs (sum/sumsq/first/last/count) | **Supprimé** |
| **P2 scan** | propagate + SVD conditionnelle via `lax.map + lax.cond` | propagate + couche A (vmap B) + couche B (`lax.map + lax.cond`) |
| **Features P2** | 14 spectrales NaN-éparses + re-propagation | 11 couche A (timelines complètes) + 15 couche B (NaN-éparses sous cond) |
| **Mode asymmetry** | en couche spectrale (B) | en couche A (A.7) |
| **Frob norm / shannon** | doublés (dyn + spec) | unique en couche A |
| **F4 JVP** | en P1 à chaque step pour tous | en couche B sous `lax.cond` (payé aux virages) |
| **DMD** | sur sigmas_buf (virages spectrales) | sur vecteur observables couche A (timeline uniforme) |
| **F6** | sur mix (timelines P1 + signal masque-étendu) | sur paires d'observables couche A uniquement |
| **Autocorr / PNN / temporal** | sur signal masque-étendu (invalide) | sur timelines couche A uniformes |
| **Phasic** | sur signal aux virages | identique, grammaire virages préservée |
| **Stationnarité** | absente | ajoutée en post-scan |
| **Parquet** | 3 timelines complètes + 99 scalaires + metadata | 11 timelines complètes + 15 tronquées + ~90 scalaires + metadata |
| **VRAM P1** | dominée par intermédiaires JVP | minimale (propagate + 4 timelines screening) |
| **VRAM P2** | dominée par re-propagation + spec_tables denses | dominée par propagate + couche A dense léger + couche B NaN-éparses conditionnelles |
| **Sémantique post-process** | mélange grammaires, non-cohérent | strictement grammaire signal sur couche A, strictement grammaire virages sur couche B |

### 3.2 Ce qui ne change pas

Pour maintenir la stabilité du pipeline et limiter la surface de changement :

- `classify_and_mask` : inchangé (c'est exactement la détection des virages qu'on veut conserver)
- Le split jobs / orchestration hub / budget VRAM : inchangé structurellement
- Le schéma parquet de base (colonnes metadata, types de base) : inchangé structurellement
- L'architecture SubBatchProcess en processus isolés : inchangée (P9 du charter)
- Les atomics gamma/encoding/modifier : inchangés
- Les timelines complètes delta_D, frob, cos_dissim : conservées (elles sont déjà dans le parquet actuel, elles rejoignent naturellement la couche A)
- Le chemin de propagation gamma en P2 : inchangé

---

## 4. SOUS-CHANTIERS ORDONNÉS

Le chantier 1 est découpé en 8 sous-chantiers. Ils suivent la méthodologie CHARTER §5.1 — Algo → Structure → Code — avec validation explicite à chaque étape. **Chaque sous-chantier produit un artefact validé avant que le suivant commence.**

### SC1 — Refonte du registre features

**Objectif** : le `features_registry.py` devient la source de vérité unique du nouveau schéma (pool couche A, pool couche B, mapping d'agrégats, constantes, NaN structurels/runtime).

**Livrables** :
- Nouvelle version du registre avec les 11 observables couche A déclarés
- 15 features couche B déclarées
- Mapping d'agrégation par feature (quel sous-ensemble de mean/std/final/delta/total)
- Listes FEATURE_NAMES mises à jour (timelines_complete, timelines_tronquees, scalaires)
- Constantes scientifiques (EPS, seuils, facteurs) conservées, ajoutées si nécessaire pour IPR / volume_proxy / stationnarité
- FEATURES_STRUCTURAL_NAN et FEATURES_RUNTIME_NAN mis à jour

**Dépendances** : DOC1 validé (fait).

**Critère de sortie** : registre relu, validé, auto-consistent (chaque nom de feature apparaît exactement une fois, chaque sous-catégorie est cohérente avec les autres).

**Note** : ce sous-chantier se fait en Structure directement (le registre est de la déclaration, pas du calcul). Pas besoin d'étape Algo séparée.

---

### SC2 — Refonte P1 minimal

**Objectif** : `run_pass1` ne fait plus que propagate + screening. Les 4 timelines (delta_D, frob, cos_dissim, is_finite) sont les seules sorties GPU de P1. Aucune référence à `dynamic_fn`, aucun JVP, aucun accumulateur.

**Livrables** :
- Nouvelle `run_pass1` avec carry minimal
- Suppression de `init_dynamic_acc`, `update_dynamic_acc`, `DynamicAccumulator` dans `jit_builders.py`
- Suppression de `build_dynamic_vmap` dans sa forme actuelle (remplacée par les builders couche A et couche B)
- Timelines P1 limitées à 4 (au lieu de 5 : `lyap_timeline` devient `lyap_empirical` dans la couche A et n'est plus en P1)

**Dépendances** : SC1.

**Algo → Structure → Code** : cycle complet.
- **Algo** : décrire en prose "P1 prend D, gp, keys, propage max_it fois via gamma, stocke 4 timelines screening, retourne les timelines + last_states + active_mask".
- **Structure** : signatures, types I/O, carry shape, squelette du scan.
- **Code** : implémentation dans le squelette validé.

**Critère de sortie** : le nouveau P1 est testé standalone sur 1-2 runs de petite taille (rank 2, n_dof 10, B=4, max_it=50) et produit les 4 timelines attendues, sans aucun appel à gamma_fn en mode JVP.

---

### SC3 — Builder couche A (observables O(n²))

**Objectif** : une fonction vmappée `build_layer_a_vmap()` qui calcule les 11 observables de la couche A à partir de `(state, prev_state)`. Zéro dépendance à gamma_fn, zéro SVD, zéro différentiation. Utilisée à chaque step de P2.

**Livrables** :
- Nouvelle fonction `build_layer_a_vmap()` dans `jit_builders.py`
- Tests unitaires sur un state forgé pour vérifier la valeur numérique de chaque observable sur un cas simple (identity, rotation, contraction)
- Vérification que les 11 observables sont canoniques (invariants par permutation d'axes)

**Dépendances** : SC1 (registre).

**Algo → Structure → Code** : cycle complet.
- **Algo** : pour chaque observable, rédiger en prose "ce qu'on calcule et pourquoi" en référençant DOC1 section 3.2.
- **Structure** : signature `fn(state, prev_state) → {11 scalaires}`, contrats numériques (eps, dtype, cas limites).
- **Code** : implémentation.

**Critère de sortie** : 11 observables calculés, résultats cohérents sur cas de test, aucun NaN/Inf sur états valides, NaN contrôlé pour les cas limites (volume_proxy sur état nul, etc.).

**Note scientifique** : `frob_gradient` (rank ≥ 3) et `volume_proxy` sont les deux observables qui ont besoin d'un traitement conditionnel au rank. Cette conditionnalité est au Python-level (static_argnums), pas au JAX runtime.

---

### SC4 — Builder couche B (features coûteuses conditionnelles)

**Objectif** : une fonction unitaire `build_layer_b_fn(rank_eff, is_diff)` qui calcule les 15 features de la couche B **pour un sample**, sous `lax.cond(active, compute, NaN)`. Utilisée dans `lax.map` à l'intérieur du scan P2.

**Livrables** :
- Nouvelle fonction `build_layer_b_fn()` dans `jit_builders.py` qui retourne une fonction unitaire prête pour `lax.map`
- Factorisation de la SVD : une seule SVD principale (mode 0) qui sert à F1 + F2 + F3 mode0 ; SVD mode 1 calculée une seule fois si rank ≥ 3
- Intégration du JVP F4 à l'intérieur du `lax.cond` (branches true = JVP + SVD, branche false = NaN partout)
- Tests unitaires sur un state forgé

**Dépendances** : SC1.

**Algo → Structure → Code** : cycle complet.
- **Algo** : rédiger en prose "à un instant masqué, on calcule la structure spectrale complète de D (F1 × 6, F2 × 2, F3 × 3 selon rank) et si différentiable on calcule aussi F4 × 4 via JVP. Si pas masqué, tout est NaN."
- **Structure** : signature `fn(state, prev, gp, key, active) → ({15 scalaires}, sigmas)`, cas branches conditionnelles, closures JVP.
- **Code** : implémentation avec `lax.cond` unique.

**Critère de sortie** : 15 features calculées pour un state forgé, comparées aux valeurs actuelles (référence pipeline actuel sur un run simple) pour non-régression scientifique. Les valeurs doivent être **identiques** ou n'avoir qu'un écart numérique sous le seuil eps.

**Point critique** : le `lax.cond` doit englober TOUT le calcul (SVD + F1 + F2 + F3 + JVP F4). Si la SVD est hors du cond, elle est payée pour tous les steps même non-masqués — défaite du masque. La branche true fait le calcul complet, la branche false retourne des NaN.

---

### SC5 — Refonte P2 avec couches A et B

**Objectif** : nouvelle `run_pass2` qui remplace l'ancienne. Fait propagate vmap B, puis couche A vmap B, puis couche B via `lax.map + lax.cond`. Produit les timelines couche A (denses) et les tables couche B (denses NaN-éparses) pour transfert unique GPU → CPU.

**Livrables** :
- Nouvelle `run_pass2` dans `subbatch_process.py`
- Carry shape explicite : `(state, prev, keys, timeline_tables_A, feature_tables_B, sigmas_buf)`
- Le masque `mask (B, T)` est consommé par le scan via le `t` courant
- Output : dict JAX contenant les 11 timelines couche A `(B, T)` + 15 tables couche B `(B, T)` NaN-éparses + sigmas_buf

**Dépendances** : SC1, SC3, SC4.

**Algo → Structure → Code** : cycle complet.
- **Algo** : "P2 re-propage la trajectoire identique à P1, et à chaque step : (a) calcule les 11 observables couche A pour tous les samples ; (b) sous lax.map, chaque sample calcule ses 15 features couche B sous lax.cond conditionné par mask[b,t]."
- **Structure** : carry complet, step function, shapes, output dict.
- **Code** : scan unique avec lax.map interne pour couche B.

**Critère de sortie** : P2 exécute sans erreur sur un run de test (rank 2, n_dof 10, B=4, max_it=50), produit les buffers attendus. Mesure de la VRAM peak vs. l'ancien pipeline sur la même configuration — attendu : réduction nette.

**Point de vigilance** : à ce stade, les 4 timelines P1 + 11 timelines couche A font un certain recouvrement (delta_D, frob, cos_dissim, is_finite sont dans les deux). Les timelines P1 servent à classify, les timelines couche A servent au post-process signal. Elles peuvent être calculées deux fois (acceptable — P1 est rapide) ou factorisées (plus complexe). **Décision recommandée : les recalculer en P2**, pour que P1 reste minimal et que la couche A soit autonome. La duplication est ~4 timelines × (B, T) × 4 bytes = quelques Mo, négligeable.

---

### SC6 — Refonte postprocess CPU

**Objectif** : le `postprocess.py` consomme la nouvelle structure de données (couches A et B) et produit les scalaires décrits dans DOC1 section 5. Chaque feature post-scan est clairement ancrée dans sa catégorie (C1 agrégats, C2 signal, C3 virages).

**Livrables** :
- Nouveau `postprocess.py` organisé par catégorie (C1, C2, C3)
- Fonction `compute_dmd_on_observables()` remplace l'ancienne `compute_dmd()` sur sigmas
- Fonction `compute_f6_on_observables()` avec les 4 paires validées
- Fonction `compute_stationarity_deltas()` nouvelle
- Fonctions `compute_temporal_features()` et `compute_first_min_autocorr()` adaptées aux noms renommés
- Fonction `compute_phasic_features()` adaptée pour consommer les timelines tronquées (option 2 : sous-échantillonnage de mode_asymmetry_o2 au masque pour phasic_asym)
- Fonction `build_col_data()` adaptée à la nouvelle structure parquet (timelines complètes + tronquées + scalaires)
- Extraction des timelines tronquées en `list<f32>` par run avec conservation des `mask_t_indices` partagés

**Dépendances** : SC1, SC5.

**Algo → Structure → Code** : cycle complet.
- **Algo** : pour chaque nouvelle fonction, décrire ce qu'elle consomme et ce qu'elle produit, avec référence à DOC1 section 5.
- **Structure** : signatures des fonctions, dict de sortie, format des listes variables par run.
- **Code** : implémentation.

**Critère de sortie** : postprocess produit tous les scalaires attendus du schéma cible sans erreur sur un batch de test, parquet écrit correctement avec les 3 niveaux.

**Points délicats** :
- La gestion des timelines tronquées par run (longueurs variables) nécessite de passer du dense NaN-éparse `(B, T)` aux listes `list<f32>` de longueur K_i lors de la construction de col_data. C'est un travail CPU, pas GPU.
- Les `mask_t_indices` partagés permettent à l'analysing de ré-ancrer chaque point tronqué dans le temps si besoin. Un seul tableau par run pour toutes les 15 features couche B.

---

### SC7 — Refonte SubBatchProcess (orchestration)

**Objectif** : le `subbatch_process.py` orchestre le nouveau flux. Discovery + matérialisation + compilation des builders + P1 + classify + P2 + transfert + postprocess + assemblage col_data + envoi queue.

**Livrables** :
- Nouvelle classe `SubBatchProcess.run()` avec les 10 étapes adaptées au nouveau flux
- Appel aux nouveaux builders (`build_layer_a_vmap`, `build_layer_b_fn`)
- Appel aux nouvelles fonctions postprocess
- Gestion budget VRAM avec nouvelle estimation (voir SC8)

**Dépendances** : SC2, SC3, SC4, SC5, SC6.

**Critère de sortie** : un run complet passe de bout en bout, produit un col_data valide envoyé à la queue. Profiling temps/étape pour vérifier que la distribution des coûts est cohérente (P1 léger, P2 dominé par couche B aux virages, postprocess léger).

---

### SC8 — Refonte de l'estimation VRAM et du hub

**Objectif** : `estimate_gpu_peak` dans `hub.py` est recalibré pour le nouveau modèle de buffers. Le split_job utilise la nouvelle estimation. **Aucun changement d'architecture du hub** — juste la fonction d'estimation et son utilisation.

**Livrables** :
- Nouvelle `estimate_gpu_peak(rank, n_dof, max_it, B, is_diff)` qui modélise :
  - État scan P1 : 3 × B × state + 4 × B × max_it (timelines screening) + intermédiaires XLA
  - État scan P2 : 3 × B × state + 11 × B × max_it (couche A dense) + 15 × B × max_it (couche B NaN-éparses) + sigmas_buf + coût JVP conditionnel (amortiz par K_moyen estimé)
- Nouvelle constante `EXPECTED_MASK_DENSITY` (heuristique initiale ~25% sur la base de l'état de l'art actuel, à calibrer)
- Tests de l'estimation sur plusieurs configurations (rank 2, n_dof 10, B 128 ; rank 3, n_dof 100, B 16) vs. mesure VRAM réelle

**Dépendances** : SC5 (pour avoir le modèle réel des buffers).

**Critère de sortie** : sur les configurations limites actuelles qui OOM, le nouveau pipeline passe. La marge d'estimation est ≤ 20% par rapport à la VRAM réelle mesurée.

---

### 4.1 Ordre d'exécution

```
SC1 (registre) 
   ├→ SC2 (P1 minimal)
   ├→ SC3 (builder couche A)
   └→ SC4 (builder couche B)
         └→ SC5 (P2 complet)
               └→ SC6 (postprocess)
                     └→ SC7 (SubBatchProcess)
                           └→ SC8 (hub + estimation)
```

**SC2, SC3, SC4 peuvent être menés en parallèle** après SC1 validé, car ils ne dépendent que du registre. SC5 est le point de convergence — il attend SC3 et SC4 minimum, SC2 optionnellement (on peut garder l'ancien P1 temporairement et bascule à la fin).

**Recommandation : ne pas paralléliser**. Travailler séquentiellement SC1 → SC2 → SC3 → SC4 → SC5 → SC6 → SC7 → SC8 pour garder le contrôle et éviter les interactions non maîtrisées entre changements. La vitesse d'itération vient de la **relecture courte** (CHARTER §5.6 — modifications séquentielles), pas du parallélisme.

---

## 5. EXPÉRIENCES STANDALONE RECOMMANDÉES

Le CHARTER §5.2 et l'ancrage §V.2 imposent des expériences standalone avant tout changement radical. Voici celles qui sont pertinentes pour le chantier 1 :

### EXP-C1-1 — Profil VRAM avant / après retrait JVP de P1

**Objectif** : vérifier empiriquement que le retrait du JVP de P1 est la principale source de réduction VRAM attendue.

**Protocole** :
1. Prendre le pipeline actuel, mesurer la VRAM peak sur une configuration connue pour OOM aux limites (candidats : rank 3, n_dof 100, max_it 1000, B=16 ; et rank 2, n_dof 20, max_it 1500, B=128).
2. Désactiver manuellement `build_dynamic_vmap` dans P1 (retourner des NaN à la place du JVP), ne toucher à rien d'autre.
3. Re-mesurer la VRAM peak.
4. Comparer.

**Critère de succès** : la réduction VRAM est ≥ 30% sur au moins une configuration. Si ≥ 30% → D3 est validé empiriquement. Si < 30% → la pression VRAM vient d'ailleurs et il faudra instrumenter davantage avant de s'engager.

**Quand la faire** : **avant SC2**. C'est un prérequis de validation de D3.

**Effort** : ~2 heures (modification temporaire dans `jit_builders.py`, profiling, rollback).

---

### EXP-C1-2 — Distribution K_i sur données réelles

**Objectif** : valider empiriquement l'hétérogénéité intra-batch des K_i qui a motivé le choix de `lax.cond` par sample et non d'un buffer compact `(B, K_max)`.

**Protocole** :
1. Sur le parquet `test_v9_baseline` (640 runs disponibles), extraire le `meta_n_svd` pour tous les runs.
2. Grouper par kernel_group (gamma_id × rank × n_dof) pour retrouver les batchs réels.
3. Pour chaque batch, calculer K_min, K_max, K_mean, K_std, histogramme.
4. Calculer le ratio `K_max / K_mean` par batch — c'est le facteur de gaspillage si on utilisait un buffer compact.

**Critère de succès** : le ratio K_max / K_mean est ≥ 5 en moyenne, confirmant l'hétérogénéité extrême. Si < 5 → le buffer compact redevient viable et on peut simplifier SC4.

**Quand la faire** : **avant SC4**. Si le ratio est faible, SC4 peut adopter une architecture plus simple.

**Effort** : ~3 heures (script de lecture parquet + statistiques + graphique).

---

### EXP-C1-3 — Non-régression scientifique des features couche B

**Objectif** : vérifier que les features couche B (F1, F2, F3, F4) produisent les **mêmes valeurs** dans le nouveau pipeline que dans l'ancien, pour un même state forgé.

**Protocole** :
1. Prendre un state JAX fixé (seed déterministe, rank 2 et rank 3, différentiable et non-différentiable)
2. Calculer les 15 features couche B avec l'ancien code (`_f1_spectral`, `_f2_informational`, `_f3_rank2/3`, `build_dynamic_vmap` partie F4)
3. Calculer les mêmes 15 features avec le nouveau `build_layer_b_fn()`
4. Comparer valeur par valeur, tolérance `1e-6`.

**Critère de succès** : toutes les valeurs sont identiques à `1e-6` près. Aucune divergence numérique.

**Quand la faire** : **pendant SC4**, comme critère de sortie.

**Effort** : ~2 heures (script comparatif).

---

### EXP-C1-4 — Non-régression scientifique du post-process sur timelines couche A

**Objectif** : vérifier que DMD sur observables, F6 sur paires, autocorr, PNN, temporal sur les signaux renommés produisent des valeurs **différentes de l'ancien pipeline mais cohérentes dans la nouvelle sémantique**.

**Protocole** :
1. Prendre un run OK du pipeline actuel avec ses scalaires post-scan (DMD, F6, autocorr, PNN, temporal sur VN/asym/frob).
2. Reconstituer les timelines couche A à partir des states stockés (si disponibles) ou relancer le même run avec le nouveau P2 pour extraire les timelines.
3. Appliquer les nouvelles fonctions post-process.
4. Comparer les valeurs obtenues. Elles seront **différentes** (c'est attendu — le signal source change) mais doivent être **cohérentes en ordre de grandeur et en signe**.

**Critère de succès** : aucun crash, aucun NaN imprévu, ordre de grandeur plausible, et **cohérence qualitative** : un run qui était "oscillating" en analysing doit toujours avoir une période détectée et un DMD radius < 1 ; un run "flat" doit avoir un DMD radius ≈ 1.

**Quand la faire** : **pendant SC6**, comme critère de sortie.

**Effort** : ~1 jour (script + interprétation).

---

### EXP-C1-5 — Calibration EXPECTED_MASK_DENSITY

**Objectif** : calibrer la constante machine `EXPECTED_MASK_DENSITY` utilisée par l'estimation VRAM pour amortir le coût JVP en couche B.

**Protocole** :
1. Sur le même parquet de référence que EXP-C1-2, calculer la densité moyenne du masque = K_i / t_effective pour chaque run.
2. Calculer la médiane, le 75e percentile, le max.
3. Utiliser le 75e percentile comme constante machine initiale (marge de sécurité).

**Critère de succès** : la constante est documentée avec sa provenance, et peut être ajustée ultérieurement si les prochaines phases montrent une dérive.

**Quand la faire** : **pendant SC8**.

**Effort** : ~1 heure.

---

## 6. CRITÈRES DE SUCCÈS DU CHANTIER 1

Le chantier 1 est considéré terminé quand **tous** les critères suivants sont satisfaits :

### 6.1 Critères fonctionnels

- [ ] Un run de référence (rank 2, n_dof 10, max_it 100, B=16) passe de bout en bout avec le nouveau pipeline
- [ ] Un run différentiable et un run non-différentiable passent sans erreur
- [ ] Les 4 statuts OK/OK_TRUNCATED/EXPLOSION/COLLAPSED sont tous correctement produits
- [ ] Parquet produit contient les 11 timelines complètes + 15 tronquées + ~90 scalaires
- [ ] Les NaN structurels et runtime sont correctement placés (pas d'artefacts)
- [ ] Les doublons (frob_norm, shannon_entropy, SVD dupliquée) sont éliminés

### 6.2 Critères scientifiques (non-régression)

- [ ] Les 15 features couche B produisent des valeurs identiques (à `1e-6`) à l'ancien pipeline pour le même state (EXP-C1-3)
- [ ] Les features post-process en grammaire signal produisent des valeurs **nouvelles** (c'est attendu — nouveau signal source) mais **cohérentes qualitativement** (EXP-C1-4)
- [ ] Les features phasic produisent les mêmes valeurs que l'ancien pipeline (grammaire virages inchangée)

### 6.3 Critères de performance

- [ ] La VRAM peak sur une configuration limite est réduite d'au moins 30% par rapport à l'ancien pipeline (EXP-C1-1)
- [ ] Le temps total d'un run de référence est **au mieux 10% plus long** que l'ancien (dégradation acceptable car le gain est ailleurs). **Idéalement plus rapide** grâce au retrait du JVP de P1.
- [ ] La fraction de compile time / run time est stable ou améliorée
- [ ] Aucun OOM sur les configurations qui OOM aujourd'hui aux limites (rank 3 n_dof 100 max_it 1000+)

### 6.4 Critères d'architecture

- [ ] Aucun import cross-layer non documenté (P8)
- [ ] Aucun paramètre hardcodé en code Python (P4)
- [ ] Aucune SVD en dehors de la couche B (pas de double SVD résiduelle)
- [ ] Aucun JVP en dehors de la couche B
- [ ] Les docstrings de tous les modules modifiés respectent le format CHARTER §6

### 6.5 Critères documentaires

- [ ] DOC1 (features) est à jour avec la structure finale implémentée
- [ ] `features_registry.py` contient les commentaires détaillés correspondant à DOC1
- [ ] Ce document (DOC2 roadmap) est mis à jour avec les décisions prises pendant l'implémentation
- [ ] Le DOC3 (patchs charter/ancrage) est appliqué si le chantier a révélé des clarifications philosophiques à inscrire

---

## 7. RISQUES IDENTIFIÉS ET MITIGATIONS

### R1 — Le retrait du JVP de P1 ne libère pas autant de VRAM qu'attendu

**Probabilité** : faible (mon estimation est que c'est le vecteur principal, mais je n'ai pas de profilage chiffré).
**Impact** : moyen — il faut alors trouver la vraie source (buffers XLA ? spec_tables ? sigmas_buf ?) et potentiellement réintroduire le chunking de P2.
**Mitigation** : EXP-C1-1 est **obligatoire avant SC2**. Si elle échoue, on instrumente davantage et on revoit D3.

### R2 — DMD sur observables est qualitativement différent de DMD sur sigmas au point de perdre l'ancrage §1.4

**Probabilité** : modérée.
**Impact** : élevé — c'est le test d'universalité, cœur de la thèse PRC.
**Mitigation** : EXP-C1-4 doit vérifier que le nouveau DMD détecte les régimes attendus (oscillants, stables, divergents). Si le DMD sur observables est sémantiquement trop faible, on peut garder un **DMD de backup sur sigmas aux virages** en plus du DMD sur observables. Coût : un champ scalaire supplémentaire + une fonction CPU post-scan. Pas une catastrophe.

### R3 — Le nouveau F6 sur paires d'observables perd le pouvoir discriminant de l'ancien F6

**Probabilité** : modérée (les nouvelles paires sont différentes des anciennes).
**Impact** : moyen — F6 sert à la causalité émergente, Q4.
**Mitigation** : sur quelques runs de test, comparer le pouvoir discriminant du nouveau F6 vs l'ancien (en analysing). Si le nouveau est trop faible, on peut ajouter d'autres paires ou garder un subset des anciennes sur des signaux approximés.

### R4 — Les timelines tronquées (list<f32> variable) sont difficiles à manipuler côté analysing

**Probabilité** : faible (parquet list<float32> est bien supporté, pyarrow le gère)
**Impact** : moyen — si l'analysing doit les charger massivement, ça peut ralentir.
**Mitigation** : l'analysing peut d'abord ne charger que les scalaires (qui sont calculés à partir des timelines tronquées mais disponibles directement). Les timelines ne sont chargées que pour les analyses qui en ont spécifiquement besoin (chantier 2).

### R5 — Le chunking P2 s'avère nécessaire malgré tout

**Probabilité** : modérée (si EXP-C1-1 montre que le JVP n'est pas le dominant)
**Impact** : élevé — ça ajoute un sous-chantier SC9 non planifié.
**Mitigation** : si EXP-C1-1 montre < 30% de réduction, on planifie SC9 Chunking avant SC7. Pas de bifurcation après SC7 (ce serait trop tard).

### R6 — Le recouvrement P1/P2 des 4 timelines screening crée une incohérence

**Probabilité** : faible
**Impact** : faible — c'est juste 4 timelines calculées deux fois, quelques Mo de surcoût VRAM en P2.
**Mitigation** : acceptable. La simplicité architecturale (couche A autonome, P1 minimal) justifie la redondance.

---

## 8. HOOKS VERS LES CHANTIERS SUIVANTS

Le chantier 1 prépare le terrain pour les 5 chantiers suivants. Voici les **contrats explicites** que le chantier 1 doit respecter pour que ces chantiers soient possibles sans re-architecturer.

### → Chantier 2 — Analyse des timelines complètes

**Ce que le chantier 1 produit** : 11 timelines complètes en parquet, uniformément échantillonnées jusqu'à `t_effective`, nommées et typées explicitement dans le schéma.

**Ce que le chantier 2 recevra** : un outillage d'analyse des trajectoires. Détection de régimes par morceaux (HMM, changepoint), DTW entre runs, clustering temporel, identification de motifs.

**Hook explicite** : le schéma parquet du chantier 1 **doit** permettre à l'analysing de charger uniquement les timelines couche A par run sans charger le reste. Les colonnes `timeline_A_*` sont des listes de longueur `t_effective`, indépendantes les unes des autres.

**Piège à éviter** : ne pas stocker les timelines couche A en mode "matrice dense paddée" (qui gaspillerait pour les runs tronqués). Stockage par run en longueur variable.

### → Chantier 3 — Strates pathologiques / frontières de viabilité

**Ce que le chantier 1 produit** : les runs OK_TRUNCATED, EXPLOSION, COLLAPSED conservent leurs timelines couche A jusqu'à `t_effective`, leur `t_effective` est explicite, leur trajectoire avant la pathologie est intacte.

**Ce que le chantier 3 recevra** : un outillage pour analyser les frontières — comparer un OK_TRUNCATED à ses cousins OK, détecter les signatures de pré-transition, cartographier les zones de non-viabilité dans l'espace des atomics.

**Hook explicite** : les timelines couche A sont calculées **jusqu'à t_effective** et pas seulement sur la partie sûrement stationnaire. L'état juste avant la pathologie est préservé dans les timelines. Le statut est clairement différencié entre OK_TRUNCATED (transition détectée) et EXPLOSION (corruption directe).

**Piège à éviter** : ne pas exclure les runs pathologiques du calcul couche A "pour économiser". Ce sont précisément les runs les plus informatifs pour ce chantier.

### → Chantier 4 — Cluster namer enrichi

**Ce que le chantier 1 produit** : un pool de scalaires enrichi (stationnarité, IPR, volume_proxy, DMD sur observables, F6 sur paires) qui donne au namer des dimensions d'observation supplémentaires.

**Ce que le chantier 4 recevra** : un namer qui peut construire des noms à partir des nouvelles features (par exemple : "IPR-high · DMD-stable · STAT-converged" pour un run à information localisée, linéarisable stable, stationnaire).

**Hook explicite** : les noms exacts des nouveaux scalaires dans le parquet sont figés pendant le chantier 1 et ne changeront plus. Le namer peut s'appuyer dessus sans re-codage.

**Piège à éviter** : ne pas multiplier les scalaires au-delà de ce qui a un sens scientifique clair, sinon le namer produit des noms inexploitables.

### → Chantier 5 — Visualiseur santé atomique

**Ce que le chantier 1 produit** : les features structurées par les 4 questions du charter (tableau récapitulatif DOC1 section 8).

**Ce que le chantier 5 recevra** : un visualiseur qui affiche la santé de chaque atomic en écart au point sain théorique, par question scientifique. Q1 (rang effectif, entropie, asymétrie), Q2 (lyapunov, jacobien, géométrie), Q3 (stationnarité, phasic, convergence), Q4 (DMD universalité, F6 causalité, holographie VN).

**Hook explicite** : chaque feature du parquet est **étiquetée** par sa question de charter principale. Cette étiquette peut être extraite du `features_registry.py` (métadonnée `question` pour chaque feature) ou reconstruite depuis DOC1 section 8.

**Piège à éviter** : ne pas figer l'association feature ↔ question dans le chantier 1 au point où l'ajout d'une feature dans le chantier 2+ serait bloqué. Laisser le mapping métadonné pour être mis à jour facilement.

### → Chantier 6 — Verdict et rapport par contraintes

**Ce que le chantier 1 produit** : le vocabulaire complet de ce qu'on peut observer d'un atomic (via l'enrichissement couche A) et la cohérence grammaticale du post-process (qui rend les observations fiables).

**Ce que le chantier 6 recevra** : un système de verdict basé sur des contraintes du type "l'analyse montre que les gammas de la famille X ne peuvent pas satisfaire au critère Y parce que la feature Z est systématiquement pathologique pour cette famille".

**Hook explicite** : les features sont interprétables individuellement (DOC1 section 3 à 5 donne "une valeur élevée signifie X"). Le verdict peut construire des règles "si feature ∈ zone pathologique → contrainte sur l'atomic".

**Piège à éviter** : ne pas produire de features dont l'interprétation est ambiguë. Chaque scalaire doit avoir une direction de préférence explicite (plus grand = mieux ? pire ? neutre ?).

---

## 9. GLOSSAIRE OPÉRATIONNEL DU CHANTIER

| Terme | Définition |
|---|---|
| **Couche A** | Pool de 11 observables canoniques O(n²) calculés à chaque step P2 pour tous les runs, en timeline complète. Nourrit la grammaire signal. |
| **Couche B** | Pool de 15 features coûteuses (spectrales O(n³) + JVP F4) calculées aux virages via `lax.cond`. Nourrit la grammaire virages. |
| **Grammaire signal** | Méthodes de traitement du signal (DMD, F6 TE, autocorr, PNN, temporal) qui supposent un échantillonnage uniforme. Appliquée uniquement à la couche A. |
| **Grammaire virages** | Méthodes ordinales (agrégats, phasic) qui ne supposent pas d'équidistance. Appliquée à la couche B et aux sous-échantillonnages de couche A. |
| **P1 minimal** | Scan temporel réduit à propagate + screening (4 scalaires). Aucune différentiation, aucun JVP. |
| **Masque** | Sélection des instants d'intérêt produite par classify. Schématisation des virages, pas sous-échantillonnage. |
| **Observable canonique** | Fonction du state invariante par reparamétrisation triviale (permutation d'axes, changement d'échelle) et calculable en O(n²). |
| **K_i** | Nombre de points du masque pour le run i. Variable, hétérogène intra-batch (min 5, max 1000, mean 40 sur test_v9_baseline). |
| **t_effective** | Durée effective du run : max_it pour les OK, instant de la pathologie pour EXPLOSION et OK_TRUNCATED. |
| **Stationnarité** | Feature ajoutée en post-scan : `|mean(last 20%) - mean(first 20%)| / (std + eps)` par observable couche A. |
| **IPR** | Inverse Participation Ratio, mesure de localisation de l'information dans les composantes de D. |
| **Volume proxy** | Mesure O(n²) de contraction/expansion sans différentiation, accessible aux gammas non-différentiables. |
| **Mode asymmetry O(n²)** | Asymétrie structurelle `||M - M^T||_F / ||M||_F` de D replié en matrice carrée. Déplacé de la couche spectrale à la couche A. |
| **Mask_t_indices** | Tableau des indices temporels du masque pour un run, partagé entre toutes les features couche B tronquées de ce run. |

---

## FIN DU DOCUMENT 2

*Ce document est la roadmap opérationnelle du chantier 1. Il est vivant : à chaque sous-chantier terminé, mettre à jour la section correspondante avec les décisions prises et les écarts constatés. À chaque bifurcation non planifiée, ajouter un paragraphe dans la section Risques et mettre à jour les hooks si nécessaire.*

*Prochaine étape après validation de ce document : rédaction du DOC3 — patchs charter + ancrage.*
