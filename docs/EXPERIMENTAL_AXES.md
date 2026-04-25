# AXES EXPÉRIMENTAUX PRC — Résultats d'analyse et guide YAML

> Document de référence — résultats empiriques sur l'influence des axes
> du produit cartésien et template YAML annoté pour les futures phases.
> Basé sur deux campagnes : bourrin_scaled (100k runs, n_dof≤50)
> et bourrin_scaled2 (8.4k runs, n_dof≤200).

---

## SECTION 1 — RÉSULTATS D'ANALYSE

### 1.1 Méthodologie

Métrique : ω² (omega-squared), variante corrigée de η² qui pénalise
le nombre de niveaux d'un facteur. Dans un design factoriel complet
(produit cartésien), ω² isole l'effet principal de chaque facteur.

Complément : Cramér's V sur la distribution des statuts (OK, EXPLOSION,
COLLAPSED) pour détecter les axes qui ne changent pas les features
moyennes mais changent la survie du run.

Campagnes d'analyse :

| Campagne | Runs | n_dof | max_it | seeds | modifiers |
|----------|------|-------|--------|-------|-----------|
| bourrin_scaled | 100,920 | 10, 20, 50 | 100 | CI×2, run×2 | M0, M1×2, M2×2 |
| bourrin_scaled2 | 8,410 | 10, 20, 50, 100, 200 | 100 | CI×1, run×1 | M0 seul |

### 1.2 Axes globaux — résultats consolidés

| Axe | ω² médian | ω² max | Régime (V) | Verdict | Confiance |
|-----|-----------|--------|------------|---------|-----------|
| gamma_id | 14–27% | 77–83% | V=0.49–0.52 | **DOMINANT** | Très haute |
| encoding_id | 0.9–2.8% | 57–72% | V=0.18–0.21 | **Modéré, croissant avec n_dof** | Haute |
| n_dof | 0.1% | 55–56% | V=0.03 | Normalisé — effet absorbé | Voir §1.5 |
| modifier_id | 0.0% | 5.5% | V=0.06 | **Élagable** | Haute |
| seed_CI | 0.0% | 0.1% | V<0.01 | **Élagable** | Très haute |
| seed_run | 0.0% | 0.0% | V<0.01 | **Élagable** | Très haute |

### 1.3 Interprétation axiomatique

**Γ domine (gamma_id = 14–27%)** :
Le mécanisme d'organisation de l'information est le facteur principal.
Changer Γ change radicalement le régime comportemental : convergence,
oscillation, explosion, collapse. Ses paramètres de couplage (beta, omega,
eta) modulent la force et la nature de l'organisation. C'est cohérent
avec le rôle fondamental de Γ dans les axiomes — c'est le moteur.

**D a un effet structurel, pas scalaire (encoding_id = 0.9–2.8%)** :
L'encodage de l'information initiale n'influence presque pas les invariants
normalisés (ω² médian faible), mais influence fortement certaines features
géométriques (ω² max = 72% sur f3_mode_asymmetry, f1_spectral_gap) et
surtout la survie du run (V=0.21, COLLAPSED varie de 0% à 22% selon
l'encoding). La forme de D détermine si D résiste à Γ (A3), pas comment
Γ agit (A2).

**Tendance dimensionnelle** :
L'effet de l'encoding est stable entre n_dof=10 et n_dof=200 (ω² médian
1.0–4.3%, pas de tendance claire). La topologie de D ne devient pas
radicalement plus importante à haute dimension dans la plage testée.
Ce résultat reste à confirmer à n_dof=500+.

**Les seeds ne comptent pas** :
Γ "oublie" les conditions initiales. Deux runs qui ne diffèrent que par
la seed CI ou la seed stochastique de propagation convergent vers les
mêmes invariants. Cela signifie que les attracteurs de Γ sur D sont
robustes — ce n'est pas du chaos sensible aux conditions initiales,
c'est de l'organisation structurelle.

**Les modifiers ne comptent pas** :
Perturbation de D avant propagation (bruit, scaling) n'a aucun effet
détectable. Cohérent avec l'insensibilité aux seeds — si les conditions
initiales exactes ne comptent pas, les perturbations de ces conditions
ne comptent pas non plus.

### 1.4 Params gamma — résultats par atomic

**Params à varier (ω² > 2%, effet scientifique confirmé)** :

| Atomic.Param | ω² médian | ω² max | Effet |
|--------------|-----------|--------|-------|
| GAM-005.omega | 6–65% | 100% | Fréquence d'oscillation — change radicalement le régime |
| GAM-006.beta | 3–8% | 99% | Couplage fort → collapse avec certains encodings |
| GAM-011.scale | 0.1–7% | 100% | Échelle du mécanisme |
| GAM-001.beta | 3–6% | 99% | Couplage — beta=3 collapse avec SYM haute corrélation |
| GAM-009.beta | 5–6% | 55–72% | Couplage dans le régime stochastique |
| GAM-012.beta | 4–5% | 81–95% | Couplage structurel — beta=3 collapse |
| GAM-004.gamma | 3–4% | 100% | Intensité |
| GAM-007.epsilon | 1–4% | 100% | Perturbation non-markovienne |
| GAM-010.sigma | 1–2% | 99–100% | Amplitude stochastique |
| GAM-002.alpha | 0.3–2% | 100% | Couplage rank-2 |

**Params élagables (ω² < 1%, confirmé sur les deux campagnes)** :

| Atomic.Param | ω² médian | Fixer à | Raison |
|--------------|-----------|---------|--------|
| GAM-008.gamma | 0.0% | 0.3 | Ni beta ni gamma ne changent le collapse (26% fixe) |
| GAM-008.beta | 0.1% | 1.0 | Idem — le collapse est structurel, pas paramétrique |
| GAM-006.alpha | 0.0% | 0.3 | Seul beta compte pour GAM-006 |
| GAM-003.gamma | 0.0% | 0.03 | ω² max=100% sur une feature mais médian=0 |

### 1.5 Params encoding — résultats par atomic

Tous les params d'encoding ont ω² médian < 0.5% sur les deux campagnes.
Aucun param d'encoding individuel n'influence significativement les
features moyennes. L'identité de l'encoding (SYM vs ASY vs RN) compte
plus que ses paramètres internes.

**Params élagables (tous)** :

SYM-005.amplitude, SYM-005.bandwidth, SYM-006.n_blocks, SYM-006.intra,
SYM-006.inter, SYM-007.correlation, SYM-008.mean, SYM-008.std,
SYM-003.sigma, ASY-004.gradient, ASY-004.noise, ASY-006.density,
R3-003.radius, R3-007.n_blocks, R3-007.intra, R3-007.inter,
RN-*.rank (mais voir §1.7 sur R3-007.inter et SYM-006.inter).

**Exception régime** : R3-007.inter (COLLAPSED écart 17–28%) et
SYM-006.inter (COLLAPSED écart 7–8%) influencent la survie du run
malgré ω² ≈ 0% sur les features. Si on veut étudier la résistance
de D à Γ (A3), conserver ces variations. Si on veut seulement
caractériser Γ, fixer.

### 1.6 n_dof — interprétation

ω² = 0.1% ne signifie PAS que n_dof est sans importance.
Les features sont normalisées : entropies, ratios, rang effectif.
Un tenseur 10×10 et un tenseur 200×200 avec la même distribution
spectrale relative donneront les mêmes valeurs normalisées.

Ce que n_dof change :
- La richesse structurelle possible (modes couplés, attracteurs complexes)
- Le coût computationnel (O(n³) pour les SVD)
- Le régime de convergence de certains Γ

Ce que n_dof ne change pas (dans la plage testée) :
- Les invariants normalisés du spectre
- Le taux de pathologie (V=0.03 stable)

Recommandation : garder au moins 2 valeurs (un petit pour le dev rapide,
un grand pour la validation structurelle). Ne pas élaguer à 1 valeur.

### 1.7 Atomics pathologiques

**GAM-013 (hebbien) — RETIRER** :

Formule : `T_{n+1} = T + η·(T @ T)`. Rétroaction quadratique positive
sans dissipation. L'explosion est mathématiquement garantie — les runs
"OK" sont en transitoire. Diverge en temps fini pour tout η > 0.
Seul gamma à produire des EXPLOSION (100% des explosions du pipeline).
Pas un candidat Γ viable en solo. Utilisable en composition pondérée
(composante expansive) dans les phases futures.

| Param | Runs | %EXP (scaled) | %EXP (scaled2) |
|-------|------|---------------|-----------------|
| eta=0.005 | 2220 / 185 | 50% | 26% |
| eta=0.05 | 2220 / 185 | 94% | 91% |

**GAM-008 — CONSERVER mais documenter** :

Collapse à 16–26% constant, indépendant de beta et gamma.
Le collapse est structurel au mécanisme, pas paramétrique.
Fortement dépendant de l'encoding : SYM-007 = 95–100% collapse,
ASY-* = 0% collapse. Information A3 précieuse.

**GAM-001, GAM-006, GAM-012 — CONSERVER** :

Collapse uniquement à couplage fort (beta ≥ 2) ET avec encodings
symétriques à haute corrélation (SYM-007, SYM-006).
À beta=1, zéro collapse. Le couplage fort + symétrie → D s'annule.
Le couplage faible ou encodings asymétriques → D résiste.
Résultat scientifiquement riche pour A3.

**Encodings pathologiques — CONSERVER** :

SYM-007 (22% collapse) et SYM-006 (10–19% collapse) ne sont PAS
défectueux — leur symétrie les rend vulnérables aux Γ contractants.
C'est une observable sur la relation D-Γ, pas un artefact.

---

## SECTION 2 — TEMPLATE YAML ANNOTÉ

### 2.1 Axes fixables

```yaml
# ─── AXES À FIXER (confirmé élagable sur 2 campagnes) ─────────────
# seed_CI : l'état initial de D est oublié par Γ en ~100 itérations.
# Pas d'effet sur les features (ω²=0.0%) ni sur les régimes (V<0.01).
seed_CI: [0]

# seed_run : la trajectoire stochastique ne change pas les invariants.
# Deux réalisations du même Γ stochastique produisent les mêmes observables.
# [0] = seed déterministe via _make_run_seed. [null] = aléatoire (reproductible
# au sein d'un run mais pas cross-runs).
seed_run: [0]

# modifier : la perturbation de D avant propagation n'a pas d'effet.
# Cohérent avec seed_CI=0% — si les conditions initiales ne comptent pas,
# les perturbations ne comptent pas.
modifier:
  - id: M0
    params: {}
```

### 2.2 Axes à varier — gammas

```yaml
# ─── GAMMAS — l'axe dominant ──────────────────────────────────────
# gamma_id + gamma_params = 40–70% de la variance des features.
# TOUJOURS varier : l'identité du gamma ET ses paramètres de couplage.
#
# RETIRÉS :
#   GAM-013 (hebbien) — explosion structurelle, pas de dissipation
#
# PARAMS FIXES (confirmé ω² < 0.1%) :
#   GAM-008.beta → 1.0, GAM-008.gamma → 0.3 (collapse structurel, pas paramétrique)
#   GAM-006.alpha → 0.3 (seul beta compte)
#
# PARAMS À VARIER (ω² > 2%) :
#   Tous les autres params de couplage (beta, omega, gamma, epsilon, sigma, scale, eta)
gamma:
  - id: GAM-001
    params:
      beta: [1.0, 3.0]            # 6% — couplage faible vs fort, collapse à 3.0

  - id: GAM-002
    params:
      alpha: [0.05, 0.10]         # 2% — couplage rank-2

  - id: GAM-003
    params:
      gamma: [0.03]               # fixer — ω²=0%, 100% OK

  - id: GAM-004
    params:
      gamma: [0.03, 0.08]         # 4% — intensité

  - id: GAM-005
    params:
      omega: [0, 0.785, 1.571]    # 65% — LE param le plus discriminant du pipeline

  - id: GAM-006
    params:
      beta: [1.0, 2.0]            # 8% — collapse à beta=2
      alpha: [0.3]                # fixer — ω²=0%

  - id: GAM-007
    params:
      epsilon: [0.1, 0.3]         # 4% — perturbation non-markovienne

  - id: GAM-008
    params:
      gamma: [0.3]                # fixer — ω²=0%, collapse structurel
      beta: [1.0]                 # fixer — ω²=0.1%

  - id: GAM-009
    params:
      beta: [1.0, 3.0]            # 6%
      sigma: [0.01, 0.1]          # 2% — frontière élagable, conserver par prudence

  - id: GAM-010
    params:
      sigma: [0.01, 0.1]          # 2%

  - id: GAM-011
    params:
      scale: [0.8, 1.2]           # 7%

  - id: GAM-012
    params:
      beta: [1.0, 3.0]            # 5% — collapse à 3.0

  # GAM-013 : RETIRÉ — explosion structurelle (hebbien sans dissipation)

  - id: GAM-014
    params: {}

  - id: GAM-015
    params:
      k: [2, 4]                   # 1% — frontière, conserver (troncature SVD)
```

### 2.3 Axes à varier — encodings

```yaml
# ─── ENCODINGS — effet modéré, surtout géométrique ───────────────
# encoding_id = 1–3% médian mais 72% max sur f3_mode_asymmetry.
# Influence la survie (COLLAPSED varie 0–22% selon encoding).
# TOUJOURS varier l'identité de l'encoding.
# Les params internes sont élagables (tous ω² < 0.5%).
#
# Si l'objectif est la caractérisation de Γ : fixer tous les params.
# Si l'objectif est l'étude de la résistance de D (A3) : conserver
# R3-007.inter et SYM-006.inter (écart COLLAPSED > 5%).
encoding:
  - id: SYM-001
  - id: SYM-002
  - id: SYM-003
    params: { sigma: 0.3 }              # fixer — ω²=0%
  - id: SYM-004
  - id: SYM-005
    params: { bandwidth: 2, amplitude: 0.3 }  # fixer
  - id: SYM-006
    params: { n_blocks: 4, intra: 0.5, inter: 0.05 }  # fixer (ou varier inter pour A3)
  - id: SYM-007
    params: { correlation: 0.3 }         # fixer
  - id: SYM-008
    params: { mean: 0.0, std: 0.3 }     # fixer

  - id: ASY-001
  - id: ASY-002
  - id: ASY-003
  - id: ASY-004
    params: { gradient: 0.1, noise: 0.1 }  # fixer
  - id: ASY-005
  - id: ASY-006
    params: { density: 0.1 }            # fixer

  - id: RN-001
    params: { rank: 2 }                 # fixer (ou varier pour cross-rank)
  - id: RN-002
    params: { rank: 2 }
  - id: RN-003
    params: { rank: 2 }
  - id: RN-004
    params: { rank: 2 }

  - id: R3-003
    params: { radius: 1 }               # fixer
  - id: R3-007
    params: { n_blocks: 3, intra: 0.6, inter: 0.05 }  # fixer (ou varier inter pour A3)
```

### 2.4 Axes structurels

```yaml
# ─── n_dof ────────────────────────────────────────────────────────
# ω²=0.1% sur les features normalisées — pas d'effet détectable.
# Mais n_dof change le coût (O(n³)) et la richesse structurelle.
# Garder au moins 2 valeurs : une petite (dev), une grande (validation).
# Phase dev : [10] ou [10, 50]
# Phase validation : [10, 50, 200]
# Phase stress : [10, 50, 200, 500]
n_dof: [10, 50, 200]

# ─── max_it ───────────────────────────────────────────────────────
# Non testé comme axe variable (toujours fixé à 100).
# Minimum fonctionnel : 20 (autocorrélation FFT crashe en-dessous).
# Recommandé : 100–200 pour la convergence, 500+ pour les transitoires longs.
# Les runs qui convergent vite (FLAT) n'ont pas besoin de plus.
# Les runs oscillants ou transitionnels bénéficient de max_it élevé.
max_it: [100]
```

---

## SECTION 3 — ESTIMATION COMBINATOIRE

### 3.1 Config maximale (exploration)

```
15 gammas × ~30 param combos × 20 encodings × 3 n_dof × 1 seed × 1 modifier
= ~27,000 runs
```

### 3.2 Config élaguée (caractérisation Γ)

```
14 gammas × ~20 param combos × 20 encodings × 2 n_dof × 1 seed × 1 modifier
= ~11,200 runs
```

### 3.3 Config minimale (dev rapide)

```
14 gammas × ~20 param combos × 5 encodings × 1 n_dof × 1 seed × 1 modifier
= ~1,400 runs
```

### 3.4 Facteurs de réduction appliqués

| Réduction | Facteur | Justification |
|-----------|---------|---------------|
| Seeds CI et run | ×4 | ω²=0.0% confirmé sur 100k+ runs |
| Modifiers | ×5 | ω²=0.0%, V<0.01, cohérent avec seeds |
| Params encoding fixés | ×2–8 selon l'encoding | ω² < 0.5% sur tous les params |
| GAM-013 retiré | -1 gamma | Explosion structurelle |
| Params gamma fixés | ×1.5 | GAM-008.beta/gamma, GAM-006.alpha, GAM-003.gamma |
| **Total estimé** | **×40–60** | 100k → 2–3k pour le même espace scientifique |

---

## SECTION 4 — QUESTIONS OUVERTES

**Q1 — Encodings à n_dof=500+ :**
L'effet de l'encoding est stable à n_dof≤200 (ω² médian 1–4%).
À n_dof=500 rank=3, la topologie de D pourrait créer des structures
que Γ exploite différemment. Non testable sans run dédié.

**Q2 — max_it comme axe :**
Non testé. Si Γ a des attracteurs lents, max_it=100 pourrait ne pas
suffire pour les observer. Un run à max_it=[100, 500] avec les gammas
non-markoviens (GAM-005, GAM-006, GAM-007) répondrait à la question.

**Q3 — GAM-013 en composition :**
Le hebbien amplifie toujours. En séquence pondérée avec un gamma
contractant (GAM-001 beta=1), le couple pourrait produire un équilibre
dynamique intéressant. Non testable avant les compositions.

**Q4 — Rank comme axe :**
RN-*.rank a ω²=0.5–1.2% — à la frontière. Le rank change la
géométrie tensorielle fondamentale (matrice vs cube). Un run dédié
rank=[2,3] avec un subset de gammas clarifierait.

---

**FIN DOCUMENT**

*Basé sur bourrin_scaled (100k runs) et bourrin_scaled2 (8.4k runs).*
*Métriques : ω² (omega-squared) et Cramér's V.*
*Scripts : analyse_axes_influence.py v3, diagnostic_atomics.py.*
*Reproductible via les YAML et parquets archivés.*
