# MANIFESTO PRC PIPELINE v7
> Refonte complète JAX — document milestone référence
> Remplace toutes versions antérieures
> Date : 2026-03-06

---

## PRÉAMBULE : Pourquoi ce refactor existe

Le pipeline PRC v6 reposait sur catch22 — un outil généraliste de time-series, conçu pour
détecter des motifs statistiques dans des signaux 1D. C'est un marteau universel.
PRC n'a pas besoin d'un marteau universel. PRC a besoin d'instruments de mesure
taillés sur ses axiomes.

Ce manifesto documente la décision de remplacer catch22 et toute l'infrastructure numpy
par JAX natif, et surtout de remplacer les features statistiques génériques par
**des invariants informationnels et géométriques directement ancrés dans A1, A2, A3**.

Le banc de test actuel n'est pas un moteur de simulation physique.
C'est un instrument de validation de faisabilité :
**les axiomes PRC peuvent-ils générer des structures émergentes mesurables ?**
Les features doivent répondre à cette question, pas caractériser des signaux.

---

## SECTION 1 — FONDATIONS THÉORIQUES

### 1.1 Les axiomes comme boussole computationnelle

```
A1 — Dissymétrie informationnelle D irréductible
A2 — Mécanisme Γ agissant sur D
A3 — Aucun Γ stable ne peut annuler D complètement
```

Ces trois axiomes définissent exactement ce qu'on doit mesurer :

- **A1** → mesurer D elle-même : son rang, son asymétrie, sa capacité d'encodage informationnel
- **A2** → mesurer l'action de Γ sur D : géométrie locale, contraction, expansion, conservation
- **A3** → mesurer la résistance de D sous Γ : est-ce que Γ dissipe D ? À quelle vitesse ?
  Existe-t-il un état attracteur où D ≠ 0 ?

### 1.2 Connexions théoriques qui informent les familles de features

PRC soutient que les théories actuelles (GR, QM, holographie, entropie entropique)
étudient le même objet sous des angles différents. Si les axiomes sont valides,
les invariants mesurés sur le pipeline doivent résonner avec chacun de ces angles.

**Holographie (Ryu-Takayanagi, Susskind)** :
L'entropie d'enchevêtrement entre régions est proportionnelle à l'aire de la surface
minimale. Si D encode une réalité pré-géométrique, l'entropie de Von Neumann de ses
bipartitions mesure la "quantité de géométrie" encodée. Sous Γ, si S_VN tend vers un
minimum stable → attracteur holographique candidat.

**Gravité entropique (Verlinde)** :
La masse émerge comme gradient de densité de corrélation. Si D est une matrice de
corrélation, le gradient de sa norme de Frobenius par mode est le précurseur de la
masse dans ce cadre. Pas une métaphore — une quantité mesurable.

**Géométrie relationnelle (Rovelli)** :
La géométrie de l'espace émerge des relations, pas des positions.
L'information mutuelle entre modes tensoriels de D est exactement cette structure
relationnelle. Sa densification sous Γ → émergence candidate d'une géométrie.

**Réseaux de causalité (Wolfram)** :
La causalité émerge de règles locales. La transfer entropy dirigée entre degrés de
liberté de D mesure si Γ crée du flux informationnel orienté — précurseur de
causalité émergente.

**Opérateur de Koopman** :
Toute dynamique non linéaire admet une représentation linéaire dans un espace
fonctionnel. Le spectre DMD de Γ sur D est la signature de cette linéarisation.
Si deux doublets (encoding, gamma) différents donnent le même spectre DMD →
universalité de Γ candidate.

### 1.3 La vraie question computationnelle

Pour chaque doublet candidat (encoding, gamma) :
```
1. D peut-il encoder de l'information structurée ? (rang, entropie, asymétrie)
2. Γ préserve-t-il, amplifie-t-il ou dissipe-t-il cette information ? (dynamique locale)
3. La dynamique produit-elle des structures émergentes stables ? (attracteurs, régimes)
4. Ces structures ont-elles les propriétés attendues d'une réalité pré-émergente ?
   (holographie, transport, causalité)
```

Le pipeline répond à ces 4 questions par des invariants numériques.
HDBSCAN regroupe les doublets qui répondent de la même façon.

---

## SECTION 2 — ARCHITECTURE GLOBALE v7

### 2.1 Principes architecturaux

**Rupture propre** : pipeline numpy conservé intact pour poc/poc2 existants.
v7 = pipeline parallèle, sélectionné via `engine: jax` dans le YAML de run.
Zéro conversion numpy ↔ JAX dans le chemin chaud.

**K1-K5 esprit conservé** : le kernel reste aveugle au domaine.
Implémentation change — lax.scan au lieu de boucle Python.
Aucun branchement sur D ou Γ, aucune validation sémantique, aucune classe State.

**Signature atomique unifiée** :
```python
# Gamma
def apply(state: jnp.ndarray, params: dict, key: jax.Array) -> jnp.ndarray: ...

# Encoding
def create(params: dict, n_dof: int, key: jax.Array) -> jnp.ndarray: ...

# Modifier
def apply(state: jnp.ndarray, params: dict, key: jax.Array) -> jnp.ndarray: ...

METADATA = {'id': 'GAM-001', 'stochastic': False, 'differentiable': True}
```

`key` PRNG JAX partout — reproducibilité garantie par construction.
`stochastic: False` → key ignoré. `differentiable: True` → jacfwd disponible.

### 2.2 Flux pipeline

```
YAML config (inchangé)
    ↓
compositions.py (inchangé — seed_CI/seed_run → PRNGKey)
    ↓
runner_jax.py
    ├── state_preparation_jax.py
    │     encoding.create(params, n_dof, key) → jnp D_initial
    │     modifier.apply(D, params, key)      → jnp D_modified
    │
    ├── kernel_jax.py
    │     jit(lax.scan(step_fn))
    │     step_fn(carry, t):
    │       state, key = carry
    │       key, subkey = jax.random.split(key)
    │       state_next = gamma.apply(state, params, subkey)
    │       measures   = jax_features.measure_state(state_next, state, gamma, params)
    │       return (state_next, key), measures
    │     → signals_array (T, n_features), last_state
    │
    ├── jax_features.py
    │     post_scan(signals_np, last_state_np) → features scalaires post-scan
    │
    └── features dict → hub_running → write_parquet

hub_running.py  (inchangé — boucle, progress, checkpoint)
write_parquet() (inchangé — colonnes plates)
analysing/      (branché phase 2 — lit parquet, colonnes différentes)
```

### 2.3 vmap — parallélisation seeds/params

```python
# Grouper par (gamma_id, encoding_id, n_dof, max_it)
# Batcher les seeds et params variant → vmap

run_batch = jax.vmap(run_single_jax, in_axes=(0, 0, None, None))
results   = run_batch(seed_keys, sigma_values, fixed_gamma_params, fixed_encoding_params)

# Poc2 bourrin 128M : vmap sur 25 seeds/params simultanés → ~5x gain minimal
```

### 2.4 Ce qui saute définitivement

| Fichier / dépendance | Remplacé par |
|----------------------|--------------|
| `kernel.py` | `kernel_jax.py` |
| `state_preparation.py` | `state_preparation_jax.py` |
| `runner.py` | `runner_jax.py` |
| `hub_featuring.py` | absorbé dans `runner_jax.py` |
| `extractor_lite.py` | inutile |
| `timeline_lite.py` | `jax_features.py` |
| `pycatch22` | supprimé |
| `regimes_lite.py` | déjà deprecated |
| Tous les atomics numpy | atomics JAX (même IDs) |
| Étape B calibration timeline | obsolète |

---

## SECTION 3 — FAMILLES DE FEATURES

Sept familles, sept angles sur la même réalité.
Toutes calculées sur l'état courant à chaque itération (dans lax.scan)
ou sur les signaux accumulés (post-scan, une fois par run).

Convention universelle d'entrée : mode-0 unfolding pour tout tenseur.
```python
M = state.reshape(state.shape[0], -1)   # (n_dof, n_dof^(rank-1))
sigmas = jnp.linalg.svd(M, compute_uv=False)  # σ₁ ≥ σ₂ ≥ ... ≥ 0
```
Les σᵢ sont le point de départ de la plupart des familles.
Calculés une fois par état, réutilisés dans F1, F2, F3.

---

### FAMILLE F1 — SPECTRALE
*Angle : structure de D, capacité d'encodage, rang informationnel*

**Pourquoi** : A1 dit que D est irréductible et asymétrique. La décomposition spectrale
de D révèle combien de "dimensions informationnelles" D utilise réellement.
Un D de rang effectif 1 est trivial — il n'encode qu'une direction.
Un D de rang plein avec distribution spectrale riche peut encoder une structure complexe.
La question "quel rang minimal pour encoder la réalité pré-émergente" se lit directement ici.

**Dans lax.scan — à chaque état :**

```python
# Depuis sigmas déjà calculés (réutilisés)
frob_norm = jnp.linalg.norm(state)   # ||D||_F — normalisation

# F1.1 — Rang effectif entropique (Roy & Vetterli 2007)
p = sigmas / (jnp.sum(sigmas) + eps)
effective_rank = jnp.exp(-jnp.sum(p * jnp.log(p + eps)))
# Interprétation : nb de dimensions spectrales actives

# F1.2 — Spectral gap
spectral_gap = (sigmas[0] - sigmas[1]) / (sigmas[0] + eps)
# Interprétation : dominance rang-1. Gap élevé → D quasi-séparable

# F1.3 — Nuclear/Frobenius ratio (rang effectif linéaire)
nuclear_frobenius_ratio = jnp.sum(sigmas) / (jnp.sqrt(M.shape[0]) * jnp.linalg.norm(sigmas) + eps)
# Interprétation : 1.0 = toutes directions égales (bruit). Faible = structure dominante

# F1.4 — Taux de décroissance spectrale
# polyfit log(σᵢ) vs i → pente = sv_decay_rate
log_sigmas = jnp.log(sigmas + eps)
i_values = jnp.arange(len(sigmas), dtype=float)
sv_decay_rate = -(jnp.cov(i_values, log_sigmas)[0,1] / (jnp.var(i_values) + eps))
# Interprétation : décroissance rapide → D très structuré (rang faible effectif)

# F1.5 — Rang-1 résiduel
sigma1_outer = jnp.outer(... U[:,0]*sigmas[0], ... V[0,:])  # T_rank1 via SVD complet si dispo
rank1_residual = jnp.linalg.norm(M - rank1_approx) / (frob_norm + eps)
# Interprétation : 0 = D parfaitement séparable. 1 = aucune structure rang-1

# F1.6 — Condition number
condition_number = sigmas[0] / (sigmas[-1] + eps)
# Interprétation : stabilité numérique + degré de singularité de D
```

**Signal PRC** : si sous Γ, le rang effectif augmente → Γ crée de l'information.
Si le spectral gap augmente → Γ sélectionne une direction dominante (émergence d'un mode).
Si sv_decay_rate accélère → Γ comprime D vers un sous-espace.

---

### FAMILLE F2 — INFORMATIONELLE (entropies)
*Angle : capacité d'encodage, production d'entropie, irréversibilité*

**Pourquoi** : L'entropie de Von Neumann est LE lien entre information et géométrie
dans le contexte holographique. Ryu-Takayanagi dit que l'entropie d'enchevêtrement
est proportionnelle à l'aire — si D encode une réalité pré-géométrique, S_VN est
le proxy de cette "aire". La production d'entropie dS_VN/dt sous Γ est la flèche
du temps dans ce cadre. A3 devient : Γ ne peut pas ramener S_VN à zéro.

**Dans lax.scan — à chaque état :**

```python
# F2.1 — Entropie de Von Neumann (depuis sigmas normalisés)
# D normalisé à trace 1 : p_i = σᵢ² / Σσⱼ²  (valeurs propres de D†D normalisées)
p_sq = sigmas**2 / (jnp.sum(sigmas**2) + eps)
von_neumann_entropy = -jnp.sum(p_sq * jnp.log(p_sq + eps))
# Interprétation : max = log(n) (bruit pur). Min = 0 (état pur, rang 1).
# Connexion Ryu-Takayanagi : S_VN ∝ aire de la surface minimale.

# F2.2 — Entropie de Rényi ordre 2 (plus robuste aux valeurs propres rares)
renyi2_entropy = -jnp.log(jnp.sum(p_sq**2) + eps)
# Interprétation : moins sensible aux petits σᵢ que Von Neumann.

# F2.3 — Entropie de Shannon sur distribution des éléments (existant, reformulé)
flat = state.reshape(-1)
hist_vals = jnp.sort(flat)  # JAX-compatible binning via histogramme différentiable
# Approximation JAX : moments de la distribution → entropie approchée
# ou : entropie de la distribution des valeurs absolues (invariant au signe)
shannon_entropy = entropy_from_moments(flat)  # détaillé dans jax_features.py

# F2.4 — Production d'entropie (post-scan, sur signal von_neumann_entropy)
# dS/dt ≈ (S_T - S_0) / T
# Signe : positif = irréversible. Négatif = auto-organisation (A3 candidat).
# Calculé en post-scan sur le signal accumulé.
```

**Signal PRC** : si S_VN décroît monotonement sous Γ → Γ auto-organise D
(candidat A3 fort). Si S_VN oscille → régime quasi-conservatif.
Si S_VN → 0 → Γ effondre D vers un état pur (collapse).
La production d'entropie négative est le signal le plus intéressant du pipeline.

---

### FAMILLE F3 — ENCHEVÊTREMENT INTER-MODES
*Angle : structure relationnelle de D, émergence géométrique*

**Pourquoi** : Si l'espace géométrique émerge de la densité de corrélation (Rovelli,
holographie), la structure relationnelle entre les modes tensoriels de D en est
le précurseur direct. L'entropie d'enchevêtrement d'une bipartition de D mesure
combien de géométrie est "encodée" dans cette bipartition.
Pour un tenseur rang-3, trois bipartitions → trois angles géométriques distincts.

**Dans lax.scan — à chaque état :**

```python
# Pour tenseur de rank r, modes k = 0, 1, ..., r-1
# Unfolding mode-k : M_k = state.reshape(shape[k], prod(autres dims))

# F3.1 — Entropie d'enchevêtrement par mode
for k in range(rank):
    M_k = jnp_unfold(state, mode=k)
    sigmas_k = jnp.linalg.svd(M_k, compute_uv=False)
    p_k = sigmas_k**2 / (jnp.sum(sigmas_k**2) + eps)
    entanglement_entropy_k = -jnp.sum(p_k * jnp.log(p_k + eps))
    # Interprétation : séparabilité du mode k par rapport au reste.
    # Élevée → mode k fortement corrélé aux autres → structure géométrique candidate.

# F3.2 — Asymétrie mode_asymmetry (A1 direct)
# Mesure directe de la dissymétrie informationnelle de D
min_dim = min(M_k.shape)
M_sq = M_k[:min_dim, :min_dim]
mode_asymmetry_k = jnp.linalg.norm(M_sq - M_sq.T) / (jnp.linalg.norm(M_k) + eps)
mode_asymmetry_mean = jnp.mean(jnp.stack([mode_asymmetry_k for k in range(rank)]))
# Interprétation : 0 = D symétrique (pas de D au sens A1). >0 = dissymétrie présente.
# Evolution sous Γ : si Γ réduit mode_asymmetry → candidat A3 (Γ tente d'annuler D).

# F3.3 — Variance inter-mode des valeurs singulières dominantes
sv1_per_mode = jnp.array([svd(unfold(state, k))[0] for k in range(rank)])
inter_mode_sv_var = jnp.var(sv1_per_mode)
# Interprétation : couplage anisotrope entre dimensions tensorielles.
```

**Signal PRC** : si entanglement_entropy_k augmente sous Γ → Γ corrèle les modes
(densification relationnelle → émergence géométrique candidate).
Si mode_asymmetry diminue → Γ symmétrise D (contre A1 → candidat à exclure).
La combinaison (entanglement croissant + asymétrie stable) est le signal le plus
intéressant pour valider le doublet (encoding, gamma).

---

### FAMILLE F4 — DYNAMIQUE LOCALE (Jacobien approximé)
*Angle : géométrie de l'action de Γ, universalité, contraction/expansion*

**Pourquoi** : Γ est une transformation de D. Sa dérivée locale (Jacobien) décrit
exactement comment Γ déforme l'espace au voisinage de D_t.
C'est la mesure directe de A2 — l'action de Γ.
La divergence trace(J) = ∂Γ/∂D est la contraction/expansion de l'espace des états.
Si trace(J) < 0 partout → Γ est contractant → attracteur garanti (Γ stable au sens A3).

**Contrainte technique** : J est de taille (n²×n²) pour un state (n×n).
Pour n=100, J est 10000×10000 = 800MB. Impraticable.

**Solution : estimateur Hutchinson via JVP** :
```python
# v ~ N(0,I), clé PRNG du scan
v = jax.random.normal(subkey, state.shape)

# jvp : calcule gamma(state+εv) direction sans construire J
_, jvp_v = jax.jvp(lambda s: gamma.apply(s, params, key), (state,), (v,))

# F4.1 — Trace(J) ≈ E[v · J · v] = E[v · jvp(v)] (Hutchinson)
trace_J = jnp.sum(v * jvp_v)
# Interprétation : >0 = expansion locale. <0 = contraction locale. |val| = intensité.
# Connexion Verlinde : contraction = pression entropique = masse émergente candidate.

# F4.2 — ||J·v|| ≈ spectral radius estimation
jvp_norm = jnp.linalg.norm(jvp_v)
# Interprétation : taux d'amplification local de Γ.

# F4.3 — Asymétrie du Jacobien
_, jvp_vT = jax.jvp(lambda s: gamma.apply(s, params, key), (state,), (jvp_v,))
jacobian_asymmetry = jnp.linalg.norm(jvp_v - jvp_vT) / (jvp_norm + eps)
# Interprétation : si Γ est symétrique localement → dynamique conservative.
# Si asymétrique → flux directionnel → causalité émergente candidate.

# F4.4 — Lyapunov local approximé
local_lyapunov = jnp.log(jvp_norm + eps)
# Post-scan : mean(local_lyapunov) → exposant de Lyapunov global approximé
# >0 → chaos. <0 → attracteur. ≈0 → marginal (transition).
```

**Pour gammas marqués `differentiable: True`**, jacfwd complet est disponible
pour les petits n_dof (≤50) où J est tractable :
```python
J = jax.jacfwd(gamma.apply)(state, params, key)
# F4.5 — Spectre singulier complet de J
s_J = jnp.linalg.svd(J.reshape(state.size, state.size), compute_uv=False)
lyapunov_spectrum_approx = jnp.log(s_J)  # λ₁ ≥ λ₂ ≥ ...
```

**Signal PRC** : trace(J) < 0 constant → Γ contractant, attracteur garanti.
local_lyapunov moyen > 0 → Γ chaotique sur D.
jacobian_asymmetry élevé → Γ crée un flux orienté (causalité candidate).
La combinaison (trace_J < 0, mode_asymmetry stable) est le signal d'un Γ
qui organise D sans la détruire — candidat A3 direct.

---

### FAMILLE F5 — TRANSPORT
*Angle : coût dynamique, précurseur de métrique, gradient de densité*

**Pourquoi** : Si la métrique de l'espace-temps émerge comme structure minimisant
le transport d'information (analogie avec Perelman, optimal transport),
le coût de Γ(D) → D' est une mesure directe de ce potentiel métrique.
Le gradient de la norme de Frobenius par mode est le précurseur de la masse
dans le cadre Verlinde : masse ∝ ∇||D||_F.

**Dans lax.scan — à chaque état (nécessite carry de l'état précédent) :**

```python
# carry : (state_t, key, state_prev)
state_prev = carry[2]

# F5.1 — Distance de Frobenius pas-à-pas
delta_D = jnp.linalg.norm(state - state_prev) / (jnp.linalg.norm(state) + eps)
# Interprétation : vitesse de déplacement dans l'espace des états.
# Grand δ → Γ déplace D rapidement. Petit → dynamique lente / attracteur proche.

# F5.2 — Gradient de norme par mode (précurseur masse Verlinde)
frob_per_mode = jnp.array([
    jnp.linalg.norm(jnp_unfold(state, k)) for k in range(rank)
])
frob_gradient = jnp.std(frob_per_mode) / (jnp.mean(frob_per_mode) + eps)
# Interprétation : hétérogénéité de la densité de corrélation entre modes.
# Élevé → gradient de densité fort → masse émergente candidate si Verlinde.

# F5.3 — Divergence de Bregman (transport sur matrices PSD)
# Si D est positive semi-définie (encodings symétriques garantissent cela) :
# D_Bregman(A||B) = ||A - B||_F² / ||A||_F  (approximation computationnelle)
bregman_cost = jnp.sum((state - state_prev)**2) / (jnp.linalg.norm(state)**2 + eps)
# Interprétation : coût "géodésique" du pas Γ sur l'espace des matrices PSD.

# Post-scan — coût total de transport de la trajectoire
# cumsum(delta_D) → transport total = longueur de trajectoire dans l'espace de D
```

**Signal PRC** : si delta_D décroît monotonement → convergence vers attracteur.
Si delta_D oscille avec amplitude stable → cycle dans l'espace de D.
Si frob_gradient augmente sous Γ → Γ crée des gradients de densité (masse candidate).

---

### FAMILLE F6 — CAUSALE (flux informationnel dirigé)
*Angle : émergence de la causalité, flux orienté dans D*

**Pourquoi** : Dans le cadre Wolfram, la causalité émerge de règles locales
qui créent un réseau orienté. Si D encode une réalité pré-causale, sous Γ on
devrait voir émerger un flux d'information dirigé entre degrés de liberté.
La transfer entropy mesure exactement cela : est-ce que le passé de X prédit
le futur de Y mieux que le passé de Y seul ?

**Contrainte** : transfer entropy exacte = O(T²) sur histogrammes.
**Solution JAX** : approximation linéaire via corrélations croisées décalées.

```python
# Post-scan uniquement — sur signaux accumulés (T, n_features)

# F6.1 — Corrélation croisée dirigée entre signaux (approximation transfer entropy)
# signal_A = frobenius_norm, signal_B = von_neumann_entropy
# TE(A→B) ≈ |corr(A_t, B_{t+1})| - |corr(A_{t+1}, B_t)|   (différence de causalité Granger)

def approx_transfer_entropy(signal_A, signal_B):
    A_lag = signal_A[:-1]
    B_next = signal_B[1:]
    B_lag = signal_B[:-1]
    A_next = signal_A[1:]
    te_AB = jnp.abs(jnp.corrcoef(A_lag, B_next)[0,1])
    te_BA = jnp.abs(jnp.corrcoef(B_lag, A_next)[0,1])
    return te_AB - te_BA   # >0 : A cause B. <0 : B cause A.

# Paires pertinentes :
# te_norm_to_entropy     : la norme cause-t-elle l'entropie ?
# te_entropy_to_asymmetry : l'entropie cause-t-elle la perte d'asymétrie ?
# te_lyapunov_to_rank    : l'expansion cause-t-elle un changement de rang ?

# F6.2 — Index de causalité globale
causal_asymmetry_index = jnp.mean(jnp.abs(jnp.array([te_AB - te_BA
                          for (A,B) in signal_pairs])))
# Interprétation : 0 = symétrique (pas de direction causale).
# >0 = flux informationnel orienté (causalité émergente candidate).
```

**Signal PRC** : si te_entropy_to_asymmetry > 0 → l'entropie cause la perte d'asymétrie
→ A3 émerge naturellement de la dynamique informationnelle.

---

### FAMILLE F7 — SPECTRALE DYNAMIQUE (DMD)
*Angle : universalité de Γ, signature spectrale du régime*

**Pourquoi** : L'opérateur de Koopman est la représentation linéaire exacte
d'une dynamique non linéaire. Le spectre DMD de Γ sur D est invariant à l'encoding :
si deux doublets (enc1, gamma) et (enc2, gamma) donnent le même spectre DMD,
Γ est universel indépendamment de l'encoding. C'est un test d'universalité direct.

**Implémentation streaming (compatible lax.scan)** :

```python
# Accumuler les corrélations dans le carry — pas de matrice X, X' en RAM

# carry : (state, key, C_xx, C_xy)   où C_xx, C_xy sont (n_dof², n_dof²)
# Note : n_dof² peut être grand — DMD pratique sur les premières r SVD modes

# Version réduite (tractable) :
# Accumuler (state_flat, prev_flat) → post-scan : DMD via SVD de corrélation

# Post-scan uniquement :
# X = signals[:-1]   # (T-1, n_dof²)
# X' = signals[1:]   # (T-1, n_dof²)
# via truncated SVD → eigenvalues DMD

# F7.1 — Rayon spectral (max |λᵢ|)
dmd_spectral_radius = jnp.max(jnp.abs(dmd_eigenvalues))
# |λ| < 1 → attracteur. |λ| ≈ 1 → cycle/oscillation. |λ| > 1 → divergence.

# F7.2 — Paires complexes (oscillations)
n_complex_pairs = jnp.sum(jnp.abs(jnp.imag(dmd_eigenvalues)) > threshold)
# >0 → Γ crée des oscillations dans D → régime ondulatoire.

# F7.3 — Entropie spectrale DMD
p_dmd = jnp.abs(dmd_eigenvalues) / (jnp.sum(jnp.abs(dmd_eigenvalues)) + eps)
dmd_spectral_entropy = -jnp.sum(p_dmd * jnp.log(p_dmd + eps))
# Interprétation : faible = un seul mode dominant. Élevée = dynamique complexe.

# F7.4 — Taux de décroissance dominant (Im(log(λ₁)))
dmd_decay_rate = jnp.real(jnp.log(dmd_eigenvalues[0] + eps))
# <0 = décroissance. >0 = croissance. 0 = marginal.
```

**Signal PRC** : le spectre DMD est la **signature de Γ indépendante de D**.
Deux runs avec encoding différent mais même spectre DMD → universalité de Γ.
C'est le test le plus direct de l'hypothèse que Γ est le mécanisme organisateur
indépendamment de la façon dont D encode l'information.

---

## SECTION 4 — FEATURES POST-SCAN

Calculées une fois par run sur les signaux accumulés (T valeurs par feature dans-scan).

```python
# Signaux disponibles : T valeurs de chaque feature dans-scan
# (effective_rank_sig, spectral_gap_sig, von_neumann_sig, mode_asymmetry_sig,
#  trace_J_sig, local_lyapunov_sig, delta_D_sig, frob_gradient_sig, ...)

# P1 — Dérivées globales (delta initial → final)
von_neumann_delta   = S_VN[-1] - S_VN[0]    # production entropie totale
mode_asymmetry_delta = asym[-1] - asym[0]    # évolution asymétrie sous Γ
rank_delta          = rank[-1] - rank[0]      # évolution rang sous Γ
frob_delta          = frob[-1] - frob[0]      # conservation norme ?

# P2 — Ratios finaux/initiaux
norm_ratio          = frob[-1] / (frob[0] + eps)
condition_ratio     = cond[-1] / (cond[0] + eps)

# P3 — Statistiques temporelles sur signaux
lyapunov_mean       = jnp.mean(local_lyapunov_sig)  # exposant global approximé
trace_J_mean        = jnp.mean(trace_J_sig)          # contraction moyenne
delta_D_total       = jnp.sum(delta_D_sig)            # longueur trajectoire totale

# P4 — Autocorrélation (équivalent F5 temporelle)
def first_min_autocorr(signal):
    ac = jnp.correlate(signal - signal.mean(), signal - signal.mean(), mode='full')
    ac = ac[len(ac)//2:]
    ac = ac / (ac[0] + eps)
    # chercher premier minimum
    diffs = jnp.diff(ac)
    first_min = jnp.argmax(diffs > 0)   # premier sign change (approximation JAX)
    return first_min.astype(float)

first_min_ac_vn     = first_min_autocorr(von_neumann_sig)    # lag entropie
first_min_ac_asym   = first_min_autocorr(mode_asymmetry_sig) # lag asymétrie
pnn40_vn            = jnp.mean(jnp.abs(jnp.diff(von_neumann_sig)) > 0.4 * jnp.std(von_neumann_sig))

# P5 — DMD (post-scan sur états subsampled si nécessaire)
dmd_eigenvalues     = _dmd_from_signals(states_subsample)

# P6 — Causalité (post-scan sur paires de signaux)
te_norm_entropy     = approx_transfer_entropy(frob_sig, von_neumann_sig)
te_entropy_asymmetry = approx_transfer_entropy(von_neumann_sig, mode_asymmetry_sig)

# P7 — Santé (inchangés)
has_nan_inf         = bool(any explosion detected in scan)
is_collapsed        = jnp.std(last_state) < eps
```

---

## SECTION 5 — COLONNES PARQUET CIBLES

**Axes (inchangés) :**
```
phase, gamma_id, encoding_id, modifier_id, n_dof, max_iterations,
gamma_params (JSON), encoding_params (JSON)
```

**Features dans-scan (~15 signaux × ~6 stats = ~90 colonnes brutes) :**
```
# F1 — Spectrale (par état → signal T valeurs → stats post-scan)
effective_rank_{mean, delta, final}
spectral_gap_{mean, delta, final}
nuclear_frobenius_ratio_{mean, final}
sv_decay_rate_{mean, final}
rank1_residual_{mean, final}
condition_number_{mean, delta}

# F2 — Informationnelle
von_neumann_entropy_{mean, delta, final}
renyi2_entropy_{mean, final}
shannon_entropy_{mean, delta}
entropy_production_rate     # (S_T - S_0) / T

# F3 — Enchevêtrement
entanglement_entropy_mode0_{mean, final}
entanglement_entropy_mode1_{mean, final}   # rang 3 seulement
mode_asymmetry_{mean, delta, final}
inter_mode_sv_var_{mean, final}

# F4 — Dynamique locale (Hutchinson)
trace_J_{mean, std, final}
jvp_norm_{mean, final}
jacobian_asymmetry_{mean, final}
local_lyapunov_{mean, std}   # exposant approximé

# F5 — Transport
delta_D_{mean, total}
frob_gradient_{mean, final}
bregman_cost_{mean, total}

# F6 — Causal (post-scan)
te_norm_to_entropy
te_entropy_to_asymmetry
te_lyapunov_to_rank
causal_asymmetry_index

# F7 — DMD spectral (post-scan)
dmd_spectral_radius
dmd_n_complex_pairs
dmd_spectral_entropy
dmd_decay_rate

# Post-scan dérivées et autocorr
first_min_ac_{von_neumann, mode_asymmetry, frob}
pnn40_{von_neumann, mode_asymmetry}
norm_ratio, condition_ratio, rank_delta, frob_delta

# Santé
has_nan_inf, is_collapsed
```

**Total estimé : ~120-130 colonnes brutes.**
Après `select_orthogonal_features` (seuil corrélation 0.85) : **~40-55 colonnes utiles.**
Ce volume est confortable pour HDBSCAN avec n_runs ≥ 500.

---

## SECTION 6 — COUCHE ANALYSING (phase 2)

Analysing n'est pas touché dans la phase 1 (compute).
Le pipeline lit le parquet, les colonnes sont différentes, mais le flux est identique.
Les adaptations nécessaires sont listées ici pour planification.

### 6.1 Ce qui est agnostique aux colonnes (inchangé)

- `clustering_lite.py` : `prepare_matrix`, `select_orthogonal_features`, `compute_tsne`
- `clustering_peeling.py` : algorithme peeling — lit des floats, agnostique
- `outliers_lite.py` : IsolationForest sur features communes — agnostique
- `hub_analysing.py` : orchestration — inchangé
- `verdict.py` : flux — inchangé

### 6.2 Ce qui doit être adapté

**`cluster_namer.yaml` — recalibration complète des slots**

Les anciens slots (VITESSE, TEXTURE) référençaient des features catch22 qui n'existent plus.
Nouveaux slots à calibrer après premiers runs JAX :

| Slot | Feature v7 | Sémantique |
|------|-----------|------------|
| ENTROPIE | `von_neumann_entropy_delta` | Production d'entropie sous Γ |
| ASYMÉTRIE | `mode_asymmetry_delta` | Évolution A1 sous Γ |
| RANG | `effective_rank_delta` | Γ crée ou comprime l'information |
| CONTRACTION | `trace_J_mean` | Γ contractant ou expansif |
| CHAOS | `local_lyapunov_mean` | Régime dynamique de Γ |
| TRANSPORT | `delta_D_total` | Coût de la trajectoire |
| UNIVERSALITÉ | `dmd_spectral_radius` | Signature spectrale de Γ |

Format continuous (comme mode_asymmetry implémenté) pour les slots numériques.

**`profiling_lite.py`** : inchangé — median/Q1/Q3 sur toutes colonnes numériques.

**`concordance_lite.py`** : décommenter cohen_kappa_score (déjà prévu).

### 6.3 Nouveaux verdicts possibles grâce aux features v7

Avec les features F2/F3/F4, le verdict peut répondre à des questions nouvelles :
- *Quels doublets (encoding, gamma) produisent une entropie décroissante ?* → A3 candidats
- *Quels gammas ont un spectre DMD identique cross-encodings ?* → universalité
- *Quels runs montrent un flux causal orienté ?* → causalité émergente

Ces questions ne nécessitent pas de nouveau code analysing — elles se lisent
directement dans les clusters nommés avec les nouveaux slots.

---

## SECTION 7 — SÉQUENCE D'IMPLÉMENTATION

### Phase 1 — Compute (prioritaire)

**Étape 0 — Nettoyage (1 session)**
```
- Supprimer regimes_lite.py + 3 YAMLs regimes/
- Corriger verdict.py layers hardcodé
- Factoriser _extract_common_features, _fix_str, _INF_SENTINEL
- Supprimer sys.stdout redirect cluster_namer.py
```
Pipeline numpy reste fonctionnel pour poc/poc2 existants.

**Étape 1 — Validation JAX machine (1 session)**
```
pip install jax[cpu]   # Windows natif — jaxlib 0.4+ wheels officiels PyPI

test_jax_prc.py :
  - lax.scan + jit : norme sur (10,10), (50,50), (100,100) × 1000 iter
  - jnp.linalg.svd mode-0 unfolding
  - jax.jvp sur lambda déterministe
  - jax.random.PRNGKey + split dans scan
  - vmap sur 5 seeds simultanés
  - Benchmark : temps séquentiel vs vmap vs jit
```
Livrable : `test_jax_prc.py` + tableau de performances sur machine cible.

**Étape 2 — `jax_features.py` standalone (1-2 sessions)**
```
featuring/jax_features.py
  measure_state(state, prev_state, gamma_fn, params, key) → dict scalaires
    → F1, F2, F3, F4, F5 (dans-scan)
  post_scan(signals_dict, states_subsample) → dict scalaires
    → F6, F7, autocorr, dérivées, santé
  FEATURE_NAMES : List[str]   # colonnes parquet
```
Testé standalone sur arrays JAX random — aucun autre fichier touché.
Validation : toutes features retournent float ou NaN, jamais crash.

**Étape 3 — Premiers atomics JAX (1-2 sessions)**
```
atomics/operators/gam_001_jax.py   # expm, déterministe, differentiable: True
atomics/operators/gam_010_jax.py   # random.normal, stochastic: True
atomics/D_encodings/sym_001_jax.py # déterministe
atomics/D_encodings/sym_002_jax.py # PRNGKey(seed_CI)
atomics/modifiers/m0_jax.py        # identité
atomics/modifiers/m1_jax.py        # bruit gaussien JAX

data_loading_lite.py : discover_gammas_jax(), discover_encodings_jax()
Convention : METADATA['id'] identique, METADATA['jax'] = True
```

**Étape 4 — `kernel_jax.py` + `runner_jax.py` (1-2 sessions)**
```
core/kernel_jax.py
  run_kernel_jax(D_initial, gamma_fn, params, key, max_it)
  → jit(lax.scan(step_fn)) → (signals_dict, last_state)
  K1-K5 esprit : aveugle domaine, no-branch sur D/Γ

running/runner_jax.py
  run_single_jax(composition, config) → {features, layers}
  Interface identique runner.py
  hub_running.py : flag YAML 'engine: jax' → bascule runner

compositions.py : seed_CI/seed_run → jax.random.PRNGKey
```
Validation : poc YAML → parquet → 216 runs → spot check colonnes.

**Étape 5 — vmap seeds/params (1 session)**
```
hub_running.py : grouper compositions vmappables → run_batch_jax
Validation : poc1 3360 runs → < 10 minutes
```

**Étape 6 — Atomics restants + F4 jacfwd (2-3 sessions)**
```
Compléter tous gammas JAX : GAM-002/003/004/005/006/007/008/009/013
Compléter tous encodings JAX : ASY-*, R3-*
Activer jacfwd pour gammas differentiable:True sur n_dof ≤ 50
```

### Phase 2 — Analysing (après Phase 1 validée)

**Étape 7 — Brancher analysing sur parquet v7**
```
Lancer poc avec engine:jax → parquet v7
Vérifier select_orthogonal_features sur nouvelles colonnes
Recalibrer cluster_namer.yaml (slots ENTROPIE, ASYMÉTRIE, CONTRACTION, etc.)
Décommenter cohen_kappa_score dans concordance_lite.py
```

**Étape 8 — Validation scientifique**
```
E1 — Variance stochastique : N seeds × même composition
E2 — Universalité : clusters DMD cross-encodings
E3 — Null model : GAM random (sigma élevé) vs GAM candidats
E4 — Concordance inter-phases
E5 — Premier run poc2b complet avec engine:jax
```

---

## SECTION 8 — PARKING ET VISION LONG TERME

### 8.1 Parking technique (hors scope immédiat)

| Item | Raison | Quand |
|------|--------|-------|
| GPU support | `jax[cuda]` zéro touche code | Quand les runs durent des heures |
| TDA par run | Intractable en haute dimension | Post-clustering seulement |
| EDMD | Bénéfice marginal sur DMD pour usage actuel | Après E8 si DMD insuffisant |
| Spectre Lyapunov QR complet | Requiert jacfwd sur grand n_dof | Après validation F4 Hutchinson |
| Multi-layer HDBSCAN | Réévaluer si F3 enchevêtrement justifie layer rang-3 | Après E8 |
| Wasserstein exact (Sinkhorn) | Coûteux, Bregman suffit | Si F5 transport discriminant |

### 8.2 Vision banc de test long terme

Le pipeline actuel répond à : *les axiomes génèrent-ils des structures émergentes mesurables ?*

La progression naturelle :
```
Phase actuelle : faisabilité axiomes
    ↓
Phase R* : caractérisation systématique des candidats Γ
    ↓
Phase plugin : encodings domaine-spécifiques (relativité, QM, ...)
    ↓
Phase validation : retrouver des propriétés physiques connues
                  ex: mouvement de Mercure avec encoding système solaire + GR
```

Le pipeline JAX v7 est le fondement de cette progression :
- `lax.scan` + vmap : explorer 128M compositions en heures
- features informationnelles : discriminer universalité vs encoding-dépendance
- DMD universalité test : identifier quels Γ sont "physiquement universels"

### 8.3 Questions ouvertes que le pipeline peut commencer à adresser

**Quelle dimensionnalité minimale pour D ?**
Si les clusters HDBSCAN se fragmentent identiquement pour n_dof=10 et n_dof=100
→ les axiomes ne nécessitent pas de haute dimension.
Feature : rank_delta vs n_dof cross-runs.

**Γ peut-il être universel ?**
Si dmd_spectral_radius est identique cross-encodings pour un gamma →
Γ est universel indépendamment de la façon dont D encode l'information.
C'est la thèse centrale vérifiable numériquement.

**Y a-t-il des encodings "plus naturels" que d'autres ?**
Si certains encodings convergent vers les mêmes attracteurs quelle que soit Γ
→ ces encodings ont une structure intrinsèque qui surpasse Γ.
Feature : entanglement_entropy_delta cross-gammas pour un encoding fixé.

**La production d'entropie est-elle bornée inférieurement ?**
A3 dit que Γ ne peut pas annuler D. Si von_neumann_entropy_delta ≥ 0 pour tous les Γ
→ A3 est observé empiriquement sur le pipeline.
C'est un test de cohérence des axiomes, pas juste une feature.

---

## ANNEXE — GLOSSAIRE v7

| Terme | Définition v7 |
|-------|--------------|
| **Atomic JAX** | Gamma/encoding/modifier avec signature `(state, params, key) → jnp.ndarray` |
| **Dans-scan** | Feature calculée à chaque itération kernel, compilée dans `lax.scan` |
| **Post-scan** | Feature calculée une fois sur signaux accumulés après la boucle |
| **Hutchinson** | Estimateur stochastique de trace(J) via JVP — O(1) mémoire |
| **Signal** | Série temporelle (T valeurs) d'une feature dans-scan sur un run |
| **DMD universalité** | Test : même spectre DMD cross-encodings pour Γ fixé |
| **S_VN** | Entropie de Von Neumann de D normalisé — proxy aire holographique |
| **mode_asymmetry** | Mesure directe A1 — dissymétrie informationnelle de D sous Γ |
| **trace_J** | Divergence locale de Γ via Hutchinson — contraction/expansion |
| **Engine** | Flag YAML `engine: jax` ou `engine: numpy` — sélectionne le runner |
| **differentiable** | Flag METADATA atomic — jacfwd disponible si True |

---

**FIN MANIFESTO PRC v7**

*Roadmap vivante — les sections compute (1-6) sont l'autorité.*
*Sections analysing et vision long terme : orientations, pas contrats.*
*Pour le code : sources + jax_features.py + atomics_catalog.*
