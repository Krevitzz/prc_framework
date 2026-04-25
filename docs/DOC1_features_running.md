# DOCUMENT 1 — FEATURES SCIENTIFIQUES DU PIPELINE PRC
## Chantier 1 — Layer Running

> Document de référence **scientifique** pour les features produites par le running.
> Code-agnostique sur les formules précises (elles iront en commentaires dans `features_registry.py`).
> Ancré sur CHARTER §1.1 à §1.5, ancrage.md partie I, et les décisions actées lors de la discussion préparatoire au chantier 1.
>
> Ce document répond à la question : **"qu'est-ce qu'on calcule, pourquoi, et ce qu'on peut en déduire scientifiquement ?"**
> Il sert de cadre à l'analysing : pour chaque feature, il dit ce qu'elle raconte et ce qu'elle ne raconte pas.

---

## TABLE DES MATIÈRES

1. Principes directeurs
2. Architecture des mesures (deux couches)
3. Couche A — Observables continus O(n²)
4. Couche B — Features coûteuses aux virages
5. Features post-scan dérivées
6. Features de classification et géométrie du masque
7. Statuts pathologiques et leur traitement
8. Tableau récapitulatif par question scientifique
9. Caveats et NaN

---

## 1. PRINCIPES DIRECTEURS

### 1.1 Les deux grammaires

Le pipeline manipule des trajectoires temporelles de tenseurs, et il en extrait des mesures selon **deux grammaires différentes** qui coexistent et sont chacune cohérente avec ce qu'elle mesure :

**Grammaire signal** — applicable aux séquences temporelles à **pas uniforme**. Les méthodes classiques du traitement du signal (autocorrélation, transfer entropy, DMD, statistiques temporelles) supposent toutes l'uniformité du pas. Elles mesurent des propriétés globales d'une trajectoire : mémoire, causalité, linéarisation spectrale, stationnarité.

**Grammaire virages** — applicable aux **séquences d'instants ordonnés non-uniformes**. Elle ne suppose pas d'équidistance temporelle. Elle mesure des propriétés **ordinales** : nombre de virages, forme du parcours, valeurs aux points d'intérêt, agrégats statistiques insensibles au pas.

Le pipeline produit et exploite les deux, mais **jamais l'une sur les observables de l'autre**. Une grammaire signal appliquée à des virages espacés irrégulièrement donne des résultats numériquement définis mais scientifiquement invalides. Cette erreur est interdite.

### 1.2 Rôle du masque

Le **masque** est la liste des instants où "il se passe quelque chose d'intéressant" dans la trajectoire d'un run. Il est déterminé par classify à partir des timelines screening (delta_D, frob, cos_dissim) et capture : les transitions abruptes détectées par Tukey fence, les points de cycles oscillatoires détectés par autocorrélation, un plancher minimum de points pour garantir une couverture, et les extrémités valides pour les runs monotones ou pathologiques.

**Le masque n'est pas un sous-échantillonnage temporel.** C'est une **schématisation** de la trajectoire : *"on note les virages, de combien ça tourne, dans quelle direction"*. Les valeurs aux points masqués ne sont pas des échantillons équidistants d'une trajectoire cachée — ce sont des **instants qualitativement significatifs**.

### 1.3 Séparation O(n²) / O(n³)

Le charter §1.5 acte la distinction entre mesures dynamiques O(n²) peu coûteuses et mesures spectrales O(n³) coûteuses. Le chantier 1 opérationnalise cette distinction :

- Les mesures **O(n²)** sont calculées à **chaque itération** de P2 pour tous les runs. Elles forment des timelines à pas uniforme → exploitables en grammaire signal.
- Les mesures **O(n³)** et les mesures nécessitant différentiation (JVP) sont calculées **uniquement aux points masqués**. Elles forment des séquences ordonnées non-uniformes → exploitables en grammaire virages uniquement.

### 1.4 Ce que chaque mesure doit justifier

Toute feature présente dans le pipeline doit pouvoir répondre à trois questions :

1. **À quelle question du charter contribue-t-elle ?** (Q1 : D encode-t-il ? / Q2 : Γ agit-il ? / Q3 : structures stables ? / Q4 : propriétés pré-émergentes ?)
2. **Quel axiome ancre son interprétation ?** (A1 dissymétrie / A2 mécanisme / A3 résistance)
3. **Dans quelle grammaire opère-t-elle ?** (signal sur timelines uniformes / virages sur points masqués)

Une feature qui ne peut pas répondre à ces trois questions est un candidat à retirer.

### 1.5 Politique du pool

> *"Je préfère avoir une feature en trop qu'une en moins de ce point de vue, après de quoi a-t-on besoin côté analysing pour traduire ces informations, on sait qu'une feature seule ne raconte parfois pas toute l'histoire."*

Le pool de features est volontairement **enrichi au-delà du strict minimum**. Chaque ajout est justifié soit par un ancrage axiomatique direct, soit par son rôle de feature **intermédiaire** qui permet à l'analysing de séparer plus proprement des comportements qu'une feature unique confondrait. Un ajout coûte quelques Mo en parquet par phase ; l'absence d'une feature coûte un re-run complet de tous les YAMLs pour uniformiser les résultats.

---

## 2. ARCHITECTURE DES MESURES

### 2.1 Flux général d'un run

```
    D initial (encoding + modifier)
           │
           ▼
   ┌───────────────────────────┐
   │ P1 — propagate + screening│   timelines screening, 4 observables
   │ (O(n²), pas de JVP)       │   → nourrit classify
   └───────────────────────────┘
           │
           ▼
   ┌───────────────────────────┐
   │ classify + masque          │   statuses, mask (B,T), p1/mask features
   │ (JIT unique, GPU)          │
   └───────────────────────────┘
           │
           ▼
   ┌───────────────────────────┐
   │ P2 — propagate             │   à chaque step, pour tous les runs:
   │      + couche A (O(n²))    │     • propagate state
   │      + couche B (cond)     │     • couche A observables ~11
   │                           │     • couche B features si mask[b,t]
   └───────────────────────────┘
           │
           ▼ transfert unique GPU → CPU
           │
   ┌───────────────────────────┐
   │ post-process CPU (numpy)  │   agrégats + DMD + F6 + autocorr + pnn
   │ par run                   │   + temporal + phasic + postscan
   └───────────────────────────┘
           │
           ▼
       parquet
```

### 2.2 Les 3 niveaux de stockage en parquet

**Niveau 1 — Timelines complètes (uniformes)**
Environ **11 signaux** issus de la couche A, stockés comme listes de longueur `t_effective` par run. Ce sont les observables O(n²) calculés à chaque step P2. Ils nourrissent la grammaire signal dans le post-process.

**Niveau 2 — Timelines tronquées (virages)**
Environ **19 features** issues de la couche B, stockées comme listes de longueur `K_i` variable par run, alignées sur les indices du masque. Un tableau unique `mask_t_indices` par run donne les indices t correspondants (un seul pour toutes les features tronquées du run, puisque le masque est global).

**Niveau 3 — Scalaires résumés**
Environ **90 à 110 scalaires** pré-calculés côté running : agrégats (mean, std, final, delta, total), DMD sur observables, F6 transfer entropy, autocorrélations, PNN40, temporal features, phasic features, P1 caractérisation, mask géométrie. Permettent à l'analysing de travailler sans charger les timelines quand seul le scalaire suffit.

### 2.3 Pourquoi deux passes de propagation

Classify a besoin de la vue complète de la trajectoire screening pour détecter les virages (Tukey fence sur `|diff(cos_dissim)|`, FFT autocorrélation, cascade de régimes, dilatation avec marges). Cette décision est non-causale : elle demande d'avoir vu toute la trajectoire avant de décider où sont les virages. **Fusionner P1 et P2 en un seul scan est impossible sans sacrifier la qualité du masque.**

P2 re-propage donc la trajectoire à partir des mêmes D, gp, keys PRNG que P1 (déterminisme garanti par la seed PRNG identique et la séquence de splits identique). C'est le coût accepté pour préserver la qualité du masque.

---

## 3. COUCHE A — OBSERVABLES CONTINUS O(n²)

### 3.1 Principe

La couche A regroupe les observables **canoniques** (invariants aux reparamétrisations triviales), **indépendants** (non-redondants entre eux), et **peu coûteux** (O(n²) ou moins) qui sont calculés à chaque itération de P2 pour tous les runs. Ces observables forment les timelines complètes sur lesquelles la grammaire signal est appliquée (DMD, F6, autocorrélations, PNN, temporal features).

Le choix des observables est motivé par trois critères :
- **Canonicité** : chaque observable doit être invariant aux transformations qui ne changent pas la structure informationnelle de D (permutations d'indices, changements d'échelle triviaux).
- **Suffisance** : le pool doit permettre à l'analysing de distinguer des structures différentes. Deux D distinctes ne doivent pas produire le même vecteur d'observables.
- **Ancrage axiomatique** : chaque observable pointe vers au moins un axiome A1/A2/A3.

### 3.2 Liste des observables de la couche A

#### A.1 — `frob`

**Mesure** : norme de Frobenius de D, ||D||_F.
**Ancrage** : A1 (D existe et a une magnitude mesurable).
**Grammaire** : signal.
**Rôle** : observable de volume. Nourrit temporal features (iqr, plateau, cusum, changepoint), autocorrélation, DMD sur observables.
**Interprétation** : une décroissance → D se contracte sous Γ (régime dissipatif). Une croissance → D s'amplifie (candidat d'explosion ou d'expansion stable). Un plateau → état stationnaire en amplitude.

#### A.2 — `delta_D`

**Mesure** : variation relative ||D_t - D_{t-1}|| / ||D_t||.
**Ancrage** : A1 + A2 (D change sous l'action de Γ).
**Grammaire** : signal.
**Rôle** : détecte les virages locaux. Nourrit classify (fence Tukey), temporal features sur les variations.
**Interprétation** : proche de 0 → trajectoire localement plate. Grand → transition brutale. Valeurs irrégulières → trajectoire turbulente.

#### A.3 — `cos_dissim`

**Mesure** : 1 - cos(D_t, D_{t-1}), dissimilarité directionnelle.
**Ancrage** : A2 (rotation de D dans son espace sous Γ).
**Grammaire** : signal.
**Rôle** : détecte les changements de direction indépendamment de l'amplitude. Nourrit classify (détection de régime FLAT, OSCILLATING, TRANSITIONAL via autocorrélation et Tukey fence), autocorrélation post-scan.
**Interprétation** : proche de 0 → D évolue dans une direction quasi-constante (rotation lente ou absente). Périodicité détectée → cycle limite. Grandes valeurs irrégulières → mouvement chaotique directionnel.

#### A.4 — `is_finite`

**Mesure** : indicateur 0/1 — tous les éléments de D sont-ils finis ?
**Ancrage** : santé numérique, pas d'ancrage axiomatique direct.
**Grammaire** : signal (utilisé pour détecter explosion).
**Rôle** : sert à classify pour détecter les EXPLOSIONS. Son passage de 1 à 0 définit `t_effective` pour un run explosif.

#### A.5 — `shannon_comp`

**Mesure** : entropie de Shannon sur les composantes de D normalisées en distribution de probabilité (|x_i| / Σ|x_j|).
**Ancrage** : A1 (distribution de l'information dans les composantes de D).
**Grammaire** : signal.
**Rôle** : **proxy O(n²) de la dispersion informationnelle** dans D. Calculé en timeline complète, il sert de signal d'entrée pour F6 (transfer entropy) et pour les temporal features quand on veut mesurer "combien D disperse son information". Renommé pour indiquer explicitement qu'il opère sur les **composantes** et non sur le spectre (la vraie entropie de Von Neumann est en couche B aux virages).
**Interprétation** : croissant → D disperse son information sur plus de composantes (diffusion). Décroissant → D concentre son information sur peu de composantes (localisation). Stable → équilibre informationnel.
**Caveat** : n'est pas équivalent à l'entropie de Von Neumann. C'est un proxy pour la grammaire signal. L'entropie de Von Neumann vraie (spectrale) est mesurée aux virages en couche B.

#### A.6 — `ipr`

**Mesure** : Inverse Participation Ratio, `(Σ|x_i|²)² / Σ|x_i|⁴`, normalisé par `n`.
**Ancrage** : A1 (localisation de l'information dans D).
**Grammaire** : signal.
**Rôle** : **mesure de localisation complémentaire à shannon_comp**. Permet à l'analysing de distinguer "information diffuse uniformément" de "information concentrée sur quelques composantes dominantes".
**Interprétation** : IPR ≈ 1/n → information uniformément répartie. IPR ≈ 1 → information concentrée sur une seule composante. Un run peut avoir shannon_comp moyen et IPR très faible s'il a de nombreuses composantes à distribution unimodale.
**Justification** : shannon_comp et IPR sont corrélés mais ne capturent pas exactement la même chose. IPR est **plus sensible aux queues** de distribution, shannon à la forme globale. Une feature intermédiaire en grammaire A1.

#### A.7 — `mode_asymmetry_o2`

**Mesure** : asymétrie O(n²) de D repliée en matrice carrée : `||M - M^T||_F / ||M||_F` où `M` est le repli de D en matrice approximativement carrée.
**Ancrage** : A1 (structure relationnelle non triviale dans D) + lien avec la géométrie relationnelle de Rovelli (charter §1.4).
**Grammaire** : signal.
**Rôle** : mesure l'asymétrie structurelle de D sans SVD. Point critique : **cette mesure est O(n²) et n'a pas besoin d'aller en couche B**. Aujourd'hui elle est calculée aux virages via `_f3_rank2/3` — cette position sera corrigée dans le chantier 1.
**Interprétation** : 0 → D parfaitement symétrique (géométrie relationnelle auto-dualisée). Grand → D asymétrique (relations orientées, candidat pour causalité émergente).
**Note** : même observable que le `f3_mode_asymmetry` actuel, mais déplacé en couche A et renommé pour cohérence.

#### A.8 — `bregman_cost`

**Mesure** : coût de Bregman entre états consécutifs, `||D_t - D_{t-1}||² / ||D_t||²`.
**Ancrage** : A2 (coût informationnel du déplacement de D sous Γ).
**Grammaire** : signal.
**Rôle** : mesure une forme de "distance informationnelle" entre états consécutifs, différente de delta_D (qui est une norme brute). Nourrit temporal features et sert de signal pour F6 (quelle quantité d'information est déplacée par step).
**Interprétation** : grand → coût informationnel élevé de la transition. Stable → transport homogène. Pic → événement de transport concentré.

#### A.9 — `lyap_empirical`

**Mesure** : `log(||D_{t+1} - D_t|| / ||D_t - D_{t-1}||)`.
**Ancrage** : A2 (expansion ou contraction exponentielle de la dynamique sous Γ).
**Grammaire** : signal.
**Rôle** : **seul proxy de Lyapunov disponible pour les gammas non différentiables**. Les gammas différentiables ont en plus le Lyapunov local via JVP en couche B, mais le lyap_empirical est le fondement commun à tous les gammas pour mesurer A2.
**Interprétation** : > 0 → la variation croît entre steps successifs (divergence, sensibilité aux conditions initiales). < 0 → la variation décroît (contraction, convergence). ≈ 0 → régime marginal.

#### A.10 — `frob_gradient` (rank ≥ 3 uniquement)

**Mesure** : ratio des écarts-types des normes des lignes entre deux modes tensoriels : `std(row_norms_M0) / std(row_norms_M1)` où M0 et M1 sont les replis de D selon les modes 0 et 1.
**Ancrage** : A2 (gradient inter-modes, précurseur de masse au sens de Verlinde — charter §1.4, gravité entropique).
**Grammaire** : signal.
**Rôle** : mesure O(n²) de l'asymétrie entre modes tensoriels. Applicable seulement à rank ≥ 3 (rank 2 n'a qu'un seul mode non trivial).
**Interprétation** : ≈ 1 → modes équilibrés. Très grand ou très petit → un mode domine l'autre. Un gradient qui évolue → candidat Verlinde.
**NaN structurel** : rank_eff == 2.

#### A.11 — `volume_proxy` (nouvelle feature, non-diff friendly)

**Mesure** : proxy de volume sans différentiation, défini comme `||D||_F · ||D||_∞ / ||D||_1` où les normes sont prises sur le tenseur aplati.
**Ancrage** : A2 (contraction/expansion informationnelle sans passer par le Jacobien).
**Grammaire** : signal.
**Rôle** : donne aux gammas non différentiables une mesure de compression/dilatation, absente aujourd'hui. Complémentaire à lyap_empirical (qui mesure la variation, pas l'occupation).
**Interprétation** : ratio bas → D étalé (information diluée). Ratio élevé → D concentré (information dense). Variation temporelle → Γ compacte ou étale D.
**Justification** : Q-sci-2 validé — la théorie PRC veut une mesure O(n²) de "volume" informationnel accessible aux gammas non différentiables.
**Caveat** : ce n'est pas un vrai volume mathématique. C'est un proxy géométrique grossier mais canonique (invariant aux permutations de composantes). L'interprétation doit être prudente.

### 3.3 Résumé couche A

| # | Observable | Ancrage | Note |
|---|---|---|---|
| A.1 | frob | A1 | volume |
| A.2 | delta_D | A1+A2 | variation relative |
| A.3 | cos_dissim | A2 | rotation directionnelle |
| A.4 | is_finite | — | santé numérique |
| A.5 | shannon_comp | A1 | dispersion info, proxy VN |
| A.6 | ipr | A1 | localisation info |
| A.7 | mode_asymmetry_o2 | A1 | asymétrie structurelle |
| A.8 | bregman_cost | A2 | coût transport |
| A.9 | lyap_empirical | A2 | expansion/contraction |
| A.10 | frob_gradient | A2 | gradient inter-modes (rank≥3) |
| A.11 | volume_proxy | A2 | compression sans différentiation |

**Total : 11 observables**, dont 10 universels et 1 conditionnel au rank.

---

## 4. COUCHE B — FEATURES COÛTEUSES AUX VIRAGES

### 4.1 Principe

La couche B regroupe les features dont le calcul est soit **O(n³)** (spectrales, nécessitant SVD), soit **coûteux en VRAM/compute** (JVP pour la différentiation du Jacobien gamma). Ces features sont calculées **uniquement aux points du masque** via `lax.cond(mask[b, t], compute, NaN)`, ce qui signifie que :

- les runs à masque dense paient beaucoup
- les runs à masque parcimonieux paient peu
- les runs pathologiques EXPLOSION/COLLAPSED ont un masque vide et ne paient rien

Le calcul est conditionnel par sample (grâce à `lax.map + lax.cond`), pas par batch : deux samples d'un même batch peuvent avoir des K_i très différents et chacun paie son propre coût sans contamination.

### 4.2 Sous-groupe F1 — Structure spectrale (A1)

#### B.1 — `f1_effective_rank`

**Mesure** : exp(entropie de Shannon de la distribution de probabilité des valeurs singulières au carré).
**Ancrage** : A1 (combien de dimensions spectrales actives).
**Grammaire** : virages.
**Interprétation** : donne le "rang effectif" au sens entropique. Rang effectif proche de 1 → D dominé par un seul mode. Rang effectif proche de n → D spectralement plein.

#### B.2 — `f1_spectral_gap`

**Mesure** : différence relative entre les deux premières valeurs singulières, `(σ_0 - σ_1) / σ_0`.
**Ancrage** : A1 (séparation du mode dominant).
**Grammaire** : virages.
**Interprétation** : grand → le premier mode est isolé (attracteur de bas rang). Petit → plusieurs modes compétitifs.

#### B.3 — `f1_nuclear_frobenius_ratio`

**Mesure** : `Σσ_i / sqrt(Σσ_i²)`, ratio entre norme nucléaire et norme de Frobenius.
**Ancrage** : A1 (compression spectrale).
**Grammaire** : virages.
**Interprétation** : décroît de sqrt(n) (spectre plat) vers 1 (un seul mode). Mesure de "concentration spectrale" complémentaire à l'entropie.

#### B.4 — `f1_sv_decay_rate`

**Mesure** : pente de régression linéaire de `log(σ_i)` en fonction de `i`.
**Ancrage** : A1 (loi de décroissance des modes).
**Grammaire** : virages.
**Interprétation** : pente forte → décroissance rapide (peu de modes effectifs). Pente faible → spectre plat. Signe du régime spectral.

#### B.5 — `f1_rank1_residual`

**Mesure** : `sqrt(Σ_{i≥1} σ_i²) / ||D||_F` — combien de l'énergie reste hors du premier mode.
**Ancrage** : A1 (dominance du premier mode).
**Grammaire** : virages.
**Interprétation** : proche de 0 → D est quasi rank-1. Proche de 1 → D multi-modal.

#### B.6 — `f1_condition_number`

**Mesure** : `σ_0 / σ_{-1}`, nombre de condition spectral.
**Ancrage** : A1 (dispersion spectrale).
**Grammaire** : virages.
**Interprétation** : grand → spectre très étiré (modes d'amplitudes très différentes). Petit → spectre compact. Utile pour détecter les régimes dégénérés.

### 4.3 Sous-groupe F2 — Entropies spectrales (A1 + lien holographique)

#### B.7 — `f2_von_neumann_entropy`

**Mesure** : entropie de Von Neumann sur la distribution spectrale au carré : `-Σ p_i log p_i` avec `p_i = σ_i² / Σσ_j²`.
**Ancrage** : A1 + **charter §1.4 Ryu-Takayanagi — proxy direct de l'aire holographique**.
**Grammaire** : virages.
**Rôle** : c'est la feature la plus fondamentale du pipeline pour l'ancrage holographique. Son évolution sous Γ teste **A3** : D résiste-t-il à l'annihilation ? Son delta (VN_final - VN_initial) est le signal principal.
**Interprétation** : stable et > 0 → D a une structure spectrale persistante (candidat d'attracteur holographique). Décroissant vers 0 → D se dégrade en rank bas (risque de collapse informationnel). Croissant → D se complexifie.
**Importance** : **test d'A3**. Si VN reste > 0 pour tous les Γ stables → A3 est observé empiriquement.

#### B.8 — `f2_renyi2_entropy`

**Mesure** : entropie de Rényi d'ordre 2, `-log(Σ p_i²)`.
**Ancrage** : A1 (entropie sensible aux modes dominants).
**Grammaire** : virages.
**Interprétation** : complément de VN. Rényi-2 est plus sensible aux modes dominants ; la différence (VN - Rényi-2) caractérise la queue du spectre.

### 4.4 Sous-groupe F3 — Enchevêtrement multi-modal (A1, rank ≥ 3)

#### B.9 — `f3_entanglement_entropy_mode0`

**Mesure** : entropie de Von Neumann de D replié en matrice selon le mode 0 (M0 = reshape(D, (n, -1))).
**Ancrage** : A1 + géométrie relationnelle Rovelli (charter §1.4).
**Grammaire** : virages.
**Interprétation** : mesure le couplage du mode 0 avec les autres modes. Faible → mode 0 isolé. Élevé → mode 0 fortement couplé.

#### B.10 — `f3_entanglement_entropy_mode1`

**Mesure** : entropie de Von Neumann de D replié selon le mode 1 (M1 = moveaxis puis reshape).
**Ancrage** : A1.
**Grammaire** : virages.
**Interprétation** : symétrique de mode0 mais sur l'autre mode. Mode0 et mode1 comparés donnent l'asymétrie entre les deux modes.
**NaN structurel** : rank_eff == 2 (un seul mode non trivial).

#### B.11 — `f3_inter_mode_sv_var`

**Mesure** : variance entre les premières valeurs singulières des deux modes : `var([σ_0^{mode0}, σ_0^{mode1}])`.
**Ancrage** : A1.
**Grammaire** : virages.
**Interprétation** : quantifie la disparité entre les deux modes. 0 → équilibre parfait. Grand → un mode porte l'essentiel.
**NaN structurel** : rank_eff == 2.

### 4.5 Sous-groupe F4 — Géométrie locale par différentiation (A2)

Ce sous-groupe est calculé **uniquement si `is_differentiable == True`** et **uniquement aux virages**. Le JVP gamma est coûteux et sa position en couche B sous `lax.cond` est la source principale de réduction VRAM du chantier 1 par rapport à l'architecture actuelle.

#### B.12 — `f4_trace_J`

**Mesure** : estimation Hutchinson de `trace(J)` où J est le Jacobien de Γ, via `v^T · (J·v)` pour v ~ N(0, I).
**Ancrage** : A2 (divergence locale — Γ contracte-t-elle ou dilate-t-elle l'espace ?).
**Grammaire** : virages.
**Interprétation** : > 0 → expansion locale. < 0 → contraction locale. ≈ 0 → volume préservé (candidat d'hamiltonien).
**NaN structurel** : `is_differentiable == False`.

#### B.13 — `f4_jvp_norm`

**Mesure** : norme du Jacobian-vector product, `||J·v||`.
**Ancrage** : A2 (amplitude de l'action de Γ sur une perturbation).
**Grammaire** : virages.
**Interprétation** : donne une mesure d'échelle de Γ localement. Combiné avec `||v||`, on obtient un ratio de contraction/dilatation.
**NaN structurel** : `is_differentiable == False`.

#### B.14 — `f4_jacobian_asymmetry`

**Mesure** : `||J·v - J^T·v|| / ||J·v||`, approximation de l'asymétrie du Jacobien via deux JVPs.
**Ancrage** : A2 (réversibilité locale de Γ).
**Grammaire** : virages.
**Interprétation** : 0 → J symétrique → Γ localement réversible. Grand → Γ localement irréversible (production d'entropie, flèche du temps émergente).
**NaN structurel** : `is_differentiable == False`.
**Lien théorique** : l'asymétrie du Jacobien est directement liée à la production d'entropie et donc à A3. Un Γ avec asymétrie systématique > 0 est un candidat pour "A3 observé" — D résiste parce que Γ ne peut pas localement l'annuler réversiblement.

#### B.15 — `f4_local_lyapunov`

**Mesure** : `log(||J·v||)`, exposant de Lyapunov local linéarisé.
**Ancrage** : A2 (Lyapunov local via Jacobien).
**Grammaire** : virages.
**Interprétation** : comparé à `lyap_empirical` (couche A continue), ils doivent être cohérents pour les gammas bien linéarisés. Un écart systématique signale une non-linéarité forte.
**NaN structurel** : `is_differentiable == False`.

### 4.6 Résumé couche B

| # | Feature | Sous-groupe | Ancrage |
|---|---|---|---|
| B.1 | f1_effective_rank | F1 | A1 |
| B.2 | f1_spectral_gap | F1 | A1 |
| B.3 | f1_nuclear_frobenius_ratio | F1 | A1 |
| B.4 | f1_sv_decay_rate | F1 | A1 |
| B.5 | f1_rank1_residual | F1 | A1 |
| B.6 | f1_condition_number | F1 | A1 |
| B.7 | f2_von_neumann_entropy | F2 | A1 + holographie |
| B.8 | f2_renyi2_entropy | F2 | A1 |
| B.9 | f3_entanglement_entropy_mode0 | F3 | A1 |
| B.10 | f3_entanglement_entropy_mode1 | F3 | A1 (rank≥3) |
| B.11 | f3_inter_mode_sv_var | F3 | A1 (rank≥3) |
| B.12 | f4_trace_J | F4 | A2 (diff) |
| B.13 | f4_jvp_norm | F4 | A2 (diff) |
| B.14 | f4_jacobian_asymmetry | F4 | A2 (diff), lien A3 |
| B.15 | f4_local_lyapunov | F4 | A2 (diff) |

**Total : 15 features** dans la couche B. (Rank1_residual et condition_number sont en F1 mais parfois classés en "stabilité" — ici je les garde en F1 pour cohérence avec le registry actuel.)

**Retrait** par rapport au pipeline actuel :
- `f3_mode_asymmetry` → déplacé en couche A (`mode_asymmetry_o2`)
- `f5_*` → intégrés en couche A (delta_D, bregman_cost, frob_gradient)
- `frob_norm`, `shannon_entropy` (dyn) → unifiés dans la couche A (frob, shannon_comp)
- Doublons éliminés.

---

## 5. FEATURES POST-SCAN DÉRIVÉES

### 5.1 Principe

Les features post-scan sont calculées côté CPU, par run, après le transfert GPU → CPU. Elles tombent en trois catégories :

**Catégorie C1 — Agrégats scalaires des timelines**
Pour chaque feature de couche A (grammaire signal) et chaque feature de couche B (grammaire virages), on produit les scalaires mean, std, final, delta, total selon la pertinence de chacun.

**Catégorie C2 — Features post-process en grammaire signal**
DMD sur observables, F6 transfer entropy, autocorrélations, PNN40, temporal features. Elles consomment **exclusivement les timelines couche A** (uniformes), jamais les timelines couche B (virages).

**Catégorie C3 — Features post-process en grammaire virages**
Phasic features (n_reversals, max_monotone_frac, range_ratio). Elles consomment **exclusivement les timelines couche B** (virages), sur lesquelles elles opèrent de manière ordinale.

Cette séparation stricte entre les catégories C2 et C3 garantit qu'aucune méthode signal n'est appliquée à un signal non-uniforme, et qu'aucune méthode ordinale n'est appliquée à un signal continu (ce qui serait gaspiller l'information).

### 5.2 C1 — Agrégats

Pour chaque timeline (couche A et couche B), l'agrégation produit un sous-ensemble de `{mean, std, final, delta, total}` selon la sémantique de la feature. Le mapping exact est :

- **Timelines couche A** : sont calculées jusqu'à `t_effective` puis agrégées. Pour chaque, la combinaison d'agrégats est choisie selon le rôle (par exemple `frob` produit mean, final, delta ; `lyap_empirical` produit mean, std, final).
- **Timelines couche B** : sont calculées aux K_i points du masque. Les agrégats sont calculés sur ces K_i valeurs sans interpolation temporelle (par cohérence grammaire virages). Pour chaque feature, combinaison pertinente de mean, std, final, delta.

Le mapping précis des combinaisons sera défini dans `features_registry.py` (équivalent des DYNAMIC_AGG_MAP et SPECTRAL_AGG_MAP actuels). **Ce mapping est une décision opérationnelle, pas scientifique** ; il sera figé au début du chantier et versionné.

### 5.3 C2 — Features post-process en grammaire signal

#### C2.1 — DMD sur observables (f7_dmd_*)

**Entrée** : le vecteur des 11 observables couche A à chaque step, formant une trajectoire `(T, 11)` uniforme pour chaque run.
**Méthode** : DMD streaming (RLS) sur la séquence de vecteurs d'observables.
**Sortie** :
- `f7_dmd_spectral_radius` — rayon spectral dominant, teste la stabilité de la linéarisation
- `f7_dmd_n_complex_pairs` — nombre de paires conjuguées complexes, mesure les modes oscillatoires
- `f7_dmd_spectral_entropy` — entropie de Shannon du spectre DMD
- `f7_dmd_decay_rate` — pente de décroissance des modes
**Ancrage** : **A2 — charter §1.4 universalité de Γ**. Si deux doublets (enc₁, Γ) et (enc₂, Γ) produisent le même spectre DMD sur observables canoniques → Γ est universel indépendamment de l'encodage, à la précision du pool d'observables.
**Grammaire** : signal — les observables sont uniformément échantillonnés à chaque step P2.
**Différence avec l'actuel** : le DMD actuel opère sur les `sigmas_buf` (spectre SVD aux virages). Le nouveau DMD opère sur les observables canoniques O(n²) uniformément échantillonnés. **Sémantique différente** : l'ancien testait l'universalité spectrale, le nouveau teste l'universalité par observables. Les deux sont valides mais ne sont pas équivalents (voir section 1 du document).
**Nomenclature** : les noms de feature sont conservés pour continuité (`f7_dmd_spectral_radius`, etc.), mais la sémantique est **redéfinie explicitement** comme "DMD sur observables canoniques". Un warning sera ajouté dans les métadonnées du schéma pour éviter toute confusion avec l'ancien DMD spectral.

#### C2.2 — F6 Transfer Entropy (f6_*)

**Entrée** : paires d'observables couche A, en timeline complète.
**Paires retenues** (Q-sci-3 validé) :
- `frob → shannon_comp` : le volume cause-t-il la diffusion informationnelle ?
- `lyap_empirical → frob` : l'expansion cause-t-elle le volume ?
- `cos_dissim → mode_asymmetry_o2` : la rotation cause-t-elle l'asymétrie ?
- `delta_D → bregman_cost` : la variation cause-t-elle le coût de transport ?
**Sortie** : quatre scalaires de transfer entropy + un scalaire d'asymétrie causale globale (`f6_causal_asymmetry_index`) agrégeant les 4 paires.
**Ancrage** : **A2 — charter §1.4 causalité Wolfram**. Γ crée-t-il un flux informationnel directionnel entre les observables ?
**Grammaire** : signal — TE classique sur séries temporelles uniformes. **Opération sur timelines couche A uniquement**, jamais sur couche B.

#### C2.3 — Autocorrélations (ps_first_min_ac_*)

**Entrée** : timelines couche A des observables `frob`, `shannon_comp`, `mode_asymmetry_o2`.
**Méthode** : détection du premier minimum de l'autocorrélation empirique.
**Sortie** :
- `ps_first_min_ac_frob`
- `ps_first_min_ac_shannon_comp` (renommé depuis `ps_first_min_ac_von_neumann`)
- `ps_first_min_ac_mode_asym_o2` (renommé depuis `ps_first_min_ac_mode_asymmetry`)
**Ancrage** : Q3 — détection de périodicité et de régimes stables.
**Grammaire** : signal.
**Nomenclature** : les renommages `von_neumann → shannon_comp` et `mode_asymmetry → mode_asym_o2` sont **essentiels** pour déclarer honnêtement que le signal analysé est l'observable O(n²), pas la vraie entropie spectrale.

#### C2.4 — PNN40 (ps_pnn40_*)

**Entrée** : timelines couche A.
**Sortie** :
- `ps_pnn40_shannon_comp` (renommé)
- `ps_pnn40_mode_asym_o2` (renommé)
**Ancrage** : Q3 — variabilité point-à-point des observables.
**Grammaire** : signal.

#### C2.5 — Temporal features (temporal_*)

**Entrée** : timelines couche A, principalement `frob` et `shannon_comp`.
**Méthode** : IQR, plateau_frac, cusum_delta, changepoint_t_norm calculés sur la timeline complète.
**Sortie** :
- `temporal_frob_iqr`, `temporal_frob_plateau_frac`, `temporal_frob_cusum_delta`, `temporal_frob_changepoint_t_norm`
- `temporal_shannon_comp_iqr`, etc. (renommé depuis `temporal_svn_*`)
**Ancrage** : Q3 — stationnarité et changements de phase dans les observables.
**Grammaire** : signal.

#### C2.6 — Stationnarité (nouvelle feature, Q-sci-5)

**Entrée** : chaque timeline couche A.
**Mesure** : pour chaque observable, `stationarity_delta = |mean(last 20%) - mean(first 20%)| / (std(all) + eps)`.
**Sortie** : un scalaire par observable de couche A, nommé `stat_delta_{observable}`.
**Ancrage** : Q3 — le run atteint-il un régime stationnaire ?
**Grammaire** : signal.
**Justification** : Q-sci-5 validé. Formalise explicitement "atteint-il un attracteur" au niveau de chaque observable, permet de filtrer les runs non stationnaires pendant l'analysing.

#### C2.7 — Entropy production rate (f2_entropy_production_rate)

**Entrée** : timeline couche A de `shannon_comp`.
**Mesure** : taux moyen de croissance de l'entropie (pente de régression linéaire).
**Ancrage** : A3 — production d'entropie irréductible.
**Grammaire** : signal.
**Note** : cette feature migre de spectral (VN) vers couche A (shannon_comp). Sémantique cohérente avec grammaire signal.

### 5.4 C3 — Features post-process en grammaire virages

#### C3.1 — Phasic features (phasic_*)

**Entrée** : timelines tronquées couche B sur `f2_von_neumann_entropy`, `f1_effective_rank`, `f3_mode_asymmetry` (celles qui sont en couche B).
**Méthode** : opère sur la séquence ordonnée des K valeurs aux points du masque **sans supposer d'uniformité temporelle**.
**Sortie** :
- `phasic_svn_n_reversals`, `phasic_svn_max_monotone_frac`, `phasic_svn_range_ratio`
- `phasic_rank_n_reversals`, `phasic_rank_max_monotone_frac`, `phasic_rank_range_ratio`
- `phasic_asym_n_reversals`, `phasic_asym_max_monotone_frac`, `phasic_asym_range_ratio`
**Ancrage** : Q3 + grammaire virages — forme du parcours aux virages.
**Grammaire** : virages.
**NaN runtime** : K_i < 2 pour un run donné → les n_reversals et autres ordinales n'ont pas de sens, NaN.

**Attention** : `phasic_asym` opère sur `f3_mode_asymmetry` qui est **désormais en couche A** (`mode_asymmetry_o2`). Deux choix possibles :

- **Option 1** : renommer `phasic_asym_*` en `phasic_mode_asym_o2_*` et les calculer en grammaire signal sur la timeline complète. Mais c'est incohérent (phasic est une grammaire virages).
- **Option 2** : garder phasic_asym en grammaire virages, en sous-échantillonnant mode_asymmetry_o2 aux points du masque au moment du post-process. C'est une projection virages d'un signal continu, valide si on l'assume explicitement.

**Décision à prendre pendant la phase structure** : je recommande Option 2 pour cohérence. La phasic est définie comme "comment l'observable varie **aux virages**", et appliquée à n'importe quel signal dont on prend la valeur aux virages. C'est propre.

### 5.5 Résumé post-scan

**Catégorie C1 — Agrégats** : ~50-60 scalaires (mean/std/final/delta/total pour chaque feature A et B selon le mapping).
**Catégorie C2 — Signal post-process** : ~20-25 scalaires (DMD × 4, F6 × 5, autocorr × 3, PNN × 2, temporal × 8, stationnarité × 11, entropy production × 1).
**Catégorie C3 — Virages post-process** : 9 scalaires (phasic).

**Total post-scan** : ~80-95 scalaires selon les choix de mapping d'agrégation.

---

## 6. FEATURES DE CLASSIFICATION ET GÉOMÉTRIE DU MASQUE

Ces features sont produites par `classify_and_mask` entre P1 et P2. Elles caractérisent le **régime du run** et la **géométrie du masque**. Elles ne sont ni de la couche A ni de la couche B — elles sont des métadonnées sur le run.

### 6.1 Classification P1

**Statut du run** : OK / OK_TRUNCATED / EXPLOSION / COLLAPSED
- OK : run sain, t_effective = max_it
- OK_TRUNCATED : run qui a dépassé un seuil d'explosion après une phase saine (transition vers la pathologie détectable). `t_effective` = instant du dépassement. **Feature scientifique importante** pour l'analyse des frontières de viabilité (chantier 3).
- EXPLOSION : is_finite passe à 0 à un moment, `t_effective` = instant de perte
- COLLAPSED : D devient quasi-constant (std/mean < seuil), `t_effective` = max_it

**Régime P1** : FLAT / OSCILLATING / TRANSITIONAL / EXPLOSIVE / MIXED
- Classification comportementale basée sur l'analyse de cos_dissim (CV, autocorrélation, Tukey fence)

### 6.2 Features P1 de caractérisation

- `p1_cos_dissim_mean`, `p1_cos_dissim_std`, `p1_cos_dissim_cv` — statistiques de l'activité directionnelle
- `p1_estimated_period` — période détectée par autocorrélation (NaN si non-périodique)
- `p1_n_zero_crossings` — zero-crossings de diff(cos_dissim), indicateur d'oscillation

### 6.3 Features mask de géométrie

- `mask_n_transitions` — nombre de segments contigus dans le masque
- `mask_t_first_norm`, `mask_t_last_norm` — positions temporelles normalisées (NaN si masque vide)
- `mask_mean_amplitude`, `mask_max_amplitude` — cos_dissim aux itérations actives
- `mask_mean_spacing_norm` — espacement moyen entre virages normalisé (NaN si < 2 virages)
- `mask_coverage_frac` — fraction de la timeline dans les zones actives

### 6.4 Meta features

- `meta_n_svd` — nombre effectif de points où la couche B a été calculée (= K_i)
- `meta_turbulence` — ratio k/max_it × constante
- `meta_t_effective` — durée effective du run (`min(max_it, t_cutoff)`)

---

## 7. STATUTS PATHOLOGIQUES ET LEUR TRAITEMENT

**Principe fondamental** : les statuts pathologiques ne sont **pas des erreurs**. Ils sont des observations scientifiques importantes, en particulier pour le chantier 3 (mapping des frontières de viabilité). Le pipeline doit les produire proprement, avec le maximum d'information préservée.

### 7.1 OK

Comportement standard. Tous les calculs sont faits jusqu'à `max_it`. Couche A complète, couche B aux virages, post-scan complet, toutes les features valides.

### 7.2 OK_TRUNCATED

**Le plus informatif scientifiquement**. Le run a passé une partie saine avant de diverger de manière monotone. Le `t_effective` marque l'instant de transition vers la pathologie.

Traitement :
- Couche A calculée **jusqu'à `t_effective`**, timelines complètes sur cette plage
- Couche B calculée aux virages **dans la plage `[0, t_effective)`**
- Post-scan calculé sur les données saines
- **L'information de transition est préservée** : `t_effective` + dernier état avant explosion + couche A jusqu'à t_effective

**Rôle dans l'analyse de frontières** (chantier 3) : un OK_TRUNCATED est un run qui *était viable puis est devenu non-viable*. Il définit une frontière empirique. Les features de ces runs, notamment leur couche A près de `t_effective`, sont la signature du passage à la non-viabilité.

### 7.3 EXPLOSION

`is_finite` passe à 0 sans phase saine suffisante pour être qualifié OK_TRUNCATED. L'état est corrompu.

Traitement :
- Couche A calculée jusqu'à `t_effective` (= instant où is_finite passe à 0)
- Couche B : masque vide, aucune mesure coûteuse calculée → économie compute
- Post-scan en grammaire signal sur couche A jusqu'à t_effective
- Agrégats : calculés sur la partie finie

### 7.4 COLLAPSED

L'état final est quasi-constant (std/mean < seuil). Le run a convergé vers un état dégénéré.

Traitement :
- Couche A calculée jusqu'à `max_it` (rien n'empêche le calcul, c'est juste plat)
- Couche B : masque vide (ou très réduit par les règles monotones) → peu de mesures coûteuses
- Post-scan normal

### 7.5 Préservation pour chantier 3

Tous les pathologiques conservent leur **couche A en timeline complète** jusqu'à `t_effective`. C'est ce qui permettra au chantier 3 d'étudier les signatures de transition vers la pathologie — notamment via des comparaisons du type "les OK_TRUNCATED ressemblent à quels OK au début de leur trajectoire ?" ou "quelle valeur de shannon_comp précède typiquement un collapse ?".

---

## 8. TABLEAU RÉCAPITULATIF PAR QUESTION SCIENTIFIQUE

| Question charter | Features dominantes | Couche | Grammaire |
|---|---|---|---|
| **Q1 — D encode-t-il ?** | frob, shannon_comp, ipr, mode_asymmetry_o2 | A | signal |
| | f1_effective_rank, f2_von_neumann_entropy, f2_renyi2_entropy, f1_sv_decay_rate | B | virages |
| | f3_entanglement × 2, f3_inter_mode_sv_var | B | virages (rank≥3) |
| **Q2 — Γ agit-il ?** | delta_D, cos_dissim, lyap_empirical, bregman_cost, frob_gradient, volume_proxy | A | signal |
| | f4_trace_J, f4_jvp_norm, f4_jacobian_asymmetry, f4_local_lyapunov | B | virages (diff) |
| | f7_dmd_* (universalité via observables) | post-scan | signal |
| | f6_* (causalité) | post-scan | signal |
| **Q3 — Structures stables ?** | temporal_*, ps_first_min_ac_*, ps_pnn40_*, stat_delta_* | post-scan | signal |
| | phasic_* (forme ordinale du parcours) | post-scan | virages |
| | p1_regime_class, p1_estimated_period | classify | — |
| | agrégats final/delta (convergence) | post-scan | — |
| **Q4 — Propriétés pré-émergentes** | f2_von_neumann_entropy (holographie) | B | virages |
| | f7_dmd_* (universalité) | post-scan | signal |
| | f6_* (causalité Wolfram) | post-scan | signal |
| | f5_* / bregman_cost / frob_gradient (transport) | A | signal |

**Lecture du tableau** : chaque question scientifique est couverte par **plusieurs features de plusieurs couches et grammaires**. Une seule feature ne répond jamais à une question entière. L'analysing doit croiser les signaux — c'est le rôle du clustering, du cluster namer et du verdict (chantiers 4, 5, 6).

---

## 9. CAVEATS ET NaN

### 9.1 NaN structurels (connus à la compilation)

Les NaN structurels sont prévisibles depuis les métadonnées du gamma et de l'encoding. Ils ne sont **pas** des observations — ce sont des absences attendues. L'analysing les masque avant clustering (charter P7).

| Condition | Features concernées |
|---|---|
| `rank_eff == 2` | f3_entanglement_entropy_mode1_*, f3_inter_mode_sv_var_*, frob_gradient_* |
| `is_differentiable == False` | f4_trace_J_*, f4_jvp_norm_*, f4_jacobian_asymmetry_*, f4_local_lyapunov_* |

### 9.2 NaN runtime (dépendants des données)

Les NaN runtime sont des **observations**. Ils disent quelque chose sur le run. L'analysing peut les traiter comme une classe à part entière.

| Condition | Features concernées |
|---|---|
| `K_i < 2` (masque trop parcimonieux) | phasic_*_n_reversals, phasic_*_max_monotone_frac, phasic_*_range_ratio |
| `n_transitions == 0` | mask_t_first_norm, mask_t_last_norm |
| `n_transitions < 2` | mask_mean_spacing_norm |
| autocorrélation sans pic | p1_estimated_period |
| run explosif avant couche B | features couche B vides |

### 9.3 Cas limites numériques

Tous les calculs utilisent `EPS = 1e-8` comme protection. Les cas limites critiques à surveiller dans l'implémentation :

- `shannon_comp` : protection contre `Σ|x| = 0`
- `ipr` : protection contre `Σ|x|² = 0`
- `lyap_empirical` : protection log contre `delta_prev = 0`
- `volume_proxy` : protection contre `||D||_1 = 0`
- `f1_condition_number` : protection contre `σ_{min} = 0`

---

## FIN DU DOCUMENT 1

*Ce document est la référence scientifique du chantier 1. Il sera mis à jour si des décisions scientifiques ultérieures modifient le pool ou la sémantique des features. Les détails d'implémentation numérique iront en commentaires dans `features_registry.py` lors de la phase structure/code du chantier 1.*
