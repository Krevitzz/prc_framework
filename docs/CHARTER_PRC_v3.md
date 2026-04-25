# CHARTER PRC v3
## Document d'ancrage cognitif unifié

> Ce document est la **source de vérité philosophique et méthodologique** du projet PRC.
> Ce document est **code-agnostique** : il ne fait référence à aucun fichier, aucun langage,
> aucune implémentation précise. Il décrit ce qui reste vrai quel que soit l'état du code.

---

## COMMENT UTILISER CE DOCUMENT

### Ce qu'il est

- **La philosophie scientifique** : pourquoi PRC existe, quels axiomes guident les mesures
- **Les principes architecturaux immuables** : ce qui reste vrai quel que soit le langage ou la version
- **La méthodologie de travail** : comment on raisonne, conçoit, valide, implémente
- **Le cycle d'exploration** : comment les phases s'enchaînent et pilotent l'évolution du pool
- **Le référentiel documentaire** : qui dit quoi et qui est autoritatif sur quel sujet

### Ce qu'il n'est pas

- Une source de vérité sur le code — les fichiers sources le sont
- Un manuel d'implémentation — les documents de chantier le sont
- Un catalogue de fonctions ou de modules — les inventaires le sont
- Un état du projet — les roadmaps de chantier le sont
- Une documentation utilisateur — elle n'existe pas encore, elle viendra à part

### Règle fondamentale

**Charter = boussole cognitive. Sources + catalogues = réalité. Chantiers = chemins.**

Si un principe du charter est contredit par le code, c'est soit que le code contient de la dette
à corriger, soit que le principe doit être révisé par discussion explicite. L'arbitrage se fait
toujours au niveau du charter, jamais par contournement silencieux.

### Hiérarchie de lecture

1. **Ce charter** — toujours en contexte pour toute conversation PRC
2. **La roadmap du chantier en cours** — pour savoir où on en est opérationnellement
3. **Le document scientifique des features du chantier** — pour savoir ce qu'on mesure et pourquoi
4. **Les sources et catalogues** — pour la vérité technique du moment

---

# PARTIE I — FONDATIONS SCIENTIFIQUES

## 1.1 Ce que PRC cherche à faire

PRC est un **banc de test de faisabilité axiomatique**, pas un moteur de simulation physique
ni un outil de fouille statistique généraliste.

La question centrale :

> **Les axiomes PRC peuvent-ils générer des structures émergentes mesurables ?**

Il ne s'agit pas de caractériser des signaux statistiques génériques. Il s'agit de mesurer
des **invariants informationnels et géométriques** directement ancrés dans les axiomes —
des instruments taillés sur la théorie, pas des marteaux universels appliqués à la théorie.

Une conséquence directe : chaque feature du pipeline doit pouvoir répondre à trois questions
avant d'y être admise.

1. À quelle question scientifique contribue-t-elle ? (voir §1.3)
2. Quel axiome ancre son interprétation ? (voir §1.2)
3. Dans quelle grammaire opère-t-elle ? (voir §1.5)

Une feature qui ne peut pas répondre à ces trois questions est un candidat à retirer.

## 1.2 Les trois axiomes

```
A1 — Dissymétrie informationnelle D irréductible
A2 — Mécanisme Γ agissant sur D
A3 — Aucun Γ stable ne peut annuler D complètement
```

Ces axiomes définissent exactement ce que le pipeline doit mesurer :

- **A1** → mesurer D elle-même : rang, asymétrie, capacité d'encodage informationnel,
  dispersion, localisation
- **A2** → mesurer l'action de Γ sur D : géométrie locale, contraction, expansion,
  conservation, signature spectrale, flux causaux
- **A3** → mesurer la résistance de D sous Γ : Γ dissipe-t-il D ? À quelle vitesse ?
  Existe-t-il un état attracteur où D ≠ 0 ?

Les trois axiomes ne sont pas indépendants : A3 est une condition d'existence posée sur A2,
et A2 ne prend son sens qu'à condition que A1 soit mesurable. Le pipeline mesure donc les
trois conjointement, jamais isolément.

## 1.3 Les quatre questions computationnelles

Pour chaque doublet candidat (encoding D, mécanisme Γ), le pipeline doit répondre
numériquement aux quatre questions suivantes :

```
1. D peut-il encoder de l'information structurée ?
   → rang effectif, entropie, asymétrie, localisation

2. Γ préserve-t-il, amplifie-t-il ou dissipe-t-il cette information ?
   → géométrie locale, Jacobien, Lyapunov, volume, transport

3. La dynamique produit-elle des structures émergentes stables ?
   → attracteurs, régimes, clustering, stationnarité, périodicité

4. Ces structures ont-elles les propriétés attendues d'une réalité pré-émergente ?
   → holographie, transport, causalité, universalité
```

Le pipeline répond à ces quatre questions par des **invariants numériques**. Le clustering
regroupe les doublets qui répondent de la même façon. Les verdicts opèrent sur ces réponses
pour dériver des contraintes sur le pool d'atomics.

**Une feature ne répond jamais seule à une question.** Chaque question est couverte par
plusieurs features de plusieurs couches et grammaires, qui doivent être croisées. Ce croisement
est le travail de l'analysing.

## 1.4 Connexions théoriques qui informent les familles de mesure

PRC soutient que les grandes théories physiques (relativité générale, mécanique quantique,
holographie, gravité entropique) étudient le même objet sous des angles différents. Si les
axiomes sont valides, les invariants mesurés doivent résonner avec chaque angle. Chaque
connexion théorique informe une famille de mesure — et donne son ancrage scientifique à
un sous-ensemble de features.

### Holographie (Ryu-Takayanagi)

L'entropie de Von Neumann de D est le proxy de l'aire holographique. Si S_VN tend vers un
minimum stable sous Γ, on a un attracteur holographique candidat. Si S_VN reste positif pour
tout Γ stable, A3 est observé empiriquement. C'est le test direct de A3.

### Gravité entropique (Verlinde)

La masse émerge comme gradient de densité de corrélation. Un gradient mesurable des normes
par mode tensoriel de D est le précurseur de ce phénomène — pas une métaphore, une quantité
calculable.

### Géométrie relationnelle (Rovelli)

La géométrie émerge des relations, pas des positions. L'information mutuelle entre modes
tensoriels de D est cette structure relationnelle. Sa densification sous Γ est une émergence
candidate de géométrie.

### Causalité (Wolfram)

La causalité émerge de règles locales. La transfer entropy dirigée entre observables de D
mesure si Γ crée un flux informationnel orienté — précurseur de causalité émergente.

### Universalité (Koopman / DMD)

Toute dynamique non linéaire admet une représentation linéaire dans un espace fonctionnel.
Le spectre DMD de Γ calculé sur un ensemble d'observables canoniques de D est la signature
de cette linéarisation. Si deux doublets (enc₁, Γ) et (enc₂, Γ) donnent le même spectre DMD
sur le même pool d'observables, alors Γ est universel indépendamment de la façon dont D
encode l'information. **C'est le test le plus direct de la thèse centrale de PRC.**

Pour que ce test soit rigoureux, le pool d'observables sur lequel DMD opère doit être :

- **Canonique** : chaque observable est invariant aux opérations qui ne changent pas
  la structure informationnelle de D (permutations de composantes, changements de base
  triviaux)
- **Suffisant** : le pool doit permettre de distinguer des structures différentes — deux D
  distinctes ne doivent pas produire le même vecteur d'observables
- **Indépendant** : pas de redondance excessive entre observables

## 1.5 Les deux grammaires

Le pipeline produit et exploite deux grammaires de mesure, qui coexistent et sont chacune
**cohérente avec ce qu'elle mesure**. Elles ne doivent **jamais** être mélangées : une
grammaire appliquée aux objets de l'autre produit des résultats numériquement définis mais
scientifiquement invalides.

### Grammaire signal

Applicable aux séquences temporelles à **pas uniforme**. Les méthodes classiques du traitement
du signal — autocorrélation, transfer entropy, décomposition en modes dynamiques (DMD), PNN,
statistiques temporelles (changepoint, cusum, plateau, IQR) — supposent toutes l'équidistance
temporelle. Elles mesurent des propriétés globales d'une trajectoire : mémoire, causalité,
linéarisation spectrale, stationnarité, régularité.

La grammaire signal s'applique uniquement à des observables calculés à **chaque itération**
de la dynamique, produisant des timelines à pas uniforme jusqu'à la fin effective du run.

### Grammaire virages

Applicable aux **séquences d'instants ordonnés non-uniformes**. Elle ne suppose pas
d'équidistance temporelle. Elle mesure des propriétés **ordinales** : nombre de virages,
forme du parcours, valeurs aux points d'intérêt, agrégats statistiques insensibles au pas.

La grammaire virages opère sur les points d'intérêt identifiés par le masque (voir §1.7) :
une séquence ordonnée de valeurs à des instants non uniformément espacés. Les méthodes
applicables en grammaire virages sont : agrégats scalaires (mean, std, final, delta, total),
n_reversals, max_monotone_frac, range_ratio, et toute mesure qui dépend uniquement de l'ordre
des points et non de leur espacement.

### Règle d'hygiène grammaticale

**Interdit** : appliquer une méthode signal à une séquence de points masqués non uniformes.
Une autocorrélation sur les K points d'un masque est numériquement définie mais n'a pas de
sens — l'espacement irrégulier fausse la notion même de lag. De même pour la transfer entropy,
pour DMD classique, pour les temporal features basées sur changepoint, etc.

**Interdit** : appliquer une méthode virages à une timeline continue quand l'information
est dans la forme complète du signal. Calculer uniquement n_reversals sur un signal continu
sans regarder son spectre est gaspiller l'information disponible.

**Obligatoire** : chaque feature du pipeline déclare explicitement sa grammaire. Cette
déclaration est vérifiable et auditable.

## 1.6 Les deux classes de mesure

Les invariants numériques du pipeline se répartissent en deux classes de coût radicalement
différent. Cette distinction est fondamentale pour la scalabilité et pour l'organisation
architecturale du pipeline.

### Mesures dynamiques O(n²)

Ces mesures observent l'action de Γ sur D **sans décomposer la structure interne de D**.
Elles sont calculables à chaque itération sans coût significatif. Elles produisent des
timelines à pas uniforme et relèvent donc naturellement de la grammaire signal.

Exemples d'invariants O(n²) :

| Famille | Mesure | Ancrage |
|---------|--------|---------|
| Norme | ||D||_F, ||D||_∞, ||D||_1 | A1 — D a une magnitude |
| Variation | ||D_t − D_{t−1}|| relative | A1+A2 — D change |
| Direction | cos_dissim entre D_t et D_{t−1} | A2 — rotation sous Γ |
| Information | Shannon sur composantes | A1 — dispersion |
| Localisation | IPR sur composantes | A1 — concentration |
| Asymétrie | ||M − Mᵀ||_F / ||M||_F | A1 — structure relationnelle |
| Transport | Coût de Bregman | A2 — coût de déplacement |
| Expansion | log(Δ_next/Δ_prev) empirique | A2 — Lyapunov sans Jacobien |
| Volume | Proxy ||D||_F · ||D||_∞ / ||D||_1 | A2 — contraction sans différentiation |
| Gradient inter-modes | Ratio des écarts-types par mode | A2 — Verlinde |

Ces observables partagent trois propriétés essentielles : **canonicité** (invariance par
permutation d'axes), **indépendance** (pas de redondance forte), **coût négligeable** par
rapport à la propagation elle-même.

### Mesures spectrales et différentiation O(n³)

Ces mesures nécessitent soit la décomposition en valeurs singulières (SVD), soit la
différentiation de Γ (Jacobien via JVP). Elles caractérisent respectivement la structure
interne et la géométrie locale de l'opérateur.

Exemples :

| Famille | Mesure | Ancrage |
|---------|--------|---------|
| F1 spectrale | Rang effectif, gap, condition, décroissance | A1 — structure spectrale |
| F2 entropique | Von Neumann, Rényi-2 sur le spectre | A1 + holographie |
| F3 multi-modale | Entanglement par mode, variance inter-mode | A1 + géométrie relationnelle |
| F4 Jacobien | trace(J), jvp_norm, jacobian_asymmetry, Lyapunov local | A2 — action différentielle de Γ |

Ces mesures sont coûteuses : O(n³) pour la SVD, et la différentiation Jacobienne allouent
des intermédiaires significatifs en mémoire. Leur calcul systématique à chaque itération est
prohibitif à haute dimension.

### Conséquence architecturale

La propagation de Γ (O(n²) ou plus selon le gamma) et la mesure spectrale (O(n³)) n'ont pas
besoin d'être synchrones. Si la dynamique de D est observable via des indicateurs O(n²) en
continu, les mesures O(n³) peuvent être restreintes aux instants où **il se passe quelque
chose d'intéressant** — et rien ne garantit que ces instants soient également espacés.

Les mesures dynamiques O(n²) fournissent un détecteur de variation qui permet d'identifier
les instants de transition significative. Les mesures spectrales et différentielles sont
alors calculées uniquement à ces instants — réalisant l'économie de compute visée.

Cette séparation ne perd pas d'information : les mesures coûteuses aux instants de transition
sont plus informatives que leur moyenne diluée sur des plateaux. Les mesures O(n²) restent
à pleine résolution temporelle, ce qui préserve la grammaire signal pour le post-process.

## 1.7 Le rôle du masque

Le **masque** est la liste des instants où "il se passe quelque chose d'intéressant" dans
la trajectoire d'un run. Il est déterminé par l'étape de classification à partir des timelines
O(n²) : transitions détectées par fence statistique sur les variations, points de cycles
détectés par autocorrélation, plancher minimum de couverture, extrémités valides pour les
runs monotones ou pathologiques.

### Ce que le masque est

Le masque est une **schématisation de la trajectoire** : *"on note les virages, de combien
ça tourne, dans quelle direction"*. Les valeurs aux points masqués ne sont pas des échantillons
équidistants d'une trajectoire cachée — ce sont des **instants qualitativement significatifs**.

Le masque est **global par run** : un seul masque gouverne où toutes les mesures O(n³) sont
calculées pour un run donné. Il est calculé à partir d'un ensemble d'observables O(n²) de
référence qui capturent le régime global du run (variation directionnelle, variation relative,
santé numérique, amplitude).

### Ce que le masque n'est pas

Le masque n'est pas un sous-échantillonnage temporel. Il n'est pas un outil d'approximation
d'une trajectoire complète. Il n'est pas utilisable pour reconstruire la dynamique entre
deux instants masqués.

**Conséquence grammaticale stricte** : les mesures calculées aux points masqués relèvent
exclusivement de la grammaire virages. Les méthodes signal ne peuvent pas s'appliquer aux
points masqués, même avec des reformulations — si le post-process a besoin d'une méthode
signal, son signal source doit être une timeline O(n²) continue, pas une extraction aux
points masqués.

### Exceptions contrôlées

Un observable O(n²) disponible en timeline continue peut être sous-échantillonné aux points
du masque pour nourrir une mesure en grammaire virages (par exemple, une mesure phasique de
l'asymétrie aux virages). Cette opération est **licite** : on projette une grammaire signal
vers la grammaire virages, en acceptant explicitement la perte d'information. L'opération
inverse (extrapoler une trajectoire continue à partir de points virages) est **interdite**.

## 1.8 Logique expérimentale — ce que le pipeline produit

Le pipeline crée des **candidats atomiques** (Γ, encoding D, modifier) et les caractérise
par des invariants numériques. Chaque phase teste des candidats et produit des verdicts
avec trois issues possibles :

- **Conserver** : pas de preuve négative, le candidat reste dans le pool
- **Explorer** : le verdict pointe un comportement particulier à approfondir
- **Exclure** : preuve d'incompatibilité avec un axiome ou avec une contrainte dérivée

### Statuts de run

Chaque run produit un statut qui reflète son déroulement numérique. Les statuts ne sont
pas des erreurs — ils sont des **observations scientifiques** à traiter différemment selon
leur nature.

```
OK          → run normal, toutes les features calculées
OK_TRUNCATED → transition vers pathologie détectée — partie saine préservée
EXPLOSION   → état non fini détecté, trajectoire corrompue
COLLAPSED   → état quasi-constant, perte d'information
INVALID     → composition structurellement incompatible
FAIL        → erreur technique pendant le calcul
```

**OK_TRUNCATED est le statut le plus informatif scientifiquement.** Il identifie un run qui
était viable puis est devenu non viable. Il définit empiriquement une frontière de viabilité.
Les features de ces runs, notamment juste avant la transition, sont la signature du passage
à la non-viabilité et doivent être étudiées comme telles.

### NaN ≠ erreur

Certaines features sont structurellement NaN selon le rang du tenseur ou la différentiabilité
du gamma. D'autres sont NaN au runtime parce que le run n'a pas produit l'information
nécessaire (par exemple, pas de transition → pas de spacing entre virages calculable).

**Les deux types de NaN sont des observations valides**, pas des artefacts d'erreur. Le
pipeline les distingue explicitement :

- **NaN structurel** — connu à la compilation depuis les métadonnées de l'atomic
  (rank_eff == 2, is_differentiable == False). L'analysing les masque a priori — ce ne
  sont pas des observations, ce sont des absences attendues.
- **NaN runtime** — dépendant des données du run (K < 2, n_transitions == 0). C'est une
  observation — un NaN runtime porte de l'information (le run n'a pas produit de transition
  = signal, pas artefact).


## 1.09 Vision long terme

```
Phase actuelle  : faisabilité axiomes
                  → les axiomes génèrent-ils des structures émergentes mesurables ?
    ↓
Phase R*        : caractérisation systématique des candidats Γ
                  → quels Γ sont universels, contractants, holographiques ?
    ↓
Phase plugin    : encodings domaine-spécifiques (relativité, QM, biologie, ...)
                  → D encode-t-il une réalité physique connue ?
    ↓
Phase validation : retrouver des propriétés physiques connues
                  → ex : mouvement de Mercure avec encoding système solaire + GR
```

Chaque phase a un critère de transition explicite vers la suivante. Le pipeline ne passe pas
à la phase R* tant que la phase actuelle n'a pas produit un pool d'atomics stable et une
cartographie des contraintes de viabilité. Il ne passe pas à la phase plugin tant que R*
n'a pas identifié au moins un Γ candidat universel.

---

# PARTIE II — PRINCIPES ARCHITECTURAUX

## 2.1 Niveaux de dépendance

```
L0 (ontologique)   → axiomes A1, A2, A3
L1 (épistémique)   → théories physiques qui informent les mesures
L2 (théorique)     → familles de features ancrées dans L0 et L1
L3 (méthodologique) → grammaires, masque, distinction des coûts
L4 (opérationnel)  → pipeline : kernel, atomics, running, analysing
L5 (documentaire)  → YAML, configs, catalogues, registres
```

**Règle de dépendance** : L(n) référence uniquement L(≤n−1). L(n) ne peut pas dériver des
règles de L(n+1) ni de L(n).

**Exception contrôlée** : L4 peut consommer L5 comme descriptif (lire un registre, charger
un YAML), mais jamais en dériver des règles philosophiques. Un YAML peut dire qu'un gamma
est différentiable ; il ne peut pas définir ce que signifie "différentiable" scientifiquement.

## 2.2 Principes architecturaux P1 à P12

Ces principes s'appliquent à toute implémentation du pipeline, indépendamment du langage,
du framework ou de la version. Ils sont numérotés pour traçabilité ; leur ordre n'impose
pas de hiérarchie.

### P1 — Intra-run / Inter-run

**Intra-run** : tout ce qui opère sur un run unique en mémoire vive (simulation, features,
statut). Doit pouvoir être calculé et libéré immédiatement.

**Inter-run** : tout ce qui nécessite le contexte cross-runs (clustering, profiling,
concordance). Opère depuis le stockage persistant.

**Règle** : ce qui peut être calculé intra-run l'est. L'inter-run est réservé à ce qui
requiert explicitement plusieurs runs.

### P2 — Gestion des états pathologiques

| Type | Cause | Action |
|------|-------|--------|
| Erreur de calcul | Opération non applicable | Lever une exception explicite |
| Valeur hors domaine mathématique | NaN/Inf sur calcul applicable | Lever une exception explicite |
| Explosion physique du système | Comportement réel du candidat | Retourner NaN (signal, pas erreur) |

Le pipeline ne s'arrête pas sur une explosion. Il la signale et continue. Arrêter le calcul
dès la détection d'une explosion (early stopping) est une optimisation valide : propager
l'état sans recalcul jusqu'à la fin du run, puis marquer le statut en conséquence.

### P3 — Kernel aveugle au domaine

Le kernel d'exécution (boucle de simulation temporelle) est la seule boucle temporelle du
pipeline. Il doit respecter ces invariants :

```
K1 — Aucune validation sémantique du contenu des tenseurs
K2 — Aucune classe State ou Operator
K3 — Aucun branchement dépendant du contenu de D ou Γ
K4 — Aucune connaissance des atomics (reçoit des fonctions opaques)
K5 — Fonction pure : mêmes entrées → mêmes sorties
```

Le kernel ne sait pas ce que représente D. Il ne sait pas si Γ modélise de la gravité ou
de la biologie. Il prend des tenseurs, retourne des tenseurs.

### P4 — Zéro paramètre hardcodé

Tout paramètre qui influence un résultat doit vivre dans un fichier de configuration. Pas
de constantes numériques dans le code de production. Tout axe d'itération du plan expérimental
est configurable via YAML sans modifier le code.

Les constantes sont séparées par nature, pas par commodité :

- **Constantes machine** (performances, ressources) — nombre de workers, tailles de batch,
  seuils VRAM, back-pressure. Leur modification affecte les performances mais **jamais
  les résultats scientifiques**. Deux machines avec des constantes machine différentes
  produisent les mêmes données.
- **Constantes scientifiques** (mesure, seuils, epsilon) — vivent dans les configs du layer
  qui les consomme. Leur modification change les valeurs numériques produites et doit être
  versionnée. Deux instances avec les mêmes constantes scientifiques produisent les mêmes
  résultats, indépendamment de la machine, du nombre de cœurs ou du type d'accélérateur.

**Règle** : chaque constante est accompagnée d'un commentaire expliquant son rôle, sa
provenance (expérience, calibration, convention), et les conséquences d'un changement.

### P5 — Fonctions pures dans les registres

Les atomics (gammas, encodings, modifiers) et les registres de features sont des fonctions
pures : elles ne lisent pas de fichiers, ne modifient pas d'état global, ne dépendent que
de leurs arguments. Leur comportement est déterministe à seed égale.

### P6 — Persistance avant analyse

Le stockage persistant (parquet) est écrit et fermé avant toute analyse. Si le pipeline
crashe pendant l'analyse, les données de simulation sont intactes. L'analyse peut être
relancée sans réexécuter les runs.

### P7 — Clustering stratifié

Pas de clustering global sur des features hétérogènes. Le clustering opère sur des
sous-ensembles homogènes (même rank effectif, même applicabilité des features). Les features
structurellement NaN pour un sous-groupe sont masquées avant le clustering de ce sous-groupe.

### P8 — Séparation stricte intra/inter layer

Un module d'un layer ne peut pas importer un module d'un autre layer de même niveau. Seuls
les hubs peuvent orchestrer : ils importent depuis leur layer uniquement et délèguent vers
les hubs des layers inférieurs.

Les dérogations architecturales nécessaires (par exemple, accès aux noms de features depuis
le layer d'exécution) doivent être documentées explicitement dans les docstrings des modules
concernés.

### P9 — Isolation mémoire des unités d'exécution

Chaque unité d'exécution (compilation + calcul d'un groupe de runs) doit vivre dans un
espace mémoire isolé. Si une unité fuit, sa terminaison doit garantir la purge totale de
ses allocations, y compris les caches du compilateur.

Conséquences :

- L'isolation par processus est préférée à l'isolation par thread (un thread partage la
  mémoire du parent ; un processus peut être tué proprement)
- La compilation et l'exécution sur accélérateur sont des ressources exclusives : un seul
  calcul accède à l'accélérateur à un instant donné **pour une même unité**. Plusieurs
  unités peuvent coexister si leurs budgets VRAM le permettent (voir P11)
- Le processus orchestrateur ne compile pas et n'exécute pas : il coordonne, collecte les
  résultats, et écrit le stockage persistant
- Aucune pré-allocation fixe de mémoire accélérateur : laisser le runtime allouer et libérer
  dynamiquement pour absorber les overlaps transitoires entre entrée et sortie

**Règle** : si on constate une fuite, la terminaison du processus isolé est un dernier
recours valide et suffisant. Si la fuite persiste après terminaison, elle vient de
l'orchestrateur — ce qui réduit drastiquement la surface de diagnostic.

### P10 — Cohérence grammaticale des mesures

Toute feature du pipeline déclare explicitement sa grammaire (§1.5). Une méthode signal
ne peut pas être appliquée à une séquence virages, et inversement. Les features post-process
qui consomment des timelines déclarent explicitement quelle timeline continue elles
consomment, sans jamais extraire de signal signal à partir d'une séquence masquée.

L'audit statique du pipeline vérifie cette cohérence : pour chaque feature, il trace les
signaux sources et vérifie qu'une méthode signal ne consomme que des timelines O(n²)
uniformes, et qu'une méthode virages ne consomme que des séquences ordonnées au masque.

### P11 — Vases communicants et utilisation des ressources

Le pipeline fonctionne comme un système de vases communicants. Chaque module est un vase
qui déverse ses résultats dans une queue, et chaque module aval consomme depuis cette queue.
Tous les modules tournent en parallèle sur un même liquide (les données). Deux séries
indépendantes permettent de traiter deux liquides en alternance pour maximiser l'occupation
de la ressource goulot (typiquement l'accélérateur).

Objectif d'optimisation :

- **L'accélérateur est le goulot** : coût de calcul élevé, ressources limitées
- **Il ne doit jamais attendre** : pendant qu'un calcul GPU est en cours, la préparation
  CPU du calcul suivant est déjà en route
- **Les flux CPU alimentent l'accélérateur en continu** : chargement, matérialisation,
  compilation, transfert — tout est pipeliné

Rôles stricts :

- **Orchestrateur** (processus principal) : ne compile pas, n'exécute pas. Coordonne,
  distribue, collecte, écrit le stockage persistant
- **Workers CPU** : dupliqués pour gérer le débit. Effectuent matérialisation, pré-traitements,
  post-process, assemblage des résultats
- **Worker GPU** (ou équivalent accélérateur) : exécute les compilations et les scans.
  Un calcul à la fois par unité d'exécution isolée, plusieurs unités en parallèle si le
  budget le permet

Règle : les constantes machine sont calibrées pour maximiser l'occupation de l'accélérateur
sans provoquer d'OOM. Les constantes scientifiques sont indépendantes de cette calibration.

### P12 — Budget VRAM dynamique et découpage adaptatif

Le pipeline ne présume pas de la taille d'un job avant de l'avoir estimé. Les jobs trop
gros pour le budget VRAM disponible sont **pré-découpés** par l'orchestrateur avant
distribution. Le découpage divise la dimension batch jusqu'à ce que l'estimation rentre dans
le budget.

Un job qui ne rentre pas même à batch = 1 est lancé avec un warning explicite (risque OOM
accepté) plutôt que silencieusement rejeté. L'estimation VRAM utilise un modèle analytique
conservateur incluant une marge XLA et une marge additionnelle pour les rangs élevés.

Le budget VRAM est **partagé entre toutes les unités d'exécution actives**. L'acquisition
est bloquante avec timeout : une unité attend jusqu'à ce que son budget soit disponible, ou
meurt proprement si le timeout expire. Le budget est libéré dans un bloc finally pour
garantir sa restitution même en cas de crash.

## 2.3 Philosophie vases communicants (approfondissement P11)

Cette section détaille comment P11 se concrétise dans un pipeline PRC typique. Les termes
utilisés ne référencent aucune implémentation — ils décrivent le schéma général.

### Flux général

```
    YAML plan expérimental
          │
          ▼
    Orchestrateur — génère les jobs
          │
          ├──→ Unité A (processus isolé)
          │      CPU prep → GPU scans → CPU post → Queue
          │
          ├──→ Unité B (processus isolé)
          │      CPU prep → GPU scans → CPU post → Queue
          │
          └──→ ...
                                                    │
                                                    ▼
                                         Orchestrateur collecte → parquet
```

Chaque unité est indépendante : elle a sa propre compilation, son propre budget, son propre
cycle de vie. L'orchestrateur ne fait que distribuer et collecter.

### Préparation avant exécution GPU

Avant même d'acquérir le budget GPU, une unité peut effectuer tout son travail CPU :
discovery des atomics, matérialisation des tenseurs initiaux, compilation des fonctions
JIT. Cette phase CPU est **hors budget VRAM**, ce qui permet à plusieurs unités de se
préparer en parallèle pendant qu'une autre utilise effectivement le GPU.

### Acquisition et libération du budget

Le budget VRAM est acquis **juste avant** la phase GPU (scans), et libéré **juste après**
(transfert GPU→CPU terminé). Le post-process numpy se fait hors budget. Cette discipline
maximise la rotation des unités sur le GPU et minimise les temps d'attente.

### Pas de mémoire partagée directe

Les workers communiquent par queues thread-safes / process-safes. Aucun accès direct aux
variables d'un autre worker. Les sémaphores et les queues sont les seuls points de
synchronisation.

### Nettoyage complet entre groupes

Lorsqu'une unité termine, sa terminaison purge toute sa mémoire (y compris les caches
compilateur). Un nouveau groupe peut être introduit sans risque d'accumulation inter-groupe.
Si une fuite est détectée, la terminaison forcée du processus est un dernier recours valide.

## 2.4 Interdictions architecturales critiques

### Sur le kernel

```
❌  Validation sémantique du contenu des tenseurs
❌  Classes State ou Operator
❌  Branchements conditionnels sur D ou Γ dans la boucle temporelle
❌  Connaissance des atomics (reçoit des callables opaques)
```

### Sur les features

```
❌  Valeurs de retour PASS/FAIL, bon/mauvais, jugements normatifs
❌  Paramètres hardcodés dans les fonctions de calcul
❌  I/O (fichiers, parquet) dans les registres de features
❌  État global mutable dans les fonctions pures
❌  Application d'une méthode signal à une séquence virages
❌  Extrapolation d'une trajectoire continue à partir de points virages
```

### Sur l'architecture

```
❌  Dépendances circulaires entre modules
❌  Import d'un module de même layer depuis un non-hub
❌  Paramètres numériques hardcodés dans le code
❌  Code produit sans validation Algo → Structure → Code
❌  Déviation aux principes sans discussion et documentation explicites
❌  except/catch silencieux : toujours logger et signaler
❌  Pré-allocation fixe de mémoire accélérateur dans le code de production
❌  Compilation et exécution dans le processus orchestrateur
❌  Accumulation de données inter-groupes sans purge
```

### Sur la nomenclature

```
❌  "matrice de corrélation"   →  ✅  "tenseur rang 2"
❌  "graphe"                   →  ✅  "patterns dans tenseur"
❌  "test verdict"             →  ✅  "observation" (feature) / "verdict" (analysing)
❌  "résultat bon/mauvais"     →  ✅  valeur numérique interprétable
❌  "sous-échantillonnage"     →  ✅  "schématisation des virages" (masque)
❌  "autocorr sur virages"     →  ✅  méthode signal interdite en grammaire virages
```

# PARTIE III — MÉTHODOLOGIE DE TRAVAIL

## 3.1 Protocole d'audit systémique

Avant toute proposition de modification architecturale ou d'optimisation, un protocole
d'audit en cinq phases doit être suivi. Ce protocole neutralise les corrections locales
aveugles et garantit que chaque changement est aligné avec les principes scientifiques
et architecturaux.

### Phase 0 — Cartographie du flux de données

Pour la version courante du code et pour toute version de référence :

- Dessiner le cheminement des données depuis l'entrée (YAML) jusqu'à la sortie (parquet)
- Pour chaque étape, noter :
  - **Format** des données (Python natif, numpy, JAX, pandas, polars)
  - **Taille** approximative (nombre d'éléments, mémoire)
  - **Nombre de passages** sur les données (une, deux, plusieurs passes)
  - **Opération** effectuée (calcul, conversion, tri, indexation, agrégation)
- Identifier les points de conversion de type (list → numpy → JAX → numpy → list)

**Règle** : aucune modification de code n'est discutée tant que ce schéma n'est pas
explicite et validé. La cartographie doit être lisible par quelqu'un qui ne connaît pas
l'implémentation actuelle.

### Phase 1 — Grille des cinq pourquoi

Pour chaque étape identifiée en Phase 0, poser systématiquement :

1. **Pourquoi ce calcul est-il fait à cet endroit précis ?** Ne peut-il pas être fusionné
   avec l'étape précédente ou suivante ?
2. **Pourquoi ce résultat est-il stocké en mémoire** alors qu'il pourrait être consommé
   immédiatement ? (principe du streaming / pipeline lazy)
3. **Pourquoi utilisons-nous un format de tableau** (numpy/JAX/pandas) plutôt qu'une
   structure plus simple (scalaire, vecteur natif, générateur) ? Chaque conversion a un
   coût — est-elle indispensable ?
4. **Pourquoi faisons-nous plusieurs passes sur les données ?** Une seule passe peut-elle
   suffire en accumulant des résultats partiels ?
5. **Pourquoi cette valeur est-elle mise à jour** alors qu'elle n'est jamais lue avant
   la fin ? (calcul paresseux, suppression de mises à jour inutiles)

**Règle** : on ne passe à la phase suivante que lorsque toutes ces questions ont reçu une
réponse claire, documentée dans le cahier d'audit. Les réponses peuvent être "parce que
X est une contrainte de l'architecture cible" — mais jamais "parce que c'est comme ça dans
la version actuelle".

### Phase 2 — Identification des dérives architecturales

Confronter le schéma et les réponses aux principes du charter (Partie II) et aux règles
grammaticales (§1.5) :

**Dérives scientifiques à détecter** :
- Violation de la distinction mesures O(n²) vs O(n³) (§1.6)
- Application d'une méthode signal à des points masqués (§1.5, P10)
- Extrapolation d'une trajectoire à partir de virages
- Interprétation d'une feature comme "bon" ou "mauvais" (P2)
- Mesures spectrales calculées à chaque itération "au cas où"
- Mélange de mesures dynamiques et spectrales dans la même boucle sans condition

**Dérives architecturales à détecter** :
- Violation d'un principe P1-P12
- Goulot GPU qui attend passivement (violation P11)
- Orchestrateur qui effectue des calculs lourds (P9 + P11)
- Accumulation de données inter-groupes (fuite P9)
- Constantes scientifiques mélangées aux constantes machine (P4)
- Conversions de format inutiles (JAX → numpy → JAX)
- Dépendances cross-layer non documentées (P8)

**Règle** : chaque dérive est listée et priorisée en "critique" (casse l'architecture) vs
"optimisation locale". Aucune modification n'est proposée avant d'avoir classé les dérives.

### Phase 3 — Proposition de restructuration sans code

Pour chaque dérive critique, rédiger une **intention de changement** en langage naturel :

- **Objectif** : ce qu'on cherche à obtenir
- **Principe** : l'idée de la solution, indépendante de l'implémentation
- **Impact attendu** : réduction de temps, de VRAM, simplification, cohérence scientifique
- **Compatibilité avec les axiomes** : ce qui ne change pas scientifiquement

**Règle** : aucune ligne de code n'est écrite tant que cette intention n'a pas été validée
par l'humain. L'intention est un contrat qui protège contre la dérive en cours
d'implémentation.

### Phase 4 — Implémentation par patchs atomiques

Chaque patch doit :

- Correspondre à une **seule** intention
- Contenir dans son en-tête :
  - La référence à la dérive corrigée
  - Le principe de la solution (rappel)
  - Les tests minimaux de non-régression
- Être présenté sous forme de diff commenté, jamais comme un bloc de code brut sans
  explication

**Règle** : ne jamais recopier un fragment de code provenant d'une autre version sans
avoir explicitement justifié que chaque ligne est compatible avec l'architecture cible et
les principes du charter.

## 3.2 Algo → Structure → Code

Au-delà du protocole d'audit (qui s'applique aux refactors), tout **nouveau développement**
suit un cycle de trois étapes avec validation obligatoire à chaque étape. Aucun passage à
l'étape suivante sans validation explicite par l'utilisateur.

### Étape 1 — ALGO (langage courant)

Décrire ce qu'on fait et pourquoi, **sans code**, **sans jargon technique**. Un non-programmeur
doit pouvoir détecter une dérive métier à ce stade.

Questions à répondre :
- Qu'est-ce qui entre ?
- Qu'est-ce qui sort ?
- Pourquoi cette transformation est-elle correcte du point de vue des axiomes ?
- Quelle grammaire est utilisée ?
- Quelle question scientifique (Q1-Q4) est servie ?

Sortie : un paragraphe ou quelques paragraphes en prose. Pas de pseudo-code.

### Étape 2 — STRUCTURE (squelette ancré)

Signatures des fonctions, types I/O, dépendances, ancré sur le code existant réel. **Pas
de code fonctionnel** — juste la forme.

Identifier les points de décision structurelle (SD) et les trancher explicitement avant
de passer au code. Une SD est un choix qui a plusieurs réponses viables et dont la
conséquence est difficile à défaire.

Exemples de SD typiques :
- Taille d'un buffer fixe vs variable
- Accès par index vs par itération
- Allocation anticipée vs lazy
- Granularité de découpage temporel ou batch

Sortie : signatures, types, squelette de flux, liste des SD tranchées.

### Étape 3 — CODE

Implémentation dans le squelette validé. **Seulement après** validation des deux étapes
précédentes. Le code remplit les corps de fonctions dont les signatures ont été validées.

Aucune invention d'interface à ce stade. Si un besoin d'interface non prévu apparaît, on
retourne à l'étape structure avec mention explicite du changement.

### Tests à chaque sous-étape

Chaque module dispose d'un fichier de tests dédié. Les tests sont écrits **en même temps**
que le code, sous-étape par sous-étape. On ne passe pas à la sous-étape suivante avant
que les tests de la sous-étape courante passent.

## 3.3 Expériences standalone avant changements radicaux

Tout changement architectural radical (fusion de modules, nouvelle politique de cache,
nouveau mécanisme de compilation, nouveau schéma de données) doit être validé par une
**expérience standalone** avant d'être intégré.

Une expérience standalone :

- Est un fichier autonome, zéro dépendance au pipeline de production
- A un protocole précis et un critère de succès binaire
- Est conservée dans un répertoire dédié pour traçabilité historique
- Doit passer avant que le chantier correspondant commence

Les expériences standalone sont un filet de sécurité : elles valident empiriquement
qu'une intention architecturale produit le gain attendu **avant** qu'on s'engage dans
l'implémentation complète. Elles protègent contre les raisonnements *a priori* qui
paraissent solides mais qui sont contredits par le comportement réel du compilateur ou
du matériel.

## 3.4 Règles de travail

- **Toujours lire la documentation** avant de coder (charter, roadmap de chantier, features
  du chantier, catalogues, sources)
- **Jamais supposer** — demander les fichiers manquants plutôt qu'inventer
- **Jamais de quick fix** sans vérification dans les sources
- **Toute déviation** aux principes → discussion explicite avant implémentation
- **Toute dérogation architecturale** → documentée dans la docstring du module
- **Modifications séquentielles uniquement** (voir §3.5)

## 3.5 Modifications séquentielles

Ne jamais appliquer deux modifications au même code dans la même passe.

Si deux modifications B et C doivent être appliquées sur un état A :

1. Appliquer A → B
2. **Relire B** (le résultat réel, pas le souvenir de A)
3. Évaluer si C s'applique encore à B, ou s'il faut écrire B → C'
4. Appliquer B → C' (ou C si toujours applicable)

Le risque : appliquer A → B puis A → C (en travaillant mentalement sur l'ancien état),
perdant B entièrement. Ce phénomène est systématique quand on enchaîne les modifications
sans relecture intermédiaire.

**Règle stricte** : une modification, une relecture, puis la suivante. Cette règle
s'applique autant à l'assistant qu'à l'humain.

## 3.6 Vérification objective par audit statique

Les docstrings décrivent l'intention. Le code implémente la réalité. Ces deux vérités
peuvent dériver l'une de l'autre — et le font inévitablement.

Le pipeline maintient un outil d'audit statique qui extrait la structure réelle du code
(fonctions, imports, constantes, dépendances, objets) par analyse syntaxique, **sans
exécuter le code ni lire les docstrings**. Cet outil produit une carte objective de l'état
du pipeline.

### Fonctions épistémiques de l'audit statique

1. **Détection de la dette technique** — code mort, imports inutilisés, constantes non
   paramétrées, objets orphelins
2. **Vérification des principes** — les interdictions du charter (P4, P8, P10) sont
   testables mécaniquement. Les constantes hardcodées et les imports cross-layer sont
   détectables sans interprétation humaine
3. **Indépendance des docstrings** — la carte générée ne dépend pas de la bonne volonté
   des développeurs à maintenir les annotations

### Règles d'utilisation

- L'outil d'audit est **exécuté après chaque phase** de développement pour mesurer la
  réduction des violations et détecter les régressions
- L'audit ne remplace pas la relecture humaine, il la **complète**
- Un audit propre ne garantit pas la qualité scientifique, seulement la conformité
  architecturale

### Limitations connues

L'analyse statique ne suit pas les appels indirects (via partial, vmap, registres de
fonctions, décorateurs). Les faux positifs sur les fonctions "mortes" sont attendus pour
les fonctions utilisées par le runtime via ces mécanismes. **L'outil signale, l'humain
tranche.**

## 3.7 Règles de blocage (contraintes sur l'assistant)

Pour éviter les dérives dans les échanges assistant-humain, l'assistant s'interdit de :

- Proposer une optimisation qui n'a pas été précédée des phases 0 à 2 de l'audit
- Utiliser des valeurs numériques (taille de batch, seuils, epsilon) sans les distinguer
  comme constantes machine ou constantes scientifiques
- Ignorer une question des cinq pourquoi en prétextant que "c'est comme ça dans la
  version actuelle"
- Écrire du code qui contourne la philosophie vases communicants sans justifier pourquoi
  le contournement est nécessaire
- Inventer une interface qui n'est pas dans la structure validée
- Présenter du code avant validation explicite de l'algo puis de la structure
- Appliquer deux modifications au même code sans relecture intermédiaire
- Supposer qu'un fichier a un contenu non lu ; demander à le voir si nécessaire

## 3.8 Style de collaboration et format des réponses

Cette section fixe le cadre d'interaction entre l'assistant et l'humain. Elle est stable à travers les chantiers et n'a pas à être redéclarée dans les documents de transition.

### 3.8.1 Profil de l'interlocuteur

L'humain qui dirige le projet PRC n'est ni scientifique professionnel ni développeur de métier. Il a en revanche une compréhension fine du projet qu'il conçoit. Son rôle est de concevoir, questionner, trancher. Le rôle de l'assistant est d'apporter la rigueur scientifique et architecturale, de détecter les incohérences entre les intuitions proposées et les contraintes du charter, et de refuser les raccourcis qui contourneraient la méthodologie.

Cette asymétrie a une conséquence directe : l'assistant ne se censure pas par politesse. Si une idée proposée par l'humain est incompatible avec un principe, l'assistant le signale clairement, avec la référence au principe concerné. Inversement, si une proposition de l'assistant est contredite par une observation, l'assistant l'abandonne sans tergiverser. L'honnêteté intellectuelle est bidirectionnelle et prime sur le confort de l'échange.

### 3.8.2 Préférences d'itération

L'itération efficace repose sur quelques règles stables que l'assistant applique sans attendre de rappel.

**Densité avant verbosité.** Un document long qui évite de recharger un autre document ou de refaire une explication vaut mieux qu'un résumé trop succinct qui force des re-demandes. La même règle s'applique aux réponses conversationnelles : préférer le dense au creux, sans tomber dans le succinct qui oblige à relancer. Les échanges creux sont proscrits — si l'assistant n'a rien de neuf à apporter, il pose une question plutôt que de remplir.

**Questions précises et peu nombreuses.** Trois questions ciblées valent mieux que dix questions vagues. Quand plusieurs questions sont nécessaires, l'assistant les numérote et indique leur ordre d'importance.

**Options chiffrées plutôt que formulations vagues.** Quand plusieurs chemins sont possibles, l'assistant les présente avec leurs coûts, gains et risques estimés — jamais "il y a plusieurs approches possibles" sans quantification.

**Validation intermédiaire systématique.** L'humain aime trancher par petites étapes. L'assistant découpe les décisions non triviales en points de validation successifs plutôt que de présenter un plan monolithique à valider d'un bloc.

**Reformulation en cas d'ambiguïté.** Quand une demande est ambiguë, l'assistant reformule avant d'agir. Cette reformulation est plus sûre que de partir sur un malentendu — l'humain corrige volontiers.

**Politique d'enrichissement.** "Une feature de trop plutôt qu'une de moins." Chaque re-run complet coûte cher, l'asymétrie du regret penche vers l'inclusion. Cette règle s'applique aussi bien aux features qu'aux observables intermédiaires et aux métadonnées persistées.

**Signaux de décision.** "Je valide" signifie que la décision est actée et que l'assistant peut avancer. "Je pense que..." signifie que l'humain attend l'avis de l'assistant avant de trancher — ce n'est pas une décision, c'est une proposition soumise à contre-expertise.

**Interdictions stables d'interaction.** Pas de code produit sans validation Algo → Structure → Code. Pas de suppositions sur des fichiers non lus — l'assistant demande à les voir. Pas de quick fix qui cache un problème — l'assistant signale le problème au lieu de le patcher. Pas de surconfiance — si l'assistant n'est pas sûr d'une information, il le dit.

### 3.8.3 Format des réponses selon le type de demande

Le format de réponse n'est pas cosmétique : il conditionne la capacité de l'humain à exploiter la réponse. Quatre régimes sont distingués.

**Pour les questions conceptuelles.** L'assistant reformule d'abord la question si elle est ambiguë. La conclusion ou la recommandation est clairement identifiable, pas noyée dans la prose. Si plusieurs options existent, elles sont numérotées et chiffrées. La réponse se termine par une question unique ou un choix précis à valider, jamais par un résumé creux ou une relance vague.

**Pour les propositions d'architecture.** Les étapes Algo, Structure et Code sont visiblement séparées — jamais mélangées dans le même bloc. Les points de décision structurelle sont identifiés et tranchés explicitement. Les risques et contre-arguments sont signalés par l'assistant lui-même — une proposition n'est jamais vendue comme parfaite. Chaque étape attend validation explicite avant passage à la suivante.

**Pour le code.** L'assistant commence par rappeler quelle intention valide il implémente (référence à l'étape Structure préalable). Les modifications passent par les outils d'édition, pas par des blocs inline à recopier. Après chaque modification, l'assistant relit le fichier avant d'en proposer une autre — application stricte de §3.5.

**Pour les artefacts longs.** L'assistant utilise la création de fichiers et les présente explicitement. Il découpe en plusieurs fichiers quand cela aide à la relecture. Après livraison, il signale les points qui méritent l'attention prioritaire de l'humain plutôt que de laisser celui-ci trier seul.

### 3.8.4 Périmètre de cette section

Ces règles sont **stables à travers les chantiers**. Un document de transition (hand-off) entre deux conversations ne les redéclare pas : il se contente de pointer vers cette section et de fournir le contexte spécifique au chantier en cours — décisions actées, données factuelles, fichiers sources, point d'entrée, pièges propres au chantier. Tout ce qui est constant vit dans le charter ; tout ce qui est variable vit dans le hand-off.

---

# PARTIE IV — CYCLE D'EXPLORATION

## 4.1 Principe général

Le travail scientifique PRC s'organise en **phases majeures** qui produisent des données
réutilisables (parquet, verdicts, cartographies) et en **sous-phases locales** qui
calibrent des atomics individuels sans produire de données persistantes.

Ce découpage permet de séparer deux types d'activité :

- L'**exploration à grande échelle** (phases majeures) qui teste le pool complet et
  produit des verdicts
- La **calibration ciblée** (sous-phases locales) qui affine un atomic problématique
  avant qu'il contamine une phase majeure

## 4.2 Phases majeures R0, R1, R2, ...

### Rôle

Les phases majeures représentent l'évolution du **pool d'atomics** et de l'**architecture
de mesure**. Chaque phase majeure teste un pool particulier avec une configuration
particulière (n_dof, max_it, modifiers) et produit un parquet persistant.

### Caractéristiques

- **Parquet écrit** : une phase majeure produit un fichier parquet durable, identifié
  par son numéro (r0, r1, r2, ...)
- **Concordance inter-phases possible** : on peut comparer les clusters de R0 à ceux de
  R1 sur un subset commun
- **Pool requirements évolutifs** : le pool peut changer d'une phase à l'autre (atomics
  ajoutés, retirés, paramètres modifiés)
- **Axes d'itération stables** dans une phase : gamma × encoding × modifier × n_dof ×
  max_it × seeds

### Triggers de passage à la phase suivante

Une phase majeure R_x déclenche R_{x+1} quand :

- **Des atomics sont deprecated validés** (≥ 3 gammas ou encodings retirés du pool après
  calibration et décision)
- **Des contraintes pool sont identifiées** (par exemple : n_dof > 100 nécessaire, ou
  max_it > 500 requis pour conclure)
- **L'architecture a évolué** (nouvelles features, nouvelles couches, nouveaux observables
  — comme dans le chantier 1)
- **Une question scientifique nouvelle** exige un réexécution avec une configuration
  différente

R_{x+1} ne se déclenche **pas** pour 1 ou 2 changements mineurs. Les modifications mineures
sont accumulées jusqu'à un seuil de valeur, puis déclenchent la phase suivante d'un coup.

### Exemples indicatifs

```
R0 — Baseline exploration
  Pool : pool complet initial
  n_dof : valeur modeste (par exemple 50)
  max_it : valeur modérée (par exemple 1000)

R1 — Après calibration R0
  Pool : réduit (deprecated exclus)
  n_dof : augmenté selon contraintes identifiées
  max_it : ajusté selon les besoins Q3
  Architecture : potentielles nouvelles features

R2 — Validation et universalité
  Pool : stable (identique ou quasi-identique à R1)
  Concordance R0 ↔ R1 ↔ R2 sur subset commun
  Test universalité DMD sur observables canoniques
```

## 4.3 Sous-phases locales R_x.y.z

### Rôle

Les sous-phases locales calibrent des atomics spécifiques en dehors du cycle des phases
majeures. Elles sont **jetables** : les runs servent à valider ou invalider un atomic,
puis sont écartés.

### Nomenclature

```
R_x.y.z
  │ │ └── Micro-ajustement (params atomics, bins, seuils)
  │ └──── Itération calibration (tentative 1, 2, 3, ...)
  └────── Phase majeure de référence
```

Exemples :
- `R0.1.0` — première tentative de calibration du premier atomic problématique détecté
  dans R0
- `R0.1.1` — deuxième tentative sur le même atomic, avec des paramètres différents
- `R0.2.0` — première tentative sur un autre atomic

### Caractéristiques

- **Pas de parquet persistant** : les sous-phases locales produisent des logs et des
  analyses ad hoc, pas de données réutilisables à long terme
- **Modifications YAML locales** : on modifie temporairement les paramètres d'un atomic
  pour tester son comportement sur une plage de valeurs
- **Objectif binaire** : valider l'atomic (ses paramètres par défaut sont corrects) ou
  l'invalider (le marquer deprecated dans le pool_requirements)

### Workflow type

```
Problème détecté dans R0 : un gamma explose systématiquement

R0.1.0 — Tester ce gamma avec des paramètres plus conservateurs
  Résultat : stable avec params conservateurs mais comportement trivial

R0.1.1 — Tester ce gamma avec différents max_it
  Résultat : explosion même avec max_it très court

Décision : gamma marqué deprecated
  Mise à jour du pool_requirements avec la raison et les calibrations testées
  R1 lancé sans ce gamma
```

### Règles de décision

**Créer une sous-phase R_x.y.z** si :
- Calibration d'un atomic individuel (1 à 3 atomics)
- Modification de paramètres locale temporaire
- Tests de seuils ou de configurations de features
- Runs jetables (moins de 1000 observations)

**Ne pas créer une sous-phase** si :
- Changement de pool (deprecated définitifs) → attendre R_{x+1}
- Modification d'architecture (nouveaux layers, nouvelles couches) → chantier dédié
- Runs de production (plus de 1000 observations) → R_{x+1}
- Concordance inter-phases nécessaire → R_{x+1}

## 4.4 Concordance inter-phases

### Principe

Quand deux phases majeures R_x et R_y ont été produites avec des configurations
différentes (pool, n_dof, max_it), on ne peut comparer leurs résultats que sur le
**subset commun** : l'ensemble des doublets qui satisfont les contraintes des deux
phases.

### Filtrage automatique

Le verdict inter-phases filtre automatiquement chaque parquet par les contraintes du pool
cible. Si les contraintes ont évolué (par exemple, n_dof minimum passé de 50 à 100), les
runs de R0 avec n_dof = 50 sont exclus du subset commun pour la comparaison avec R1.

Quand le subset commun devient trop petit (par exemple moins de 10 observations), la
concordance est considérée **peu fiable** et un warning explicite est émis. La concordance
n'est pas calculée silencieusement sur un échantillon statistiquement insuffisant.

### Usage

La concordance inter-phases sert à :

- Vérifier la **stabilité** des clusters identifiés en R_x quand on passe à R_{x+1}
- Détecter la **dérive** des profils d'atomics entre phases (un gamma qui a changé de
  cluster signale soit une instabilité intrinsèque, soit un changement de contexte
  significatif)
- Valider que les **verdicts par atomic** sont cohérents d'une phase à l'autre
- Identifier les atomics dont le comportement est **dépendant** de configurations
  particulières (ce qui est une information scientifique en soi)

## 4.5 Réduction du pool par contraintes

### Principe central

L'objectif de l'exploration PRC est de construire un **tableau de contraintes** sur les
atomics. Ce tableau est la sortie scientifique principale du travail. Il se présente sous
des formulations du type :

> *"Si un Γ universel existe, d'après l'analyse il ne peut pas être : ... (liste de
> familles d'atomics exclues ou de zones de paramètres pathologiques)"*

> *"L'analyse montre que cette famille d'atomics ne peut satisfaire aux critères de cette
> contrainte : ... (liste de critères définitifs)"*

Chaque contrainte est dérivée de l'analyse d'une ou plusieurs phases majeures, et est
ancrée dans une observation reproductible. Les contraintes sont **cumulatives** : une
contrainte dérivée de R0 peut être affinée ou complétée par R1, mais rarement retirée.

### Rôle des statuts pathologiques

Les runs OK_TRUNCATED, EXPLOSION et COLLAPSED sont **essentiels** pour construire les
contraintes. Ils marquent les **frontières de viabilité** : un atomic qui produit des
OK_TRUNCATED dans certaines plages de paramètres et des OK dans d'autres révèle une
transition de phase empirique entre zone viable et zone non-viable.

Les statuts pathologiques ne sont pas du déchet à éviter. Ils sont la matière première
de la cartographie des contraintes.

### Usage du tableau de contraintes

Le tableau de contraintes pilote :

- La **réduction du pool** pour les phases majeures suivantes (exclusion des atomics
  systématiquement pathologiques)
- Le **raffinement des YAMLs d'analyse** (focus sur les zones intéressantes identifiées)
- Les **décisions de composition** (quels atomics peuvent être composés et selon quelles
  règles)
- L'**interprétation scientifique finale** (quelles classes de Γ sont candidates
  universelles, quelles sont exclues)

---

# PARTIE V — RÉFÉRENTIEL DOCUMENTAIRE

## 5.1 Hiérarchie des documents

Le projet PRC maintient plusieurs documents aux rôles distincts et non redondants. Cette
section établit la hiérarchie et définit qui est **autoritatif** sur quel sujet.

### Documents permanents (en contexte de toute conversation)

**CHARTER PRC v3 (ce document)**
Rôle : ancrage philosophique, scientifique et architectural. Source de vérité pour les
principes, les axiomes, la méthodologie, le cycle d'exploration. Code-agnostique. Stable :
mis à jour uniquement par discussion explicite.

### Documents de chantier (actualisés par chantier actif)

**Roadmap du chantier en cours**
Rôle : plan d'action opérationnel pour un chantier. Décisions actées, sous-chantiers
ordonnés, expériences standalone, critères de succès, hooks vers les chantiers suivants.
Vivante : mise à jour après chaque sous-chantier terminé.

Un seul document de roadmap est actif à la fois. Quand un chantier est terminé, sa
roadmap est archivée et la roadmap du chantier suivant devient active.

**Document scientifique des features du chantier en cours**
Rôle : référence scientifique détaillée des features produites par le chantier. Explicite
la sémantique, l'ancrage axiomatique, la grammaire, l'interprétation de chaque feature.
Stable pendant un chantier, mis à jour entre chantiers si de nouvelles features sont
introduites.

### Documents de référence (actualisables à la demande)

**Architecture (audit statique)**
Rôle : carte objective du code réel, générée par analyse syntaxique. Extrait les
dépendances, les imports, les constantes, les violations de principes. Régénéré après
chaque modification de structure.

**Inventaires d'atomics**
Rôle : liste exhaustive des gammas, encodings et modifiers disponibles avec leurs
signatures, métadonnées, paramètres. Régénéré quand des atomics sont ajoutés ou retirés.

**Pool requirements**
Rôle : contraintes sur le pool d'atomics utilisables en phase majeure. Contient les
atomics deprecated avec leurs raisons, les contraintes de n_dof minimum, les exclusions
de combinaisons. Mis à jour après chaque calibration validée.

### Documents archivés

Les documents antérieurs remplacés par le charter v3 — CHARTER PRC v2, ancrage.md,
PHASES_GUIDE.md — sont archivés sans être réutilisés. Leur contenu est intégré dans le
charter v3. Ils ne doivent plus être consultés comme sources actives.

## 5.2 Quand consulter quoi

| Situation | Document à consulter |
|-----------|----------------------|
| Démarrer une session de travail | Charter v3 + roadmap du chantier actif |
| Comprendre un principe ou une décision philosophique | Charter v3 |
| Savoir ce qu'une feature mesure et pourquoi | Document scientifique des features |
| Savoir où on en est dans le chantier | Roadmap du chantier actif |
| Ajouter une feature | Registre features + document scientifique des features |
| Modifier l'architecture | Charter v3 §2 (principes) + roadmap |
| Débugger le pipeline | Audit statique + sources |
| Analyser les résultats d'une phase | Document scientifique des features + analyses ad hoc |
| Réduire le pool d'atomics | Pool requirements + contraintes dérivées des phases |
| Passer à une phase majeure suivante | Pool requirements + charter v3 §4 |
| Décider d'une sous-phase locale | Charter v3 §4.3 |
| Planifier un changement architectural | Charter v3 §3.1 (protocole d'audit) |

## 5.3 Qui est autoritatif sur quoi

| Sujet | Document autoritatif |
|-------|---------------------|
| Axiomes A1, A2, A3 | Charter v3 §1.2 |
| Questions scientifiques Q1-Q4 | Charter v3 §1.3 |
| Grammaires signal / virages | Charter v3 §1.5 |
| Rôle du masque | Charter v3 §1.7 |
| Principes architecturaux P1-P12 | Charter v3 §2.2 |
| Philosophie vases communicants | Charter v3 §2.3 |
| Protocole d'audit | Charter v3 §3.1 |
| Méthodologie Algo/Structure/Code | Charter v3 §3.2 |
| Cycle phases majeures / sous-phases | Charter v3 §4 |
| Définition et sémantique d'une feature | Document scientifique des features du chantier |
| Formules numériques précises des features | Commentaires dans le registre features (code) |
| Ordre d'implémentation et sous-chantiers | Roadmap du chantier actif |
| État réel du code | Audit statique + sources |
| Atomics disponibles | Inventaire d'atomics |
| Contraintes du pool | Pool requirements |

### Règle d'arbitrage en cas de contradiction

Si deux documents se contredisent :

1. **Charter v3 prévaut** sur tous les autres pour les questions philosophiques,
   scientifiques et méthodologiques.
2. **Les sources prévalent** sur tous les documents pour la vérité technique instantanée.
3. **Le document scientifique des features prévaut** sur la roadmap pour les questions
   de sémantique des features.
4. **La roadmap prévaut** sur le charter pour les questions d'ordre opérationnel d'un
   chantier, mais **jamais** pour les questions de principe.

Si aucune de ces règles ne tranche, la contradiction est remontée en discussion
explicite avec l'humain.

---

# ANNEXES

## Annexe A — Glossaire

| Terme | Définition |
|-------|------------|
| **Atomic** | Gamma, encoding ou modifier — fonction pure avec signature unifiée |
| **Attracteur holographique** | État stable où S_VN tend vers un minimum non-nul |
| **Audit statique** | Analyse syntaxique du code source, indépendante des docstrings |
| **Candidat A3** | Γ pour lequel D résiste à l'annihilation — D ≠ 0 à l'attracteur |
| **Cluster** | Groupe de runs ayant des profils de features similaires |
| **Concordance** | Stabilité d'un verdict cross-phases |
| **Constantes machine** | Paramètres de performance (workers, batch, VRAM) — leur changement n'affecte pas les résultats scientifiques |
| **Constantes scientifiques** | Paramètres de mesure (epsilon, seuils, quantiles) — leur changement modifie les valeurs produites |
| **Composition** | Dict des axes d'un run : gamma_id, encoding_id, modifier_id, params, seeds |
| **Dissymétrie D** | Tenseur initial représentant l'information structurelle à mesurer (A1) |
| **DMD universalité** | Test : même spectre DMD cross-encodings pour Γ fixé sur un pool d'observables canoniques |
| **Encoding** | Fonction qui produit le tenseur initial D pour un run |
| **Feature** | Valeur numérique brute issue du pipeline — une observation, jamais un verdict |
| **Γ (Gamma)** | Mécanisme de transformation appliqué à D à chaque itération |
| **Grammaire signal** | Méthodes applicables aux séquences temporelles uniformes |
| **Grammaire virages** | Méthodes ordinales applicables aux séquences de points non uniformes |
| **Inter-run** | Calculs cross-runs, depuis le stockage persistant |
| **Intra-run** | Calculs sur un run unique, en mémoire vive |
| **Kernel** | Boucle de simulation temporelle — aveugle au domaine |
| **Kernel group** | Ensemble de runs partageant une même shape, compilés ensemble |
| **K_i** | Nombre de points du masque pour un run — variable par run, hétérogène intra-batch |
| **Masque** | Séquence des instants d'intérêt d'un run — schématisation des virages |
| **Mesure dynamique** | Mesure O(n²) calculée à chaque itération, relève de la grammaire signal |
| **Mesure spectrale** | Mesure O(n³) calculée aux virages, relève de la grammaire virages |
| **Modifier** | Transformation pré-Γ appliquée à D avant le scan |
| **Observable canonique** | Fonction du state invariante par reparamétrisation triviale |
| **Observation** | Valeur numérique brute issue du pipeline — jamais un jugement PASS/FAIL |
| **OK_TRUNCATED** | Run qui a passé une phase saine avant transition vers pathologie — signature de frontière de viabilité |
| **Phase majeure (R_x)** | Exploration à grande échelle avec parquet persistant |
| **Sous-phase locale (R_x.y.z)** | Calibration ciblée sans parquet persistant |
| **Profil entité** | Agrégation des features par gamma_id ou encoding_id dans un cluster |
| **Régime** | Classification comportementale d'un run (cluster nommé) |
| **Schématisation des virages** | Rôle du masque — noter les instants qualitativement significatifs, pas sous-échantillonner |
| **Signal dans-scan** | Valeur calculée à chaque itération et accumulée |
| **Signal post-scan** | Scalaire calculé une fois sur les signaux accumulés |
| **Timeline complète** | Signal à pas uniforme sur toute la durée du run — grammaire signal |
| **Timeline tronquée** | Séquence aux virages du run — grammaire virages |
| **Tableau de contraintes** | Sortie scientifique principale — liste des exclusions dérivées de l'analyse |
| **t_effective** | Durée effective d'un run — max_it pour les OK, instant de pathologie sinon |
| **Unité d'exécution** | Processus isolé qui compile et exécute un groupe de runs |
| **Vases communicants** | Philosophie d'exécution — modules parallèles alimentant le goulot en continu |
| **Verdict** | Décision scientifique issue de l'analyse d'un cluster ou d'un atomic |
| **Virage** | Instant où un observable présente une transition significative détectée par classify |

## Annexe B — Format des docstrings

Le pipeline dispose de deux sources complémentaires de vérité structurelle :

- **L'audit statique** — extrait la réalité du code par analyse syntaxique. Il voit ce
  qui existe, pas ce qui est déclaré. C'est le socle objectif.
- **Les docstrings structurées** — déclarent l'intention, le rôle, le lifecycle, la
  conformité. Elles sont lues par l'outil de génération d'architecture et par les humains.
  C'est le socle déclaratif.

Les deux se complètent. L'audit détecte les écarts entre intention et réalité. Les
docstrings fournissent le contexte sémantique que l'AST ne peut pas déduire (pourquoi un
objet est créé, pas seulement qu'il l'est).

### Structure complète des docstrings

```
"""
Résumé une ligne — ce que fait ce module.

Description optionnelle en prose (2-5 lignes max) si le résumé ne suffit pas.

@ROLE    Description concise du rôle dans le pipeline (une ligne)
@LAYER   running | analysing | utils | atomics | root

@EXPORTS
  nom_fonction(args) → Type   | description une ligne
  NOM_CONSTANTE      → Type   | description une ligne

@LIFECYCLE
  CREATES  nom_objet   description, type, destination
  RECEIVES nom_objet   depuis quel module, dans quelle fonction
  PASSES   nom_objet   vers quel module
  DELETES  nom_objet   où et quand

@CONFORMITY
  OK        description de la conformité à un principe nommé
  WARN      avertissement non bloquant
  VIOLATION description de la violation charter

@BUGS
  B_NOM     description concise du bug

@TODOS
  T_NOM [blocks:B_NOM]   description de ce qui est à faire

@QUESTIONS
  Q1   question ouverte sur ce module

@RESOLVED
  B_NOM   description du fix appliqué
"""
```

### Règles de format

**@ROLE** : une seule ligne, pas de point final.

**@LAYER** : exactement un des termes définis. Si un module appartient à un sous-layer
(par exemple `analysing/pipeline`), noter le layer parent.

**@EXPORTS** : uniquement les symboles publics utiles aux autres modules. Ne pas lister
les fonctions privées préfixées par underscore. Format : `nom(args) → Type | description`
ou `NOM_CONSTANTE → Type | description`.

**@LIFECYCLE** : tracer tous les objets coûteux (arrays numpy, tenseurs JAX, dicts larges,
fichiers ouverts, connexions). Ne pas tracer les primitives (int, str, bool). Un objet
CREATES sans DELETES correspondant indique une fuite potentielle à documenter.

**@CONFORMITY** : déclarer explicitement les conformités et violations au charter.
`OK` = conforme à un principe nommé. `WARN` = non-bloquant mais à surveiller. `VIOLATION`
= dérogation au charter, **toujours accompagnée d'une justification**.

**@BUGS** : bugs connus dans ce module, identifiés par un ID court unique. Ne pas lister
les bugs d'autres modules ici.

**@TODOS** : actions à réaliser dans ce module. `[blocks:B_NOM]` si le TODO résout un bug.

**@QUESTIONS** : questions ouvertes sur la conception de ce module, numérotées Q1, Q2, ...

**@RESOLVED** : bugs ou todos résolus dans la version courante, pour traçabilité.

### Ce que génère l'outil à partir des docstrings

L'outil de génération d'architecture analyse les docstrings et produit :

- Arborescence des modules par layer
- Flux topologique (PASSES/RECEIVES entre modules)
- Tableau lifecycle global (CREATES/DELETES/fuites potentielles)
- Tableau conformité charter (OK/WARN/VIOLATION agrégés)
- Inventaire bugs, todos, questions par module
- Couverture roadmap (liens `[blocks:X]`)

La qualité de la carte cognitive dépend directement de la complétude des docstrings. Un
module sans @LIFECYCLE apparaît comme une boîte noire dans la carte.

## Annexe C — Tableau récapitulatif des interdictions critiques

### Kernel

| Interdit | Pourquoi |
|----------|----------|
| Validation sémantique des tenseurs | P3 K1 — kernel aveugle au domaine |
| Classes State ou Operator | P3 K2 — pas d'abstraction métier |
| Branchements sur contenu de D ou Γ | P3 K3 — compilation déterministe |
| Connaissance des atomics | P3 K4 — callables opaques uniquement |
| État mutable | P3 K5 — fonction pure |

### Features

| Interdit | Pourquoi |
|----------|----------|
| Valeurs PASS/FAIL, bon/mauvais | §1.8 — observations, pas verdicts |
| Paramètres hardcodés | P4 — tout vient des configs |
| I/O dans les registres | P5 — fonctions pures |
| État global mutable | P5 — fonctions pures |
| Méthode signal sur virages | §1.5, P10 — grammaire violée |
| Extrapolation trajectoire à partir de virages | §1.7 — le masque n'est pas un sous-échantillonnage |

### Architecture

| Interdit | Pourquoi |
|----------|----------|
| Dépendances circulaires | P8 — layers stricts |
| Import cross-layer hors hub | P8 — seuls les hubs orchestrent |
| Code sans Algo → Structure → Code | §3.2 — validation obligatoire |
| except silencieux | P2 — toujours logger |
| Pré-allocation fixe VRAM | P11, P12 — allocation dynamique |
| Calcul dans orchestrateur | P9, P11 — séparation des rôles |
| Accumulation inter-groupes sans purge | P9 — isolation mémoire |
| Déviation sans discussion | §3.4 — règles de travail |

### Nomenclature

| Éviter | Préférer |
|--------|----------|
| "matrice de corrélation" | "tenseur rang 2" |
| "graphe" | "patterns dans tenseur" |
| "test verdict" | "observation" (feature) / "verdict" (analysing) |
| "résultat bon/mauvais" | "valeur numérique interprétable" |
| "sous-échantillonnage temporel" | "schématisation des virages" (masque) |
| "autocorr sur virages" | interdit — méthode signal en grammaire virages |

---

**FIN DU CHARTER PRC v3**

*Ce document est code-agnostique. Il ne vieillit pas avec les refactors.*
*Si un principe doit changer → discussion explicite avant modification.*
*Charter = boussole. Sources = réalité. Chantiers = chemins.*
