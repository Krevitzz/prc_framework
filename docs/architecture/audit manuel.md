## Audit du fichier `batch.py`

### Résumé général
`batch.py` est le point d'entrée CLI d’un programme (PRC v7 JAX). Il analyse les arguments, puis oriente l’exécution vers trois modes possibles :
1. **Exécution d’une phase** (`run_phase` depuis `running.hub_running_v8`)
2. **Verdict sur une phase unique** à partir d’un fichier Parquet (`run_verdict_from_parquet`)
3. **Verdict croisé entre plusieurs phases** (`run_verdict_cross_phases`)

Le code est propre, bien structuré, et tous les éléments importés ou définis sont utilisés. Une seule incohérence mineure entre la documentation et le code est relevée.

---

### Analyse détaillée

#### 1. Imports
| Module importé | Éléments utilisés | Obsolescence |
|----------------|-------------------|--------------|
| `warnings` | `filterwarnings` (utilisé) | ✅ utilisé |
| `os` | `environ` (utilisé) | ✅ utilisé |
| `argparse` | `ArgumentParser`, `RawDescriptionHelpFormatter` (utilisés) | ✅ utilisé |
| `pathlib.Path` | `Path`, `mkdir`, `exists`, `resolve`, `/` (utilisés) | ✅ utilisé |
| `running.hub_running_v8` | `run_phase` | ✅ utilisé (mode 1) |
| `analysing.hub_analysing_v8` | `run_verdict_from_parquet`, `run_verdict_cross_phases` | ✅ utilisés (modes 2 et 3) |

**Aucun import inutile.**

#### 2. Variables d’environnement et filtres
- `warnings.filterwarnings('ignore', message='.*SLASCLS.*')` : modification du comportement des warnings (pas d’objet persistant).
- `os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'` et `os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')` : configuration globale (pas d’objet Python à suivre).

#### 3. Fonction `main()`
**Création d’objets :**
- `parser` (ArgumentParser) : détruit à la sortie de `main()`.
- `args` (Namespace) : idem.
- Multiples objets `Path` : `output_dir`, `_default_cfg`, `cfg_path`, `parquet_path`, `yaml_path` – tous temporaires.
- Aucune allocation mémoire explicite (pas de `__new__` sans `__init__`, pas de gestion manuelle).

**Flot des données :**
```
Arguments CLI → parser → args
│
├─ si args.verdict:
│    ├─ si args.phase présent : construction de parquet_path → appel run_verdict_from_parquet(parquet_path, cfg_path, output_dir/reports/phase, plot, debug)
│    └─ sinon : appel run_verdict_cross_phases(results_dir, cfg_path, output_dir/reports, plot)
│    → return
│
└─ sinon (pas verdict) :
     validation args.phase non nul
     construction yaml_path → appel run_phase(yaml_path, output_dir, auto_confirm, verbose)
```

**Points clés :**
- Les chemins sont construits avec `pathlib`, garantissant une manipulation robuste.
- Vérification d’existence des fichiers (`parquet_path.exists()`, `yaml_path.exists()`) avant appel, avec message d’erreur explicite.
- `output_dir.mkdir(parents=True, exist_ok=True)` crée le dossier parent si nécessaire (effet de bord, pas d’objet retourné).

#### 4. Utilisation des arguments CLI
| Argument | Utilisation |
|----------|-------------|
| `phase` | Nom de phase pour construire `parquet_path` (mode 2) ou `yaml_path` (mode 1). |
| `--verdict` | Bascule en mode verdict. |
| `--verbose` | Transmis à `run_phase` (mode 1) ; ignoré en mode verdict. *(comportement intentionnel)* |
| `--auto-confirm` | Transmis à `run_phase` (mode 1). |
| `--cfg` | Surcharge le chemin de config pour l’analyse (modes 2 et 3). |
| `--plot` | Transmis aux fonctions de verdict (modes 2 et 3). |
| `--debug` | Transmis aux fonctions de verdict (mode 2 principalement, mais aussi mode 3 si la fonction l’accepte). |

Tous les arguments sont exploités ; aucun n’est inutilisé.

#### 5. Gestion mémoire et durée de vie
- Les objets créés dans `main()` sont locaux à la fonction. Le garbage collector les libère après l’appel.
- Les appels aux fonctions externes (`run_phase`, etc.) peuvent créer leurs propres objets, mais cela dépasse le cadre de l’audit de ce fichier.
- Pas de fuite mémoire évidente (pas de variables globales modifiées, pas de ressources non fermées).

#### 6. Cohérence documentation / code
- **Docstring** : mentionne `analysing/configs/analysing_default.yaml` comme défaut pour `--cfg`.
- **Code** : utilise `analysing/configs/analysing_v8.yaml` comme défaut (`_default_cfg`).
  → **Incohérence mineure** : la docstring devrait refléter le chemin réel, ou le code devrait être aligné.

---

### Conclusion pour `batch.py`
- ✅ Tous les imports sont nécessaires et utilisés.
- ✅ Toutes les variables et fonctions définies sont utilisées.
- ✅ Le flux de données est clair et linéaire.
- ⚠️ Une incohérence docstring/code sur le chemin par défaut du fichier de configuration est à corriger.

**Aucun objet ou fonction inutilisé n’a été détecté dans ce fichier.**

Audit du fichier hub_running_v8.py
Résumé général

Ce module orchestre l'exécution d'une phase : à partir d'un fichier de configuration YAML, il génère des groupes de kernels (via generate_kernel_groups), utilise une fenêtre glissante de compilation (rolling compile window) pour pré-compiler les groupes en parallèle, exécute les batches sur GPU, construit des lignes de résultats (agrégées ou timeline) et écrit dans un fichier Parquet.

Le code est complexe mais bien structuré. Plusieurs fonctions internes, gestion de threads, compteurs, gestion d'erreurs.
Analyse détaillée
1. Imports

    os, sys, time, warnings, collections.deque, concurrent.futures.ThreadPoolExecutor, pathlib.Path, typing, numpy : tous standards et utilisés.

    utils.io_v8 : plusieurs fonctions (discover_gammas_jax, discover_encodings_jax, discover_modifiers_jax, load_yaml, open_parquet_writer, write_rows_to_parquet, close_parquet_writer) – toutes utilisées.

    running.plan_v8 : generate_kernel_groups, split_into_batches, dry_run_stats – utilisées.

    running.kernel_v8 : execute_batch, pre_compile_for_group, FEATURE_NAMES, FEATURES_STRUCTURAL_NAN, DANS_SCAN_KEYS – utilisées.

Aucun import inutile.
2. Fonctions internes (privées)
_run_status_from_features(features_row)

    Utilisée uniquement dans _build_rows_aggregated.

    Détermine le statut d'un run à partir des features agrégées.

    Retourne 'EXPLOSION', 'COLLAPSED', 'NAN_ALL', 'OK'.

    Logique basée sur health_has_inf, health_is_collapsed, et vérification que toutes les features non-structurelles sont NaN.

_run_status_from_timeline(signals_run, max_it)

    Utilisée dans _build_rows_timeline.

    Heuristique : si toutes les features sont gelées sur les 10 derniers pas (ou un tail adaptatif), considère comme EXPLOSION.

    Retourne 'EXPLOSION' ou 'OK'.

_build_rows_aggregated(batch, features_sync, phase)

    Construit une ligne par échantillon (run) pour le mode agrégé.

    Appelle _run_status_from_features.

    Retourne une liste de dictionnaires (rows).

_build_rows_timeline(batch, features_sync, phase)

    Déplie en max_it lignes par run.

    Appelle _run_status_from_timeline pour le statut global du run.

    Inclut l'itération et toutes les features.

_build_rows(batch, features_sync, phase, recording_mode)

    Dispatch simple : appelle la fonction appropriée selon recording_mode.

_count_statuses(rows, counters)

    Incrémente les compteurs (n_ok, n_explosion, n_collapsed, n_nan_all) à partir des rows.

    Gère le cas timeline pour ne compter chaque run qu'une seule fois (en utilisant un set basé sur les clés et en ne comptant que sur iteration=0).

    Remarque : La logique pour éviter de compter plusieurs fois le même run en timeline est correcte, mais elle utilise iteration=0 comme condition. Cela suppose que les rows sont générées dans l'ordre croissant des itérations (ce qui est le cas). Cependant, si jamais les rows n'étaient pas ordonnées, cela pourrait compter plusieurs fois. Mais c'est peu probable.

make_nan_rows(batch, run_status, phase, recording_mode)

    Génère des lignes de remplacement en cas d'échec d'exécution (FAIL).

    Gère les deux modes : pour timeline, produit max_it lignes par run avec NaN pour toutes les features ; pour agrégé, une ligne par run avec NaN.

    Utilise FEATURE_NAMES pour agrégé, DANS_SCAN_KEYS pour timeline.

_auto_batch_size(n_samples, timeline_mode=False)

    Détermine une taille de batch adaptative selon le nombre d'échantillons et le mode.

    Utilisée dans la boucle principale pour chaque groupe.

_rolling_compile_window(kg_gen, dmd_rank, batch_size, n_workers, timeline_mode=False)

    C'est un générateur qui prend un flux de groupes de kernels (kg_gen), soumet chaque groupe à une pré-compilation en parallèle via ThreadPoolExecutor, et les renvoie dans l'ordre (via une fenêtre glissante) une fois compilés.

    Utilise deque pour stocker les futures.

    Yield (kg_ready, kernel_fn).

    Important : La fonction pre_compile_for_group est appelée avec kg, dmd_rank, bs, timeline_mode. Le résultat est une fonction kernel compilée.

    Gestion de la mémoire : les futures sont libérés après popleft().

run_phase(...)

C'est la fonction principale exportée. Elle est appelée par batch.py (mode 1). Analysons en détail.

Déroulement :

    Charge la config YAML.

    Extrait phase, dmd_rank, recording_mode, déduit timeline_mode.

    Découvre les registres (gammas, encodings, modifiers) via discover_*_jax().

    Dry run (si auto_confirm est False) : affiche les statistiques via dry_run_stats et demande confirmation. Si refus, retourne un dict avec n_ok=0.

    Détermine les noms des features pour le schéma Parquet (parquet_feature_names).

    Initialise les compteurs, buffer de lignes, seuil de flush, writer, chemin parquet, chronomètre, nombre de workers.

    Définit une fonction interne _flush() qui écrit le buffer dans le Parquet (ouvre le writer si nécessaire).

    Boucle principale : obtient un flux de groupes via generate_kernel_groups(run_config, registries). Pour chaque groupe, il récupère un kernel compilé via _rolling_compile_window. Pour chaque groupe, il détermine batch_size avec _auto_batch_size, puis découpe en batches via split_into_batches. Pour chaque batch :

        Appelle execute_batch avec le kernel compilé.

        Construit les rows via _build_rows.

        Met à jour les compteurs via _count_statuses.

        Étend le buffer.

        En cas d'exception, génère des rows NaN via make_nan_rows et incrémente n_fail (nombre de runs en échec).

        Flush si nécessaire.

        Supprime kernel_fn (pour libérer ?).

        Affiche la progression sur stderr.

        Supprime kg (pour libérer).

    Finally : flush final, fermeture du writer.

    Récupère generate_kernel_groups._last_n_invalid (variable de module ?) pour le nombre d'invalid filtrés.

    Calcule les stats finales et affiche.

    Retourne un dict de résultats.

Points clés :

    Utilisation de _rolling_compile_window pour paralléliser la compilation.

    Gestion des exceptions : en cas d'échec d'un batch, on écrit des lignes NaN avec statut 'FAIL' et on continue.

    Le buffer de lignes est vidé périodiquement pour éviter une consommation mémoire excessive.

    Le writer est ouvert à la première écriture et fermé à la fin.

    La variable generate_kernel_groups._last_n_invalid est un attribut statique de la fonction (ou un module-level) – cela suppose que generate_kernel_groups expose un compteur. C'est un peu fragile mais utilisé.

    La fonction _count_parquet_rows est appelée à la fin pour afficher le nombre de lignes. Elle utilise pyarrow.parquet – importée à l'intérieur pour éviter une dépendance obligatoire ? (gère l'exception).

Problèmes potentiels :

    generate_kernel_groups._last_n_invalid : si la fonction est modifiée ou si plusieurs appels, ce compteur pourrait ne pas être fiable. Mais ici c'est un usage unique dans run_phase.

    _count_parquet_rows importe pyarrow à l'intérieur : c'est bien pour éviter un import global si non disponible, mais en cas d'échec, il affiche '?'. Acceptable.

    Dans _rolling_compile_window, la variable window_size est calculée comme max(n_workers * 2, 4). C'est correct.

    Gestion des ressources : ThreadPoolExecutor est utilisé dans un contexte with, donc fermé automatiquement. Les futures sont récupérés avec result().

    Dans run_phase, après la boucle, il y a del kernel_fn et del kg pour aider le GC, mais ce n'est pas nécessaire en Python (sauf pour libérer des références cycliques). Cela ne fait pas de mal.

    La fonction _flush utilise nonlocal writer – correct.

    write_rows_to_parquet et close_parquet_writer sont appelées ; on suppose qu'elles gèrent correctement les ressources.

    execute_batch est appelé avec kernel_fn – attention, kernel_fn peut être None ? Normalement pre_compile_for_group retourne une fonction. Si la compilation échoue, pre_compile_for_group pourrait lever une exception, mais elle est exécutée dans un thread, et future.result() relèvera l'exception. Dans ce cas, le groupe entier ne sera pas yield. C'est correct car on ne peut pas exécuter sans kernel.

3. Variables globales et constantes

    FEATURE_NAMES, FEATURES_STRUCTURAL_NAN, DANS_SCAN_KEYS importées de kernel_v8 – utilisées.

    Aucune autre variable globale définie dans ce module.

4. Flux de données
text

run_phase(yaml_path, ...)
│
├─ load_yaml → run_config
├─ discover_*_jax() → registries
│
├─ dry_run_stats(run_config, registries) → stats
│
├─ generate_kernel_groups(run_config, registries) → kg_stream (générateur)
│
├─ _rolling_compile_window(kg_stream, dmd_rank, batch_size_default, n_workers, timeline_mode)
│   └─ pour chaque kg, yield (kg, kernel_fn)
│
└─ pour chaque (kg, kernel_fn) :
    ├─ _auto_batch_size(len(kg['samples']), timeline_mode) → batch_size
    ├─ split_into_batches(kg, batch_size) → batches
    └─ pour chaque batch :
        ├─ execute_batch(batch, dmd_rank, timeline_mode, kernel_fn) → features_sync (dict de arrays)
        ├─ _build_rows(batch, features_sync, phase, recording_mode) → rows
        ├─ _count_statuses(rows, counters)
        ├─ rows_buffer.extend(rows)
        │
        └─ (en cas d'exception) make_nan_rows(batch, 'FAIL', ...) → rows de NaN, incrément n_fail

Les données produites sont écrites dans un fichier Parquet via le buffer et le writer.
5. Utilisation des fonctions et objets

    Toutes les fonctions définies sont utilisées (certaines seulement dans ce module, d'autres exportées).

    Les fonctions importées sont toutes utilisées.

    Les constantes importées sont utilisées.

    Aucune fonction ou variable inutilisée.

6. Gestion mémoire

    Les batches, rows, et buffers sont créés et libérés.

    Le buffer de lignes (rows_buffer) est vidé périodiquement pour éviter une accumulation.

    Les objets kg et kernel_fn sont explicitement supprimés après usage (del), mais Python les aurait de toute façon collectés quand ils sortent de la portée. Cela peut aider à casser des cycles éventuels.

    Le ThreadPoolExecutor est géré par contexte.

    Les fichiers Parquet sont fermés.

    Pas de fuite mémoire évidente.

7. Dépendances et appel

    Ce module est importé par batch.py (uniquement run_phase).

    Il importe de running.plan_v8 et running.kernel_v8, ainsi que utils.io_v8.

    Il utilise generate_kernel_groups._last_n_invalid – c'est un couplage implicite. Il faudrait vérifier que plan_v8 expose bien cette variable.

8. Cohérence documentation / code

    La docstring en tête décrit bien le rôle.

    Les fonctions internes ont des docstrings ou commentaires.

    Rien d'incohérent.

Conclusion pour hub_running_v8.py

    ✅ Tous les imports sont nécessaires et utilisés.

    ✅ Toutes les fonctions définies sont utilisées (soit exportées, soit internes appelées).

    ✅ Le flux de données est clair.

    ⚠️ Dépendance à generate_kernel_groups._last_n_invalid – à vérifier dans l'audit de plan_v8.py si cette variable est bien définie.

    ✅ Gestion mémoire correcte.

    ✅ Aucun objet inutilisé.

Analyse de plan_v8.py

Résumé général
Ce fichier génère des "kernel groups" à partir d'un fichier de configuration YAML et de registres (gamma, encoding, modifier). Il gère la compatibilité (contrainte de rang), normalise les configurations v7/v8, résout les axes (y compris les séquences composées et les poids), et fournit des fonctions pour diviser en lots et faire un dry run. Il utilise des fonctions de utils.io_v8 (importé) et des bibliothèques standard.

Imports

    random as _random : utilisé pour générer des clés aléatoires (fallback) et pour random dans la résolution d'axe.

    time : utilisé pour générer une clé basée sur le temps.

    itertools.product : utilisé intensivement pour les produits cartésiens.

    typing : pour les annotations.

    jax, jax.numpy as jnp : utilisés pour les clés PRNG, les opérations sur les tenseurs.

    utils.io_v8.load_yaml : importé mais pas utilisé dans ce fichier (peut-être pour un usage futur ou oubli). À vérifier : il est importé mais n'est pas appelé dans le code. C'est un élément potentiellement inutilisé.

Variables globales / fonctions

    is_compatible : utilisée dans generate_kernel_groups.

    _wrap_gamma_v7_to_v8 : utilisée dans _resolve_gamma_axis et _resolve_gamma_sequence.

    _normalize_to_list : utilisée dans generate_kernel_groups et _normalize_max_it (dans hub_running_v8, mais ici aussi utilisée dans dry_run_stats? non, dry_run_stats appelle generate_kernel_groups, mais pas directement _normalize_to_list dans ce fichier sauf dans generate_kernel_groups).

    _seed_to_key : utilisée dans generate_kernel_groups.

    _make_run_key : utilisée dans generate_kernel_groups.

    _get_rank_eff : utilisée dans generate_kernel_groups.

    _normalize_yaml_config : utilisée dans generate_kernel_groups (et aussi dans hub_running_v8 via _normalize_max_it qui l'importe).

    resolve_axis_atomic : utilisée dans _resolve_gamma_axis, _resolve_gamma_sequence et generate_kernel_groups.

    _format_gamma_id : utilisée dans _resolve_gamma_sequence.

    make_composed_gamma : utilisée dans _resolve_gamma_sequence.

    _generate_weights_grid : utilisée dans _resolve_gamma_sequence.

    _resolve_gamma_axis : utilisée dans generate_kernel_groups.

    _resolve_gamma_sequence : appelée par _resolve_gamma_axis.

    generate_kernel_groups : fonction principale, utilisée dans dry_run_stats et dans hub_running_v8.

    split_into_batches : utilisée dans hub_running_v8.

    dry_run_stats : utilisée dans hub_running_v8.

Flux de données et création d'objets

    generate_kernel_groups : c'est un générateur. Il crée des dictionnaires kernel_group contenant des listes samples. Chaque sample contient des clés PRNG JAX (objets JAX), des paramètres, etc. Les objets JAX sont créés via jax.random.PRNGKey, jax.random.fold_in. Ces objets sont légers mais doivent être suivis. La fonction yield des groupes et supprime la liste samples après chaque yield (del samples) pour libérer la mémoire. L'attribut _last_n_invalid est mis à jour à chaque appel.

    dry_run_stats : itère sur le générateur, compte les groupes et échantillons, et supprime chaque groupe (del kg). Elle retourne un dictionnaire de statistiques.

    split_into_batches : crée une liste de sous-groupes (copies superficielles avec la même référence aux samples, mais découpés). Utilisée dans hub_running.

Gestion mémoire

    Les générateurs permettent de ne garder qu'un groupe à la fois.

    Les del explicites aident le garbage collector.

    Les objets JAX (PRNGKey, tableaux) sont gérés par JAX (peut-être sur le périphérique). Mais dans ce fichier, on ne fait que les créer et les passer.

Éléments inutilisés

    load_yaml importé mais jamais utilisé. Cela pourrait être un oubli (peut-être utilisé dans une version antérieure). À signaler.

    La fonction _normalize_yaml_config est utilisée dans generate_kernel_groups et aussi importée dans hub_running_v8 (via from running.plan_v8 import _normalize_yaml_config dans _normalize_max_it). Donc elle est utilisée indirectement.

    Les fonctions _wrap_gamma_v7_to_v8 et make_composed_gamma sont utilisées. Aucune autre fonction inutilisée.

Dépendances

    Ce fichier dépend de utils.io_v8 (seulement load_yaml), mais cette dépendance n'est pas utilisée. Donc on pourrait l'enlever.

Points d'attention

    generate_kernel_groups utilise un attribut de fonction _last_n_invalid pour stocker un état global. C'est un peu inhabituel mais fonctionnel. Cela pourrait poser problème en cas d'appels concurrents, mais ici c'est séquentiel.

    La gestion des seeds : _seed_to_key utilise time.time_ns() et random.randint si seed est None. Cela génère des clés potentiellement non reproductibles, mais c'est probablement voulu.

Conclusion pour plan_v8.py

    ✅ La plupart des fonctions sont utilisées.

    ⚠️ L'import inutilisé de load_yaml est à nettoyer.

    ✅ Le flux de données est clair : les configurations sont transformées en groupes via des produits cartésiens avec filtrage.

    ✅ Pas de fuite mémoire évidente.

Analyse de kernel_v8.py

Ce fichier est le cœur du calcul JAX/XLA. Il définit toutes les fonctions de calcul des features, le step function pour le scan, la fonction de post-traitement, et l'exécution batch avec précompilation. C'est un fichier volumineux et complexe.
1. Imports

    jax, jax.numpy as jnp, lax, partial : utilisés partout.

    typing : utilisé pour les annotations.

    gc, numpy as np : utilisés dans execute_batch pour la gestion mémoire et conversion.

Tous les imports sont utilisés.
2. Constantes et listes globales

    DANS_SCAN_KEYS : liste de 22 noms de features in-scan. Utilisée dans _measure_state (construction du dict), dans _step_fn (pour le gel per-feature), et dans _post_scan (via accès aux signaux). Aussi utilisée dans hub_running_v8.py via import.

    FEATURE_NAMES : liste de ~75 noms de features finales. Utilisée dans _post_scan (construction du dict) et dans hub_running_v8.py.

    FEATURES_STRUCTURAL_NAN : dict décrivant les conditions de NaN structurel. Utilisé dans hub_running_v8.py (pour déterminer le statut NAN_ALL). Non utilisé ailleurs dans ce fichier.

    _FEATURE_NAMES_FOR_HEALTH : liste des features sans health, utilisée seulement pour définir une variable locale non exportée ? En fait, elle est définie mais jamais utilisée dans ce fichier. Elle pourrait être un vestige ou destinée à un usage externe. Elle n'est pas exportée, donc si elle n'est pas utilisée ailleurs, elle est inutile. À vérifier dans d'autres fichiers (mais probablement pas). Élément possiblement inutilisé : _FEATURE_NAMES_FOR_HEALTH n'est pas utilisé dans kernel_v8.py. Il faudra vérifier si elle est importée ailleurs (dans hub_running_v8 par exemple) – mais elle n'est pas exportée car pas dans __all__. Donc probablement morte.

3. Fonctions helpers

    _mode0_unfolding, _svd_no_vectors, _entropy_from_probs : utilisées.

    _f1_spectral, _f2_informational, _f3_entanglement, _f4_lyapunov_empirical, _f4_hutchinson, _f4_dynamics, _f5_transport : toutes utilisées dans _measure_state.

    _dmd_streaming_update : utilisé dans _measure_state.

    _measure_state : utilisée dans _step_fn (branche compute).

    _step_fn : utilisée comme fonction de scan dans _run_fn.

    _temporal_features : utilisée dans _post_scan.

    _approx_transfer_entropy, _first_min_autocorr, _f7_safe, _f7_nan : utilisées dans _post_scan.

    _post_scan : utilisée dans _run_fn (mode aggregated).

    _run_fn : fonction principale vmappable, utilisée dans _build_kernel_fn et run_one.

    run_one : wrapper JIT pour debug, probablement utilisé pour des tests mais pas dans le flux principal. Il est exporté (pas de __all__ mais accessible). Est-il utilisé ailleurs ? À vérifier dans hub_running_v8.py ou autre. Si non utilisé, il pourrait être mort. Pour l'instant, on le note comme potentiellement inutilisé dans le contexte de l'exécution batch (car execute_batch utilise _build_kernel_fn directement). Cependant, il peut être appelé depuis un script de test ou autre. À vérifier globalement.

    build_batch_inputs : utilisé dans pre_compile_for_group et execute_batch.

    execute_batch : utilisé dans hub_running_v8.py.

    _build_kernel_fn : utilisé dans execute_batch et pre_compile_for_group.

    pre_compile_for_group : utilisé dans hub_running_v8.py (rolling compile).

Toutes les fonctions semblent être utilisées localement ou exportées. Cependant, il faut vérifier l'utilisation de run_one et de _FEATURE_NAMES_FOR_HEALTH.
4. Variables et fonctions non utilisées dans ce fichier

    _FEATURE_NAMES_FOR_HEALTH : définie mais non utilisée. Si elle n'est pas importée ailleurs, elle est morte.

    run_one : peut-être utilisé dans des tests, mais dans le code principal, on voit que execute_batch utilise _build_kernel_fn et non run_one. Cependant, run_one est une fonction publique (pas de _ au début) et pourrait être appelée depuis un autre module non encore analysé (par exemple, un script de debug). Pour l'instant, on ne peut pas conclure.

5. Gestion mémoire

    Dans execute_batch, il y a une gestion explicite : del kernel_fn, del features_async, D_b, gp_b, mp_b, keys_b, puis gc.collect(). Cela libère la mémoire GPU/CPU.

    Dans pre_compile_for_group, on appelle jax.block_until_ready pour forcer la compilation, puis on del result, D_b, gp_b, mp_b, keys_b.

    Dans _rolling_compile_window (hub_running), on a del kernel_fn et del kg après usage.

    Pas de fuite évidente.

6. Flux de données

    Les données entrent via batch (contenant samples avec D_initial, gamma_params, mod_params, key_run). build_batch_inputs les convertit en tableaux JAX.

    _build_kernel_fn crée une fonction vmappée JIT.

    execute_batch exécute et récupère les résultats, les convertit en numpy et les retourne.

    Les features sont structurées en dictionnaires.

7. Cohérence

    Les noms de features dans DANS_SCAN_KEYS et FEATURE_NAMES sont cohérents avec les clés produites par _measure_state et _post_scan.

    FEATURES_STRUCTURAL_NAN est utilisé dans hub_running_v8.py pour déterminer le statut NAN_ALL. C'est cohérent.

Analyse de io_v8.py

Ce fichier contient des fonctions d'I/O génériques : discovery des atomics, chargement YAML, écriture/lecture Parquet avec schéma explicite.
1. Imports

    importlib, json, warnings, pathlib, typing, yaml, pyarrow, pyarrow.parquet. Tous utilisés.

2. Exceptions

    CriticalDiscoveryError : utilisée dans _discover_from_dir.

3. Discovery

    _discover_from_dir : fonction interne utilisée par les trois fonctions de découverte.

    discover_gammas_jax, discover_encodings_jax, discover_modifiers_jax : utilisées dans hub_running_v8.py et plan_v8.py. Toutes sont appelées.

4. YAML

    load_yaml : utilisée dans hub_running_v8.py et plan_v8.py.

    merge_configs : définie mais apparemment non utilisée dans les fichiers analysés jusqu'à présent. Peut-être utilisée dans analysing ou ailleurs. À vérifier. Pour l'instant, elle est potentiellement inutilisée dans le code actuel.

5. Parquet

    build_schema_v8 : utilisée dans open_parquet_writer.

    open_parquet_writer, write_rows_to_parquet, close_parquet_writer : utilisées dans hub_running_v8.py.

    read_parquet : utilisée probablement dans analysing (non encore vu). Pour l'instant, non utilisée dans les fichiers analysés, mais elle est exportée et sera probablement utilisée.

6. Variables inutilisées

    _META_COLS_V8 : défini mais non utilisé dans ce fichier. Il pourrait être utilisé ailleurs (par exemple dans analysing). À vérifier. Si non utilisé, il est mort.

Éléments à vérifier globalement

    _FEATURE_NAMES_FOR_HEALTH dans kernel_v8.py : probablement inutile.

    merge_configs dans io_v8.py : peut-être utilisé dans analysing.

    read_parquet dans io_v8.py : probablement utilisé dans analysing.

    _META_COLS_V8 dans io_v8.py : peut-être utilisé ailleurs.

    run_one dans kernel_v8.py : peut-être utilisé dans des tests ou scripts.

Audit du fichier hub_analysing_v8.py
Résumé général

hub_analysing_v8.py est le point d’entrée de la couche analysing. Il orchestre le pipeline d’analyse complet : chargement des données depuis un fichier Parquet, profiling, détection d’outliers, préparation de la matrice, clustering, nommage des clusters, visualisation et production de rapports (JSON, TXT). Deux modes sont proposés : verdict sur une phase unique (run_verdict_from_parquet) et verdict croisé entre plusieurs phases (run_verdict_cross_phases). Le fichier est bien structuré, avec un découpage en fonctions claires, et ne contient pas de code mort évident.
Analyse détaillée
1. Imports
Module importé	Éléments utilisés	Obsolescence
pathlib.Path	Path, exists, glob, mkdir, parent, stem, name (utilisés)	✅ utilisé
typing.Dict, List, Optional	utilisés pour les annotations de type	✅ utilisé
numpy as np	non utilisé directement dans ce fichier (mais présent pour compatibilité ?)	⚠️ inutilisé dans ce fichier (peut-être un vestige)
utils.io_v8.load_yaml	load_yaml	✅ utilisé
analysing.data_v8	load_analysing_data, prepare_matrix, AnalysingData	✅ utilisés
analysing.clustering_v8	run_profiling, analyze_outliers, run_clustering, ClusterNamer	✅ utilisés
analysing.outputs_v8	write_verdict_report, write_verdict_report_txt, ClusterVisualizer	✅ utilisés
try: from analysing.data_v8 import compute_projection	compute_projection (utilisé conditionnellement dans run_analysing_pipeline si plot est vrai)	⚠️ import protégé, mais fonction utilisée ; l’import en try est correct

Observation : L’import de numpy (import numpy as np) n’est jamais utilisé dans ce fichier. Il pourrait être supprimé sans conséquence.
2. Variables globales et fonctions

    Aucune variable globale n’est définie (seulement des fonctions).

    La fonction _default_cfg_path() est utilisée dans run_verdict_from_parquet (et potentiellement ailleurs) → ✅ utilisée.

    scan_major_phases() est utilisée dans run_verdict_cross_phases → ✅ utilisée.

    _empty_result() est utilisée dans run_analysing_pipeline → ✅ utilisée.

    _print_naming_summary() est utilisée dans run_analysing_pipeline → ✅ utilisée.

    run_analysing_pipeline() est utilisée dans les deux fonctions de verdict → ✅ utilisée.

    run_verdict_from_parquet() est appelée depuis batch.py (mode 2) → ✅ utilisée (externe).

    run_verdict_cross_phases() est appelée depuis batch.py (mode 3) → ✅ utilisée (externe).

Aucune fonction ou variable définie dans ce fichier n’est inutilisée en interne. Toutes sont appelées soit localement, soit exportées pour être utilisées par batch.py.
3. Flux de données (orchestration)
text

run_verdict_from_parquet(parquet_path, cfg_path, output_dir, label, plot, debug)
│
├─ cfg = load_yaml(cfg_path)
├─ data = load_analysing_data(parquet_path, scope, apply_pool)  # retourne un objet AnalysingData
├─ result = run_analysing_pipeline(data, cfg, output_dir, label, plot, debug)
│   │
│   ├─ profiling_results = run_profiling(data)
│   ├─ outliers_results = analyze_outliers(data, contamination)
│   ├─ M_ortho, feat_names, nan_mask, matrix_meta = prepare_matrix(data, cfg)
│   ├─ (si plot) M_2d = compute_projection(M_ortho, cfg['projection'], cache_path)
│   ├─ clustering_result = run_clustering(M_ortho, feat_names, peeling_cfg, verbose)
│   ├─ named_clusters = ClusterNamer.from_yaml().name_all(peeling_result, M_ortho, feat_names)
│   ├─ (si plot) ClusterVisualizer(...).plot_all(...)
│   └─ retourne dict résultat
│
├─ write_verdict_report(result, ...)  # JSON
└─ write_verdict_report_txt(result, ...)  # TXT

Le flux est clair et linéaire. Les appels aux modules externes sont bien séparés.
4. Gestion mémoire et durée de vie

    Les objets créés dans les fonctions sont locaux.

    Dans run_analysing_pipeline, on voit des del explicites pour libérer de la mémoire :

        del data.M (si le dataset est gros)

        del M_ortho

        del M_2d (après visualisation)

        Cela montre une attention à la gestion mémoire, surtout pour les grandes matrices.

    Les appels aux fonctions externes (clustering, etc.) peuvent créer leurs propres objets, mais ils sont libérés à la sortie des fonctions (ou via garbage collector).

    Pas de fuite mémoire évidente dans ce fichier.

5. Cohérence documentation / code

    La docstring en tête est correcte et décrit le rôle.

    Les commentaires internes sont présents et utiles.

    Note : L’import de numpy est inutile, ce qui pourrait être considéré comme une petite incohérence (code mort).

    Le try/except pour compute_projection est bien géré : si l’import échoue, la fonction est définie à None, et l’appel est conditionné à plot and output_dir, ce qui évite une erreur.

Conclusion pour hub_analysing_v8.py

    ✅ Tous les imports (sauf numpy) sont nécessaires et utilisés.

    ✅ Toutes les fonctions définies sont utilisées (soit localement, soit exportées).

    ⚠️ L’import de numpy est inutile et pourrait être supprimé (code mort mineur).

    ✅ La gestion mémoire est proactive avec des del explicites.

    ✅ Le flux de données est clair et bien orchestré.

Aucun objet ou fonction inutilisé n’a été détecté dans ce fichier, à l’exception de l’import superflux de numpy.

Audit du fichier data_v8.py
Résumé général

data_v8.py est le module de la couche analysing responsable du chargement des données depuis un fichier Parquet, de leur filtrage, de la préparation de la matrice pour l'apprentissage automatique (imputation des NaN, transformations, standardisation, orthogonalisation) et du calcul de projection 2D (t-SNE/UMAP). Il définit la classe AnalysingData comme conteneur principal. Le code est bien organisé, avec des fonctions dédiées et une gestion mémoire explicite. Aucune fonction ou classe définie n'est inutilisée dans le module, mais il y a quelques importations redondantes ou inutilisées.
Analyse détaillée
1. Imports
Module importé	Éléments utilisés	Obsolescence
dataclasses	dataclass, field	✅ utilisés pour AnalysingData
pathlib.Path	Path	✅ utilisé
typing	Dict, List, Optional, Set, Tuple	✅ utilisés pour les annotations
warnings	warn	✅ utilisé
numpy as np	très utilisé	✅
pyarrow.parquet as pq	read_table	✅
sklearn.preprocessing.RobustScaler	RobustScaler	✅
utils.io_v8	load_yaml, _META_COLS_V8	✅ load_yaml utilisé, _META_COLS_V8 utilisé pour filtrer les colonnes feature
sklearn.decomposition.PCA	utilisé dans compute_projection	✅ (import à l'intérieur de la fonction)
sklearn.manifold.TSNE	utilisé dans compute_projection	✅
umap	tenté dans compute_projection (avec fallback)	✅ import conditionnel

Aucun import inutile : tous les imports sont nécessaires. Les imports de PCA, TSNE et umap sont faits à l'intérieur de compute_projection pour éviter des dépendances inutiles si la fonction n'est pas appelée.
2. Classe AnalysingData

    Attributs bien définis.

    Méthodes : __post_init__, n (property), F (property), features_for_ml().

    Toutes les méthodes sont utilisées :

        n utilisé dans hub_analysing_v8.py pour afficher le nombre d'observations.

        F non utilisé dans ce module mais pourrait l'être ailleurs ; reste utile.

        features_for_ml() utilisé dans prepare_matrix.

    Pas d'attribut ou méthode inutilisée.

3. Fonctions de conversion et filtrage

    _df_to_analysing_data(df, feature_cols) : utilisée dans load_analysing_data.

    build_pyarrow_filters(scope) : utilisée dans load_analysing_data.

    _mask_seeds_one(data) : utilisée dans load_analysing_data si scope['seeds'] == 'one'.

    _apply_mask(data, mask) : utilisée dans load_analysing_data pour appliquer les masques.

    load_pool_requirements(path=None) : utilisée dans load_analysing_data si apply_pool=True.

    _mask_pool_requirements(data, req) : utilisée dans load_analysing_data après chargement des requirements.

Toutes ces fonctions sont internes et appelées. Aucune n'est inutilisée.
4. Fonction principale load_analysing_data

    Point d'entrée pour charger les données depuis un Parquet.

    Utilise les fonctions ci-dessus.

    Gère le filtrage Arrow pushdown, le filtrage "seeds=one", et les pool_requirements.

    Affiche des informations de debug.

    RAS : tout est utilisé.

5. Préparation de matrice prepare_matrix

    Fonction complexe mais bien découpée.

    Utilise data.features_for_ml() pour obtenir les features (sans health).

    Appelle _log_transform (définie plus haut) et _select_orthogonal_features.

    Gère les NaN, transformations, scaling, orthogonalisation.

    Retourne un tuple complet.

    Tous les appels internes sont utilisés. Les fonctions auxiliaires _log_transform et _select_orthogonal_features sont définies dans le module et appelées.

Remarque : _select_orthogonal_features utilise PROTECTED_FEATURES_V8, un ensemble défini plus haut. Cet ensemble est bien utilisé.
6. Projection compute_projection

    Fonction conditionnelle (imports à l'intérieur).

    Utilise PCA, t-SNE, et éventuellement UMAP.

    Gère le sous-échantillonnage et le cache.

    Appelée dans hub_analysing_v8.py seulement si plot=True.

    RAS : tout est utilisé.

7. Variables globales et constantes

    DYNAMIC_THRESHOLD = 1e6 : utilisé dans _log_transform.

    CORRELATION_THRESHOLD_DEFAULT = 0.98 : utilisé dans prepare_matrix comme valeur par défaut.

    PROTECTED_FEATURES_V8 : utilisé dans _select_orthogonal_features.

    Toutes ces constantes sont utilisées.

8. Gestion mémoire

    Dans load_analysing_data, après conversion en AnalysingData, del df libère la table PyArrow.

    Dans prepare_matrix, création de matrices intermédiaires, mais pas de del explicite (les variables locales seront garbage collectées à la sortie).

    Dans compute_projection, les matrices intermédiaires sont également locales.

    Pas de fuite évidente.

9. Cohérence et code mort

    L'import de load_yaml depuis utils.io_v8 est utilisé dans load_pool_requirements.

    _META_COLS_V8 est utilisé pour identifier les colonnes non-feature.

    Aucune fonction ou variable inutilisée.

Conclusion pour data_v8.py

    ✅ Tous les imports sont nécessaires.

    ✅ Toutes les fonctions et classes définies sont utilisées.

    ✅ Les constantes sont utilisées.

    ✅ Gestion mémoire correcte.

    Aucun objet ou fonction inutilisé détecté.

Le fichier est propre et bien architecturé.

Audit du fichier clustering_v8.py
Résumé général

clustering_v8.py implémente l’analyse ML proprement dite pour la couche analysing :

    Profiling : agrégation des statistiques par entité (gamma, encoding, modifier).

    Outliers : détection d’anomalies via IsolationForest.

    Peeling : clustering multi-niveaux par HDBSCAN avec extraction itérative des clusters homogènes.

    Nommage : via ClusterNamer, qui utilise un fichier de configuration YAML pour attribuer des noms interprétables aux clusters en fonction de leurs profils.

Le fichier est volumineux mais bien organisé. Toutes les fonctions sont utilisées et le code est cohérent.
Analyse détaillée
1. Imports
Module importé	Éléments utilisés	Obsolescence
warnings	warn	✅ utilisé
numpy as np	nombreuses fonctions	✅ utilisé
pathlib.Path	Path	✅ utilisé
collections.Counter	non utilisé directement ?	⚠️ Counter est importé mais jamais utilisé dans ce fichier.
typing	Dict, List, Optional, Set, Tuple	✅ utilisés (annotations)
sklearn.cluster.HDBSCAN	HDBSCAN	✅ utilisé
sklearn.decomposition.PCA	PCA	✅ utilisé
sklearn.ensemble.IsolationForest	IsolationForest	✅ utilisé
sklearn.metrics.silhouette_score	silhouette_score	✅ utilisé
utils.io_v8	load_yaml	✅ utilisé
analysing.data_v8	AnalysingData	✅ utilisé

Observation : from collections import Counter est inutile (non utilisé). À supprimer.
2. Variables globales

    EPS = 1e-10 : utilisé dans les calculs de percentile → ✅

3. Fonctions
a. Profiling

    _aggregate_by_entity(M, feat_names, entity_arr) : agrège les stats par entité. Utilisée dans run_profiling → ✅

    run_profiling(data) : point d’entrée du profiling. Appelée depuis hub_analysing_v8.py → ✅

b. Outliers

    _compute_atomic_recurrence(entity_arr, mask) : calcule la récurrence des entités parmi les outliers. Utilisée dans analyze_outliers → ✅

    analyze_outliers(data, contamination=0.1) : détecte les outliers avec IsolationForest. Appelée depuis hub_analysing_v8.py → ✅

c. Peeling (clustering multi-niveaux)

    _threshold(cfg, level) : calcule le seuil d’homogénéité pour un niveau donné. Utilisée dans _run_level → ✅

    _mcs_from_n(n, cfg) : calcule min_cluster_size en fonction du nombre d’échantillons. Utilisée dans run_peeling et _mcs_residual → ✅

    _ms_from_n(n, cfg) : calcule min_samples. Utilisée dans run_peeling → ✅

    _mcs_residual(cfg, mcs_global, n_total, n_residual, M_residual, M_global) : ajuste min_cluster_size pour le niveau résiduel. Utilisée dans run_peeling → ✅

    _homogeneity_score(M_cluster, proba_cluster, M_others, cfg) : calcule le score d’homogénéité d’un cluster. Utilisée dans _run_level → ✅

    _run_level(M_level, global_idx, M_global, cfg, level, mcs, ms, configs, verbose) : exécute HDBSCAN sur un niveau et extrait les clusters candidats. Utilisée dans run_peeling → ✅

    run_peeling(M_ortho, cfg, verbose=False) : algorithme principal de peeling. Utilisée dans run_clustering → ✅

    run_clustering(M_ortho, feat_names, peeling_cfg, verbose=False) : wrapper pour run_peeling. Appelée depuis hub_analysing_v8.py → ✅

d. Nommage des clusters

    build_cluster_profile(cluster_mask, M_ortho, feat_names) : construit le profil statistique d’un cluster. Utilisée dans ClusterNamer.name_all → ✅

    build_layer_distribution(M_ortho, feat_names) : construit la distribution de chaque feature sur l’ensemble des données (pour calculs de percentiles). Utilisée dans ClusterNamer.name_all → ✅

    _percentile_rank(value, dist) : calcule le rang percentile d’une valeur. Utilisée dans les handlers de nommage → ✅

    _conf_from_percentile(pct, direction, conf_at_edge=1.0, conf_at_center=0.5) : convertit un percentile en niveau de confiance. Utilisée dans les handlers → ✅

    _get_median(profil, feature_key) et _get_std(profil, feature_key) : extraient médiane/écart-type d’un profil. Utilisées dans les handlers → ✅

    _eval_zones, _eval_delta, _eval_threshold : trois handlers pour évaluer les conditions des slots de nommage. Utilisés dans ClusterNamer.name_cluster via dispatch → ✅

    _EVAL_DISPATCH : dictionnaire de dispatch → ✅

e. Classe ClusterNamer

    __init__ : initialise avec la config.

    from_yaml(cls, path=None) : charge la configuration depuis un fichier YAML. Utilisée dans hub_analysing_v8.py → ✅

    name_cluster(self, profil, layer_dist, cluster_homogeneity, n, cluster_id=None) : génère un nom pour un cluster. Utilisée dans name_all → ✅

    name_all(self, peeling_result, M_ortho, feat_names) : nomme tous les clusters extraits et le résidu. Appelée depuis hub_analysing_v8.py → ✅

4. Flux de données

Le flux est celui décrit dans hub_analysing_v8.py :

    run_profiling → stats par entité.

    analyze_outliers → masque outliers.

    run_clustering → appel à run_peeling → production de peeling_result.

    ClusterNamer.name_all → génération des noms.

5. Gestion mémoire

    Dans run_peeling, on manipule des sous-ensembles de la matrice originale par indexation, sans créer de copies inutiles (sauf dans _run_level où une PCA peut créer une copie réduite).

    _homogeneity_score peut sous-échantillonner pour le calcul du silhouette score si trop d’échantillons.

    Pas de fuite mémoire évidente.

6. Cohérence documentation / code

    Docstrings présentes.

    Nommage des variables clair.

    Le code est complexe mais bien commenté.

Conclusion pour clustering_v8.py

    ✅ Tous les imports sauf Counter sont nécessaires.

    ✅ Toutes les fonctions et classes sont utilisées.

    ⚠️ L’import from collections import Counter est inutile (code mort) et devrait être supprimé.

    ✅ Gestion mémoire correcte.

    ✅ Logique métier complexe mais bien implémentée.

Un seul élément inutilisé : l’import de Counter.

Audit du fichier outputs_v8.py
Résumé général

outputs_v8.py est responsable de la génération des sorties du pipeline d’analyse :

    Sérialisation JSON des résultats de verdict (write_verdict_report).

    Génération d’un rapport texte lisible (write_verdict_report_txt).

    Visualisations graphiques via la classe ClusterVisualizer (projection 2D, résumé du peeling, heatmap des signatures, etc.).

Le code est propre, bien isolé (zéro calcul métier), et toutes les fonctions sont utilisées. Aucun élément inutilisé n’est détecté.
Analyse détaillée
1. Imports
Module importé	Éléments utilisés	Obsolescence
json	dump	✅ utilisé
numpy as np	nombreuses fonctions (array, median, etc.)	✅ utilisé
matplotlib	use('Agg'), colormaps, pyplot as plt	✅ utilisé
pathlib.Path	Path, mkdir, parent	✅ utilisé
typing	Dict, List, Optional	✅ utilisés (annotations)

Aucun import inutile. L’utilisation de matplotlib.use('Agg') avant l’import de pyplot est correcte pour un environnement sans affichage.
2. Variables globales et constantes

    DARK_BG, PANEL_BG, GRID_COLOR, TEXT_COLOR, DIM_COLOR, LEVEL_COLORS : constantes de style pour les graphiques, utilisées dans les fonctions de visualisation → ✅

    La fonction _style, _save, _cluster_colormap sont des helpers internes, tous utilisés dans la classe ClusterVisualizer → ✅

3. Fonctions
a. JSON

    write_verdict_report(verdict_results, output_path) : écrit les résultats au format JSON, en excluant M_2d (non sérialisable) et en convertissant les types numpy. Appelée depuis hub_analysing_v8.py → ✅

b. TXT

    write_verdict_report_txt(verdict_results, output_path) : génère un rapport texte lisible avec un résumé des clusters, outliers, etc. Appelée depuis hub_analysing_v8.py → ✅

c. Visualisation

    _cluster_colormap(n) : retourne une colormap resampled. Utilisée dans ClusterVisualizer.__init__ → ✅

    _style(ax, title='') : applique le style sombre à un axe. Utilisée dans les méthodes de ClusterVisualizer → ✅

    _save(fig, path, dpi=140) : sauvegarde une figure et la ferme. Utilisée dans les méthodes de ClusterVisualizer → ✅

d. Classe ClusterVisualizer

    __init__(self, M_2d, named_clusters, peeling_result, run_regimes=None, gammas=None) : initialise avec les données et prépare la colormap. Les paramètres run_regimes et gammas ne sont pas utilisés dans les méthodes (obsolètes ?). Ils sont stockés dans l’instance mais jamais référencés.

        run_regimes : non utilisé (pourrait être supprimé).

        gammas : non utilisé (idem).

        Cela constitue un code mort mineur (paramètres inutilisés).

    plot_peeling_summary(self, output_dir, label) : génère une figure avec deux sous-graphes : clusters par ID et clusters par niveau. Utilisée dans plot_all → ✅

    plot_signature_heatmap(self, output_dir, label) : génère une heatmap des signatures des clusters. Utilisée dans plot_all → ✅

    plot_layer(self, layer_name, output_dir, label) : génère une vue par couche (ici toujours appelée avec 'universal'). Utilisée dans plot_all → ✅

    plot_all(self, output_dir, label) : appelle les trois méthodes de visualisation. Appelée depuis hub_analysing_v8.py → ✅

Remarque : Les paramètres run_regimes et gammas dans __init__ ne sont pas utilisés. Ils pourraient être supprimés sans impact, car ils ne sont jamais lus.
4. Flux de données

Les fonctions de ce module ne font que recevoir des structures de données déjà calculées et les écrire sur disque ou générer des images. Aucune transformation métier.
5. Gestion mémoire

    Les figures matplotlib sont fermées après sauvegarde avec plt.close(fig) dans _save, ce qui libère la mémoire.

    Pas d’autres allocations notables.

6. Cohérence documentation / code

    Docstrings présentes.

    Utilisation de matplotlib.colormaps (moderne) au lieu de plt.cm.get_cmap (déprécié), conforme au commentaire.

    Les noms de fonctions sont clairs.

Conclusion pour outputs_v8.py

    ✅ Tous les imports sont nécessaires.

    ✅ Toutes les fonctions et classes sont utilisées (soit localement, soit exportées).

    ⚠️ Dans ClusterVisualizer.__init__, les paramètres run_regimes et gammas sont stockés mais jamais utilisés. Cela constitue du code mort mineur.

    ✅ Gestion mémoire correcte.

Seul point mineur : paramètres inutilisés dans le constructeur de ClusterVisualizer.



