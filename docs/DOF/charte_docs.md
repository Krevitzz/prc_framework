CHARTE_DOCS.md

Charte de documentation du projet PRC

0. Statut du document

Ce document définit les règles formelles, épistémiques et structurelles applicables à toute documentation produite dans le cadre du projet PRC, indépendamment de son statut (exploration, élimination, validation, synthèse).

Cette charte est :

normative (elle impose des règles),

transversale (elle s’applique à tous les fichiers),

prioritaire sur toute convention locale ou implicite.

Toute entrée non conforme à cette charte est considérée comme non valide et ne doit pas être utilisée comme référence.

1. Finalité de la documentation PRC

La documentation PRC n’a pas pour objectif :

de convaincre,

de démontrer une thèse,

de construire une narration cohérente a posteriori,

de préserver une intuition fondatrice.

Elle a pour objectif exclusif :

la traçabilité cumulative et falsifiable d’un processus d’exploration par élimination,
conforme aux principes ontologiques et épistémiques du cadre PRC.

La documentation est un artefact opératoire, pas un support rhétorique.

2. Principe fondamental : primauté du négatif

Le projet PRC adopte explicitement une épistémologie négative.

En conséquence :

l’élimination est une information,

l’échec est un résultat,

l’absence de candidat valide est un état acceptable,

la persistance d’un résidu est plus informative qu’une convergence rapide.

Aucun document ne doit :

masquer une élimination,

reformuler un rejet comme un “presque succès”,

réintroduire implicitement une hypothèse éliminée.

3. Séparation stricte des niveaux de description

Toute documentation doit explicitement respecter la séparation suivante :

Niveau ontologique : ce qui est supposé être (axiomes, principes)

Niveau épistémique : ce qui est accessible, testable, mesurable

Niveau opératoire : ce qui est implémenté, simulé, diagnostiqué

Toute confusion entre ces niveaux est considérée comme un glissement invalide.

Lorsqu’un énoncé relève d’un niveau précis, cela doit être clairement identifiable par le contexte ou par une mention explicite.

4. Format obligatoire des entrées documentaires

Toute entrée documentée (hypothèse, famille, contrainte, test, élimination) doit respecter les règles suivantes.

4.1 Identification

Chaque entrée doit comporter :

un identifiant stable (H1, C3, R2, etc.),

un intitulé descriptif non ambigu.

Aucun intitulé ne doit suggérer un succès ou une direction privilégiée.

4.2 Datation

Toute entrée doit comporter :

une date explicite (YYYY-MM-DD),

correspondant à la date de constat, pas de formulation ultérieure.

4.3 Référencement des protocoles

Toute conclusion doit référencer :

un ou plusieurs scripts,

ou un protocole formalisé,

ou une configuration expérimentale identifiable.

Aucune conclusion ne peut être fondée sur une intuition non testée sans être explicitement marquée comme telle.

4.4 Statut épistémique obligatoire

Toute entrée doit préciser son statut parmi les catégories suivantes :

EXPLORATOIRE

ÉLIMINÉ (opératoire)

ÉLIMINÉ (théorique)

CONTRAINTE DÉDUITE

RÉSIDUEL (non éliminé)

Toute absence de statut rend l’entrée invalide.

5. Irreversibilité contrôlée des éliminations

Toute élimination enregistrée est irréversible par défaut.

Une élimination ne peut être remise en question que si :

le cadre de test change explicitement,

ou si une hypothèse implicite du protocole est invalidée.

Dans ce cas :

l’entrée originale n’est jamais supprimée,

une nouvelle entrée est créée, explicitant la révision.

6. Interdiction des réintroductions implicites

Il est strictement interdit de :

réintroduire une hypothèse éliminée sous une autre terminologie,

recomposer plusieurs éléments éliminés sans justification explicite,

faire réapparaître une famille rejetée par hybridation ad hoc.

Toute réintroduction doit être :

explicitement signalée,

justifiée par une modification du cadre,

documentée comme telle.

7. Neutralité narrative

La documentation PRC doit maintenir une neutralité narrative stricte.

Sont proscrits :

les formulations téléologiques,

les termes valorisants (“prometteur”, “élégant”, “intéressant”),

toute projection vers une thèse finale.

Le langage doit rester :

descriptif,

factuel,

non affectif.

8. Rapport à la thèse

La thèse n’est pas un document actif du projet.

Elle constitue, le cas échéant :

une compilation a posteriori,

d’un ensemble de documents déjà stabilisés,

sans modification rétroactive du contenu source.

La documentation PRC doit être compréhensible et exploitable indépendamment de toute thèse.

9. Principe de clôture locale

Un document peut être considéré comme localement clos lorsqu’il :

ne nécessite plus de modification interne,

reste compatible avec les documents futurs,

ne présuppose aucun résultat à venir.

La clôture locale est préférée à la complétude globale.

10. Clause finale

Cette charte peut évoluer, mais :

toute modification doit être documentée,

datée,

et ne s’applique jamais rétroactivement sans mention explicite.

Le respect de cette charte est une condition nécessaire à la cohérence du projet PRC.

## Conventions documentaires

### Statut du présent document
- **Version** : 1.0
- **Nature** : Normatif documentaire
- **Champ** : Méthodologie du projet PRC
- **Portée** : Organisation, traçabilité et qualification des documents
- **Exclusion** : Ce document n’introduit aucune hypothèse ontologique,
  dynamique ou formelle sur PRC, D ou Γ.

Toute modification de cette charte implique une incrémentation de version.
Les documents existants ne sont pas modifiés rétroactivement.

---

### 1. Identification des entrées

Chaque entrée documentaire doit posséder un identifiant unique et stable.

**Format recommandé** :
[TYPE]-[NUMÉRO]-[DATE]


Exemples :
- `HYP-03-2025-01-12`
- `GAM-07-2025-02-04`
- `TEST-12-2025-02-18`

Les identifiants :
- ne sont jamais réutilisés,
- ne sont jamais renommés,
- restent valides même si le contenu est ultérieurement invalidé.

---

### 2. Typologie des documents

Les types autorisés sont :

- `HYP` : hypothèse explicite
- `FAM` : famille d’opérateurs ou de mécanismes
- `GAM` : règle Γ candidate ou mécanisme apparenté
- `TEST` : protocole expérimental ou diagnostic
- `RES` : résultat brut ou synthèse de résultats
- `ELIM` : élimination formelle (hypothèse, famille ou règle)
- `CONS` : contrainte émergente déduite par élimination
- `META` : note méthodologique ou réflexive (hors ontologie)

Tout nouveau type doit être ajouté explicitement à cette liste.

---

### 3. Statuts autorisés

Chaque entrée doit porter **un statut unique** parmi :

- **ACTIF** : en cours d’exploration
- **PARTIEL** : valide sous conditions limitées
- **ÉLIMINÉ** : rejeté par protocole documenté
- **OBSOLETE** : remplacé par une formulation plus récente
- **RÉSERVÉ** : indéterminé, données insuffisantes

Le statut ne préjuge pas de la vérité ontologique,
uniquement de l’état méthodologique.

---

### 4. Qualification D / Γ

Toute entrée doit préciser explicitement son domaine principal :

- `D` : dissymétrie, résidu, structure observée
- `Γ` : mécanisme, règle, opérateur de compensation
- `MIXTE` : interaction D–Γ
- `META` : niveau méthodologique ou épistémique

Cette qualification est **obligatoire**.

---

### 5. Règle de non-effacement

Aucune entrée ne peut être supprimée.

Les éliminations, échecs ou impasses :
- sont conservés,
- documentés,
- utilisés comme information négative.

Une entrée éliminée peut être référencée ultérieurement,
mais jamais réactivée sans création d’une nouvelle entrée.

---

### 6. Principe de neutralité méthodologique

Les documents de registre :
- décrivent ce qui a été testé,
- ce qui a été observé,
- et ce qui a été éliminé.

Ils ne cherchent ni à sauver une hypothèse,
ni à forcer une interprétation cohérente a posteriori.

La cohérence globale est un **résultat**, jamais un prérequis.
