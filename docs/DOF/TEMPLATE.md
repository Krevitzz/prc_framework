0. Convention adoptée (engagement ferme)
0.1 Convention d’identification
PRC-[TYPE]-[NUM]


TYPE ∈ {HYP, GAM, FAM, CON, PRO, DIA}

NUM : incrémental, non réutilisé, non sémantique

Exemples :

PRC-HYP-001

PRC-GAM-004

PRC-FAM-002

PRC-CON-003

0.2 Convention de statut

ELIMINÉE_DÉFINITIVE : invalide quel que soit l’habillage futur

ELIMINÉE_CADRE_ACTUEL : invalide sous hypothèses actuelles

PARTIELLEMENT_FERTILE : intéressant mais insuffisant

SURVIVANTE_FAIBLE : passe les tests minimaux

OBSOLETE_DÉFINITION : éliminée car définition de Γ a changé

0.3 Convention D / Γ (très important)

D doit toujours être qualifié :

D^ont

D^epi

D^mixte

Γ doit toujours être qualifié :

spatial : local / non-local

temporel : markovien / à mémoire

causalité : instantané / étalé


Métadonnées générales

ID entrée : PRC-XXXX

Date : YYYY-MM-DD

Auteur :

Statut :

EN_COURS | ELIMINÉE | PARTIELLE | SURVIVANTE | OBSOLETE

Type d’entrée :

HYPOTHÈSE

CANDIDAT_Γ

FAMILLE

CONTRAINTE_DÉDUITE

PROTOCOLE

DIAGNOSTIC

Niveau :

ONTOLOGIQUE

OPÉRATIONNEL

MÉTA-MÉTHODOLOGIQUE

1. Contexte et motivation

But de l’entrée :

Pourquoi cette hypothèse / ce candidat / ce protocole a été introduit.
Quel problème précis du PRC il adresse.

Origine :

intuition théorique

analogie

nécessité méthodologique

dérivation négative (élimination antérieure)

test exploratoire

2. Définition formelle (si applicable)
2.1 Objets impliqués

Définition de D :

☐ ontologique

☐ épistémique

☐ mixte / indéterminé

Définition de Γ :

locale / non-locale

markovienne / à mémoire

instantanée / étalée

Autres opérateurs / contraintes :

2.2 Formulation mathématique / algorithmique
(placeholder formalisme)


⚠️ Si la définition est volontairement incomplète, le signaler explicitement.

3. Hypothèses implicites (OBLIGATOIRE)

Lister explicitement ce que l’entrée suppose sans le démontrer.

H_imp1 :

H_imp2 :

…

Cette section est critique pour éviter les glissements non perçus.

4. Protocole de test associé

Nom du protocole :

Implémentation : (fichier / script / version)

Paramètres clés :

N_DOF :

ε :

β :

durée :

Diagnostics utilisés :

stabilité

cycles

spectre

dépendance à C₀

compression informationnelle

autres

5. Résultats observés
5.1 Résumé exécutif

Résumé factuel, sans interprétation.

5.2 Observations détaillées

attracteurs :

cycles :

dérives :

pathologies :

invariants :

6. Verdict provisoire

Décision :

☐ VALIDÉ (dans ce cadre)

☐ REJETÉ

☐ INCONCLUANT

☐ PARTIELLEMENT FERTILE

Justification :

critère(s) violé(s)

critère(s) satisfait(s)

7. Information négative extraite (CRUCIAL)

Même en cas d’échec, ce que l’entrée nous apprend sur Γ, D ou le PRC.

Contraintes induites :

Propriétés impossibles :

Comportements systématiquement observés :

8. Implications pour l’espace des possibles

Réduction de l’espace :

Nouvelles directions ouvertes :

Familles affectées :

Protocoles à ajuster :

9. Statut de réutilisation

☐ Ne doit plus être testé

☐ À re-tester sous définition élargie de Γ

☐ À requalifier (ex : de candidat → famille)

☐ À recycler comme contre-exemple

10. Liens explicites

Entrées liées :

PRC-XXXX

PRC-YYYY

Contraintes dérivées :

C1, C2, …

Documents impactés :

REGISTRE_ELIMINATIONS.md

CONTRAINTES_EMERGENTES.md

ESPACE_RESIDUEL.md

11. Commentaires méta (optionnel mais recommandé)

Doutes, tensions conceptuelles, soupçons de circularité, surcharge cognitive,
problèmes de repère ontologique/épistémique, etc.

Fin de l’entrée
Remarques importantes (hors template)

Toute entrée PRC doit pouvoir être relue isolément sans perdre son statut.

Une entrée peut devenir obsolète sans être fausse.

L’élimination n’est jamais une perte, c’est une compression du champ des possibles.