---
title: "Draft Notes"
draft: true
math: true
---

## Cadre général : on part d'un problème probabiliste

On veut modéliser une distribution conditionnelle :

$$
P(Y \mid X)
$$

ou une distribution jointe :

$$
P(X, Y)
$$

En statistiques classiques :
- on impose une forme paramétrique simple (GLM, arbres, etc.)
- structure explicite

En deep learning :

on remplace la forme explicite par une paramétrisation flexible via une fonction composée

## Hypothèse fondamentale du deep learning

On introduit une représentation latente déterministe :

$$
h = f_\theta(X)
$$

et on écrit :

$$
P(Y \mid X) = P(Y \mid h)
$$

Donc :
- toute l'information utile de \(X\) est compressée dans \(h\)
- \(h\) est une variable latente apprise

## Structure hiérarchique

On généralise :

$$
h_1 = f_1(X), \quad h_2 = f_2(h_1), \dots, h_L = f_L(h_{L-1})
$$

Finalement :

$$
P(Y \mid X) = P(Y \mid h_L)
$$

Interprétation statistique :

on remplace un modèle global par une composition de transformations latentes successives

## Vue "modèle latent"

On peut reformuler :
- \(h_1, \dots, h_L\) = variables latentes déterministes
- le modèle est :

$$
P(Y \mid X) = \int P(Y \mid h_L) \, \delta(h_L - f_\theta(X)) \, dh_L
$$

Donc :

deep learning = modèle latent avec variables latentes déterministes (Dirac)

## Rôle de la paramétrisation

Chaque couche est :

$$
h_{k+1} = \sigma(W_k h_k + b_k)
$$

Donc :
- transformation affine
- non-linéarité

Interprétation statistique :
- \(W_k\) = paramètres de dépendances
- \(\sigma\) = induction de non-linéarité dans la structure de dépendance

## Lecture probabiliste implicite

Même si ce n'est pas écrit explicitement, un réseau définit une famille :

$$
\{ P_\theta(Y \mid X), \theta \in \Theta \}
$$

où :

$$
P_\theta(Y \mid X) = \text{Softmax}(g_\theta(X))
$$

Donc :

le réseau définit une famille exponentielle implicite très flexible

## Fonction objectif = estimation statistique

Apprentissage :

$$
\max_\theta \sum_{i} \log P_\theta(Y_i \mid X_i)
$$

Donc :
- maximum de vraisemblance (ou équivalent)
- estimation de paramètres dans un modèle très riche

## Pourquoi la profondeur change tout

Une seule couche :

$$
g(X) = W X
$$

→ modèle linéaire

Profondeur :

$$
g(X) = W_L \sigma(W_{L-1} \sigma(\dots W_1 X))
$$

Interprétation statistique :

on compose des transformations qui correspondent à des reparamétrisations successives de l'espace des dépendances

Effet clé :
- induction de représentations hiérarchiques
- approximation de fonctions très complexes avec structure factorisée

## Où se placent les embeddings dans ce cadre

Les embeddings apparaissent comme cas particulier :
- entrée discrète \(X \in \mathcal{X}\)
- on définit :

$$
h = E X
$$

où \(E\) est une matrice d'embedding

Interprétation :

on transforme une variable catégorielle en variable latente continue optimisée

## Vision unifiée (le point important)

On peut résumer tout le deep learning ainsi :

1. Variables latentes déterministes

$$
h = f_\theta(X)
$$

2. Modèle probabiliste sur la sortie

$$
P(Y \mid h)
$$

3. Paramétrisation profonde de \(f_\theta\)

## Ce que le deep learning remplace en statistique

| Stat classique | Deep learning |
|---|---|
| features explicites | features apprises |
| modèle simple | fonction composée |
| structure imposée | structure apprise |
| faible dimension | haute capacité + régularisation implicite |

## Insight final (important)

Le deep learning n'est pas :

un nouveau type de modèle probabiliste

mais plutôt :

une manière de paramétrer des variables latentes déterministes extrêmement expressives, optimisées par maximum de vraisemblance.

Si tu veux aller encore plus loin, le niveau supérieur consiste à :
- introduire des variables latentes stochastiques (VAE)
- relier ça aux modèles graphiques
- et reconstruire transformers comme modèles de dépendances conditionnelles factorisées

C'est là que tout devient vraiment unifié côté stats.

---

## Raw source : RNN / LSTM (ChatGPT conversation extract)

On peut garder exactement la même grille "ENSAE / modèles probabilistes à variables latentes", et traduire les LSTM et modèles de séquences dans ce langage.

### 1. Problème de départ : modéliser une séquence

On observe une suite :

$$
X_{1:T} = (X_1, X_2, \dots, X_T)
$$

Objectif typique :
- soit prédire \(X_{t+1}\)
- soit prédire \(Y\) à partir de toute la séquence
- soit modéliser la distribution jointe

$$
P(X_{1:T}) \quad \text{ou} \quad P(X_{t+1} \mid X_{1:t})
$$

### 2. Vision probabiliste classique (référence mentale)

Le problème séquentiel exact serait :

$$
P(X_{1:T}) = \prod_{t=1}^{T} P(X_t \mid X_{1:t-1})
$$

Donc :

une séquence = produit de distributions conditionnelles dépendantes du passé

### 3. Problème fondamental

Le passé \(X_{1:t}\) est :
- de dimension croissante
- impossible à utiliser directement

Donc on introduit une variable latente compressée :

$$
h_t = \text{résumé du passé}
$$

### 4. Idée centrale des RNN

On impose :

$$
h_t = f_\theta(h_{t-1}, X_t)
$$

et :

$$
P(X_{t+1} \mid X_{1:t}) \approx P(X_{t+1} \mid h_t)
$$

Interprétation ENSAE :

On a remplacé \(X_{1:t} \to h_t\), donc \(h_t\) est une statistique suffisante apprise.

### 5. RNN = modèle latent dynamique

On peut reformuler :
- \(h_t\) = variable latente d'état
- dynamique déterministe : \(h_t = f_\theta(h_{t-1}, X_t)\)

Donc :

RNN = modèle d'état caché déterministe

### 6. Limite des RNN classiques

Problème statistique :
- gradient qui disparaît / explose
- mémoire courte

Interprétation :

le modèle n'est pas stable comme estimateur de dépendances longues

### 7. LSTM : correction structurelle

LSTM introduit une mémoire explicite :
- état mémoire \(c_t\)
- état caché \(h_t\)

avec des portes :
- forget gate
- input gate
- output gate

### 8. Lecture probabiliste du LSTM

On peut voir :

(1) mémoire stable :

$$
c_t \approx \text{estimateur lissé du passé}
$$

(2) mise à jour contrôlée :

$$
c_t = c_{t-1} + \text{nouvelle information pondérée}
$$

Interprétation statistique clé :

Le LSTM fait une mise à jour adaptative de statistique suffisante temporelle.

### 9. Vision unifiée

| Objet | Interprétation |
|---|---|
| \(c_t\) | état latent mémoire |
| \(h_t\) | représentation utilisée pour prédire |
| gates | mécanisme de pondération adaptative |

### 10. Pourquoi ça marche mieux que RNN

Les gates permettent :
- de contrôler le "poids du passé"
- d'éviter la perte d'information
- de stabiliser la propagation du gradient

Lecture ENSAE :

Le LSTM approxime un estimateur récursif non linéaire de statistique suffisante avec mémoire contrôlée.

### 11. Lien avec la grille embeddings

- embedding = variable latente statique (objet)
- LSTM = variable latente dynamique (temps)

| Type | Variable latente |
|---|---|
| embedding | \(h(x)\) fixe |
| LSTM | \(h_t\) évolutif |

### 12. Vue probabiliste complète (très importante)

On peut écrire :

$$
P(X_{1:T}) \approx \prod_t P(X_t \mid h_t)
$$

avec :

$$
h_t = f_\theta(h_{t-1}, X_t)
$$

Donc toute la complexité est déplacée dans la dynamique de \(h_t\).

### 13. Insight clé (niveau avancé)

Le LSTM n'est pas un modèle probabiliste explicite, mais une paramétrisation de la mémoire nécessaire pour approximer une factorisation conditionnelle complexe.

### 14. Pourquoi c'est puissant conceptuellement

Parce que tu passes de :
- modèle statistique classique, dépendance explicite (Markov faible)

à :
- deep learning séquentiel, dépendance compressée dans un état latent appris

### 15. Résumé ENSAE

Un LSTM est un modèle de séquence factorisé :

$$
P(X_{1:T}) = \prod_t P(X_t \mid h_t)
$$

où :

$$
h_t = f_\theta(h_{t-1}, X_t)
$$

donc une statistique suffisante dynamique apprise par optimisation.

Prochaine étape naturelle :
- transformer ce même cadre pour les Transformers (attention = modèle de dépendance globale)
- ou comparer LSTM vs attention en termes de structure probabiliste

C'est là que tu obtiens la vision unifiée complète des architectures modernes.
