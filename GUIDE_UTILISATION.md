# Guide d'utilisation — Analyseur Voice Key

Ce guide explique comment utiliser l'Analyseur Voice Key pour détecter (automatiquement ou manuellement) l'onset et l'offset de parole dans des enregistrements WAV, typiquement pour des expériences de psycholinguistique / temps de réaction vocal.

## Table des matières

1. [Installation](#1-installation)
2. [Lancement](#2-lancement)
3. [Vue d'ensemble de l'interface](#3-vue-densemble-de-linterface)
4. [Ouvrir un dossier et naviguer entre les fichiers](#4-ouvrir-un-dossier-et-naviguer-entre-les-fichiers)
5. [Mode Adaptatif (VOAT) vs Mode Manuel](#5-mode-adaptatif-voat-vs-mode-manuel)
6. [Paramètres de détection](#6-paramètres-de-détection)
7. [Corriger manuellement un onset/offset](#7-corriger-manuellement-un-onsetoffset)
8. [Filtrage](#8-filtrage)
9. [Découpage d'un fichier](#9-découpage-dun-fichier)
10. [Écoute audio](#10-écoute-audio)
11. [Sauvegarder, traiter en lot, exporter](#11-sauvegarder-traiter-en-lot-exporter)
12. [Sauvegarde automatique de la session](#12-sauvegarde-automatique-de-la-session)
13. [Comprendre le fichier CSV exporté](#13-comprendre-le-fichier-csv-exporté)
14. [Limites à connaître](#14-limites-à-connaître)
15. [Dépannage](#15-dépannage)

---

## 1. Installation

**Option A — exécutable pré-compilé (Windows, sans Python) :** utilisez `dist/VoiceKeyAnalyzer.exe` s'il est présent, ou reconstruisez-le avec `pyinstaller VoiceKeyAnalyzer.spec`.

**Option B — depuis les sources :**
```
pip install -r requirements.txt
```
Dépendances : numpy, scipy, pandas, matplotlib, PyAudio (lecture audio, optionnelle), sv-ttk (thème visuel).

## 2. Lancement

```
python analyse_voicekey.py
```

## 3. Vue d'ensemble de l'interface

La fenêtre est divisée en deux zones :
- **Colonne de gauche** (défilante à la molette) : sélection de fichiers, contrôles de détection, résultats, boutons d'action.
- **Colonne de droite** : le graphique — en haut la forme d'onde brute, en bas l'enveloppe RMS avec les seuils et les marqueurs d'onset (vert) / offset (bleu foncé).

## 4. Ouvrir un dossier et naviguer entre les fichiers

Cliquez **"Parcourir..."** en haut du panneau "Fichiers" et sélectionnez le dossier contenant vos fichiers `.wav`. La liste en dessous affiche tous les fichiers trouvés avec leur statut :

| Statut | Signification |
|---|---|
| Non revu | Le fichier n'a jamais été ouvert dans cette session |
| Personnalisé | Des paramètres ont été modifiés pour ce fichier, mais il n'a pas encore été sauvegardé |
| Revu | Le fichier a été sauvegardé (bouton "Sauvegarder ce fichier" ou "Traiter tous") |

**Navigation** : cliquez directement sur un fichier dans la liste, utilisez les boutons **"◄◄ Précédent"** / **"Suivant ►►"**, ou les flèches gauche/droite du clavier (désactivées si le focus est sur un champ texte, un curseur ou la liste elle-même, pour ne pas interférer avec leur usage normal).

Le bouton **"🔄 Reset"** réinitialise tous les paramètres du fichier courant (y compris une éventuelle correction manuelle) aux valeurs par défaut.

## 5. Mode Adaptatif (VOAT) vs Mode Manuel

La case à cocher **"Seuils adaptatifs (VOAT recommandé)"** bascule entre les deux modes :

- **Adaptatif (recommandé par défaut)** : le seuil de détection est calculé automatiquement à partir du bruit de fond et du pic d'amplitude du fichier (algorithme de type VOAT). Convient à la majorité des enregistrements sans réglage manuel.
- **Manuel** : vous fixez vous-même un seuil d'amplitude fixe pour l'onset et un pour l'offset. Utile pour les fichiers atypiques (bruit inhabituel, niveau très différent) où l'algorithme adaptatif se trompe systématiquement.

Dans les deux modes, les marqueurs affichés peuvent être **corrigés à la main** — voir section 7.

## 6. Paramètres de détection

Chaque étiquette de paramètre possède une infobulle (survolez-la avec la souris) expliquant son rôle. Résumé :

**Toujours visibles :**
- **nSD** (mode adaptatif) — sensibilité au bruit, en écarts-types au-dessus du niveau de bruit de fond. Plus élevé = détection plus stricte (moins de faux positifs, mais risque de rater un onset faible).
- **minTH** (mode adaptatif) — seuil plancher en pourcentage du pic d'amplitude du fichier, pour éviter les faux déclenchements dans un enregistrement très calme.
- **Durée minimale onset / offset** — durée pendant laquelle le signal doit rester au-dessus (onset) ou en dessous (offset) du seuil pour confirmer la détection. Évite qu'un simple clic ou bruit parasite ne soit pris pour de la parole. *(Ce réglage est partagé entre les modes adaptatif et manuel — le modifier dans un mode le modifie aussi pour l'autre.)*
- **Seuil onset / offset** (mode manuel) — l'amplitude fixe utilisée comme seuil.

**Sous "Paramètres avancés"** (repliés par défaut — cliquez pour les déplier) :
- **Fenêtre bruit initial** — portion du début de l'enregistrement utilisée pour estimer le niveau de bruit de fond ; doit correspondre à une zone réellement silencieuse.
- **Délai avant recherche** — temps ignoré avant de commencer la recherche d'un onset, pour éviter de détecter un bip de consigne ou un bruit de démarrage.
- **Longueur trame RMS / Pas de trame RMS** — paramètres techniques du calcul de l'enveloppe d'amplitude ; les valeurs par défaut conviennent à la plupart des enregistrements de parole.

## 7. Corriger manuellement un onset/offset

Si la détection (automatique ou manuelle) place le marqueur au mauvais endroit, **cliquez-glissez directement le trait vert (onset) ou bleu foncé (offset)** sur le graphique du bas — cela fonctionne dans les deux modes.

Une fois relâché, cette correction est **enregistrée pour ce fichier** et reste appliquée même si vous changez ensuite un curseur, changez de mode, ou rechargez le fichier plus tard. Un message **"✓ Onset/offset corrigé(s) manuellement"** apparaît sous les résultats pour vous le rappeler.

Pour revenir à la détection automatique/aux seuils actuels : cliquez **"Annuler la correction manuelle"**.

> Note : appliquer ou annuler une découpe (section 9) efface automatiquement toute correction manuelle en cours, car la découpe change les repères temporels du fichier.

## 8. Filtrage

La case **"Filtre passe-haut"** applique un filtre qui supprime les basses fréquences (bourdonnement secteur, bruit de manipulation du micro) avant analyse — généralement à laisser activé. La fréquence de coupure par défaut (80 Hz) convient à la plupart des voix ; les réglages de fenêtre RMS associés sont dans "Paramètres avancés".

## 9. Découpage d'un fichier

Si l'enregistrement contient une portion à ignorer (avant/après la réponse), utilisez **"Découpage du fichier"** : indiquez un début et/ou une fin en millisecondes (0 pour "jusqu'à la fin"), puis **"Appliquer la découpe"**. **"Annuler la découpe"** restaure le fichier d'origine. **"Sauvegarder fichier découpé"** exporte la portion découpée dans un nouveau fichier `.wav`.

## 10. Écoute audio

Le bouton **"▶ Jouer"** lit le fichier courant, avec un curseur rouge qui se déplace sur les deux graphiques. Si PortAudio n'est pas installé sur la machine, ce bouton est simplement désactivé (l'analyse reste utilisable normalement).

## 11. Sauvegarder, traiter en lot, exporter

- **"Sauvegarder ce fichier"** enregistre le résultat du fichier actuellement affiché.
- **"Traiter tous"** relance l'analyse sur tous les fichiers du dossier et sauvegarde chaque résultat automatiquement. Les fichiers déjà personnalisés conservent leurs réglages ; les autres utilisent les valeurs par défaut. Si un fichier pose problème (trop court, illisible), un avertissement s'affiche avec une option **"Ignorer les avertissements restants pour ce traitement"** pour ne pas avoir à cliquer sur chaque fichier problématique.
- **"Exporter CSV"** écrit tous les résultats sauvegardés dans un fichier `.csv` de votre choix.

## 12. Sauvegarde automatique de la session

Chaque fois que vous sauvegardez un fichier, terminez un traitement en lot, réinitialisez un fichier, ou corrigez manuellement un marqueur, les réglages sont **automatiquement enregistrés** dans un fichier `voicekey_session.json` à l'intérieur du dossier ouvert (également sauvegardé à la fermeture de l'application). En rouvrant ce même dossier plus tard, tous vos réglages, corrections et résultats précédents sont restaurés automatiquement — rien à faire de spécial.

Ce fichier voyage avec le dossier de données : si vous déplacez ou partagez le dossier complet, votre travail suit.

## 13. Comprendre le fichier CSV exporté

Chaque ligne correspond à un fichier traité. Colonnes clés :
- `onset_time_ms` / `offset_time_ms` / `duration_ms` — les valeurs finales retenues (y compris une éventuelle correction manuelle).
- `detection_mode` — `adaptive` ou `manual`.
- `manually_corrected_onset` / `manually_corrected_offset` — `True` si le marqueur correspondant a été corrigé à la main, pour transparence lors de vos analyses.
- `was_cut` — si une découpe a été appliquée au fichier.
- Les colonnes restantes détaillent les paramètres utilisés (nSD, minTH, filtre, etc.) pour la reproductibilité.

## 14. Limites à connaître

La détection est basée sur l'amplitude du signal (enveloppe RMS), pas sur une analyse phonétique. Comme tout "voice key" de ce type :
- Les segments à faible énergie (consonnes sourdes, nasales) peuvent être détectés en retard ou manqués.
- Un bruit non-vocal suffisamment fort (toux, raclement de gorge, bruit de fond) peut être pris pour un onset.

C'est précisément pour cela que la vérification visuelle et la correction manuelle (section 7) sont importantes sur les essais incertains, plutôt que de faire confiance aveuglément à la détection automatique sur un grand jeu de données.

## 15. Dépannage

- **Le bouton "Jouer" est grisé** : PortAudio/pyaudio n'est pas installé ; l'analyse fonctionne normalement sans lecture audio.
- **Message "fichier plus court que la fenêtre de bruit"** : réduisez la "Fenêtre bruit initial" dans les paramètres avancés, ou vérifiez que le fichier n'est pas tronqué.
- **Le CSV exporté n'ouvre pas correctement dans Excel (colonnes mal séparées)** : Excel attend parfois un séparateur `;` selon la langue régionale — utilisez "Données > Convertir" ou ouvrez le fichier depuis Excel via "Importer des données" en spécifiant la virgule comme séparateur.
- **Un fichier corrompu bloque le traitement en lot** : un message d'avertissement l'indique et le fichier est ignoré automatiquement ; le traitement continue avec les fichiers suivants.
