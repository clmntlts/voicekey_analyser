# Analyseur Voice Key

Outil graphique (Tkinter) pour détecter automatiquement ou manuellement l'onset/offset de parole dans des enregistrements WAV — pensé pour des expériences de psycholinguistique / temps de réaction vocal (voice key), en complément ou remplacement d'un dispositif matériel.

## Fonctionnalités

- Détection adaptative (seuil statistique + relatif au pic, façon VOAT) ou seuils manuels.
- Correction manuelle par glisser-déposer des marqueurs onset/offset, dans les deux modes, avec persistance par fichier.
- Liste de fichiers avec statut (non revu / personnalisé / revu), navigation au clic ou au clavier.
- Découpage de fichier, filtrage passe-haut, lecture audio intégrée.
- Traitement en lot avec export CSV.
- Sauvegarde automatique de session (par dossier de données) — reprenez votre travail là où vous l'avez laissé.

## Installation

**Windows, sans Python :** téléchargez `VoiceKeyAnalyzer.exe` depuis la [dernière version publiée](https://github.com/clmntlts/voicekey_analyser/releases/latest) et lancez-le directement.

**Depuis les sources :**
```
pip install -r requirements.txt
python analyse_voicekey.py
```

Pour reconstruire l'exécutable : `pyinstaller VoiceKeyAnalyzer.spec`.

## Documentation

Voir le [guide d'utilisation complet](GUIDE_UTILISATION.md) pour un tutoriel détaillé de chaque fonctionnalité.

## Structure du projet

- `analyse_voicekey.py` — interface graphique.
- `voice_onset_core.py` — algorithme de détection (indépendant de l'interface, testable isolément).
- `ui_widgets.py` — composants d'interface réutilisables (infobulles, sections repliables).
- `test_voice_onset_core.py` — suite de tests (`pytest test_voice_onset_core.py`).
