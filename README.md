# RAG from scratch en Python

Un projet d’apprentissage qui répond à des questions sur un corpus local. Le chargement des fichiers, le découpage, la vectorisation TF-IDF, le calcul de similarité et la construction du prompt sont implémentés avec la bibliothèque standard de Python. Ollama exécute le LLM quand on active la génération.

## Lancer la recherche

Avec Python 3.10 ou supérieur, depuis la racine du projet.

```powershell
python rag.py "Quel est le délai de retour ?"
```

La CLI affiche les extraits classés et leur fichier source. Les deux documents d’exemple sont fictifs.

## Activer la génération

Installe [Ollama](https://ollama.com/download) et démarre-le. Télécharge le modèle d’exemple, puis relance la même question.

```powershell
ollama pull qwen2.5:3b
python rag.py "Quel est le délai de retour ?" --model qwen2.5:3b
```

Le modèle reçoit les meilleurs extraits et peut formuler une réponse comme « Le délai de retour est de 30 jours à compter de la réception du produit [1]. » Les numéros renvoient aux extraits fournis au modèle. La liste des sources affiche les extraits transmis, et les citations générées restent à vérifier.

## Utiliser son corpus

Place tes fichiers UTF-8 `.md` ou `.txt` dans le dossier `data/`, ignoré par Git.

```powershell
python rag.py "Quel est le délai de retour ?" --docs data --top-k 2 --model qwen2.5:3b
```

## Suivre le pipeline dans le code

| Étape | Code | Rôle |
| --- | --- | --- |
| Ingestion et découpage | `load_chunks` | Lit les documents et crée des extraits de 120 mots avec un chevauchement de 25 mots. |
| Indexation | `TfidfIndex` | Pondère les termes fréquents dans un extrait et rares dans le corpus, puis normalise les vecteurs. |
| Recherche | `search` | Classe les extraits par similarité cosinus avec la question. |
| Génération | `generate` | Envoie la question et les extraits numérotés au LLM local. |

L’index est recalculé à chaque lancement. TF-IDF compare les mots et peut manquer un passage pertinent lorsqu’une question emploie des synonymes. Un score positif indique un recouvrement lexical. La génération est ignorée si la recherche ne trouve aucun extrait correspondant.

## Publier sur GitHub

Crée un [dépôt vide](https://github.com/new) appelé `rag-from-scratch`, avec la visibilité de ton choix. Depuis le dossier extrait de l’archive, exécute les commandes suivantes dans PowerShell. Remplace `TON_COMPTE` par ton identifiant GitHub.

```powershell
git init -b main
git add .
git commit -m "Build a RAG from scratch in Python"
git remote add origin https://github.com/TON_COMPTE/rag-from-scratch.git
git push -u origin main
```
