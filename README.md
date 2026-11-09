# ANN-playground

Projet du MSc IA & Data de La Plateforme sur le Perceptron Multicouches (MLP), appliqué à la prédiction du décrochage étudiant.

## Contexte

Le projet suit trois étapes :

1. **Comprendre le MLP** : veille sur son architecture et ses hyperparamètres, puis expériences dans le [TensorFlow Playground](https://playground.tensorflow.org) (nombre de neurones, variables d'entrée, activation, learning rate, vanishing gradient).
2. **L'appliquer à un vrai problème** : prédire si un étudiant va décrocher (*Dropout*), être encore inscrit en retard (*Enrolled*) ou être diplômé (*Graduate*), avec des MLP construits avec **Keras**.
3. **Le coder soi-même** : un MLP écrit uniquement avec **NumPy** (`mlp.py`), comparé à Keras dans des conditions identiques.

**Question métier :** repérer à temps les étudiants qui risquent d'abandonner, pour leur proposer un accompagnement.

## Données

- **Source** : [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), UCI Machine Learning Repository (id 697), licence CC BY 4.0, Instituto Politécnico de Portalegre (Portugal).
- **Taille** : 4 424 étudiants, 36 variables (profil, parcours d'entrée, résultats des 1er et 2e semestres, contexte économique).
- **Cible** : 3 classes déséquilibrées, Graduate 49,9 %, Dropout 32,1 %, Enrolled 17,9 %.
- Une copie locale est dans `data/data.csv` (séparateur `;`).

## Analyse

| Étape | Principaux résultats |
| --- | --- |
| Playground | Gaussian se résout avec 1 neurone, Circle avec 3, XOR avec 4. X1² et X2² résolvent Circle sans couche cachée. ReLU apprend plus vite que Tanh et Sigmoid. Avec 6 couches et Sigmoid, la perte reste bloquée à 0,500 (vanishing gradient), alors que ReLU apprend. |
| EDA | Aucune valeur manquante, aucun doublon. Les variables les plus liées à la réussite sont les unités validées et les notes des semestres (avec 0 unité validée au S2, 84 % de décrochage), les frais de scolarité payés (87 % de décrochage sinon) et la bourse. |
| Préparation | Découpage stratifié 70/15/15, regroupement des modalités rares et one-hot (108 variables), `StandardScaler` ajusté sur le train seulement. |
| Keras | Tous les modèles sur-apprennent en quelques epochs. Le dropout le réduit, la BatchNorm change peu, l'EarlyStopping est le plus efficace. Les poids de classes font passer le rappel d'Enrolled de 0,38 à 0,59. |
| **Meilleur modèle (test)** | [256, 128, 64], dropout 0,3, EarlyStopping et poids de classes : **accuracy 0,773, F1 macro 0,735**, rappel Enrolled 0,66. |
| MLP NumPy | Gradients vérifiés numériquement, XOR résolu à 100 %. À réglages identiques (SGD 0,01), scores et courbes quasi identiques à Keras (accuracy 0,72 sans dropout, 0,75 avec). |

Toutes les figures sont dans `images/` ; celles du Playground sont décrites dans `images/README.md`.

## Outils

Python 3.12, NumPy, pandas, Matplotlib, seaborn, scikit-learn, TensorFlow/Keras 2.21, ucimlrepo, Jupyter.

## Lancer le projet

```bash
py -3.12 -m venv .venv                 # TensorFlow ne supporte pas Python 3.13+ sous Windows
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name ann-playground --display-name "Python 3.12 (ANN-playground)"
jupyter notebook ann_playground.ipynb   # puis Kernel > Restart & Run All
```

Le MLP NumPy s'utilise seul :

```python
from mlp import MLP
modele = MLP([n_variables, 64, 32, 3], learning_rate=0.01, epochs=100, batch_size=32, dropout=0.3)
modele.fit(X_train, y_train, X_val, y_val)
modele.predict(X_test)
```

## Organisation du dépôt

| Fichier | Contenu |
| --- | --- |
| `ann_playground.ipynb` | Le notebook complet, de la veille à la conclusion |
| `mlp.py` | La classe `MLP` en NumPy (propagation, rétropropagation, mini-batchs, dropout) |
| `data/data.csv` | Les données UCI |
| `images/` | Captures du Playground et figures du notebook |
| `slides/` | La présentation |

## Conclusion

Un MLP bien régularisé prédit le devenir des étudiants avec environ 77 % d'accuracy sur des données jamais vues, et repère les décrocheurs avec une précision de 0,90. La classe *Enrolled*, petite et au profil intermédiaire, reste la plus difficile. Les meilleures variables (résultats de semestre) ne sont connues qu'après un ou deux semestres, ce qui limite une détection très précoce. Le MLP écrit à la main reproduit le comportement de Keras, ce qui confirme la compréhension de la propagation, de la rétropropagation et du dropout.

## Bibliographie

- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. *Trends and Applications in Information Systems and Technologies*, Springer. https://doi.org/10.1007/978-3-030-72657-7_16
- Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6).
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers. *ICCV*.
- Srivastava, N. et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. *JMLR*, 15.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Smilkov, D., & Carter, S. *TensorFlow Playground*. https://playground.tensorflow.org
- Documentation Keras : https://keras.io
