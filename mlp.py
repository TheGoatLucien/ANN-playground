"""Perceptron multicouches (MLP) écrit à la main avec NumPy.

Le réseau est entièrement connecté : des couches cachées avec activation ReLU,
une couche de sortie softmax, et une perte d'entropie croisée. Il apprend par
descente de gradient stochastique (SGD) sur des mini-batchs, avec un dropout
optionnel sur les couches cachées.

Conventions (les mêmes partout dans le fichier) :
    - X a la forme (n exemples, n variables) ;
    - la couche l a des poids W[l] de forme (entrées, neurones)
      et des biais b[l] de forme (1, neurones) ;
    - propagation : Z = A_précédent @ W + b, puis A = activation(Z).

Exemple :
    >>> from mlp import MLP
    >>> modele = MLP([X_train.shape[1], 64, 32, 3], learning_rate=0.01, epochs=100)
    >>> modele.fit(X_train, y_train, X_val, y_val)
    >>> modele.predict(X_val)
"""

import numpy as np


def relu(Z):
    """ReLU : garde les valeurs positives, remplace les négatives par 0."""
    return np.maximum(0.0, Z)


def relu_derivee(Z):
    """Dérivée de ReLU : 1 là où Z > 0, 0 ailleurs."""
    return (Z > 0).astype(Z.dtype)


def softmax(Z):
    """Softmax ligne par ligne : transforme des scores en probabilités qui somment à 1.

    On soustrait le maximum de chaque ligne avant l'exponentielle : le résultat
    est identique, mais exp ne déborde plus quand les scores sont grands.
    """
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class MLP:
    """Perceptron multicouches pour la classification multiclasse.

    Paramètres
    ----------
    tailles_couches : list[int]
        Nombre de neurones de chaque couche, entrée et sortie comprises.
        Exemple : [108, 64, 32, 3] pour 108 variables, deux couches cachées
        de 64 et 32 neurones, et 3 classes.
    learning_rate : float
        Pas de la descente de gradient.
    epochs : int
        Nombre de passages complets sur les données d'entraînement.
    batch_size : int
        Nombre d'exemples par mini-batch.
    dropout : float
        Proportion de neurones cachés éteints au hasard à chaque batch
        pendant l'entraînement (0 = pas de dropout).
    seed : int
        Graine du hasard (poids initiaux, mélange, masques de dropout),
        pour des résultats reproductibles.
    verbose : bool
        Affiche la perte toutes les 10 epochs si True.
    """

    def __init__(self, tailles_couches, learning_rate=0.01, epochs=100,
                 batch_size=32, dropout=0.0, seed=42, verbose=False):
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout doit être compris entre 0 (inclus) et 1 (exclu)")
        self.tailles = list(tailles_couches)
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        self._initialiser_poids()

    # ------------------------------------------------------------------ poids
    def _initialiser_poids(self):
        """Initialisation de He : W ~ N(0, 2 / n_entrées), biais à 0.

        Des poids aléatoires (et non nuls) permettent à chaque neurone de se
        spécialiser ; l'échelle sqrt(2 / n_entrées) est adaptée à ReLU et garde
        les activations à une taille raisonnable d'une couche à l'autre.
        """
        self.W, self.b = [], []
        for n_in, n_out in zip(self.tailles[:-1], self.tailles[1:]):
            self.W.append(self.rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in))
            self.b.append(np.zeros((1, n_out)))

    def nombre_parametres(self):
        """Nombre total de poids et de biais du réseau."""
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # ------------------------------------------------------------ propagation
    def _propagation(self, X, entrainement=False):
        """Calcule la sortie du réseau et garde en mémoire les Z, les A et les masques.

        En entraînement avec dropout, chaque activation cachée est multipliée par
        un masque aléatoire de 0 et de 1, divisé par (1 - taux) pour que la valeur
        moyenne reste la même (« inverted dropout »). En prédiction, pas de dropout.
        """
        A = X
        self.Z_cache, self.A_cache, self.masques = [], [X], []
        derniere = len(self.W) - 1
        for l in range(len(self.W)):
            Z = A @ self.W[l] + self.b[l]
            if l == derniere:
                A = softmax(Z)
                masque = None
            else:
                A = relu(Z)
                masque = None
                if entrainement and self.dropout > 0:
                    masque = (self.rng.random(A.shape) >= self.dropout) / (1.0 - self.dropout)
                    A = A * masque
            self.Z_cache.append(Z)
            self.A_cache.append(A)
            self.masques.append(masque)
        return A

    def predict_proba(self, X):
        """Probabilités de chaque classe, matrice (n, nombre de classes)."""
        return self._propagation(np.asarray(X, dtype=float), entrainement=False)

    def predict(self, X):
        """Classe prédite (indice de la plus grande probabilité) pour chaque exemple."""
        return self.predict_proba(X).argmax(axis=1)

    # ------------------------------------------------------------------ perte
    @staticmethod
    def _perte(Y_hat, Y):
        """Entropie croisée moyenne ; 1e-12 évite log(0)."""
        return float(-np.mean(np.sum(Y * np.log(Y_hat + 1e-12), axis=1)))

    def _one_hot(self, y):
        """Transforme des classes 0, 1, 2… en vecteurs one-hot."""
        return np.eye(self.tailles[-1])[np.asarray(y, dtype=int)]

    # -------------------------------------------------------- apprentissage
    def _retropropagation(self, Y):
        """Calcule les gradients dW et db de chaque couche, de la sortie vers l'entrée.

        Formules (m exemples dans le batch) :
            dZ_sortie = (Ŷ − Y) / m
            dW[l] = A[l−1]ᵀ @ dZ[l]        db[l] = somme de dZ[l] sur les lignes
            dA[l−1] = dZ[l] @ W[l]ᵀ        dZ[l−1] = dA[l−1] × masque × relu′(Z[l−1])
        Tous les gradients sont calculés avant de modifier les poids.
        """
        m = Y.shape[0]
        n_couches = len(self.W)
        dW, db = [None] * n_couches, [None] * n_couches
        dZ = (self.A_cache[-1] - Y) / m
        for l in range(n_couches - 1, -1, -1):
            dW[l] = self.A_cache[l].T @ dZ
            db[l] = dZ.sum(axis=0, keepdims=True)
            if l > 0:
                dA = dZ @ self.W[l].T
                if self.masques[l - 1] is not None:
                    dA = dA * self.masques[l - 1]
                dZ = dA * relu_derivee(self.Z_cache[l - 1])
        return dW, db

    def _mise_a_jour(self, dW, db):
        """Descente de gradient : on recule d'un pas lr dans le sens du gradient."""
        for l in range(len(self.W)):
            self.W[l] -= self.lr * dW[l]
            self.b[l] -= self.lr * db[l]

    def fit(self, X, y, X_val=None, y_val=None):
        """Entraîne le réseau par mini-batchs et remplit self.history.

        À chaque epoch : on mélange les exemples, on les découpe en mini-batchs,
        et pour chaque batch on fait propagation, rétropropagation et mise à jour.
        À la fin de l'epoch, on mesure la perte et l'accuracy (train et validation).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        Y = self._one_hot(y)
        n = X.shape[0]
        for epoch in range(self.epochs):
            ordre = self.rng.permutation(n)
            for debut in range(0, n, self.batch_size):
                lot = ordre[debut:debut + self.batch_size]
                self._propagation(X[lot], entrainement=True)
                dW, db = self._retropropagation(Y[lot])
                self._mise_a_jour(dW, db)

            P = self.predict_proba(X)
            self.history["loss"].append(self._perte(P, Y))
            self.history["accuracy"].append(float((P.argmax(axis=1) == y).mean()))
            if X_val is not None and y_val is not None:
                P_val = self.predict_proba(X_val)
                self.history["val_loss"].append(self._perte(P_val, self._one_hot(y_val)))
                self.history["val_accuracy"].append(float((P_val.argmax(axis=1) == np.asarray(y_val)).mean()))
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"epoch {epoch + 1:4d}  loss {self.history['loss'][-1]:.4f}")
        return self

    def score(self, X, y):
        """Accuracy sur (X, y)."""
        return float((self.predict(X) == np.asarray(y)).mean())
