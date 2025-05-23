# Les files d'attentes M/M/1, M/G/1, G/M/1
---

## 🧠 Objectif

- Simuler un système **M/M/1** pour différentes valeurs de **λ** (de 0.1 à 0.9)
- Mesurer et comparer :
  - Temps de réponse moyen
  - Temps d’attente moyen
  - Taux d’occupation du serveur
- Comparer les résultats **empiriques vs théoriques**
- Visualiser la convergence vers le **régime stationnaire**

---

## 🛠️ Fonctionnalités du code

### 1. **Classe `MM1Simulator`**

Un simulateur événementiel qui :
- Génère des arrivées selon une **loi exponentielle** de paramètre λ
- Génère des services selon une **loi exponentielle** de paramètre μ
- Gère la file avec une structure `deque` (FIFO)
- Applique une **période de chauffe** (`warmup_customers`) pour ignorer les premiers clients (transitoire)
- Calcule les métriques de performance après convergence :
  - `avg_response_time`
  - `avg_waiting_time`
  - `server_utilization`

### 2. **Fonction `run_multiple_simulations()`**

- Exécute la simulation pour **chaque valeur de λ ∈ [0.1, 0.9]**
- Compare les résultats mesurés avec les **valeurs théoriques** :
  - \( E[T] = \frac{1}{\mu - \lambda} \)
  - \( E[W] = \frac{\lambda}{\mu(\mu - \lambda)} \)
  - \( \rho = \frac{\lambda}{\mu} \)
- Retourne un `DataFrame` Pandas contenant toutes les mesures

### 3. **Fonction `plot_results()`**

Génère une figure 2×2 avec :
- Temps de réponse moyen : empirique vs théorique
- Temps d’attente moyen : empirique vs théorique
- Taux d’occupation du serveur
- Erreurs relatives (%)

---
