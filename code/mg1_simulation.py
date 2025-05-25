import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from collections import deque
from scipy import stats
import seaborn as sns
from tqdm import tqdm
import warnings


class QueueSimulator:
    """
    Classe abstraite de base pour simuler les systèmes de files d'attente.
    """
    def __init__(self, lambda_rate, mu_rate, num_customers=100000, seed=42):
        """
        Initialise le simulateur avec des validations améliorées.
        """
        # Validation des paramètres
        if lambda_rate <= 0 or mu_rate <= 0:
            raise ValueError("Les taux d'arrivée et de service doivent être positifs")
        
        if num_customers <= 0:
            raise ValueError("Le nombre de clients doit être positif")
            
        # Vérification de stabilité
        if lambda_rate >= mu_rate:
            warnings.warn(f"Système instable: λ={lambda_rate} >= μ={mu_rate}")
            
        self.lambda_rate = float(lambda_rate)
        self.mu_rate = float(mu_rate)
        self.num_customers = int(num_customers)
        self.rng = np.random.RandomState(seed)
        
        # Initialisation des métriques
        self._reset_metrics()
        
    def _reset_metrics(self):
        """Réinitialise les métriques de simulation."""
        self.waiting_times = []  # Utiliser des listes plutôt que des arrays fixes
        self.response_times = []
        self.server_busy_time = 0
        self.last_event_time = 0
        self.total_simulation_time = 0
        self.queue = deque()
        self.server_busy = False
        
    def generate_interarrival_time(self):
        """Génère le temps entre deux arrivées consécutives."""
        raise NotImplementedError("Les classes dérivées doivent implémenter cette méthode")
    
    def generate_service_time(self):
        """Génère le temps de service d'un client."""
        raise NotImplementedError("Les classes dérivées doivent implémenter cette méthode")
    
    def run_simulation(self):
        """
        Exécute la simulation d'événements discrets avec optimisations.
        """
        print(f"Démarrage de la simulation avec {self.num_customers} clients...")
        
        try:
            # Génération vectorisée des temps (plus efficace)
            print("Génération des temps d'inter-arrivée...")
            inter_arrival_times = np.array([self.generate_interarrival_time() 
                                          for _ in range(self.num_customers)])
            
            # Validation des temps générés
            if np.any(inter_arrival_times <= 0):
                raise ValueError("Temps d'inter-arrivée négatifs ou nuls détectés")
            
            print("Génération des temps de service...")
            service_times = np.array([self.generate_service_time() 
                                    for _ in range(self.num_customers)])
            
            if np.any(service_times <= 0):
                raise ValueError("Temps de service négatifs ou nuls détectés")
            
            # Calcul des temps d'arrivée absolus
            arrival_times = np.cumsum(inter_arrival_times)
            
            # Simulation optimisée
            departure_time = 0
            
            for i in tqdm(range(self.num_customers), desc="Simulation en cours"):
                arrival_time = arrival_times[i]
                service_time = service_times[i]
                
                # Calcul du temps d'attente
                if arrival_time >= departure_time:
                    waiting_time = 0
                    departure_time = arrival_time + service_time
                else:
                    waiting_time = departure_time - arrival_time
                    departure_time += service_time
                
                # Enregistrement des métriques
                self.waiting_times.append(waiting_time)
                self.response_times.append(waiting_time + service_time)
                self.server_busy_time += service_time
            
            self.total_simulation_time = departure_time
            self.calculate_metrics()
            
        except Exception as e:
            print(f"Erreur lors de la simulation: {e}")
            raise
            
    def calculate_metrics(self):
        """Calcule les métriques finales avec gestion d'erreurs."""
        try:
            # Conversion en arrays numpy pour les calculs
            waiting_times_array = np.array(self.waiting_times)
            response_times_array = np.array(self.response_times)
            
            self.avg_waiting_time = np.mean(waiting_times_array)
            self.avg_response_time = np.mean(response_times_array)
            
            # Validation de la durée de simulation
            if self.total_simulation_time <= 0:
                raise ValueError("Durée de simulation invalide")
                
            self.utilization = self.server_busy_time / self.total_simulation_time
            self.theoretical_utilization = self.lambda_rate / self.mu_rate
            
            # Métriques additionnelles
            self.max_waiting_time = np.max(waiting_times_array)
            self.max_response_time = np.max(response_times_array)
            self.std_waiting_time = np.std(waiting_times_array)
            self.std_response_time = np.std(response_times_array)
            
            # Percentiles avec gestion d'erreurs
            if len(waiting_times_array) > 0:
                self.waiting_time_95th = np.percentile(waiting_times_array, 95)
                self.response_time_95th = np.percentile(response_times_array, 95)
            else:
                self.waiting_time_95th = 0
                self.response_time_95th = 0
            
            if hasattr(self, 'get_theoretical_metrics'):
                self.theoretical_metrics = self.get_theoretical_metrics()
                
        except Exception as e:
            print(f"Erreur lors du calcul des métriques: {e}")
            raise
    
    def get_results(self):
        """Renvoie les résultats avec validation."""
        return {
            'lambda': self.lambda_rate,
            'mu': self.mu_rate,
            'rho': getattr(self, 'utilization', 0),
            'theoretical_rho': self.theoretical_utilization,
            'avg_waiting_time': getattr(self, 'avg_waiting_time', 0),
            'avg_response_time': getattr(self, 'avg_response_time', 0),
            'max_waiting_time': getattr(self, 'max_waiting_time', 0),
            'max_response_time': getattr(self, 'max_response_time', 0),
            'std_waiting_time': getattr(self, 'std_waiting_time', 0),
            'std_response_time': getattr(self, 'std_response_time', 0),
            'waiting_time_95th': getattr(self, 'waiting_time_95th', 0),
            'response_time_95th': getattr(self, 'response_time_95th', 0),
            'num_customers': self.num_customers,
            'simulation_time': getattr(self, 'total_simulation_time', 0)
        }
    
    def print_summary(self):
        """Affiche un résumé avec gestion d'erreurs."""
        try:
            print(f"\n{'='*50}")
            print(f"RÉSULTATS DE SIMULATION - {self.__class__.__name__}")
            print(f"{'='*50}")
            print(f"Paramètres:")
            print(f"  λ (taux d'arrivée): {self.lambda_rate}")
            print(f"  μ (taux de service): {self.mu_rate}")
            print(f"  Nombre de clients: {self.num_customers}")
            print(f"\nMétriques principales:")
            print(f"  Taux d'occupation (ρ): {getattr(self, 'utilization', 'N/A'):.4f}")
            print(f"  Temps d'attente moyen: {getattr(self, 'avg_waiting_time', 'N/A'):.4f}")
            print(f"  Temps de réponse moyen: {getattr(self, 'avg_response_time', 'N/A'):.4f}")
            print(f"\nMétriques de variabilité:")
            print(f"  Écart-type temps d'attente: {getattr(self, 'std_waiting_time', 'N/A'):.4f}")
            print(f"  Écart-type temps de réponse: {getattr(self, 'std_response_time', 'N/A'):.4f}")
            print(f"  95e percentile temps d'attente: {getattr(self, 'waiting_time_95th', 'N/A'):.4f}")
            print(f"  95e percentile temps de réponse: {getattr(self, 'response_time_95th', 'N/A'):.4f}")
        except Exception as e:
            print(f"Erreur lors de l'affichage: {e}")


class MG1Simulator(QueueSimulator):
    """
    Simulateur M/G/1 amélioré avec validations et optimisations.
    """
    def __init__(self, lambda_rate, mu_rate, shape=2.0, num_customers=100000, seed=42):
        super().__init__(lambda_rate, mu_rate, num_customers, seed)
        
        if shape <= 0:
            raise ValueError("Le paramètre de forme doit être positif")
            
        self.shape = float(shape)
        self.scale = 1.0 / (mu_rate * shape)
        
        if self.scale <= 0:
            raise ValueError("Paramètre d'échelle invalide")
        
    def generate_interarrival_time(self):
        """Génère un temps inter-arrivée avec validation."""
        time_val = self.rng.exponential(scale=1/self.lambda_rate)
        if time_val <= 0:
            return 1e-10  # Valeur minimale pour éviter les problèmes
        return time_val
    
    def generate_service_time(self):
        """Génère un temps de service avec validation."""
        time_val = self.rng.gamma(shape=self.shape, scale=self.scale)
        if time_val <= 0:
            return 1e-10  # Valeur minimale pour éviter les problèmes
        return time_val
    
    def get_theoretical_metrics(self):
        """Calcule les métriques théoriques avec gestion d'erreurs."""
        try:
            rho = self.lambda_rate / self.mu_rate
            
            if rho >= 1:
                return {
                    'rho': rho,
                    'avg_waiting_time': float('inf'),
                    'avg_response_time': float('inf'),
                    'avg_queue_length': float('inf')
                }
            
            # Variance du temps de service (pour la distribution gamma)
            var_service = self.shape * self.scale**2
            
            # Formule de Pollaczek-Khinchine
            numerator = self.lambda_rate * ((1/self.mu_rate)**2 + var_service)
            denominator = 2 * (1 - rho)
            
            if denominator <= 0:
                raise ValueError("Dénominateur invalide dans la formule de Pollaczek-Khinchine")
                
            avg_waiting_time = numerator / denominator
            avg_response_time = avg_waiting_time + 1/self.mu_rate
            avg_queue_length = self.lambda_rate * avg_waiting_time
            
            return {
                'rho': rho,
                'avg_waiting_time': avg_waiting_time,
                'avg_response_time': avg_response_time,
                'avg_queue_length': avg_queue_length
            }
            
        except Exception as e:
            print(f"Erreur dans le calcul des métriques théoriques: {e}")
            return {
                'rho': self.lambda_rate / self.mu_rate,
                'avg_waiting_time': float('nan'),
                'avg_response_time': float('nan'),
                'avg_queue_length': float('nan')
            }


def run_mg1_experiment(lambda_values, mu=1.0, shape=2.0, num_customers=50000, num_runs=3):
    """
    Exécute des expériences M/G/1 avec gestion d'erreurs améliorée.
    """
    results = []
    
    print(f"\nExécution d'expériences M/G/1")
    print(f"Lambda de {min(lambda_values)} à {max(lambda_values)}, μ = {mu}, shape = {shape}")
    
    for lambda_val in tqdm(lambda_values, desc="Simulation M/G/1"):
        if lambda_val >= mu:
            print(f"Attention: λ={lambda_val} >= μ={mu}, résultats peuvent être instables")
            continue  # Skip unstable configurations
            
        run_results = []
        
        for run in range(num_runs):
            try:
                simulator = MG1Simulator(
                    lambda_val, mu, 
                    shape=shape,
                    num_customers=num_customers, 
                    seed=42+run
                )
                simulator.run_simulation()
                run_results.append(simulator.get_results())
                
            except Exception as e:
                print(f"Erreur lors de l'exécution {run} pour λ={lambda_val}: {e}")
                continue
                
        if not run_results:
            print(f"Aucun résultat valide pour λ={lambda_val}")
            continue
            
        # Calculer la moyenne des résultats valides
        avg_result = {
            'lambda': lambda_val,
            'mu': mu,
            'shape': shape,
            'rho': np.mean([r['rho'] for r in run_results]),
            'theoretical_rho': lambda_val / mu,
            'avg_waiting_time': np.mean([r['avg_waiting_time'] for r in run_results]),
            'avg_response_time': np.mean([r['avg_response_time'] for r in run_results]),
            'std_waiting_time': np.std([r['avg_waiting_time'] for r in run_results]),
            'std_response_time': np.std([r['avg_response_time'] for r in run_results]),
            'max_waiting_time': np.mean([r['max_waiting_time'] for r in run_results]),
            'max_response_time': np.mean([r['max_response_time'] for r in run_results]),
            'waiting_time_95th': np.mean([r['waiting_time_95th'] for r in run_results]),
            'response_time_95th': np.mean([r['response_time_95th'] for r in run_results])
        }
        
        results.append(avg_result)
    
    return pd.DataFrame(results)


# Fonction main simplifiée pour test
def main():
    """Fonction principale avec paramètres réduits pour éviter les problèmes de mémoire."""
    print("Test M/G/1 avec paramètres réduits...")
    
    # Paramètres réduits pour éviter les problèmes de mémoire
    lambda_values = np.arange(0.1, 0.9, 0.2)  # Moins de valeurs
    mu = 1.0
    num_customers = 10000  # Nombre réduit de clients
    num_runs = 2  # Moins d'exécutions
    
    # Test simple
    mg1 = MG1Simulator(0.7, 1.0, shape=2.0, num_customers=5000, seed=42)
    mg1.run_simulation()
    mg1.print_summary()
    
    print("Test terminé avec succès!")


if __name__ == "__main__":
    main()