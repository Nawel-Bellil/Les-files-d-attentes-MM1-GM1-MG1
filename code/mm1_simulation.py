import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import time

class MM1Simulator:
    """
    Simulateur pour file d'attente M/M/1
    """
    def __init__(self, lambda_rate, mu_rate, max_customers=1000000):
        self.lambda_rate = lambda_rate  # Taux d'arrivée
        self.mu_rate = mu_rate         # Taux de service
        self.max_customers = max_customers
        self.rho = lambda_rate / mu_rate  # Taux d'occupation
        
        # Vérification de stabilité
        if self.rho >= 1:
            raise ValueError(f"Système instable: ρ = {self.rho:.3f} >= 1")
        
        # Statistiques à collecter
        self.reset_stats()
    
    def reset_stats(self):
        """Réinitialise les statistiques"""
        self.customers_served = 0
        self.total_response_time = 0
        self.total_waiting_time = 0
        self.total_service_time = 0
        self.server_busy_time = 0
        self.simulation_time = 0
        self.queue_lengths = []
        self.response_times = []
        self.waiting_times = []
        
    def generate_interarrival_time(self):
        """Génère un temps inter-arrivée selon loi exponentielle"""
        return np.random.exponential(1.0 / self.lambda_rate)
    
    def generate_service_time(self):
        """Génère un temps de service selon loi exponentielle"""
        return np.random.exponential(1.0 / self.mu_rate)
    
    def simulate(self, warmup_customers=10000, collect_stats_interval=1000):
        """
        Simule la file d'attente M/M/1
        
        Args:
            warmup_customers: Nombre de clients pour la période de chauffe
            collect_stats_interval: Intervalle pour collecter les statistiques
        """
        print(f"Simulation M/M/1: λ={self.lambda_rate}, μ={self.mu_rate}, ρ={self.rho:.3f}")
        
        # Variables de simulation
        current_time = 0
        next_arrival_time = self.generate_interarrival_time()
        next_departure_time = float('inf')  # Pas de client en service initialement
        
        queue = deque()  # File d'attente
        server_busy = False
        server_start_time = 0
        
        # Statistiques en temps réel
        customers_processed = 0
        stats_collection = []
        last_stats_time = 0
        
        while customers_processed < self.max_customers:
            # Déterminer le prochain événement
            if next_arrival_time < next_departure_time:
                # Événement: Arrivée
                current_time = next_arrival_time
                
                # Nouveau client arrive
                customer = {
                    'id': customers_processed + 1,
                    'arrival_time': current_time,
                    'service_time': self.generate_service_time()
                }
                
                if not server_busy:
                    # Serveur libre: service immédiat
                    server_busy = True
                    server_start_time = current_time
                    customer['start_service_time'] = current_time
                    next_departure_time = current_time + customer['service_time']
                else:
                    # Serveur occupé: ajout à la file
                    queue.append(customer)
                
                # Planifier la prochaine arrivée
                next_arrival_time = current_time + self.generate_interarrival_time()
                
            else:
                # Événement: Départ
                current_time = next_departure_time
                customers_processed += 1
                
                # Client qui termine son service
                if customers_processed > warmup_customers:
                    # Calculer les métriques pour ce client
                    service_time = current_time - server_start_time
                    
                    # Si on a les informations du client qui part
                    if hasattr(self, 'current_customer'):
                        response_time = current_time - self.current_customer['arrival_time']
                        waiting_time = self.current_customer['start_service_time'] - self.current_customer['arrival_time']
                        
                        self.total_response_time += response_time
                        self.total_waiting_time += waiting_time
                        self.customers_served += 1
                        
                        self.response_times.append(response_time)
                        self.waiting_times.append(waiting_time)
                
                # Mise à jour temps d'occupation serveur
                self.server_busy_time += current_time - server_start_time
                
                # Vérifier s'il y a des clients en attente
                if queue:
                    # Servir le prochain client
                    next_customer = queue.popleft()
                    next_customer['start_service_time'] = current_time
                    self.current_customer = next_customer
                    server_start_time = current_time
                    next_departure_time = current_time + next_customer['service_time']
                else:
                    # Serveur devient libre
                    server_busy = False
                    next_departure_time = float('inf')
                
                # Collecter statistiques périodiquement
                if customers_processed % collect_stats_interval == 0 and customers_processed > warmup_customers:
                    queue_length = len(queue) + (1 if server_busy else 0)
                    self.queue_lengths.append(queue_length)
                    
                    # Statistiques instantanées
                    if self.customers_served > 0:
                        avg_response = self.total_response_time / self.customers_served
                        avg_waiting = self.total_waiting_time / self.customers_served
                        utilization = self.rho  # Approximation
                        
                        stats_collection.append({
                            'time': current_time,
                            'customers_served': self.customers_served,
                            'avg_response_time': avg_response,
                            'avg_waiting_time': avg_waiting,
                            'queue_length': queue_length,
                            'utilization': utilization
                        })
        
        self.simulation_time = current_time
        self.stats_history = stats_collection
        
        # Calcul des métriques finales
        self.calculate_final_metrics()
        
    def calculate_final_metrics(self):
        """Calcule les métriques finales de performance"""
        if self.customers_served == 0:
            return
            
        # Métriques empiriques
        self.avg_response_time = self.total_response_time / self.customers_served
        self.avg_waiting_time = self.total_waiting_time / self.customers_served
        self.server_utilization = self.server_busy_time / self.simulation_time
        self.avg_queue_length = np.mean(self.queue_lengths) if self.queue_lengths else 0
        
        # Métriques théoriques (pour comparaison)
        self.theoretical_response_time = 1 / (self.mu_rate - self.lambda_rate)
        self.theoretical_waiting_time = self.lambda_rate / (self.mu_rate * (self.mu_rate - self.lambda_rate))
        self.theoretical_utilization = self.rho
        self.theoretical_queue_length = self.lambda_rate / (self.mu_rate - self.lambda_rate)
    
    def print_results(self):
        """Affiche les résultats de la simulation"""
        print(f"\n=== Résultats de simulation M/M/1 ===")
        print(f"Paramètres: λ={self.lambda_rate}, μ={self.mu_rate}, ρ={self.rho:.3f}")
        print(f"Clients servis: {self.customers_served:,}")
        print(f"Temps de simulation: {self.simulation_time:.2f}")
        
        print(f"\n--- Métriques empiriques vs théoriques ---")
        print(f"Temps de réponse moyen: {self.avg_response_time:.4f} vs {self.theoretical_response_time:.4f}")
        print(f"Temps d'attente moyen: {self.avg_waiting_time:.4f} vs {self.theoretical_waiting_time:.4f}")
        print(f"Taux d'occupation: {self.server_utilization:.4f} vs {self.theoretical_utilization:.4f}")
        print(f"Longueur moyenne file: {self.avg_queue_length:.4f} vs {self.theoretical_queue_length:.4f}")
        
        # Calcul des erreurs relatives
        response_error = abs(self.avg_response_time - self.theoretical_response_time) / self.theoretical_response_time * 100
        waiting_error = abs(self.avg_waiting_time - self.theoretical_waiting_time) / self.theoretical_waiting_time * 100
        util_error = abs(self.server_utilization - self.theoretical_utilization) / self.theoretical_utilization * 100
        
        print(f"\n--- Erreurs relatives ---")
        print(f"Temps de réponse: {response_error:.2f}%")
        print(f"Temps d'attente: {waiting_error:.2f}%")
        print(f"Taux d'occupation: {util_error:.2f}%")

def run_multiple_simulations():
    """
    Lance plusieurs simulations avec différentes valeurs de λ
    """
    mu = 1.0  # Taux de service fixe
    lambda_values = np.arange(0.1, 0.95, 0.1)  # De 0.1 à 0.9
    
    results = []
    
    print("Lancement des simulations multiples...")
    for i, lam in enumerate(lambda_values):
        print(f"\nSimulation {i+1}/{len(lambda_values)}: λ={lam:.1f}")
        
        # Créer et lancer la simulation
        simulator = MM1Simulator(lam, mu, max_customers=100000)  # Réduit pour les tests
        
        start_time = time.time()
        simulator.simulate(warmup_customers=10000)
        end_time = time.time()
        
        simulator.print_results()
        print(f"Temps d'exécution: {end_time - start_time:.2f}s")
        
        # Sauvegarder les résultats
        results.append({
            'lambda': lam,
            'mu': mu,
            'rho': simulator.rho,
            'avg_response_time_emp': simulator.avg_response_time,
            'avg_response_time_theo': simulator.theoretical_response_time,
            'avg_waiting_time_emp': simulator.avg_waiting_time,
            'avg_waiting_time_theo': simulator.theoretical_waiting_time,
            'server_utilization_emp': simulator.server_utilization,
            'server_utilization_theo': simulator.theoretical_utilization,
            'customers_served': simulator.customers_served,
            'simulation_time': simulator.simulation_time
        })
    
    return pd.DataFrame(results)

def plot_results(results_df):
    """
    Crée les graphiques de comparaison avec un espacement optimal
    """
    # Augmenter la taille de la figure et ajuster l'espacement
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Titre principal avec plus d'espace
    fig.suptitle('Simulation M/M/1: Comparaison Empirique vs Théorique', 
                 fontsize=12, y=1)
    
    # Graphique 1: Temps de réponse moyen
    axes[0,0].plot(results_df['rho'], results_df['avg_response_time_emp'], 
                   'bo-', label='Empirique', markersize=6)
    axes[0,0].plot(results_df['rho'], results_df['avg_response_time_theo'], 
                   'r--', label='Théorique', linewidth=2)
    axes[0,0].set_xlabel('Taux d\'occupation (ρ)')
    axes[0,0].set_ylabel('Temps de réponse moyen')
    axes[0,0].set_title('Temps de réponse moyen', pad=12)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Graphique 2: Temps d'attente moyen
    axes[0,1].plot(results_df['rho'], results_df['avg_waiting_time_emp'], 
                   'go-', label='Empirique', markersize=6)
    axes[0,1].plot(results_df['rho'], results_df['avg_waiting_time_theo'], 
                   'r--', label='Théorique', linewidth=2)
    axes[0,1].set_xlabel('Taux d\'occupation (ρ)')
    axes[0,1].set_ylabel('Temps d\'attente moyen')
    axes[0,1].set_title('Temps d\'attente moyen', pad=12)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Graphique 3: Taux d'occupation du serveur
    axes[1,0].plot(results_df['rho'], results_df['server_utilization_emp'], 
                   'mo-', label='Empirique', markersize=6)
    axes[1,0].plot(results_df['rho'], results_df['server_utilization_theo'], 
                   'r--', label='Théorique', linewidth=2)
    axes[1,0].set_xlabel('Taux d\'occupation théorique (ρ)')
    axes[1,0].set_ylabel('Taux d\'occupation mesuré')
    axes[1,0].set_title('Taux d\'occupation du serveur', pad=12)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Graphique 4: Erreur relative
    response_error = abs(results_df['avg_response_time_emp'] - results_df['avg_response_time_theo']) / results_df['avg_response_time_theo'] * 100
    waiting_error = abs(results_df['avg_waiting_time_emp'] - results_df['avg_waiting_time_theo']) / results_df['avg_waiting_time_theo'] * 100
    
    axes[1,1].plot(results_df['rho'], response_error, 'bo-', label='Temps de réponse', markersize=6)
    axes[1,1].plot(results_df['rho'], waiting_error, 'go-', label='Temps d\'attente', markersize=6)
    axes[1,1].set_xlabel('Taux d\'occupation (ρ)')
    axes[1,1].set_ylabel('Erreur relative (%)')
    axes[1,1].set_title('Erreur relative (Empirique vs Théorique)', pad=12)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Ajustement de l'espacement entre les sous-graphiques
    plt.subplots_adjust(
        left=0.08,      # Marge gauche
        bottom=0.08,    # Marge bas
        right=0.95,     # Marge droite
        top=0.90,       # Marge haut (laisse de la place pour le titre principal)
        wspace=0.30,    # Espacement horizontal entre les graphiques
        hspace=0.35     # Espacement vertical entre les graphiques
    )
    
    plt.show()
    return fig

# Exemple d'utilisation
if __name__ == "__main__":
    # Test simple avec une seule simulation
    print("=== Test simple ===")
    simulator = MM1Simulator(lambda_rate=0.7, mu_rate=1.0, max_customers=50000)
    simulator.simulate()
    simulator.print_results()
    
    # Simulations multiples
    print("\n=== Simulations multiples ===")
    results_df = run_multiple_simulations()
    
    # Affichage du tableau récapitulatif
    print("\n=== Tableau récapitulatif ===")
    print(results_df[['lambda', 'rho', 'avg_response_time_emp', 'avg_response_time_theo', 
                     'avg_waiting_time_emp', 'avg_waiting_time_theo', 'server_utilization_emp']].round(4))
    
    # Création des graphiques
    fig = plot_results(results_df)
    
    # Sauvegarde des résultats
    results_df.to_csv('mm1_simulation_results.csv', index=False)
    print("\nRésultats sauvegardés dans 'mm1_simulation_results.csv'")