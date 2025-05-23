import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import namedtuple
import time

# Structure pour stocker les événements
Event = namedtuple('Event', ['time', 'event_type', 'client_id'])

class GM1Simulator:
    """
    Simulateur pour le modèle de file d'attente G/M/1
    G/M/1 : Arrivées générales (loi uniforme), Service exponentiel, 1 serveur
    """
    
    def __init__(self, lambda_rate, mu_rate, num_clients=1000000):
        """
        Initialise le simulateur G/M/1
        
        Args:
            lambda_rate (float): Taux d'arrivée moyen λ
            mu_rate (float): Taux de service μ (exponentiel)
            num_clients (int): Nombre de clients à simuler
        """
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.num_clients = num_clients
        self.rho = lambda_rate / mu_rate  # Taux d'occupation théorique
        
        # Paramètres pour la loi uniforme des arrivées
        # Pour avoir une moyenne de 1/λ, on utilise U(a,b) avec (a+b)/2 = 1/λ
        # On choisit a = 0 et b = 2/λ pour simplifier
        self.uniform_min = 0
        self.uniform_max = 2.0 / lambda_rate
        
        # Vérification de stabilité
        if self.rho >= 1:
            raise ValueError(f"Système instable: ρ = {self.rho:.3f} >= 1")
    
    def generate_uniform_interarrival(self):
        """Génère le temps entre arrivées selon une loi uniforme"""
        return random.uniform(self.uniform_min, self.uniform_max)
    
    def generate_exponential_service(self):
        """Génère le temps de service selon une loi exponentielle"""
        return -np.log(1 - random.random()) / self.mu_rate
    
    def simulate(self):
        """
        Exécute la simulation G/M/1
        
        Returns:
            dict: Statistiques de performance
        """
        print(f"🚀 Début simulation G/M/1 (λ={self.lambda_rate}, μ={self.mu_rate}, ρ={self.rho:.3f})")
        start_time = time.time()
        
        # Variables de simulation
        current_time = 0.0
        server_busy = False
        server_free_time = 0.0
        queue = []
        
        # Statistiques
        total_response_time = 0.0
        total_waiting_time = 0.0
        total_service_time = 0.0
        completed_clients = 0
        
        # Temps d'arrivée du prochain client
        next_arrival_time = self.generate_uniform_interarrival()
        next_service_completion = float('inf')
        
        clients_arrived = 0
        
        while completed_clients < self.num_clients:
            # Déterminer le prochain événement
            if next_arrival_time <= next_service_completion:
                # Événement : Arrivée d'un client
                current_time = next_arrival_time
                clients_arrived += 1
                
                client_arrival_time = current_time
                
                if not server_busy:
                    # Serveur libre : service immédiat
                    service_time = self.generate_exponential_service()
                    server_busy = True
                    next_service_completion = current_time + service_time
                    
                    # Statistiques pour ce client
                    waiting_time = 0.0
                    response_time = service_time
                    
                    total_waiting_time += waiting_time
                    total_response_time += response_time
                    total_service_time += service_time
                    completed_clients += 1
                else:
                    # Serveur occupé : client entre dans la file
                    queue.append(client_arrival_time)
                
                # Programmer la prochaine arrivée
                if clients_arrived < self.num_clients * 2:  # Marge de sécurité
                    next_arrival_time = current_time + self.generate_uniform_interarrival()
                else:
                    next_arrival_time = float('inf')
            
            else:
                # Événement : Fin de service
                current_time = next_service_completion
                server_busy = False
                
                if queue:
                    # Il y a des clients en attente
                    client_arrival_time = queue.pop(0)
                    service_time = self.generate_exponential_service()
                    
                    # Calcul des temps pour ce client
                    waiting_time = current_time - client_arrival_time
                    response_time = waiting_time + service_time
                    
                    # Mise à jour des statistiques
                    total_waiting_time += waiting_time
                    total_response_time += response_time
                    total_service_time += service_time
                    completed_clients += 1
                    
                    # Le serveur reprend le service
                    server_busy = True
                    next_service_completion = current_time + service_time
                else:
                    # Pas de clients en attente : serveur devient libre
                    server_free_time += (next_arrival_time - current_time) if next_arrival_time != float('inf') else 0
                    next_service_completion = float('inf')
            
            # Affichage du progrès
            if completed_clients % 100000 == 0 and completed_clients > 0:
                progress = (completed_clients / self.num_clients) * 100
                print(f"   Progrès: {progress:.1f}% ({completed_clients:,} clients traités)")
        
        # Calcul des métriques finales
        simulation_time = time.time() - start_time
        
        mean_response_time = total_response_time / completed_clients
        mean_waiting_time = total_waiting_time / completed_clients
        mean_service_time = total_service_time / completed_clients
        
        # Taux d'occupation du serveur
        total_busy_time = total_service_time
        total_simulation_time = current_time
        server_utilization = total_busy_time / total_simulation_time
        
        results = {
            'lambda': self.lambda_rate,
            'mu': self.mu_rate,
            'rho': self.rho,
            'mean_response_time': mean_response_time,
            'mean_waiting_time': mean_waiting_time,
            'mean_service_time': mean_service_time,
            'server_utilization': server_utilization,
            'completed_clients': completed_clients,
            'simulation_time': simulation_time,
            'total_simulation_time': total_simulation_time,
            'interarrival_distribution': 'Uniform',
            'service_distribution': 'Exponential',
            'uniform_min': self.uniform_min,
            'uniform_max': self.uniform_max
        }
        
        print(f"✅ Simulation G/M/1 terminée en {simulation_time:.2f}s")
        print(f"   • Clients traités: {completed_clients:,}")
        print(f"   • Temps de réponse moyen: {mean_response_time:.4f}")
        print(f"   • Temps d'attente moyen: {mean_waiting_time:.4f}")
        print(f"   • Taux d'occupation: {server_utilization:.4f}")
        
        return results

def run_gm1_experiments():
    """
    Exécute les expériences G/M/1 pour différentes valeurs de λ
    """
    print("=" * 80)
    print("SIMULATION G/M/1 - ARRIVÉES UNIFORMES, SERVICE EXPONENTIEL")
    print("=" * 80)
    
    # Paramètres de l'expérience
    lambda_values = np.arange(0.1, 0.9, 0.1)
    mu = 1.0
    num_clients = 1000000
    num_repetitions = 3  # Répétitions pour stabilité
    
    all_results = []
    
    for lambda_rate in lambda_values:
        print(f"\n{'='*20} λ = {lambda_rate:.1f} {'='*20}")
        
        repetition_results = []
        
        # Répéter l'expérience plusieurs fois
        for rep in range(num_repetitions):
            print(f"\n--- Répétition {rep + 1}/{num_repetitions} ---")
            
            simulator = GM1Simulator(lambda_rate, mu, num_clients)
            result = simulator.simulate()
            repetition_results.append(result)
        
        # Calculer les moyennes sur les répétitions
        avg_result = {
            'lambda': lambda_rate,
            'mu': mu,
            'rho': lambda_rate / mu,
            'mean_response_time': np.mean([r['mean_response_time'] for r in repetition_results]),
            'mean_waiting_time': np.mean([r['mean_waiting_time'] for r in repetition_results]),
            'mean_service_time': np.mean([r['mean_service_time'] for r in repetition_results]),
            'server_utilization': np.mean([r['server_utilization'] for r in repetition_results]),
            'num_clients': num_clients,
            'num_repetitions': num_repetitions,
            'response_time_std': np.std([r['mean_response_time'] for r in repetition_results]),
            'waiting_time_std': np.std([r['mean_waiting_time'] for r in repetition_results]),
            'interarrival_distribution': 'Uniform',
            'service_distribution': 'Exponential'
        }
        
        all_results.append(avg_result)
        
        print(f"\n📊 RÉSULTATS MOYENS pour λ = {lambda_rate:.1f}:")
        print(f"   • Temps de réponse: {avg_result['mean_response_time']:.4f} ± {avg_result['response_time_std']:.4f}")
        print(f"   • Temps d'attente: {avg_result['mean_waiting_time']:.4f} ± {avg_result['waiting_time_std']:.4f}")
        print(f"   • Taux d'occupation: {avg_result['server_utilization']:.4f}")
    
    return all_results

def save_results_and_create_graphs(results):
    """
    Sauvegarde les résultats et crée les graphiques
    """
    # Conversion en DataFrame
    df = pd.DataFrame(results)
    
    # Sauvegarde CSV
    csv_filename = 'gm1_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Résultats sauvegardés dans {csv_filename}")
    
    # Création des graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Résultats de simulation G/M/1 (Arrivées Uniformes, Service Exponentiel)', 
                 fontsize=14, fontweight='bold')
    
    # Graphique 1: Temps de réponse vs λ
    axes[0, 0].plot(df['lambda'], df['mean_response_time'], 'o-', color='orange', linewidth=2, markersize=6)
    axes[0, 0].fill_between(df['lambda'], 
                           df['mean_response_time'] - df['response_time_std'],
                           df['mean_response_time'] + df['response_time_std'],
                           alpha=0.3, color='orange')
    axes[0, 0].set_xlabel('Taux d\'arrivée λ')
    axes[0, 0].set_ylabel('Temps de réponse moyen')
    axes[0, 0].set_title('Temps de réponse moyen vs λ')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Graphique 2: Temps d'attente vs λ
    axes[0, 1].plot(df['lambda'], df['mean_waiting_time'], 's-', color='red', linewidth=2, markersize=6)
    axes[0, 1].fill_between(df['lambda'], 
                           df['mean_waiting_time'] - df['waiting_time_std'],
                           df['mean_waiting_time'] + df['waiting_time_std'],
                           alpha=0.3, color='red')
    axes[0, 1].set_xlabel('Taux d\'arrivée λ')
    axes[0, 1].set_ylabel('Temps d\'attente moyen')
    axes[0, 1].set_title('Temps d\'attente moyen vs λ')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Graphique 3: Taux d'occupation vs λ
    axes[1, 0].plot(df['lambda'], df['server_utilization'], '^-', color='green', linewidth=2, markersize=6)
    axes[1, 0].plot(df['lambda'], df['rho'], '--', color='gray', alpha=0.7, label='ρ théorique')
    axes[1, 0].set_xlabel('Taux d\'arrivée λ')
    axes[1, 0].set_ylabel('Taux d\'occupation')
    axes[1, 0].set_title('Taux d\'occupation vs λ')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Graphique 4: Comparaison théorique (approximation)
    axes[1, 1].plot(df['lambda'], df['mean_response_time'], 'o-', color='orange', label='G/M/1 simulé')
    
    # Approximation théorique pour G/M/1 avec arrivées uniformes
    # Pour loi uniforme: coefficient de variation CA² = 1/3
    ca_squared = 1/3  # Coefficient de variation au carré pour loi uniforme
    theoretical_response = []
    for lam in df['lambda']:
        rho = lam / 1.0
        # Formule approximative G/M/1: W = (ρ²(1+CA²))/(2(1-ρ)) + 1/μ
        w_approx = (rho**2 * (1 + ca_squared)) / (2 * (1 - rho)) + 1.0
        theoretical_response.append(w_approx)
    
    axes[1, 1].plot(df['lambda'], theoretical_response, '--', color='blue', label='G/M/1 théorique')
    axes[1, 1].set_xlabel('Taux d\'arrivée λ')
    axes[1, 1].set_ylabel('Temps de réponse moyen')
    axes[1, 1].set_title('Comparaison théorique vs simulé')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde des graphiques
    graph_filename = 'gm1_graphs.png'
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Graphiques sauvegardés dans {graph_filename}")
    
    plt.show()
    
    # Affichage du tableau récapitulatif
    print(f"\n📋 TABLEAU RÉCAPITULATIF G/M/1:")
    print("=" * 80)
    print(f"{'λ':<6} {'ρ':<6} {'Temps rép.':<12} {'Temps att.':<12} {'Taux occ.':<10} {'Écart-type':<10}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['lambda']:<6.1f} {row['rho']:<6.2f} {row['mean_response_time']:<12.4f} "
              f"{row['mean_waiting_time']:<12.4f} {row['server_utilization']:<10.4f} "
              f"{row['response_time_std']:<10.4f}")
    print("=" * 80)

# Exécution principale
if __name__ == "__main__":
    # Lancement des expériences G/M/1
    results = run_gm1_experiments()
    
    # Sauvegarde et visualisation
    save_results_and_create_graphs(results)
    
    print(f"\n🎉 SIMULATION G/M/1 TERMINÉE!")
    print(f"📁 Fichiers générés:")
    print(f"   • gm1_results.csv")
    print(f"   • gm1_graphs.png")