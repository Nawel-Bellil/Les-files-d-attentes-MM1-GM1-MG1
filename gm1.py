"""
Simulation de File d'Attente G/M/1 avec Loi Uniforme
====================================================

G/M/1 : Arrivées suivent une loi Uniforme, Service suit une loi exponentielle
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class Customer:
    # structure pour stocker les infos de chaque client
    id: int
    arrival_time: float
    service_start_time: float
    service_end_time: float
    waiting_time: float
    response_time: float

class GM1UniformQueueSimulator:
    
    def __init__(self, mu=1.0, num_customers=1000000, warmup_customers=20000):
        # paramètres de base du simulateur
        self.mu = mu  # taux de service
        self.num_customers = num_customers  # nombre total de clients à simuler
        self.warmup_customers = warmup_customers  # clients ignorés pour stabiliser les stats
        self.results = {}
        
    def generate_uniform_interarrival_time(self, lambda_param):
        # génère les temps entre arrivées selon une loi uniforme
        # on utilise U(0, 2/λ) pour avoir E[X] = 1/λ
        max_value = 2.0 / lambda_param
        return np.random.uniform(0, max_value)
    
    def generate_exponential_service_time(self):
        # temps de service selon une loi exponentielle classique
        return np.random.exponential(1.0 / self.mu)
    
    def calculate_theoretical_values(self, lambda_param):
        # calcule les valeurs théoriques selon les formules G/M/1
        rho = lambda_param / self.mu  # taux d'utilisation
        
        # coefficient de variation au carré pour la distribution uniforme
        c_a_squared = 1.0/3.0  # propriété mathématique de la loi uniforme
        
        if rho < 1.0:
            # on peut utiliser la formule de Pollaczek-Khintchine pour G/M/1
            avg_waiting_time_theo = (rho**2 / (2 * (1 - rho))) * (1 + c_a_squared)
            avg_response_time_theo = avg_waiting_time_theo + (1.0 / self.mu)
            server_utilization_theo = rho
            
            # longueur moyenne du système (clients en attente + en service)
            avg_system_length_theo = (rho**2 * (1 + c_a_squared) / (2 * (1 - rho))) + rho
        else:
            # système instable quand rho >= 1
            avg_waiting_time_theo = float('inf')
            avg_response_time_theo = float('inf')
            server_utilization_theo = 1.0
            avg_system_length_theo = float('inf')
        
        return {
            'rho': rho,
            'avg_waiting_time_theo': avg_waiting_time_theo,
            'avg_response_time_theo': avg_response_time_theo,
            'server_utilization_theo': server_utilization_theo,
            'avg_system_length_theo': avg_system_length_theo
        }
    
    def simulate(self, lambda_param):
        print(f"simulation G/M/1 - λ={lambda_param:.2f}, μ={self.mu:.2f}")
        
        # récupère les valeurs théoriques pour comparaison
        theoretical = self.calculate_theoretical_values(lambda_param)
        
        customers = []
        server_busy_until = 0.0  # instant où le serveur sera libre
        
        # variables pour les statistiques après la période de warmup
        total_busy_time_analysis = 0.0
        total_waiting_time_analysis = 0.0
        total_response_time_analysis = 0.0
        customers_served_analysis = 0
        
        # génère tous les temps d'arrivée d'un coup pour optimiser
        arrival_times = [0.0]
        for i in range(1, self.num_customers):
            interarrival = self.generate_uniform_interarrival_time(lambda_param)
            arrival_times.append(arrival_times[-1] + interarrival)
        
        # traite chaque client individuellement
        for i in range(self.num_customers):
            customer_id = i + 1
            arrival_time = arrival_times[i]
            
            # détermine quand le service commence
            if arrival_time >= server_busy_until:
                # serveur libre, pas d'attente
                service_start_time = arrival_time
                waiting_time = 0.0
            else:
                # serveur occupé, il faut attendre
                service_start_time = server_busy_until
                waiting_time = service_start_time - arrival_time
            
            # génère le temps de service et calcule la fin
            service_time = self.generate_exponential_service_time()
            service_end_time = service_start_time + service_time
            server_busy_until = service_end_time  # met à jour l'occupation du serveur
            response_time = service_end_time - arrival_time  # temps total dans le système
            
            # accumule les stats seulement après le warmup
            if customer_id > self.warmup_customers:
                total_busy_time_analysis += service_time
                total_waiting_time_analysis += waiting_time
                total_response_time_analysis += response_time
                customers_served_analysis += 1
            
            # stocke les infos du client
            customer = Customer(
                id=customer_id,
                arrival_time=arrival_time,
                service_start_time=service_start_time,
                service_end_time=service_end_time,
                waiting_time=waiting_time,
                response_time=response_time
            )
            customers.append(customer)
            
            # affiche le progrès de temps en temps
            if customer_id % 100000 == 0:
                print(f"traité {customer_id:,} clients...")
        
        # calcule la période d'analyse (après warmup)
        analysis_start_time = customers[self.warmup_customers].arrival_time
        analysis_end_time = customers[-1].service_end_time
        total_analysis_time = analysis_end_time - analysis_start_time
        
        # calcule les métriques empiriques finales
        avg_waiting_time_emp = total_waiting_time_analysis / customers_served_analysis
        avg_response_time_emp = total_response_time_analysis / customers_served_analysis
        server_utilization_emp = total_busy_time_analysis / total_analysis_time
        
        # utilise la loi de Little pour calculer E[L] empiriquement
        avg_system_length_emp = lambda_param * avg_response_time_emp
        
        return {
            'lambda': lambda_param,
            'mu': self.mu,
            'rho': server_utilization_emp,
            'avg_response_time_emp': avg_response_time_emp,
            'avg_response_time_theo': theoretical['avg_response_time_theo'],
            'avg_waiting_time_emp': avg_waiting_time_emp,
            'avg_waiting_time_theo': theoretical['avg_waiting_time_theo'],
            'server_utilization_emp': server_utilization_emp,
            'server_utilization_theo': theoretical['server_utilization_theo'],
            'avg_system_length_theo': theoretical['avg_system_length_theo'],
            'avg_system_length_emp': avg_system_length_emp,
            'customers_served': customers_served_analysis,
            'simulation_time': total_analysis_time
        }
    
    def run_multiple_experiments(self, lambda_values, num_replications=1):
        # lance plusieurs expériences pour différentes valeurs de lambda
        all_results = []
        
        print(f"début des expériences G/M/1")
        print(f"clients total: {self.num_customers:,}, warmup: {self.warmup_customers:,}")
        
        for lambda_val in lambda_values:
            print(f"\nexpériences pour λ = {lambda_val:.2f}")
            
            start_total = time.time()
            replication_results = []
            
            # fait plusieurs réplications pour chaque lambda si demandé
            for rep in range(num_replications):
                result = self.simulate(lambda_val)
                result['replication'] = rep + 1
                replication_results.append(result)
            
            # calcule la moyenne des réplications pour ce lambda
            if num_replications > 1:
                # convertit en DataFrame pour faciliter les calculs
                rep_df = pd.DataFrame(replication_results)
                
                # moyenne des métriques empiriques
                mean_result = {
                    'lambda': lambda_val,
                    'mu': self.mu,
                    'rho': rep_df['rho'].mean(),
                    'avg_response_time_emp': rep_df['avg_response_time_emp'].mean(),
                    'avg_response_time_theo': rep_df['avg_response_time_theo'].iloc[0],  # les valeurs théoriques sont identiques
                    'avg_waiting_time_emp': rep_df['avg_waiting_time_emp'].mean(),
                    'avg_waiting_time_theo': rep_df['avg_waiting_time_theo'].iloc[0],
                    'server_utilization_emp': rep_df['server_utilization_emp'].mean(),
                    'server_utilization_theo': rep_df['server_utilization_theo'].iloc[0],
                    'avg_system_length_theo': rep_df['avg_system_length_theo'].iloc[0],
                    'avg_system_length_emp': rep_df['avg_system_length_emp'].mean(),
                    'customers_served': rep_df['customers_served'].mean(),
                    'simulation_time': rep_df['simulation_time'].mean()
                }
                all_results.append(mean_result)
            else:
                # une seule réplication, on ajoute directement
                all_results.append(replication_results[0])
            
            end_total = time.time()
            print(f"temps total: {end_total - start_total:.2f}s")
        
        return pd.DataFrame(all_results)

def create_performance_plots(results_df):
    # crée les graphiques de performance
    avg_results = results_df.copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('analyse de performance - file G/M/1 (arrivées uniformes)', 
                 fontsize=16, fontweight='bold')
    
    # graphique des temps de réponse
    ax = axes[0, 0]
    ax.plot(avg_results['lambda'], avg_results['avg_response_time_emp'], 'bo-', 
            label='empirique', linewidth=2, markersize=8)
    ax.plot(avg_results['lambda'], avg_results['avg_response_time_theo'], 'r--', 
            label='théorique', linewidth=2)
    ax.set_xlabel('taux d\'arrivée λ')
    ax.set_ylabel('temps de réponse E[R]')
    ax.set_title('temps de réponse moyen')
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    # graphique des temps d'attente
    ax = axes[1, 0]
    ax.plot(avg_results['lambda'], avg_results['avg_waiting_time_emp'], 'bo-', 
            label='empirique', linewidth=2, markersize=8)
    ax.plot(avg_results['lambda'], avg_results['avg_waiting_time_theo'], 'r--', 
            label='théorique', linewidth=2)
    ax.set_xlabel('taux d\'arrivée λ')
    ax.set_ylabel('temps d\'attente E[W]')
    ax.set_title('temps d\'attente moyen')
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    # graphique de l'utilisation du serveur
    ax = axes[0, 1]
    ax.plot(avg_results['lambda'], avg_results['server_utilization_emp'], 'bo-', 
            label='empirique', linewidth=2, markersize=8)
    ax.plot(avg_results['lambda'], avg_results['server_utilization_theo'], 'r--', 
            label='théorique', linewidth=2)
    ax.set_xlabel('taux d\'arrivée λ')
    ax.set_ylabel('utilisation ρ')
    ax.set_title('utilisation du serveur')
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    # graphique de la longueur du système
    ax = axes[1, 1]
    ax.plot(avg_results['lambda'], avg_results['avg_system_length_emp'], 'bo-', 
            label='empirique', linewidth=2, markersize=8)
    ax.plot(avg_results['lambda'], avg_results['avg_system_length_theo'], 'r--', 
            label='théorique', linewidth=2)
    ax.set_xlabel('taux d\'arrivée λ')
    ax.set_ylabel('longueur du système E[L]')
    ax.set_title('longueur du système moyenne')
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('gm1.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("simulation de file d'attente G/M/1")
    print("arrivées: loi uniforme U(0, 2/λ)")  
    print("service: loi exponentielle")
    
    # configuration des paramètres de simulation
    lambda_values = np.arange(0.1, 1.0, 0.1)  # teste différents taux d'arrivée
    mu = 1.0  # taux de service fixé à 1
    num_customers = 1000000  # assez de clients pour avoir des stats fiables
    warmup_customers = 20000  # ignore les premiers clients pour stabiliser
    num_replications = 3  # répète chaque expérience pour réduire la variance
    
    print(f"μ = {mu}, clients: {num_customers:,}, warmup: {warmup_customers:,}")
    print(f"réplications par λ: {num_replications}")
    
    # crée le simulateur avec les paramètres choisis
    simulator = GM1UniformQueueSimulator(mu=mu, num_customers=num_customers, 
                                        warmup_customers=warmup_customers)
    
    # lance toutes les simulations
    start_time = time.time()
    results_df = simulator.run_multiple_experiments(lambda_values, num_replications)
    end_time = time.time()
    
    print(f"\nsimulation terminée en {end_time - start_time:.2f} secondes")
    
    # définit les colonnes à sauvegarder dans le CSV
    csv_columns = [
        'lambda', 'mu', 'rho', 
        'avg_response_time_emp', 'avg_response_time_theo',
        'avg_waiting_time_emp', 'avg_waiting_time_theo',
        'server_utilization_emp', 'server_utilization_theo',
        'avg_system_length_theo', 'avg_system_length_emp',
        'customers_served', 'simulation_time'
    ]
    
    # sauvegarde les résultats dans un fichier CSV
    csv_df = results_df[csv_columns].copy()
    filename = 'gm1.csv'
    csv_df.to_csv(filename, index=False)
    print(f"résultats sauvegardés dans '{filename}'")
    print(f"nombre de lignes dans le CSV: {len(csv_df)} (une par valeur de λ)")
    
    print(f"\néchantillon des résultats:")
    print(csv_df.round(4))
    
    # vérifie que les lois de Little sont respectées
    print(f"\nvérification des lois de Little:")
    for idx, row in csv_df.iterrows():
        l_little = row['lambda'] * row['avg_response_time_emp']
        print(f"λ={row['lambda']:.1f}: E[L]={row['avg_system_length_emp']:.4f} vs λ*E[R]={l_little:.4f}")
    
    # génère les graphiques de performance
    create_performance_plots(results_df)
    
    print(f"\nsimulation G/M/1 complétée")

if __name__ == "__main__":
    # fixe la graine pour des résultats reproductibles
    np.random.seed(42)
    main()