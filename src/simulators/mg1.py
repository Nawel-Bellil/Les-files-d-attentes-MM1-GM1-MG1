import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MG1UniformQueueSimulator:
    
    def __init__(self, mu=1.0, num_customers=1000000, warmup_customers=20000):
        # paramètres de base du simulateur
        self.mu = mu  # taux de service
        self.num_customers = num_customers  # nombre total de clients à simuler
        self.warmup_customers = warmup_customers  # clients ignorés pour stabiliser les stats
        self.results = {}
        
    def generate_exponential_interarrival_time(self, lambda_param):
        # génère les temps entre arrivées selon une loi exponentielle
        return np.random.exponential(1.0 / lambda_param)
    
    def generate_uniform_service_time(self):
        # temps de service selon une loi uniforme U(0, 2/μ) pour avoir E[S] = 1/μ
        return np.random.uniform(0, 2.0 / self.mu)
    
    def calculate_theoretical_values(self, lambda_param):
    # Utilization
      rho = lambda_param / self.mu
      server_utilization_theo = rho

      if rho >= 1:
         avg_waiting_time_theo = float('inf')
         avg_response_time_theo = float('inf')
         avg_system_length_theo = float('inf')
      else:
        # Service time follows Uniform(0, 2/μ)
        # Mean service time
        mean_service_time = 1 / self.mu

        # coefficient de variation Cs^2 = 1/3
        C_s_squared = 1 / 3

        # Second moment of service time: E[S^2] = (1/μ)^2 * (1 + Cs^2)
        service_time_second_moment = (mean_service_time ** 2) * (1 + C_s_squared)

        # average waiting time using Pollaczek–Khinchine formula
        avg_waiting_time_theo = (lambda_param * service_time_second_moment) / (2 * (1 - rho))

        # Average response time
        avg_response_time_theo = avg_waiting_time_theo + mean_service_time

        # Little’s Law: L = λ * R
        avg_system_length_theo = lambda_param * avg_response_time_theo

      return {
        'rho': rho,
        'avg_waiting_time_theo': avg_waiting_time_theo,
        'avg_response_time_theo': avg_response_time_theo,
        'server_utilization_theo': server_utilization_theo,
        'avg_system_length_theo': avg_system_length_theo
    }
    
    def simulate(self, lambda_param):        
        # récupère les valeurs théoriques pour comparaison
        theoretical = self.calculate_theoretical_values(lambda_param)
        
        customers = []
        server_busy_until = 0.0  # instant où le serveur sera libre
        
        # variables pour les statistiques après la période de warmup
        total_busy_time_analysis = 0.0
        total_waiting_time_analysis = 0.0
        total_response_time_analysis = 0.0
        customers_served_analysis = 0
        
        # variables pour mesurer E[L] directement
        total_client_time = 0.0  # somme des (nombre_clients × durée)
        current_clients_in_system = 0
        last_event_time = 0.0
        analysis_started = False
        
        # génère tous les temps d'arrivée d'un coup pour optimiser
        arrival_times = [0.0]
        for i in range(1, self.num_customers):
            interarrival = self.generate_exponential_interarrival_time(lambda_param)
            arrival_times.append(arrival_times[-1] + interarrival)

        # crée une liste d'événements (arrivées et départs) pour mesurer E[L]
        events = []
        service_end_time = 0.0 
        
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
            service_time = self.generate_uniform_service_time()
            service_end_time = service_start_time + service_time
            server_busy_until = service_end_time  # met à jour l'occupation du serveur
            response_time = service_end_time - arrival_time  # temps total dans le système
            
            # ajoute les événements pour mesurer E[L]
            events.append(('arrival', arrival_time, customer_id))
            events.append(('departure', service_end_time, customer_id))
            
            # accumule les stats seulement après le warmup
            if customer_id > self.warmup_customers:
                total_busy_time_analysis += service_time
                total_waiting_time_analysis += waiting_time
                total_response_time_analysis += response_time
                customers_served_analysis += 1

            if customer_id == self.warmup_customers:
                analysis_start_time = arrival_time

        # trie les événements par temps pour mesurer E[L] correctement
        events.sort(key=lambda x: x[1])  # trie par temps
        
        # mesure E[L] directement en suivant l'évolution du système
        current_clients_in_system = 0
        last_event_time = 0.0
        total_client_time = 0.0
        analysis_started = False
        analysis_start_time = 0.0
        
        # premier passage: compter les clients présents au début de l'analyse
        clients_at_analysis_start = 0
        for event_type, event_time, customer_id in events:
            if customer_id <= self.warmup_customers:
                # traite tous les événements de warmup pour avoir le bon état initial
                if event_type == 'arrival':
                    current_clients_in_system += 1
                else:  # departure
                    current_clients_in_system -= 1
            else:
                # premier événement après warmup = début de l'analyse
                if not analysis_started:
                    analysis_started = True
                    analysis_start_time = event_time
                    last_event_time = event_time
                    clients_at_analysis_start = current_clients_in_system
                break
        
        # deuxième passage: mesure de E[L] avec le bon état initial
        current_clients_in_system = clients_at_analysis_start
        analysis_started = False
        
        for event_type, event_time, customer_id in events:
            # commence l'analyse après le warmup
            if not analysis_started and customer_id > self.warmup_customers:
                analysis_started = True
                analysis_start_time = event_time
                last_event_time = event_time
            
            # si l'analyse a commencé, accumule le temps
            if analysis_started:
                duration = event_time - last_event_time
                total_client_time += current_clients_in_system * duration
                last_event_time = event_time
                
                # met à jour le nombre de clients dans le système
                if event_type == 'arrival':
                    current_clients_in_system += 1
                else:  # departure
                    current_clients_in_system -= 1
        
        # finalise la mesure de E[L]
        total_analysis_time = service_end_time - analysis_start_time
        
        # calcule les métriques empiriques finales
        avg_waiting_time_emp = total_waiting_time_analysis / customers_served_analysis
        avg_response_time_emp = total_response_time_analysis / customers_served_analysis
        server_utilization_emp = total_busy_time_analysis / total_analysis_time
        
        # mesure directe de E[L] à partir de l'intégration temporelle
        avg_system_length_emp = total_client_time / total_analysis_time
        
        
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
            'avg_system_length_emp': avg_system_length_emp,  # mesure directe
            'customers_served': customers_served_analysis,
            'simulation_time': total_analysis_time
        }
    
    def run_multiple_experiments(self, lambda_values, num_replications=1):
        # lance plusieurs expériences pour différentes valeurs de lambda
        all_results = []
        
        for lambda_val in lambda_values:
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
        return pd.DataFrame(all_results)

def create_performance_plots(results_df):
    # crée les graphiques de performance
    avg_results = results_df.copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('analyse de performance - file M/G/1 (services uniformes)', 
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
            label='empirique (mesure directe)', linewidth=2, markersize=8)
    ax.plot(avg_results['lambda'], avg_results['avg_system_length_theo'], 'r--', 
            label='théorique', linewidth=2)
    ax.set_xlabel('taux d\'arrivée λ')
    ax.set_ylabel('longueur du système E[L]')
    ax.set_title('longueur du système moyenne')
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('mg1.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # configuration des paramètres de simulation
    lambda_values = np.arange(0.1, 1.0, 0.1)  # teste différents taux d'arrivée
    mu = 1.0  
    num_customers = 1000000  
    warmup_customers = 20000  
    num_replications = 3  
    
    # crée le simulateur avec les paramètres choisis
    simulator = MG1UniformQueueSimulator(mu=mu, num_customers=num_customers, 
                                        warmup_customers=warmup_customers)
    
    # lance toutes les simulations
    results_df = simulator.run_multiple_experiments(lambda_values, num_replications)
    
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
    filename = 'mg1.csv'
    csv_df.to_csv(filename, index=False)
    
    # génère les graphiques de performance
    create_performance_plots(results_df)

if __name__ == "__main__":
    main()