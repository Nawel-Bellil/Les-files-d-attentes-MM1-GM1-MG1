import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

class MM1ConvergenceAnalyzer:
    """
    Analyseur de convergence vers l'état stationnaire pour M/M/1
    """
    def __init__(self, lambda_rate, mu_rate):
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.rho = lambda_rate / mu_rate
        
        if self.rho >= 1:
            raise ValueError(f"Système instable: ρ = {self.rho:.3f} >= 1")
    
    def simulate_with_convergence_tracking(self, max_customers=1000000, window_size=10000):
        """
        Simule en trackant la convergence vers l'état stationnaire
        """
        print(f"Analyse de convergence: λ={self.lambda_rate}, μ={self.mu_rate}, ρ={self.rho:.3f}")
        
        # Variables de simulation
        current_time = 0
        next_arrival_time = np.random.exponential(1.0 / self.lambda_rate)
        next_departure_time = float('inf')
        
        queue = deque()
        server_busy = False
        server_start_time = 0
        
        # Tracking de convergence
        customers_processed = 0
        response_times_window = deque(maxlen=window_size)
        waiting_times_window = deque(maxlen=window_size)
        
        # Historique des moyennes mobiles
        convergence_history = []
        customers_history = []
        
        # Métriques théoriques
        theo_response = 1 / (self.mu_rate - self.lambda_rate)
        theo_waiting = self.lambda_rate / (self.mu_rate * (self.mu_rate - self.lambda_rate))
        
        current_customer = None
        
        while customers_processed < max_customers:
            # Déterminer le prochain événement
            if next_arrival_time < next_departure_time:
                # Événement: Arrivée
                current_time = next_arrival_time
                
                customer = {
                    'arrival_time': current_time,
                    'service_time': np.random.exponential(1.0 / self.mu_rate)
                }
                
                if not server_busy:
                    server_busy = True
                    server_start_time = current_time
                    customer['start_service_time'] = current_time
                    current_customer = customer
                    next_departure_time = current_time + customer['service_time']
                else:
                    queue.append(customer)
                
                next_arrival_time = current_time + np.random.exponential(1.0 / self.lambda_rate)
                
            else:
                # Événement: Départ
                current_time = next_departure_time
                customers_processed += 1
                
                if current_customer is not None:
                    # Calculer métriques pour ce client
                    response_time = current_time - current_customer['arrival_time']
                    waiting_time = current_customer['start_service_time'] - current_customer['arrival_time']
                    
                    response_times_window.append(response_time)
                    waiting_times_window.append(waiting_time)
                
                # Servir le prochain client ou libérer le serveur
                if queue:
                    current_customer = queue.popleft()
                    current_customer['start_service_time'] = current_time
                    server_start_time = current_time
                    next_departure_time = current_time + current_customer['service_time']
                else:
                    server_busy = False
                    current_customer = None
                    next_departure_time = float('inf')
                
                # Analyser convergence tous les 1000 clients
                if customers_processed % 1000 == 0 and len(response_times_window) > 100:
                    avg_response = np.mean(response_times_window)
                    avg_waiting = np.mean(waiting_times_window)
                    
                    # Calculer erreurs relatives
                    response_error = abs(avg_response - theo_response) / theo_response * 100
                    waiting_error = abs(avg_waiting - theo_waiting) / theo_waiting * 100
                    
                    convergence_history.append({
                        'customers': customers_processed,
                        'avg_response': avg_response,
                        'avg_waiting': avg_waiting,
                        'response_error': response_error,
                        'waiting_error': waiting_error,
                        'std_response': np.std(response_times_window),
                        'std_waiting': np.std(waiting_times_window)
                    })
                    
                    customers_history.append(customers_processed)
                    
                    # Critère de convergence: erreur < 1% pendant 5 mesures consécutives
                    if len(convergence_history) >= 5:
                        recent_errors = [h['response_error'] for h in convergence_history[-5:]]
                        if all(error < 1.0 for error in recent_errors):
                            print(f"Convergence atteinte après {customers_processed:,} clients")
                            break
        
        return pd.DataFrame(convergence_history), customers_history
    
    def plot_convergence(self, convergence_df, customers_history):
        """
        Graphiques de convergence
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Convergence vers l\'état stationnaire M/M/1 (λ={self.lambda_rate}, μ={self.mu_rate})', fontsize=14)
        
        # Métriques théoriques
        theo_response = 1 / (self.mu_rate - self.lambda_rate)
        theo_waiting = self.lambda_rate / (self.mu_rate * (self.mu_rate - self.lambda_rate))
        
        # Graphique 1: Convergence temps de réponse
        axes[0,0].plot(convergence_df['customers'], convergence_df['avg_response'], 'b-', linewidth=2, label='Empirique')
        axes[0,0].axhline(y=theo_response, color='r', linestyle='--', linewidth=2, label='Théorique')
        axes[0,0].set_xlabel('Nombre de clients servis')
        axes[0,0].set_ylabel('Temps de réponse moyen')
        axes[0,0].set_title('Convergence - Temps de réponse')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Graphique 2: Convergence temps d'attente
        axes[0,1].plot(convergence_df['customers'], convergence_df['avg_waiting'], 'g-', linewidth=2, label='Empirique')
        axes[0,1].axhline(y=theo_waiting, color='r', linestyle='--', linewidth=2, label='Théorique')
        axes[0,1].set_xlabel('Nombre de clients servis')
        axes[0,1].set_ylabel('Temps d\'attente moyen')
        axes[0,1].set_title('Convergence - Temps d\'attente')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Graphique 3: Erreur relative
        axes[1,0].plot(convergence_df['customers'], convergence_df['response_error'], 'b-', linewidth=2, label='Temps de réponse')
        axes[1,0].plot(convergence_df['customers'], convergence_df['waiting_error'], 'g-', linewidth=2, label='Temps d\'attente')
        axes[1,0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Seuil 1%')
        axes[1,0].set_xlabel('Nombre de clients servis')
        axes[1,0].set_ylabel('Erreur relative (%)')
        axes[1,0].set_title('Erreur relative vs théorique')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_yscale('log')
        
        # Graphique 4: Variabilité (écart-type)
        axes[1,1].plot(convergence_df['customers'], convergence_df['std_response'], 'b-', linewidth=2, label='Écart-type réponse')
        axes[1,1].plot(convergence_df['customers'], convergence_df['std_waiting'], 'g-', linewidth=2, label='Écart-type attente')
        axes[1,1].set_xlabel('Nombre de clients servis')
        axes[1,1].set_ylabel('Écart-type')
        axes[1,1].set_title('Variabilité des métriques')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def analyze_convergence_multiple_rho():
    """
    Analyse la convergence pour différentes valeurs de ρ
    """
    mu = 1.0
    rho_values = [0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Convergence pour différentes valeurs de ρ', fontsize=14)
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, rho in enumerate(rho_values):
    lambda_rate = rho * mu
    analyzer = MM1ConvergenceAnalyzer(lambda_rate, mu)
    
    print(f"\nAnalyse pour ρ = {rho}")
    convergence_df, _ = analyzer.simulate_with_convergence_tracking(max_customers=200000)
    
    # Métriques théoriques
    theo_response = 1 / (mu - lambda_rate)
    theo_waiting = lambda_rate / (mu * (mu - lambda_rate))
    
    # Graphique temps de réponse
    axes[0,0].plot(convergence_df['customers'], convergence_df['avg_response'],
                   color=colors[i], linewidth=2, label=f'ρ={rho}')
    axes[0,0].axhline(y=theo_response, color=colors[i], linestyle='--', alpha=0.7)
    
    # Graphique temps d'attente
    axes[0,1].plot(convergence_df['customers'], convergence_df['avg_waiting'],
                   color=colors[i], linewidth=2, label=f'ρ={rho}')
    axes[0,1].axhline(y=theo_waiting, color=colors[i], linestyle='--', alpha=0.7)
    
    # Erreur relative temps de réponse
    axes[1,0].plot(convergence_df['customers'], convergence_df['response_error'],
                   color=colors[i], linewidth=2, label=f'ρ={rho}')
    
    # Erreur relative temps d'attente
    axes[1,1].plot(convergence_df['customers'], convergence_df['waiting_error'],
                   color=colors[i], linewidth=2, label=f'ρ={rho}')

# Configuration des graphiques avec espacement amélioré
plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)

# Configuration graphique 1 : Temps de réponse
axes[0,0].set_xlabel('Nombre de clients', fontsize=10)
axes[0,0].set_ylabel('Temps de réponse', fontsize=10)
axes[0,0].set_title('Convergence - Temps de réponse', fontsize=12, pad=20)
axes[0,0].legend(loc='best', fontsize=9)
axes[0,0].grid(True, alpha=0.3)
axes[0,0].tick_params(axis='both', which='major', labelsize=8)

# Configuration graphique 2 : Temps d'attente
axes[0,1].set_xlabel('Nombre de clients', fontsize=10)
axes[0,1].set_ylabel('Temps d\'attente', fontsize=10)
axes[0,1].set_title('Convergence - Temps d\'attente', fontsize=12, pad=20)
axes[0,1].legend(loc='best', fontsize=9)
axes[0,1].grid(True, alpha=0.3)
axes[0,1].tick_params(axis='both', which='major', labelsize=8)

# Configuration graphique 3 : Erreur relative temps de réponse
axes[1,0].set_xlabel('Nombre de clients', fontsize=10)
axes[1,0].set_ylabel('Erreur relative (%)', fontsize=10)
axes[1,0].set_title('Erreur relative - Temps de réponse', fontsize=12, pad=20)
axes[1,0].legend(loc='best', fontsize=9)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_yscale('log')
axes[1,0].tick_params(axis='both', which='major', labelsize=8)

# Configuration graphique 4 : Erreur relative temps d'attente
axes[1,1].set_xlabel('Nombre de clients', fontsize=10)
axes[1,1].set_ylabel('Erreur relative (%)', fontsize=10)
axes[1,1].set_title('Erreur relative - Temps d\'attente', fontsize=12, pad=20)
axes[1,1].legend(loc='best', fontsize=9)
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_yscale('log')
axes[1,1].tick_params(axis='both', which='major', labelsize=8)

# Ajuster la taille de la figure si nécessaire (à ajouter avant la boucle)
# plt.figure(figsize=(14, 10))

# Sauvegarder avec une résolution élevée
plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')