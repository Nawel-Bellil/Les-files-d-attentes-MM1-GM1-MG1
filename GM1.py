import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import namedtuple
import time


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
        
        # Calculs théoriques pour G/M/1 avec arrivées uniformes
        self.ca_squared = 1/3  # Coefficient de variation au carré pour loi uniforme
        
        # Vérification de stabilité
        if self.rho >= 1:
            raise ValueError(f"Système instable: ρ = {self.rho:.3f} >= 1")
    
    def calculate_theoretical_values(self):
        """
        Calcule les valeurs théoriques pour G/M/1 avec arrivées uniformes
        
        Returns:
            dict: Valeurs théoriques
        """
        rho = self.rho
        ca_squared = self.ca_squared  # 1/3 pour loi uniforme
        
        # Temps d'attente moyen théorique (Formule de Pollaczek-Khinchine)
        # W = (ρ² × (1 + CA²)) / (2 × (1 - ρ)) × (1/μ)
        avg_waiting_time_theo = (rho**2 * (1 + ca_squared)) / (2 * (1 - rho)) * (1 / self.mu_rate)
        
        # Temps de réponse moyen théorique
        # T = W + 1/μ
        avg_response_time_theo = avg_waiting_time_theo + (1 / self.mu_rate)
        
        # Taux d'occupation serveur théorique
        server_utilization_theo = rho
        
        return {
            'avg_response_time_theo': avg_response_time_theo,
            'avg_waiting_time_theo': avg_waiting_time_theo,
            'server_utilization_theo': server_utilization_theo
        }
    
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
            dict: Statistiques de performance empiriques et théoriques
        """
        print(f"🚀 Début simulation G/M/1 (λ={self.lambda_rate}, μ={self.mu_rate}, ρ={self.rho:.3f})")
        start_time = time.time()
        
        # Variables de simulation
        current_time = 0.0
        server_busy = False
        server_free_time = 0.0
        queue = []
        
        # Statistiques (ne commencent qu'après la période de chauffe)
        warmup_clients = 20000  # Période de chauffe
        total_response_time = 0.0
        total_waiting_time = 0.0
        total_service_time = 0.0
        completed_clients = 0
        
        # Temps d'arrivée du prochain client
        next_arrival_time = self.generate_uniform_interarrival()
        next_service_completion = float('inf')
        
        clients_arrived = 0
        total_clients_completed = 0  # Total incluant la période de chauffe
        
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
                    
                    # Statistiques pour ce client (seulement après la période de chauffe)
                    waiting_time = 0.0
                    response_time = service_time
                    
                    total_clients_completed += 1
                    if total_clients_completed > warmup_clients:
                        total_waiting_time += waiting_time
                        total_response_time += response_time
                        total_service_time += service_time
                        completed_clients += 1
                else:
                    # Serveur occupé : client entre dans la file
                    queue.append(client_arrival_time)
                
                # Programmer la prochaine arrivée
                if clients_arrived < (self.num_clients + warmup_clients) * 2:  # Marge de sécurité
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
                    
                    # Mise à jour des statistiques (seulement après la période de chauffe)
                    total_clients_completed += 1
                    if total_clients_completed > warmup_clients:
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
        
        # Calcul des métriques empiriques
        simulation_elapsed_time = time.time() - start_time
        
        avg_response_time_emp = total_response_time / completed_clients
        avg_waiting_time_emp = total_waiting_time / completed_clients
        avg_service_time_emp = total_service_time / completed_clients
        
        # Taux d'occupation du serveur empirique
        total_busy_time = total_service_time
        total_simulation_time = current_time
        server_utilization_emp = total_busy_time / total_simulation_time
        rho_emp = server_utilization_emp
        
        # Calcul des valeurs théoriques
        theoretical_values = self.calculate_theoretical_values()
        
        # Compilation des résultats
        results = {
            'lambda': self.lambda_rate,
            'mu': self.mu_rate,
            'rho_emp': rho_emp,
            'avg_response_time_emp': avg_response_time_emp,
            'avg_response_time_theo': theoretical_values['avg_response_time_theo'],
            'avg_waiting_time_emp': avg_waiting_time_emp,
            'avg_waiting_time_theo': theoretical_values['avg_waiting_time_theo'],
            'server_utilization_emp': server_utilization_emp,
            'server_utilization_theo': theoretical_values['server_utilization_theo'],
            'customers_served': completed_clients,
            'simulation_time': simulation_elapsed_time
        }
        
        print(f"✅ Simulation G/M/1 terminée en {simulation_elapsed_time:.2f}s")
        print(f"   • Clients traités (après période de chauffe): {completed_clients:,}")
        print(f"   • Total clients traités: {total_clients_completed:,}")
        print(f"   • Période de chauffe: {warmup_clients:,} clients")
        print(f"   • Temps de réponse (emp/theo): {avg_response_time_emp:.4f} / {theoretical_values['avg_response_time_theo']:.4f}")
        print(f"   • Temps d'attente (emp/theo): {avg_waiting_time_emp:.4f} / {theoretical_values['avg_waiting_time_theo']:.4f}")
        print(f"   • Taux d'occupation (emp/theo): {server_utilization_emp:.4f} / {theoretical_values['server_utilization_theo']:.4f}")
        
        return results

def run_gm1_experiments():
    """
    Exécute les expériences G/M/1 pour différentes valeurs de λ
    """
    print("=" * 80)
    print("SIMULATION G/M/1 - ARRIVÉES UNIFORMES, SERVICE EXPONENTIEL")
    print("Comparaison Valeurs Empiriques vs Théoriques")
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
        
        # Calculer les moyennes sur les répétitions (pour les valeurs empiriques)
        avg_result = {
            'lambda': lambda_rate,
            'mu': mu,
            'rho_emp': np.mean([r['rho_emp'] for r in repetition_results]),
            'avg_response_time_emp': np.mean([r['avg_response_time_emp'] for r in repetition_results]),
            'avg_response_time_theo': repetition_results[0]['avg_response_time_theo'],
            'avg_waiting_time_emp': np.mean([r['avg_waiting_time_emp'] for r in repetition_results]),
            'avg_waiting_time_theo': repetition_results[0]['avg_waiting_time_theo'],
            'server_utilization_emp': np.mean([r['server_utilization_emp'] for r in repetition_results]),
            'server_utilization_theo': repetition_results[0]['server_utilization_theo'],
            'customers_served': num_clients,
            'simulation_time': np.mean([r['simulation_time'] for r in repetition_results]),
            'num_repetitions': num_repetitions,
            # Écarts-types pour analyse
            'response_time_std': np.std([r['avg_response_time_emp'] for r in repetition_results]),
            'waiting_time_std': np.std([r['avg_waiting_time_emp'] for r in repetition_results])
        }
        
        all_results.append(avg_result)
        
        print(f"\n📊 RÉSULTATS MOYENS pour λ = {lambda_rate:.1f}:")
        print(f"   • Temps de réponse (emp/theo): {avg_result['avg_response_time_emp']:.4f} / {avg_result['avg_response_time_theo']:.4f}")
        print(f"   • Temps d'attente (emp/theo): {avg_result['avg_waiting_time_emp']:.4f} / {avg_result['avg_waiting_time_theo']:.4f}")
        print(f"   • Taux d'occupation (emp/theo): {avg_result['server_utilization_emp']:.4f} / {avg_result['server_utilization_theo']:.4f}")
    
    return all_results

def save_results_and_create_graphs(results):
    """
    Sauvegarde les résultats avec les colonnes demandées et crée les graphiques comparatifs
    """
    # Création du DataFrame avec les colonnes demandées (sans rho_theo)
    df_columns = [
        'lambda', 'mu', 'rho_emp', 
        'avg_response_time_emp', 'avg_response_time_theo',
        'avg_waiting_time_emp', 'avg_waiting_time_theo',
        'server_utilization_emp', 'server_utilization_theo',
        'customers_served', 'simulation_time'
    ]
    
    df_data = []
    for result in results:
        row = {col: result[col] for col in df_columns}
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Sauvegarde CSV avec les colonnes exactes demandées
    csv_filename = 'gm1_results_theo_vs_emp.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Résultats sauvegardés dans {csv_filename}")
    
    # Création des graphiques comparatifs
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('G/M/1: Comparaison Valeurs Théoriques vs Empiriques', 
                 fontsize=16, fontweight='bold')
    
    # Couleurs pour la cohérence
    emp_color = '#FF6B35'  # Orange pour empirique
    theo_color = '#004E89'  # Bleu pour théorique
    
    # Graphique 1: Taux d'occupation ρ (seulement empirique maintenant)
    axes[0, 0].plot(df['lambda'], df['rho_emp'], 'o-', color=emp_color, 
                   linewidth=2, markersize=7, label='ρ Empirique')
    axes[0, 0].set_xlabel('Taux d\'arrivée λ', fontsize=12)
    axes[0, 0].set_ylabel('Taux d\'occupation ρ', fontsize=12)
    axes[0, 0].set_title('Taux d\'occupation du serveur', fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Graphique 2: Temps de réponse
    axes[0, 1].plot(df['lambda'], df['avg_response_time_emp'], 'o-', color=emp_color, 
                   linewidth=2, markersize=7, label='Temps réponse Empirique')
    axes[0, 1].plot(df['lambda'], df['avg_response_time_theo'], 's--', color=theo_color, 
                   linewidth=2, markersize=7, label='Temps réponse Théorique')
    axes[0, 1].set_xlabel('Taux d\'arrivée λ', fontsize=12)
    axes[0, 1].set_ylabel('Temps de réponse moyen', fontsize=12)
    axes[0, 1].set_title('Temps de réponse moyen', fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Graphique 3: Temps d'attente
    axes[1, 0].plot(df['lambda'], df['avg_waiting_time_emp'], 'o-', color=emp_color, 
                   linewidth=2, markersize=7, label='Temps attente Empirique')
    axes[1, 0].plot(df['lambda'], df['avg_waiting_time_theo'], 's--', color=theo_color, 
                   linewidth=2, markersize=7, label='Temps attente Théorique')
    axes[1, 0].set_xlabel('Taux d\'arrivée λ', fontsize=12)
    axes[1, 0].set_ylabel('Temps d\'attente moyen', fontsize=12)
    axes[1, 0].set_title('Temps d\'attente dans la file', fontsize=14)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Graphique 4: Taux d'occupation serveur
    axes[1, 1].plot(df['lambda'], df['server_utilization_emp'], 'o-', color=emp_color, 
                   linewidth=2, markersize=7, label='Utilisation Empirique')
    axes[1, 1].plot(df['lambda'], df['server_utilization_theo'], 's--', color=theo_color, 
                   linewidth=2, markersize=7, label='Utilisation Théorique')
    axes[1, 1].set_xlabel('Taux d\'arrivée λ', fontsize=12)
    axes[1, 1].set_ylabel('Taux d\'utilisation serveur', fontsize=12)
    axes[1, 1].set_title('Taux d\'utilisation du serveur', fontsize=14)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde des graphiques
    graph_filename = 'gm1_theo_vs_emp_graphs.png'
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Graphiques sauvegardés dans {graph_filename}")
    
    plt.show()
    
    # Affichage du tableau récapitulatif avec les colonnes demandées (sans rho_theo)
    print(f"\n📋 TABLEAU RÉCAPITULATIF G/M/1 - THÉORIQUE VS EMPIRIQUE:")
    print("=" * 110)
    print(f"{'λ':<5} {'μ':<5} {'ρ_emp':<8} {'T_rép_emp':<12} {'T_rép_theo':<12} "
          f"{'T_att_emp':<12} {'T_att_theo':<12} {'Util_emp':<10} {'Util_theo':<10} {'Clients':<10} {'Temps_sim':<10}")
    print("-" * 110)
    
    for _, row in df.iterrows():
        print(f"{row['lambda']:<5.1f} {row['mu']:<5.1f} {row['rho_emp']:<8.4f} "
              f"{row['avg_response_time_emp']:<12.4f} {row['avg_response_time_theo']:<12.4f} "
              f"{row['avg_waiting_time_emp']:<12.4f} {row['avg_waiting_time_theo']:<12.4f} "
              f"{row['server_utilization_emp']:<10.4f} {row['server_utilization_theo']:<10.4f} "
              f"{row['customers_served']:<10} {row['simulation_time']:<10.2f}")
    
    print("=" * 110)
    
    # Calcul des erreurs relatives
    print(f"\n📈 ANALYSE DES ERREURS RELATIVES:")
    print("=" * 60)
    
    for _, row in df.iterrows():
        err_response = abs(row['avg_response_time_emp'] - row['avg_response_time_theo']) / row['avg_response_time_theo'] * 100
        err_waiting = abs(row['avg_waiting_time_emp'] - row['avg_waiting_time_theo']) / row['avg_waiting_time_theo'] * 100 if row['avg_waiting_time_theo'] > 0 else 0
        err_util = abs(row['server_utilization_emp'] - row['server_utilization_theo']) / row['server_utilization_theo'] * 100
        
        print(f"λ={row['lambda']:.1f}: Erreur temps réponse: {err_response:.2f}%, "
              f"temps attente: {err_waiting:.2f}%, utilisation: {err_util:.2f}%")

# Exécution principale
if __name__ == "__main__":
    # Lancement des expériences G/M/1
    results = run_gm1_experiments()
    
    # Sauvegarde et visualisation
    save_results_and_create_graphs(results)
    
    print(f"\n🎉 SIMULATION G/M/1 TERMINÉE!")
    print(f"📁 Fichiers générés:")
    print(f"   • gm1_results_theo_vs_emp.csv (avec colonnes demandées)")
    print(f"   • gm1_theo_vs_emp_graphs.png (graphiques comparatifs)")
    print(f"\n📊 Le fichier CSV contient exactement les colonnes demandées:")
    print(f"   lambda, mu, rho_emp, avg_response_time_emp, avg_response_time_theo,")
    print(f"   avg_waiting_time_emp, avg_waiting_time_theo, server_utilization_emp,")
    print(f"   server_utilization_theo, customers_served, simulation_time")