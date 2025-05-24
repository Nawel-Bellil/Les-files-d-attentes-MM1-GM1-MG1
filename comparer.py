import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_process_data():
    """load the three csv files and combine them with labels"""
    files = ['mm1.csv', 'mg1.csv', 'gm1.csv']
    labels = ['m/m/1', 'm/g/1', 'g/m/1']
    
    dataframes = []
    for file, label in zip(files, labels):
        try:
            df = pd.read_csv(file)
            df['model'] = label
            dataframes.append(df)
            print(f"loaded {file}: {len(df)} rows")
        except FileNotFoundError:
            print(f"warning: {file} not found, skipping...")
    
    if not dataframes:
        raise FileNotFoundError("no csv files found!")
    
    return pd.concat(dataframes, ignore_index=True)

def create_empirical_comparison_figure(data):
    """create empirical performance metrics comparison figure"""
    
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('empirical performance metrics comparison across queue models', fontsize=16, fontweight='bold')
    
    # response time
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        ax1.plot(model_data['rho'], model_data['avg_response_time_emp'], 
                marker='o', linewidth=2, label=model, markersize=6)
    ax1.set_xlabel('server utilization (ρ)')
    ax1.set_ylabel('average response time (empirical)')
    ax1.set_title('average response time vs server utilization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # waiting time
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        ax2.plot(model_data['rho'], model_data['avg_waiting_time_emp'], 
                marker='s', linewidth=2, label=model, markersize=6)
    ax2.set_xlabel('server utilization (ρ)')
    ax2.set_ylabel('average waiting time (empirical)')
    ax2.set_title('average waiting time vs server utilization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # server utilization
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        ax3.plot(model_data['lambda'], model_data['server_utilization_emp'], 
                marker='^', linewidth=2, label=model, markersize=6)
    ax3.set_xlabel('arrival rate (λ)')
    ax3.set_ylabel('server utilization (empirical)')
    ax3.set_title('server utilization vs arrival rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # system length
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        ax4.plot(model_data['rho'], model_data['avg_system_length_emp'], 
                marker='d', linewidth=2, label=model, markersize=6)
    ax4.set_xlabel('server utilization (ρ)')
    ax4.set_ylabel('average system length (empirical)')
    ax4.set_title('average system length vs server utilization')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig1

def create_theoretical_comparison_figure(data):
    """create theoretical performance metrics comparison figure"""
    
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('theoretical performance metrics comparison across queue models', fontsize=16, fontweight='bold')
    
    # response time theoretical
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        ax1.plot(model_data['rho'], model_data['avg_response_time_theo'], 
                marker='o', linewidth=2, label=model, markersize=6)
    ax1.set_xlabel('server utilization (ρ)')
    ax1.set_ylabel('average response time (theoretical)')
    ax1.set_title('theoretical average response time vs server utilization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # waiting time theoretical
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        ax2.plot(model_data['rho'], model_data['avg_waiting_time_theo'], 
                marker='s', linewidth=2, label=model, markersize=6)
    ax2.set_xlabel('server utilization (ρ)')
    ax2.set_ylabel('average waiting time (theoretical)')
    ax2.set_title('theoretical average waiting time vs server utilization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # server utilization theoretical
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        ax3.plot(model_data['lambda'], model_data['server_utilization_theo'], 
                marker='^', linewidth=2, label=model, markersize=6)
    ax3.set_xlabel('arrival rate (λ)')
    ax3.set_ylabel('server utilization (theoretical)')
    ax3.set_title('theoretical server utilization vs arrival rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # system length theoretical
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        ax4.plot(model_data['rho'], model_data['avg_system_length_theo'], 
                marker='d', linewidth=2, label=model, markersize=6)
    ax4.set_xlabel('server utilization (ρ)')
    ax4.set_ylabel('average system length (theoretical)')
    ax4.set_title('theoretical average system length vs server utilization')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig2

def calculate_error_metrics(data):
    """calculate and display error metrics between theoretical and empirical values"""
    print("\n" + "="*50)
    print("error analysis: theoretical vs empirical")
    print("="*50)
    
    metrics = [
        ('avg_response_time', 'response time'),
        ('avg_waiting_time', 'waiting time'),
        ('server_utilization', 'server utilization'),
        ('avg_system_length', 'system length')
    ]
    
    for metric, name in metrics:
        print(f"\n{name}:")
        print("-" * len(name))
        
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            
            emp_col = f"{metric}_emp"
            theo_col = f"{metric}_theo"
            
            if emp_col in model_data.columns and theo_col in model_data.columns:
                empirical = model_data[emp_col].values
                theoretical = model_data[theo_col].values
                
                # calculate error metrics
                mae = np.mean(np.abs(empirical - theoretical))
                mse = np.mean((empirical - theoretical) ** 2)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((empirical - theoretical) / theoretical * 100))
                
                print(f"  {model}:")
                print(f"    mae:  {mae:.6f}")
                print(f"    mse:  {mse:.6f}")
                print(f"    rmse: {rmse:.6f}")
                print(f"    mape: {mape:.2f}%")

def main():
    """main function to run the analysis"""
    try:
        # load data
        print("loading csv files...")
        data = load_and_process_data()
        
        print(f"\ndata loaded successfully!")
        print(f"total rows: {len(data)}")
        print(f"models: {', '.join(data['model'].unique())}")
        print(f"lambda range: {data['lambda'].min():.1f} - {data['lambda'].max():.1f}")
        print(f"rho range: {data['rho'].min():.3f} - {data['rho'].max():.3f}")
        
        # create figures
        print("\ngenerating figures...")
        
        # empirical comparison figure
        fig1 = create_empirical_comparison_figure(data)
        
        # theoretical comparison figure
        fig2 = create_theoretical_comparison_figure(data)
        
        # calculate error metrics
        calculate_error_metrics(data)
        
        # save figures
        fig1.savefig('empirical_performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
        fig2.savefig('theoretical_performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
        
        print(f"\n✅ analysis complete!")
        print(f"📊 figures saved:")
        print(f"   - empirical_performance_metrics_comparison.png")
        print(f"   - theoretical_performance_metrics_comparison.png") 
        
        # show plots
        plt.show()
        
    except Exception as e:
        print(f"❌ error: {e}")
        print("make sure your csv files (mm1.csv, mg1.csv, gm1.csv) are in the same directory as this script.")

if __name__ == "__main__":
    main()