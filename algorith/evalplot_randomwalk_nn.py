import os
import math
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

# ==================== CONFIGURAÇÃO ====================
# Diretório onde estão os arquivos .npz gerados pelo randomwalk_nn.py
# Exemplo: 'sdata/randomwalk_nn_20x12x18/'
DATA_DIR = './sdata/randomwalk_nn_30x24x16/'

# Se quiser carregar um arquivo específico, defina-o aqui. Deixe None para carregar todos.
SPECIFIC_FILE = None   # Exemplo: 'randomwalk_nn_20x12x18.npz'

# Caminho para salvar a figura
OUTPUT_DIR = './figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ====================================================

def load_results(directory, specific_file=None):
    """
    Carrega todos os arquivos .npz do diretório ou um arquivo específico.
    Retorna uma lista de dicionários com os dados e a configuração.
    """
    if specific_file is not None:
        files = [os.path.join(directory, specific_file)]
    else:
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]
    
    results = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        # all_results: (seeds, n_eval_points)
        all_res = data['all_results']
        # configuração salva como dict
        config = data['config'].item() if 'config' in data else {}
        # extrair intervalo de avaliação (padrão 50)
        eval_interval = config.get('eval_interval', 50)
        results.append({
            'all_results': all_res,
            'eval_interval': eval_interval,
            'filename': os.path.basename(f)
        })
    return results

def compute_stats(all_results):
    """Calcula média, desvio padrão e percentis ao longo das sementes."""
    mean = np.mean(all_results, axis=0)
    std = np.std(all_results, axis=0)
    p10 = np.percentile(all_results, 10, axis=0)
    p25 = np.percentile(all_results, 25, axis=0)
    p50 = np.percentile(all_results, 50, axis=0)
    p75 = np.percentile(all_results, 75, axis=0)
    p90 = np.percentile(all_results, 90, axis=0)
    return mean, std, p10, p25, p50, p75, p90

def plot_results(results_list):
    """Plota os resultados de um ou mais arquivos."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for idx, res in enumerate(results_list):
        all_res = res['all_results']
        eval_interval = res['eval_interval']
        n_points = all_res.shape[1]
        x_episodes = np.arange(1, n_points+1) * eval_interval
        
        mean, std, p10, p25, p50, p75, p90 = compute_stats(all_res)
        
        # Nome para a legenda (usa o nome do arquivo ou arquitetura)
        label = res['filename'].replace('.npz', '').replace('randomwalk_nn_', '')
        
        # Plot das áreas sombreadas (percentis)
        ax.fill_between(x_episodes, p10, p25, color=colors[idx], alpha=0.2, label=f'{label} 10-25%')
        ax.fill_between(x_episodes, p25, p75, color=colors[idx], alpha=0.4, label=f'{label} 25-75%')
        ax.fill_between(x_episodes, p75, p90, color=colors[idx], alpha=0.6, label=f'{label} 75-90%')
        ax.plot(x_episodes, p50, color=colors[idx], linestyle='-.', linewidth=2, label=f'{label} mediana')
    
    ax.set_ylim([0, 1000])
    ax.set_xlabel('Episódios')
    ax.set_ylabel(r'$\mathcal{R}(\phi^{\theta})$')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Salvar figura
    fig.savefig(os.path.join(OUTPUT_DIR, 'randomwalk_comparison.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'randomwalk_comparison.svg'), format='svg')
    print(f'Gráfico salvo em {OUTPUT_DIR}')
    plt.show()

if __name__ == '__main__':
    results = load_results(DATA_DIR, SPECIFIC_FILE)
    if not results:
        print(f"Nenhum arquivo .npz encontrado em {DATA_DIR}")
    else:
        plot_results(results)