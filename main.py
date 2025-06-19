import numpy as np
import matplotlib.pyplot as plt

def jacobi(A, b, x0, tol, max_iter, x_exato=None):
    """
    Implementa o método de Jacobi para resolver o sistema linear Ax = b.

    Args:
        A (numpy.array): Matriz dos coeficientes (N x N).
        b (numpy.array): Vetor do lado direito (N).
        x0 (numpy.array): Estimativa inicial (N).
        tol (float): Tolerância para o critério de parada.
        max_iter (int): Número máximo de iterações.
        x_exato (numpy.array, optional): Solução exata.

    Returns:
        tuple: Contendo:
            - x_final (numpy.array): Solução aproximada final.
            - historico_iteracoes (list): Lista de tuplas (k, x_k, epsilon_k) 
                                          k: número da iteração.
                                          x_k: vetor solução na iteração k.
                                          epsilon_k: epsilon calculado na iteração k (None para k=0).
            - distancias_exatas (list): Lista de distâncias ||X* - X_k||.
    """
    n = len(b)
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError("A matriz A deve ser quadrada e de dimensão n x n.")
    if len(x0) != n:
        raise ValueError("O vetor de estimativa inicial x0 deve ter dimensão n.")

    x = np.array(x0, dtype=float) # x na iteração k-1, inicialmente x^(0)
    
    historico_iteracoes = [] 
    distancias_exatas = []

    historico_iteracoes.append((0, x.copy(), None)) 
    if x_exato is not None:
        if len(x_exato) != n:
            raise ValueError("A solução exata x_exato deve ter dimensão n.")
        distancias_exatas.append(np.linalg.norm(x_exato - x))

    for k_iter in range(1, max_iter + 1): # k_iter é a iteração atual k
        x_novo = np.zeros(n, dtype=float) # x_novo será x^(k)
        for i in range(n):
            soma_ax_j = 0
            for j in range(n):
                if i != j:
                    soma_ax_j += A[i, j] * x[j] # usa x da iteração anterior (x^(k-1))
            
            if A[i, i] == 0:
                raise ValueError(f"Elemento da diagonal principal A[{i},{i}] é zero. O método de Jacobi não pode continuar.")

            x_novo[i] = (b[i] - soma_ax_j) / A[i, i]

        # epsilon = max{|x^(k) - x^(k-1)|}
        epsilon = np.linalg.norm(x_novo - x, np.inf)
        
        x = x_novo.copy() # Atualiza x para ser x^(k) para a próxima iteração ou para o histórico
        
        historico_iteracoes.append((k_iter, x.copy(), epsilon)) # Armazena (k, x^(k), epsilon_k)
        
        if x_exato is not None:
            distancias_exatas.append(np.linalg.norm(x_exato - x))

        if epsilon <= tol:
            print(f"\nConvergência alcançada na iteração {k_iter} com epsilon = {epsilon:.8f} (<= {tol:.8f}).")
            break
    else: 
        print(f"\nO método não convergiu em {max_iter} iterações. Epsilon final = {epsilon:.8f} (> {tol:.8f}).")

    return x, historico_iteracoes, distancias_exatas

# --- Exemplo de uso ---
if __name__ == "__main__":
    # Sistema de exemplo
    #−3𝑥1 + 𝑥2 + 𝑥3 = 2
    #2𝑥1 + 5𝑥2 + 𝑥3 = 5
    #2𝑥1 + 3𝑥2 + 7𝑥3 = −17
    A_exemplo = np.array([
        [-3, 1, 1],
        [2, 5, 1],
        [2, 3, 7]
    ], dtype=float)

    b_exemplo = np.array([2,5,-17], dtype=float)
    x0_exemplo = np.array([0, 0, 0], dtype=float) #Iteração inicial

    # Parâmetros do método
    # "epsilon <= 10^-n, n dado". Se n_tol = 5, então tol = 10^-5.
    n_tol = 5 
    tolerancia = 10**(-n_tol) 
    maximo_iteracoes = 50

    # Solução exata
    solucao_exata_exemplo = np.array([-1, 2, -3], dtype=float)

    # Executar o método de Jacobi
    x_final, historico, distancias = jacobi(A_exemplo, b_exemplo, x0_exemplo, 
                                            tolerancia, maximo_iteracoes, 
                                            solucao_exata_exemplo)
    
    # 4) Plotar a solução aproximada X = [...., ....]
    print(f"\n--- Solução Aproximada Final X ---")
    print(f"X = {x_final}")

    # 5) Gerar a tabela de iterações como uma figura
    print("\n--- Gerando Tabela de Iterações como Imagem ---")
    if historico:
        num_vars = len(historico[0][1]) 
        col_labels = ['k'] + [f'x{i+1}' for i in range(num_vars)] + ['Epsilon']
        
        table_data = []
        for k_val, x_k_val, epsilon_k_val in historico:
            row = [f"{k_val}"] 
            row.extend([f"{val:.7f}" for val in x_k_val]) 
            if epsilon_k_val is None:
                row.append('-')
            else:
                row.append(f"{epsilon_k_val:.2e}") 
            table_data.append(row)

        if not table_data:
            print("Não há dados para exibir na tabela.")
        else:
            row_height_multiplier = 0.45
            
            fig_width = max(8, len(col_labels) * 1.5) 
            fig_height = max(6, len(table_data) * row_height_multiplier) 
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 
            ax.axis('tight')
            ax.axis('off') 
            
            the_table = ax.table(cellText=table_data,
                                 colLabels=col_labels,
                                 loc='center',
                                 cellLoc='center')
            
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(9) 
            
            the_table.scale(1.1, 1.7) 

            title_y_position = 1.05 
            if fig_height > 10: 
                 title_y_position = 1.01 
                 plt.subplots_adjust(top=0.97) 

            plt.title(f'Tabela de Iterações (k, x^(k), Epsilon)', fontsize=12, y=0.92)

            try:
                plt.tight_layout()
            except ValueError:
                print("Aviso: tight_layout falhou, pode haver sobreposição. Tente ajustar figsize ou a escala da tabela.")
            plt.show()
    else:
        print("Histórico de iterações está vazio, tabela não gerada.")

    # Plotar a convergência dos componentes de X
    plt.figure(figsize=(10, 6))
    iteracoes_plot = [item[0] for item in historico]
    for i in range(num_vars):
        componente_valores = [item[1][i] for item in historico]
        plt.plot(iteracoes_plot, componente_valores, marker='.', linestyle='-', label=f'x{i+1}')
    
    plt.xlabel('Iteração (k)')
    plt.ylabel('Valor do Componente da Solução')
    plt.title('Convergência dos Componentes da Solução Aproximada (X_k vs k)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6) Plotar a distância da solução exata e a solução aproximada
    # d = ||X* - X|| onde X* é a solução exata.
    if solucao_exata_exemplo is not None and distancias:
        plt.figure(figsize=(10, 6))
        plt.plot(iteracoes_plot, distancias, marker='o', linestyle='-')
        plt.xlabel('Iteração (k)')
        plt.ylabel('Distância ||X* - X_k|| (Norma Euclidiana)')
        plt.title('Distância entre Solução Exata e Aproximada vs. Iterações')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-")
        plt.tight_layout()
        plt.show()
    elif solucao_exata_exemplo is None:
        print("\nSolução exata não fornecida, gráfico de distância não gerado.")