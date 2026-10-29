import os
import sys
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra a projeção 3D)
from scipy.optimize import curve_fit

def fcm_processar_imagem(img, K, m=2.0, max_iter=100, epsilon=0.01):
    """
    Executa o pipeline completo do FCM Puro com proteções contra índice inválido e divisão por zero.
    """
    
    largura, altura = img.size
    
    # Extração segura dos pixels da imagem
    pixels_2d = list(img.getdata())
    
    # ====================================================================
    # ALVO: MISSÃO 3 (Structure of Arrays - SoA)
    # A equipe 3 deve DESTRUIR este for. Em vez de uma lista X com sublistas
    # [R, G, B], eles deverão criar três listas gigantes isoladas:
    # X_R = [], X_G = [], X_B = []
    # ====================================================================
    X = []
    for p in pixels_2d:
        X.append([float(p[0]), float(p[1]), float(p[2])])
        
    N = len(X)
    canais = 3
    
    paleta_cores = [
        [255, 0, 0],     # Cluster 0: Vermelho
        [0, 255, 0],     # Cluster 1: Verde
        [0, 0, 255],     # Cluster 2: Azul
        [255, 255, 0],   # Cluster 3: Amarelo
        [255, 0, 255],   # Cluster 4: Magenta
        [0, 255, 255]    # Cluster 5: Ciano
    ]
    
    # ====================================================================
    # ALVO: MISSÃO 1 (Memory Flattening)
    # A equipe 1 deve DESTRUIR a declaração bidimensional de U (lista de listas).
    # U deve ser um vetor achatado (1D) de tamanho N * K.
    # ====================================================================
    #pertinencia
    U = []
    for i in range(N):
        linha = []
        soma_linha = 0.0
        for j in range(K):
            val = random.random()
            linha.append(val)
            soma_linha += val
        
        for j in range(K):
            if soma_linha == 0.0:
                linha[j] = 1.0 / K
            else:
                linha[j] = linha[j] / soma_linha
        U.append(linha)
        
    # 4. LAÇO PRINCIPAL DO FCM (Puro)
    for iteracao in range(max_iter):
        
        # ====================================================================
        # ALVO: MISSÃO 2 (Fixed Point Math / Fuga da FPU)
        # A equipe 2 terá que modificar este laço e os laços subsequentes
        # para impedir o uso de float e a divisão decimal de numerador por 
        # denominador, usando shifiting (>>).
        # ====================================================================
        C = []
        for j in range(K):
            numerador = [0] * canais
            denominador = 0
            
            for i in range(N):
                #Considerando que U[i][j] está entre 0 e 1, temos uma forma de representar-lo em 
                #inteiro sem perder tanto a precisão, que no caso é de 0 a 2 << 16, fazendo com que o 
                #número caiba em 16 bits de memória.
                shift = 16
                u_int = int(U[i][j] * (2 << shift))      

                #Por padrão ** vira uma operação em ponto flutuante. O loop dribla isso fazendo várias 
                #multiplicações sucessivas.
                u_m = u_int
                for _ in range(int(m) - 1):
                    #O shift lida com o fato de que com cada multiplicação você vai estar aumentando 
                    #o tamanho da fração efetivamente. 
                    u_m = (u_m * u_int) >> shift     

                denominador += u_m
                
                # ====================================================================
                # ALVO: MISSÃO 4 (Loop Unrolling)
                # A equipe 4 deverá ELIMINAR este laço `for c in range(canais):` 
                # e desdobrar as operações manualmente (num_0 += ..., num_1 += ...)
                # ====================================================================
                for c in range(canais):
                    numerador[c] += u_m * int(X[i][c])   # Q16 * pixel inteiro = Q16
            
            centroide_j = []
            for c in range(canais):
                if denominador == 0:
                    centroide_j.append(0)
                else:
                    #Divisão inteira ao invés de para float
                    centroide_j.append(numerador[c] // denominador)
            C.append(centroide_j)
            
        U_nova = []
        max_diff = 0.0
        
        # ====================================================================
        # ALVO: MISSÃO 6 (Loop Fission)
        # A equipe 6 deverá DUPLICAR todo este bloco `for i in range(N):`.
        # Um laço roda a tela inteira APENAS calculando e salvando as distâncias 
        # em um array temporário gigante. Um segundo laço roda a tela de novo 
        # calculando a pertinência a partir desse array.
        # ====================================================================
        for i in range(N):
            linha_nova = [0.0] * K
            distancias = [0.0] * K
            
            for j in range(K):
                soma_quadrados = 0.0
                for c in range(canais):
                    diff = X[i][c] - C[j][c]
                    soma_quadrados += diff * diff
                distancias[j] = soma_quadrados ** 0.5 
                
                # ====================================================================
                # ALVO: MISSÃO 5 (Early Exit / Poda Condicional)
                # A equipe 5 deverá INJETAR aqui dentro um desvio:
                # Se 'distancias[j]' for maior que um limite arbitrário X, break.
                # ====================================================================
                
            for j in range(K):
                if distancias[j] == 0.0:
                    linha_nova[j] = 1.0
                    for k in range(K):
                        if k != j: 
                            linha_nova[k] = 0.0
                    break
                    
                soma_dist = 0.0
                for k in range(K):
                    d_k = distancias[k] if distancias[k] > 0.0 else 1e-10
                    razao = distancias[j] / d_k
                    soma_dist += razao ** (2.0 / (m - 1.0))
                
                pertinencia = 1.0 / soma_dist if soma_dist > 0.0 else (1.0 / K)
                linha_nova[j] = pertinencia
                
                diff_u = pertinencia - U[i][j]
                if diff_u < 0: 
                    diff_u = -diff_u
                if diff_u > max_diff: 
                    max_diff = diff_u
                
            U_nova.append(linha_nova)
            
        # O U TEM QUE SER ATUALIZADO AQUI FORA DO LAÇO DE PIXELS!
        U = U_nova
            
        if max_diff < epsilon:
            print(f"-> Convergiu na iteração {iteracao}!")
            break
            
    # 5. FASE 5: COLORIZAÇÃO POR PERTINÊNCIA MÁXIMA
    pixels_segmentados_lista = []
    for i in range(N):
        maior_pertinencia = -1.0
        cluster_vencedor = 0
        for j in range(K):
            if U[i][j] > maior_pertinencia:
                maior_pertinencia = U[i][j]
                cluster_vencedor = j
                
        cor = paleta_cores[cluster_vencedor % len(paleta_cores)]
        pixels_segmentados_lista.append((cor[0], cor[1], cor[2]))
        
    # 6. RESHAPE MANUAL E SALVAMENTO
    imagem_final = Image.new("RGB", (largura, altura))
    imagem_final.putdata(pixels_segmentados_lista)
    
    return C, U, imagem_final

# ==============================================================
# EXPERIMENTADOR (ANÁLISE F.O.C.A.) - UMA IMAGEM POR VEZ
# Funções adicionadas SEM modificar fcm_processar_imagem.
# Como o modelo T(n, K) precisa de tamanhos (n) diferentes, a imagem escolhida é
# reamostrada em várias escalas (ex.: 25%, 50%, 75%, 100%) e cada escala é
# cronometrada para cada K. Os resultados vão para: resultados/<nome_da_imagem>/
# ==============================================================

DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIR_RESULTADOS = os.path.join(DIR_SCRIPT, "resultados")
MOSTRAR_GRAFICOS = False   # True = também abre as janelas do matplotlib (bloqueia a execução)


def resolver_caminho(nome_arquivo):
    """Aceita caminho relativo ao terminal, relativo ao script ou absoluto."""
    for c in (nome_arquivo, os.path.join(DIR_SCRIPT, nome_arquivo)):
        if os.path.isfile(c):
            return os.path.abspath(c)
    return nome_arquivo


def preparar_pasta_resultados(nome_base):
    pasta = os.path.join(DIR_RESULTADOS, nome_base)
    os.makedirs(pasta, exist_ok=True)
    return pasta


def finalizar_figura(fig, caminho):
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    print(f"   [salvo] {caminho}")
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close(fig)


def model_n_k(X, alpha, beta, gamma, delta):
    """Modelo de complexidade: T(n, k) = alpha*(n*k) + beta*n + gamma*k + delta"""
    n, k = X
    return alpha * (n * k) + beta * n + gamma * k + delta


def calcular_tamanhos(caminho, escalas):
    """Lê só o cabeçalho da imagem e calcula (escala, largura, altura, n) de cada escala."""
    with Image.open(caminho) as img:
        w0, h0 = img.size
    tamanhos = []
    for s in sorted(set(escalas)):
        w, h = max(1, round(w0 * s)), max(1, round(h0 * s))
        tamanhos.append({'escala': s, 'largura': w, 'altura': h, 'n': w * h})
    return tamanhos


def executar_experimento(caminho, tamanhos, k_list, R, m, max_iter, epsilon):
    """Para a imagem escolhida: cada escala x cada K x R repetições."""
    resultados = []
    with Image.open(caminho) as img:
        img_base = img.convert("RGB")   # o FCM espera RGB (descarta alpha)

    for t in tamanhos:
        n = t['n']
        print(f"\nEscala {t['escala']:.0%} -> {t['largura']}x{t['altura']} (n={n})")
        if t['escala'] == 1.0:
            img_esc = img_base
        else:
            img_esc = img_base.resize((t['largura'], t['altura']), Image.LANCZOS)

        for k in k_list:
            for r in range(R):
                t0 = time.perf_counter()
                fcm_processar_imagem(img=img_esc, K=k, m=m, max_iter=max_iter, epsilon=epsilon)
                t1 = time.perf_counter()

                resultados.append({
                    'escala': t['escala'],
                    'n': n,
                    'k': k,
                    'repeticao': r,
                    'tempo': (t1 - t0)
                })

        # Libera a RAM explicitamente antes da próxima escala
        if img_esc is not img_base:
            del img_esc
    del img_base
    return pd.DataFrame(resultados)


def validacao_foca(df_raw):
    """Filtra outliers (percentis 5 e 95) e valida estabilidade (média, desvio, CV < 15%)."""
    df_filtered = pd.DataFrame()
    for (n, k), group in df_raw.groupby(['n', 'k']):
        p05 = group['tempo'].quantile(0.05)
        p95 = group['tempo'].quantile(0.95)
        valid_mask = (group['tempo'] >= p05) & (group['tempo'] <= p95)
        df_filtered = pd.concat([df_filtered, group[valid_mask]])

    resumo = df_filtered.groupby(['n', 'k']).agg(
        escala=('escala', 'first'),
        tempo_medio=('tempo', 'mean'),
        desvio=('tempo', 'std'),
        contagem_valida=('tempo', 'count')
    ).reset_index()

    resumo['CV'] = resumo['desvio'] / resumo['tempo_medio']
    resumo['estavel_CV_menor_15'] = resumo['CV'] < 0.15
    return resumo


def ajustar_curva(resumo):
    """Ajuste não linear do modelo T(n,k). Retorna parâmetros (popt) e R²."""
    X_data = np.vstack((resumo['n'], resumo['k']))
    y_data = resumo['tempo_medio']

    # O p0 pode precisar de sintonia dependendo da escala de tempo real
    popt, _ = curve_fit(model_n_k, X_data, y_data, p0=[1e-6, 1e-6, 1e-6, 0.01], maxfev=10000)
    resumo['tempo_predito'] = model_n_k(X_data, *popt)

    ss_res = np.sum((y_data - resumo['tempo_predito']) ** 2)
    ss_tot = np.sum((y_data - y_data.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return popt, r2


# ---------------------------- GRÁFICOS 2D ----------------------------
COLORS_K = ['red', 'green', 'blue', 'orange', 'purple']


def _desenhar_tempo_vs_n(ax, resumo, k_list, titulo):
    tem_pred = 'tempo_predito' in resumo.columns
    for idx, k in enumerate(k_list):
        dados_k = resumo[resumo['k'] == k]
        if not dados_k.empty:
            c = COLORS_K[idx % len(COLORS_K)]
            ax.plot(dados_k['n'], dados_k['tempo_medio'], marker='o',
                    color=c, label=f'K={k} (Empírico)')
            if tem_pred:
                ax.plot(dados_k['n'], dados_k['tempo_predito'], linestyle='--',
                        color=c, alpha=0.6)
    ax.set_xlabel('Número de Pixels (n)')
    ax.set_ylabel('Tempo Médio de Execução (s)')
    ax.set_title(f'Tempo vs Tamanho da Imagem - {titulo}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _desenhar_tempo_vs_k(ax, resumo, k_list, titulo):
    tem_pred = 'tempo_predito' in resumo.columns
    n_processados = sorted(resumo['n'].unique())
    colors_n = plt.cm.viridis(np.linspace(0, 1, len(n_processados)))
    for idx, n_val in enumerate(n_processados):
        dados_n = resumo[resumo['n'] == n_val]
        if not dados_n.empty:
            ax.plot(dados_n['k'], dados_n['tempo_medio'], marker='s',
                    color=colors_n[idx], label=f'n={n_val}')
            if tem_pred:
                ax.plot(dados_n['k'], dados_n['tempo_predito'], linestyle='--',
                        color=colors_n[idx], alpha=0.6)
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('Tempo Médio de Execução (s)')
    ax.set_title(f'Tempo vs Número de Clusters - {titulo}')
    ax.set_xticks(k_list)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plotar_2d(resumo, k_list, pasta, titulo):
    """Gráficos 2D em arquivos separados (Tempo vs n, Tempo vs K) + versão combinada."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _desenhar_tempo_vs_n(ax, resumo, k_list, titulo)
    finalizar_figura(fig, os.path.join(pasta, 'grafico_2d_tempo_vs_n.png'))

    fig, ax = plt.subplots(figsize=(8, 5))
    _desenhar_tempo_vs_k(ax, resumo, k_list, titulo)
    finalizar_figura(fig, os.path.join(pasta, 'grafico_2d_tempo_vs_k.png'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _desenhar_tempo_vs_n(ax1, resumo, k_list, titulo)
    _desenhar_tempo_vs_k(ax2, resumo, k_list, titulo)
    plt.tight_layout()
    finalizar_figura(fig, os.path.join(pasta, 'relatorio_complexidade.png'))


# ---------------------------- SUPERFÍCIE 3D ----------------------------
def plotar_superficie_3d(resumo, popt, pasta, titulo):
    """Superfície 3D T(n, k): modelo ajustado (superfície) + pontos empíricos (scatter)."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if (popt is not None and resumo['n'].min() < resumo['n'].max()
            and resumo['k'].min() < resumo['k'].max()):
        n_grade = np.linspace(resumo['n'].min(), resumo['n'].max(), 30)
        k_grade = np.linspace(resumo['k'].min(), resumo['k'].max(), 30)
        N_g, K_g = np.meshgrid(n_grade, k_grade)
        T_g = model_n_k((N_g, K_g), *popt)
        ax.plot_surface(N_g, K_g, T_g, cmap='viridis', alpha=0.55, edgecolor='none')

    ax.scatter(resumo['n'], resumo['k'], resumo['tempo_medio'],
               color='red', s=45, depthshade=False, label='Empírico (média filtrada)')

    ax.set_xlabel('Número de Pixels (n)')
    ax.set_ylabel('Número de Clusters (K)')
    ax.set_zlabel('Tempo Médio (s)')
    ax.set_title(f'Superfície de Tempo T(n, K) - {titulo}')
    ax.set_yticks(sorted(resumo['k'].unique()))
    ax.legend()
    finalizar_figura(fig, os.path.join(pasta, 'superficie_tempo_3d.png'))


# ---------------------------- TABELA DE VALIDAÇÃO ----------------------------
def salvar_tabela_png(resumo, pasta, titulo):
    """Renderiza a tabela de validação estatística como imagem (verde = estável, vermelho = instável)."""
    celulas = []
    for _, r in resumo.iterrows():
        celulas.append([
            f"{r['escala']:.0%}", f"{int(r['n'])}", f"{int(r['k'])}", f"{r['tempo_medio']:.4f}",
            "-" if pd.isna(r['desvio']) else f"{r['desvio']:.4f}",
            f"{int(r['contagem_valida'])}",
            "-" if pd.isna(r['CV']) else f"{r['CV']:.4f}",
            "Sim" if r['estavel_CV_menor_15'] else "Não"
        ])

    altura = max(2.0, 0.45 * (len(celulas) + 2))
    fig, ax = plt.subplots(figsize=(13, altura))
    ax.axis('off')
    tabela = ax.table(cellText=celulas,
                      colLabels=['Escala', 'n', 'K', 'Tempo médio (s)', 'Desvio (s)',
                                 'Amostras válidas', 'CV', 'Estável (CV<15%)'],
                      loc='center', cellLoc='center')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1, 1.4)
    for i, linha in enumerate(celulas, start=1):
        cor = '#d9f2d9' if linha[-1] == "Sim" else '#f8d7d7'
        tabela[(i, 7)].set_facecolor(cor)
    ax.set_title(f'Tabela de Validação Estatística (F.O.C.A.) - {titulo}', pad=12)
    finalizar_figura(fig, os.path.join(pasta, 'tabela_validacao_foca.png'))


def executar_analise_foca(caminho, escalas, k_list, R, m, max_iter, epsilon):
    """Pipeline completo para UMA imagem: execução -> validação F.O.C.A. -> ajuste -> gráficos -> arquivos."""
    nome_base = os.path.splitext(os.path.basename(caminho))[0]
    pasta = preparar_pasta_resultados(nome_base)

    tamanhos = calcular_tamanhos(caminho, escalas)
    lista_escalas = ", ".join("{:.0%}".format(t['escala']) for t in tamanhos)
    print(f"Imagem: {os.path.basename(caminho)} | escalas: {lista_escalas}")

    df_raw = executar_experimento(caminho, tamanhos, k_list, R, m, max_iter, epsilon)

    if df_raw.empty:
        print("\nNenhum dado processado.")
        return

    print(f"\nSalvando resultados em: {pasta}")

    # 1) Dados brutos (todas as repetições)
    caminho_brutos = os.path.join(pasta, 'dados_brutos.csv')
    df_raw.to_csv(caminho_brutos, index=False, float_format='%.6f', sep=';', decimal=',')
    print(f"   [salvo] {caminho_brutos}")

    # 2) Validação F.O.C.A.
    resumo = validacao_foca(df_raw)
    print("\n--- Tabela de Validação Estatística (F.O.C.A.) ---")
    print(resumo.to_string(index=False))

    # 3) Ajuste de curvas (precisa de pelo menos 4 combinações (n, K) para 4 parâmetros)
    popt, r2 = None, None
    if resumo['n'].nunique() < 2 or len(resumo) < 4:
        print("\n[AVISO] Ajuste de curva impossível: use pelo menos 2 escalas diferentes "
              "(e 4 combinações n x K no total).")
    else:
        popt, r2 = ajustar_curva(resumo)
        print("\n--- Modelo ajustado: T = a*(n*K) + b*n + c*K + d ---")
        print(f"a={popt[0]:.3e}  b={popt[1]:.3e}  c={popt[2]:.3e}  d={popt[3]:.3e}  |  R²={r2:.4f}")
        caminho_modelo = os.path.join(pasta, 'modelo_ajustado.txt')
        with open(caminho_modelo, 'w', encoding='utf-8') as f:
            f.write(f"Imagem: {os.path.basename(caminho)}\n")
            f.write("Modelo: T(n, K) = alpha*(n*K) + beta*n + gamma*K + delta\n")
            f.write(f"alpha = {popt[0]:.6e}\nbeta  = {popt[1]:.6e}\n")
            f.write(f"gamma = {popt[2]:.6e}\ndelta = {popt[3]:.6e}\nR2    = {r2:.6f}\n")
        print(f"   [salvo] {caminho_modelo}")

    # 4) Tabela de validação (CSV para Excel PT-BR + imagem)
    arquivo_csv = os.path.join(pasta, 'resultados_foca_aluno.csv')
    resumo.to_csv(arquivo_csv, index=False, float_format='%.6f', sep=';', decimal=',')
    print(f"   [salvo] {arquivo_csv}")
    salvar_tabela_png(resumo, pasta, nome_base)

    # 5) Gráficos 2D e superfície 3D
    k_plot = sorted(resumo['k'].unique())
    plotar_2d(resumo, k_plot, pasta, nome_base)
    plotar_superficie_3d(resumo, popt, pasta, nome_base)

    print(f"\nExperimento concluído. Todos os arquivos estão em: {pasta}")

# ==============================================================
# MAIN 
# ==============================================================
if __name__ == "__main__":
    # Defina o caminho da sua imagem real de teste e o nome da saída.
    # Também pode passar pelo terminal:  python FCM_PURO_v3_FOCA.py minha_foto.jpg
    imagem_entrada = "image3.jpg"          
    if len(sys.argv) > 1:
        imagem_entrada = sys.argv[1]
    imagem_entrada = resolver_caminho(imagem_entrada)
    
    K_clusters = 4                        
    fator_m = 2.0                         
    iteracoes_max = 100                    
    limite_epsilon = 0.001                
    
    print(f"--- INICIANDO PROCESSAMENTO DE BANCADA ---")
    print(f"Figura de Entrada: {imagem_entrada}")
    print(f"Clusters (K): {K_clusters}")

    # 1. LEITURA DA IMAGEM DO DISCO
    try:
        imagem = Image.open(imagem_entrada).convert("RGB")

        plt.figure(figsize=(8, 8))
        plt.title(f"Imagem Original")
        plt.imshow(imagem)
        plt.axis('off')
        plt.show()
        
        t0 = time.perf_counter()
        
        # ATENÇÃO: Agora pegando os 3 retornos corretamente
        centroides, pertinencias, imagem_segmentada = fcm_processar_imagem(
            img=imagem,
            K=K_clusters,
            m=fator_m,
            max_iter=iteracoes_max,
            epsilon=limite_epsilon
        )
        
        t1 = time.perf_counter()
        tempo_total = t1 - t0
        
        print(f"\n[SUCESSO] Processamento finalizado com folga!")
        print(f"Tempo Total de Execução (Bancada): {tempo_total:.4f} segundos")

        # Salva a imagem original e a segmentada na pasta de resultados da imagem
        nome_base = os.path.splitext(os.path.basename(imagem_entrada))[0]
        pasta_res = preparar_pasta_resultados(nome_base)
        imagem.save(os.path.join(pasta_res, f"{nome_base}_original.png"))
        imagem_segmentada.save(os.path.join(pasta_res, f"{nome_base}_segmentada_K{K_clusters}.png"))
        print(f"   [salvo] imagens original e segmentada em: {pasta_res}")
        
        # Plotagem usando Matplotlib
        plt.figure(figsize=(8, 8))
        plt.title(f"FCM Puro Segmentado - {K_clusters} Clusters")
        plt.imshow(imagem_segmentada)
        plt.axis('off')
        plt.show()
        
    except FileNotFoundError:
        print(f"\n[ERRO] O arquivo '{imagem_entrada}' não foi encontrado.")
    except Exception as e:
        print(f"\n[ERRO INESPERADO] Ocorreu uma falha durante o processamento: {e}")

    # ----------------------------------------------------------
    # EXPERIMENTO F.O.C.A. (gráficos 2D, superfície 3D e tabela)
    # Roda sobre a MESMA imagem escolhida acima (imagem_entrada).
    # ----------------------------------------------------------
    k_list = [2, 3, 4]
    escalas = [0.25, 0.5, 0.75, 1.0]   # frações da imagem original -> geram os diferentes 'n'
    R = 30                             # Número de repetições por combinação

    if os.path.isfile(imagem_entrada):
        print(f"\n--- INICIANDO EXPERIMENTO F.O.C.A. ---")
        try:
            executar_analise_foca(
                caminho=imagem_entrada,
                escalas=escalas,
                k_list=k_list,
                R=R,
                m=fator_m,
                max_iter=iteracoes_max,
                epsilon=limite_epsilon
            )
        except Exception as e:
            print(f"\n[ERRO INESPERADO NO EXPERIMENTO] {e}")
