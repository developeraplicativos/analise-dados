import pandas as pd
import os 
import numpy as np 
from sqlalchemy import create_engine, exc
# Imports para visualização de dados
import matplotlib.pyplot as plt
import matplotlib as m
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX 
import seaborn as sns
# import seaborn as sns 
import pymysql
# Filtra os warnings
import warnings
warnings.filterwarnings('ignore') 

def create_archive():
    # Carregando os dados  
    conexao_transacional = pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="vendas", 
        charset="utf8mb4",     # Padrão de codificação
        cursorclass=pymysql.cursors.Cursor
    ) 
    try:
        # Configurações de conexão
        host = "localhost"
        port = 3306
        usuario = "root"
        password = "root"
        bancoorigem = "vendas_analise" 
        # Cria engine de conexão SQLAlchemy com PyMySQL
        conexao_fato = create_engine(
            f"mysql+pymysql://{usuario}:{password}@{host}:{port}/{bancoorigem}"
        ) 
        # Testa a conexão
        with conexao_fato.connect() as connori:
            print(f"Conexão com o banco '{bancoorigem}' estabelecida com sucesso.")

    except exc.SQLAlchemyError as e:
        print(f"Erro ao conectar ao banco '{bancoorigem}': {e}")
        exit()

    dados = pd.read_sql(
        '''SELECT DATE_FORMAT(pedidos.datadopagamento, '%Y-%m') AS periodo, 
            SUM(itens.quantidade * itens.precounitario * ( 1 - pedidos.desconto)) as total_vendido 
            FROM pedidos 
                inner join itens on itens.idpedido = pedidos.id
            WHERE pedidos.datadopagamento is not null
            GROUP BY 1
            ORDER BY 1''' 
        , conexao_transacional)

    dados['periodo'] = pd.to_datetime(dados['periodo'])
    dados = dados.set_index('periodo').sort_index() 
    idx_completo = pd.date_range(dados.index.min(), dados.index.max(), freq='MS')
    dados_completo = dados.reindex(idx_completo, fill_value=0)
    dados_completo.index.name = 'periodo'

    print(dados_completo)
    dados_completo.to_csv('archive/new_table.csv')
    return True

'''
Dando os devidos créditos:
Esta função foi recebida no curso de Modelagem de Séries Temporais e Real-Time Analytics 
com Apache Spark e Databricks oferecido pela DSA - Data Science Academy
'''

# Função para testar a estacionaridade
def dsa_testa_estacionaridade(serie, window = 12, title = 'Estatísticas Móveis e Teste Dickey-Fuller'):
    """
    Função para testar a estacionaridade de uma série temporal.
    
    Parâmetros:
    - serie: pandas.Series. Série temporal a ser testada.
    - window: int. Janela para cálculo das estatísticas móveis.
    - title: str. Título para os gráficos.
    """
    # Calcula estatísticas móveis
    rolmean = serie.rolling(window = window).mean()
    rolstd = serie.rolling(window = window).std()

    # Plot das estatísticas móveis
    plt.figure(figsize = (14, 6))
    plt.plot(serie, color = 'blue', label = 'Original')
    plt.plot(rolmean, color = 'red', label = 'Média Móvel')
    plt.plot(rolstd, color = 'black', label = 'Desvio Padrão Móvel')
    plt.legend(loc = 'best')
    plt.title(title)
    plt.show(block = False)
    
    # Teste Dickey-Fuller
    print('\nResultado do Teste Dickey-Fuller:')
    dfteste = adfuller(serie, autolag = 'AIC')
    dfsaida = pd.Series(dfteste[0:4], index = ['Estatística do Teste', 
                                               'Valor-p', 
                                               'Número de Lags Consideradas', 
                                               'Número de Observações Usadas'])
    for key, value in dfteste[4].items():
        dfsaida['Valor Crítico (%s)' % key] = value
        
    print(dfsaida)
    
    # Conclusão baseada no valor-p
    if dfsaida['Valor-p'] > 0.05:
        print('\nConclusão:\nO valor-p é maior que 0.05 e, portanto, não temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente não é estacionária.')
    else:
        print('\nConclusão:\nO valor-p é menor que 0.05 e, portanto,temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente é estacionária.')

def tendencia_show(dados):
    dados.plot(figsize = (15, 6))
    plt.show()

def histograma_show(dados):
    # Plot
    plt.figure(1)
    # Subplot 1
    plt.subplot(211) 
    dados.total_vendido.hist()
    # Subplot 2
    plt.subplot(212)
    dados.total_vendido.plot(kind = 'kde')
    plt.show()

def boxplof_mediana_outline(dados):
    dados['periodo'] = pd.to_datetime(dados['periodo'])
    dados = dados.set_index('periodo')
    # Define a área de plotagem para os subplots (os boxplots)
    fig, ax = plt.subplots(figsize = (15,6)) 
    # Define as variáveis
    indice_ano = dados.index.year
    valor = dados.total_vendido 
    # Cria um box plot para cada ano usando o Seaborn
    # Observe que estamos extraindo o ano (year) do índice da série
    sns.boxplot(x = indice_ano, y = valor, ax = ax, data = dados) 
    plt.xlabel("\nAno")
    plt.ylabel("\nQuantidade de Acidentes")
    plt.show() 



def serie_temporal_tendencia_sazonalidade_ruido(dados):
    # Plot
    dados['periodo'] = pd.to_datetime(dados['periodo'])
    dados = dados.set_index('periodo')
    
    serie = dados['total_vendido']  # <- apenas a série numérica
    decomposicao_aditiva = seasonal_decompose(
        serie,
        model='additive',
        extrapolate_trend='freq'
    )
    decomposicao_aditiva.plot()
    plt.show() 

def modelagem_naive(dados):
    # Plot
    figure(figsize = (15, 6))
    plt.title("Previsão Usando Método Naive") 
    plt.plot(dados.index, dados['total_vendido'], label = 'Dados de Treino') 
    plt.plot(dados.index, dados['total_vendido'], label = 'Dados de Validação') 
    plt.plot(dados_cp.index, dados_cp['previsao_naive'], label = 'Naive Forecast',color='black') 
    plt.legend(loc = 'best') 
    plt.show()

def testes_sarimax():
    import itertools
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX

   
    warnings.filterwarnings("ignore")  # ignora avisos de convergência

    # ======== PARÂMETROS PARA BUSCA MANUAL ========
    p = d = q = [0,1]        # menor espaço para estabilidade
    P = D = Q = [0,1]
    s = 12                   # sazonalidade mensal (ajuste conforme sua série)

    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None

    # ======== PRÉ-PROCESSAMENTO ========
    dados['periodo'] = pd.to_datetime(dados['periodo'], errors='coerce')
    serie = dados.set_index('periodo')['total_vendido'].dropna()

    # ======== SUAVIZA OUTLIERS ========
    serie_suave = serie.clip(upper=serie.quantile(0.99))

    # ======== TREINO E TESTE ========
    n = len(serie_suave)
    corte = int(n * 0.8)
    train = serie_suave.iloc[:corte]
    test = serie_suave.iloc[corte:]

    print(f'Treino: {len(train)} períodos')
    print(f'Teste: {len(test)} períodos')

    # ======== LOG-TRANSFORM ========
    train_log = np.log1p(train)
    test_log = np.log1p(test)

    # ======== BUSCA MANUAL DO MELHOR SARIMAX ========
    for order in itertools.product(p,d,q):
        for seasonal_order in itertools.product(P,D,Q):
            try:
                model = SARIMAX(train_log,
                                order=order,
                                seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], s),
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit(disp=False)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
            except:
                continue

    print(f"Melhor AIC: {best_aic}")
    print(f"Melhor order: {best_order}")
    print(f"Melhor seasonal_order: {best_seasonal_order}")

    # ======== AJUSTE FINAL COM MELHOR MODELO ========
    modelo_final = SARIMAX(train_log,
                        order=best_order,
                        seasonal_order=(best_seasonal_order[0], best_seasonal_order[1], best_seasonal_order[2], s),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)

    # ======== PREVISÃO ========
    previsao_log = modelo_final.get_forecast(steps=len(test_log))
    previsao_media_log = previsao_log.predicted_mean
    intervalo_confianca_log = previsao_log.conf_int()

    # Revertendo log-transform
    previsao_media = np.expm1(previsao_media_log)
    intervalo_confianca = np.expm1(intervalo_confianca_log)

    # ======== AVALIAÇÃO ========
    mae = mean_absolute_error(test, previsao_media)
    rmse = np.sqrt(mean_squared_error(test, previsao_media))
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # ======== VISUALIZAÇÃO ========
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Treino')
    plt.plot(test.index, test, label='Teste', color='orange')
    plt.plot(test.index, previsao_media, label='Previsão', color='green')
    plt.fill_between(test.index,
                    intervalo_confianca.iloc[:,0],
                    intervalo_confianca.iloc[:,1],
                    color='green', alpha=0.2)
    plt.legend()
    plt.show()


                    


def modelo_sarimax(dados):
    try:
        
        # Pré-processamento
        dados['periodo'] = pd.to_datetime(dados['periodo'], errors='coerce')
        serie = dados.set_index('periodo')['total_vendido'].dropna()

        # Suavizando outliers
        serie_suave = serie.clip(upper=serie.quantile(0.99))

        # Separando treino e teste
        n = len(serie_suave)
        corte = int(n * 0.8)
        train = serie_suave.iloc[:corte]
        test = serie_suave.iloc[corte:]

        print(f'Treino: {len(train)} períodos')
        print(f'Teste: {len(test)} períodos')

        # Log-transform
        train_log = np.log1p(train)
        test_log = np.log1p(test)

        # Treinamento do modelo final no treino
        modelo_log = SARIMAX(
            train_log,
            order=(0,1,1),
            seasonal_order=(1,0,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        # Previsão para o período de teste
        previsao_log = modelo_log.get_forecast(steps=len(test_log))
        previsao_media_log = previsao_log.predicted_mean
        intervalo_confianca_log = previsao_log.conf_int()

        # Revertendo log-transform
        previsao_media = np.expm1(previsao_media_log)
        intervalo_confianca = np.expm1(intervalo_confianca_log)

        # Avaliação
        mae = mean_absolute_error(test, previsao_media)
        rmse = np.sqrt(mean_squared_error(test, previsao_media))
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        # Visualização treino/teste
        plt.figure(figsize=(12,6))
        plt.plot(train.index, train, label='Treino')
        plt.plot(test.index, test, label='Teste', color='orange')
        plt.plot(test.index, previsao_media, label='Previsão', color='green')
        plt.fill_between(test.index,
                        intervalo_confianca.iloc[:,0],
                        intervalo_confianca.iloc[:,1],
                        color='green', alpha=0.2)
        plt.legend()
        plt.show()

        # Treinamento final no total da série (para previsão futura)
        serie_log_total = np.log1p(serie_suave)
        modelo_total = SARIMAX(
            serie_log_total,
            order=(0,1,1),
            seasonal_order=(1,0,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        # Previsão para os próximos 12 períodos
        previsao_futura_log = modelo_total.get_forecast(steps=12)
        previsao_futura = np.expm1(previsao_futura_log.predicted_mean)

        # Visualização histórica + previsão futura
        plt.figure(figsize=(12,6))
        plt.plot(serie_suave.index, serie_suave, label='Histórico')
        plt.plot(previsao_futura.index, previsao_futura, label='Previsão Futura', color='red')
        plt.legend()
        plt.show()
        exit()

        # previsao = ajuste.get_forecast(steps=12).predicted_mean

        
        # plt.figure(figsize=(12,5))
        # plt.plot(serie, label='Histórico')
        # plt.plot(previsao, label='Previsão SARIMAX', color='red')
        # plt.legend()
        # plt.show()

        # print(previsao.round(2))

    except Exception as ex:
        print("Ocorreu um erro:", ex)

def aplicacao_exponencial_smooth(dados):
    try:
        # Garante que o índice é datetime
        dados['periodo'] = pd.to_datetime(dados['periodo'], errors='coerce')
        serie = dados.set_index('periodo')['total_vendido'].dropna()
        
        # Ajusta o modelo
        modelo = SimpleExpSmoothing(serie)
        ajuste = modelo.fit(smoothing_level=0.6, optimized=False)
        
        # Faz a previsão
        previsao = ajuste.forecast(12)
        
        # Plota histórico + previsão
        plt.figure(figsize=(12,5))
        plt.plot(serie, label='Histórico')
        plt.plot(previsao, label='Previsão', color='red')
        plt.title('Previsão - Simple Exponential Smoothing')
        plt.xlabel('Período')
        plt.ylabel('Total Vendido')
        plt.legend()
        plt.show()
        
        # Mostra valores numéricos
        print(previsao)

    except Exception as ex:
        print("Ocorreu um erro:", ex)


if __name__ == "__main__":
    file_path = "archive/new_table.csv" 
    while os.path.exists(file_path) == False:
       create_archive()  
    dados = pd.read_csv(file_path)
    # tendencia_show(dados) 
    # histograma_show(dados)
    # boxplof_mediana_outline(dados)
    # serie_temporal_tendencia_sazonalidade_ruido(dados) 
    # dsa_testa_estacionaridade( dados['total_vendido'] ) #é estacionária
    '''
    Conclusão:
    O valor-p é menor que 0.05 e, portanto,temos evidências para rejeitar a hipótese nula.
    Essa série provavelmente é estacionária.
    '''

    # dados_total_vendidos = np.asarray(dados.total_vendido)
    # global dados_cp = dados.copy()
    # modelagem_naive(dados)
    # print(dados_total_vendidos)
    # aplicacao_exponencial_smooth(dados)
    # testes_sarimax()
    modelo_sarimax(dados)

    # modelo_sarimax(dados)
