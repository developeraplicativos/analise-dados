import pandas as pd
import os
import numpy as np
from sqlalchemy import create_engine, exc
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
import pymysql
import warnings
warnings.filterwarnings('ignore')


def create_archive():
    # Carregando os dados
    conexao_transacional = pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="vendas",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor
    )
    try:
        # Configurações de conexão (SQLAlchemy - apenas para teste; não usado no read_sql abaixo)
        host = "localhost"
        port = 3306
        usuario = "root"
        password = "root"
        bancoorigem = "vendas_analise"
        conexao_fato = create_engine(
            f"mysql+pymysql://{usuario}:{password}@{host}:{port}/{bancoorigem}"
        )
        with conexao_fato.connect() as connori:
            print(f"Conexão com o banco '{bancoorigem}' estabelecida com sucesso.")
    except exc.SQLAlchemyError as e:
        print(f"Erro ao conectar ao banco '{bancoorigem}': {e}")
        exit()

    # Consulta original (retorna 'periodo' e 'total_vendido')
    dados = pd.read_sql(
        '''SELECT DATE_FORMAT(pedidos.datadopagamento, '%Y-%m') AS periodo, 
            SUM(itens.quantidade * itens.precounitario * ( 1 - pedidos.desconto)) as total_vendido 
            FROM pedidos 
                inner join itens on itens.idpedido = pedidos.id
            WHERE pedidos.datadopagamento is not null
            GROUP BY 1
            ORDER BY 1''',
        conexao_transacional)

    # garante tipo datetime e índice contínuo mensal
    dados['periodo'] = pd.to_datetime(dados['periodo'], errors='coerce')
    dados = dados.set_index('periodo').sort_index()
    idx_completo = pd.date_range(dados.index.min(), dados.index.max(), freq='MS')
    dados_completo = dados.reindex(idx_completo, fill_value=0)
    dados_completo.index.name = 'periodo'

    print(dados_completo.head())
    os.makedirs('archive', exist_ok=True)
    dados_completo.to_csv('archive/new_table.csv', index=True)
    return True


def dsa_testa_estacionaridade(serie, window=12, title='Estatísticas Móveis e Teste Dickey-Fuller'):
    rolmean = serie.rolling(window=window).mean()
    rolstd = serie.rolling(window=window).std()

    plt.figure(figsize=(14, 6))
    plt.plot(serie, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Média Móvel')
    plt.plot(rolstd, color='black', label='Desvio Padrão Móvel')
    plt.legend(loc='best')
    plt.title(title)
    plt.show()

    print('\nResultado do Teste Dickey-Fuller:')
    dfteste = adfuller(serie.dropna(), autolag='AIC')
    dfsaida = pd.Series(dfteste[0:4], index=[
                       'Estatística do Teste', 'Valor-p', 'Número de Lags Consideradas', 'Número de Observações Usadas'])
    for key, value in dfteste[4].items():
        dfsaida['Valor Crítico (%s)' % key] = value

    print(dfsaida)
    if dfsaida['Valor-p'] > 0.05:
        print('\nConclusão:\nO valor-p é maior que 0.05 e, portanto, não temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente não é estacionária.')
    else:
        print('\nConclusão:\nO valor-p é menor que 0.05 e, portanto,temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente é estacionária.')


def tendencia_show(dados):
    df = dados.copy()
    # aceita periodo como coluna ou como índice
    if 'periodo' in df.columns:
        df['periodo'] = pd.to_datetime(df['periodo'], errors='coerce')
        df = df.dropna(subset=['periodo'])
        df = df.set_index('periodo').sort_index()
    else:
        # tenta garantir índice datetime
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.dropna(axis=0, subset=[df.columns[0]])  # mantém somente linhas com dados
        except Exception:
            pass

    if 'total_vendido' not in df.columns:
        raise ValueError("Coluna 'total_vendido' não encontrada para plotagem de tendência.")

    df['total_vendido'].plot(figsize=(15, 6))
    plt.title('Tendência - Total Vendido')
    plt.ylabel('Total Vendido')
    plt.xlabel('Período')
    plt.tight_layout()
    plt.show()


def histograma_show(dados):
    df = dados.copy()
    # garante coluna total_vendido
    if 'total_vendido' not in df.columns:
        raise ValueError("Coluna 'total_vendido' não encontrada para histograma.")
    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    df['total_vendido'].hist()
    plt.subplot(212)
    df['total_vendido'].plot(kind='kde')
    plt.tight_layout()
    plt.show()


def boxplof_mediana_outline(dados):
    """
    Boxplot anual usando 'periodo' (coluna ou índice) e 'total_vendido'.
    Exibe boxplot por ano.
    """
    df = dados.copy()

    # Normaliza a coluna periodo: se está como coluna, converte; se está no índice, cria coluna
    if 'periodo' in df.columns:
        df['periodo'] = pd.to_datetime(df['periodo'], errors='coerce')
    else:
        # cria coluna a partir do índice (caso índice seja datetime)
        try:
            df = df.copy()
            df['periodo'] = pd.to_datetime(df.index, errors='coerce')
        except Exception:
            pass

    if 'periodo' not in df.columns or 'total_vendido' not in df.columns:
        raise ValueError("DataFrame precisa ter 'periodo' e 'total_vendido' para desenhar boxplot.")

    df = df.dropna(subset=['periodo', 'total_vendido'])
    df = df.sort_values('periodo')

    df['ano'] = df['periodo'].dt.year
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='ano', y='total_vendido', data=df)
    plt.xlabel("Ano")
    plt.ylabel("Total Vendido")
    plt.title("Boxplot anual - Total Vendido")
    plt.tight_layout()
    plt.show()


def serie_temporal_tendencia_sazonalidade_ruido(dados):
    df = dados.copy()
    # garante a série com índice periodo
    if 'periodo' in df.columns:
        df['periodo'] = pd.to_datetime(df['periodo'], errors='coerce')
        df = df.set_index('periodo').sort_index()
    else:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.sort_index()

    if 'total_vendido' not in df.columns:
        raise ValueError("Coluna 'total_vendido' não encontrada para decomposição.")

    serie = df['total_vendido'].dropna()
    if len(serie) < 24:
        raise ValueError("Série muito curta para decomposição (recomenda-se >= 24 observações).")

    decomposicao_aditiva = seasonal_decompose(serie, model='additive', extrapolate_trend='freq')
    decomposicao_aditiva.plot()
    plt.tight_layout()
    plt.show()


def modelagem_naive(dados, horizon=12):
    """
    Naive forecast: último valor observado projetado para os próximos 'horizon' períodos.
    Retorna a série de previsão (pandas.Series com índice datetime continous).
    """
    df = dados.copy()
    if 'periodo' in df.columns:
        df['periodo'] = pd.to_datetime(df['periodo'], errors='coerce')
        df = df.set_index('periodo').sort_index()
    else:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.sort_index()

    if 'total_vendido' not in df.columns:
        raise ValueError("Coluna 'total_vendido' necessária para modelagem naive.")

    last = df['total_vendido'].dropna().iloc[-1]
    start = df.index.max() + pd.offsets.MonthBegin(1)
    idx = pd.date_range(start, periods=horizon, freq='MS')
    forecast = pd.Series(last, index=idx, name='naive_forecast')

    # plot histórico + previsão
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['total_vendido'], label='Histórico')
    plt.plot(forecast.index, forecast.values, label='Naive Forecast', color='black')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return forecast


def testes_sarimax(dados, p=[0, 1], d=[0, 1], q=[0, 1], P=[0, 1], D=[0, 1], Q=[0, 1], s=12):
    """
    Busca manual simples por combinação de parâmetros SARIMAX (espaço reduzido).
    Retorna o melhor modelo ajustado (statsmodels result) e informações (order, seasonal_order).
    """
    import itertools
    warnings.filterwarnings("ignore")

    df = dados.copy()
    # normalização de periodo
    if 'periodo' in df.columns:
        df['periodo'] = pd.to_datetime(df['periodo'], errors='coerce')
        df = df.set_index('periodo').sort_index()
    else:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.sort_index()

    if 'total_vendido' not in df.columns:
        raise ValueError("Coluna 'total_vendido' necessária para testes SARIMAX.")

    serie = df['total_vendido'].dropna()

    # suaviza outliers (clipping)
    serie_suave = serie.clip(upper=serie.quantile(0.99))

    n = len(serie_suave)
    corte = int(n * 0.8)
    train = serie_suave.iloc[:corte]
    test = serie_suave.iloc[corte:]

    print(f'Treino: {len(train)} períodos')
    print(f'Teste: {len(test)} períodos')

    train_log = np.log1p(train)
    test_log = np.log1p(test)

    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None

    for order in itertools.product(p, d, q):
        for seasonal_order in itertools.product(P, D, Q):
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
            except Exception:
                continue

    print(f"Melhor AIC: {best_aic}")
    print(f"Melhor order: {best_order}")
    print(f"Melhor seasonal_order: {best_seasonal_order}")

    # Ajuste com melhor combinação (se foi encontrada)
    if best_order is None or best_seasonal_order is None:
        print("Nenhuma combinação válidas encontrada no espaço de busca.")
        return None, None, None

    modelo_final = SARIMAX(train_log,
                          order=best_order,
                          seasonal_order=(best_seasonal_order[0], best_seasonal_order[1], best_seasonal_order[2], s),
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)

    # Previsão
    previsao_log = modelo_final.get_forecast(steps=len(test_log))
    previsao_media_log = previsao_log.predicted_mean
    intervalo_confianca_log = previsao_log.conf_int()

    previsao_media = np.expm1(previsao_media_log)
    intervalo_confianca = np.expm1(intervalo_confianca_log)

    mae = mean_absolute_error(test, previsao_media)
    rmse = np.sqrt(mean_squared_error(test, previsao_media))
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Treino')
    plt.plot(test.index, test, label='Teste', color='orange')
    plt.plot(test.index, previsao_media, label='Previsão', color='green')
    plt.fill_between(test.index, intervalo_confianca.iloc[:, 0], intervalo_confianca.iloc[:, 1], color='green', alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return modelo_final, best_order, best_seasonal_order


def modelo_sarimax(dados, order=(0, 1, 1), seasonal_order=(1, 0, 1, 12)):
    try:
        df = dados.copy()
        if 'periodo' in df.columns:
            df['periodo'] = pd.to_datetime(df['periodo'], errors='coerce')
            df = df.set_index('periodo').sort_index()
        else:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.sort_index()

        if 'total_vendido' not in df.columns:
            raise ValueError("Coluna 'total_vendido' necessária para modelo SARIMAX.")

        serie = df['total_vendido'].dropna()

        # Remoção de outliers com IQR (filtragem)
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        serie_suave = serie[(serie >= limite_inferior) & (serie <= limite_superior)]

        n = len(serie_suave)
        if n < 12:
            raise ValueError("Série muito curta após remoção de outliers para treinar SARIMAX.")

        corte = int(n * 0.8)
        train = serie_suave.iloc[:corte]
        test = serie_suave.iloc[corte:]

        train_log = np.log1p(train)
        test_log = np.log1p(test)

        modelo_log = SARIMAX(train_log,
                             order=order,
                             seasonal_order=seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False)

        previsao_log = modelo_log.get_forecast(steps=len(test_log))
        previsao_media_log = previsao_log.predicted_mean
        intervalo_confianca_log = previsao_log.conf_int()

        previsao_media = np.expm1(previsao_media_log)
        intervalo_confianca = np.expm1(intervalo_confianca_log)

        mae = mean_absolute_error(test, previsao_media)
        rmse = np.sqrt(mean_squared_error(test, previsao_media))
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label='Treino')
        plt.plot(test.index, test, label='Teste', color='orange')
        plt.plot(test.index, previsao_media, label='Previsão', color='green')
        plt.fill_between(test.index, intervalo_confianca.iloc[:, 0], intervalo_confianca.iloc[:, 1], color='green', alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Treinamento final em toda a série para previsão futura
        serie_log_total = np.log1p(serie_suave)
        modelo_total = SARIMAX(serie_log_total,
                              order=order,
                              seasonal_order=seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False)

        previsao_futura_log = modelo_total.get_forecast(steps=12)
        previsao_futura = np.expm1(previsao_futura_log.predicted_mean)

        plt.figure(figsize=(12, 6))
        plt.plot(serie_suave.index, serie_suave, label='Histórico (sem outliers)')
        plt.plot(previsao_futura.index, previsao_futura, label='Previsão Futura', color='red')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return modelo_total

    except Exception as ex:
        print("Ocorreu um erro no modelo SARIMAX:", ex)
        return None 

def remove_outliers_iqr(dados, col='total_vendido', factor=1.5):
    """
    Remove outliers (abaixo e acima) usando IQR. Mantém coluna 'periodo' e 'total_vendido'.
    Retorna DataFrame limpo (com 'periodo' como coluna, não como índice).
    """
    df = dados.copy()

    # Normaliza coluna periodo: se for índice, copia para coluna
    if 'periodo' not in df.columns:
        try:
            df = df.copy()
            df['periodo'] = pd.to_datetime(df.index, errors='coerce')
        except Exception:
            pass

    # Garantir que colunas existam
    if 'periodo' not in df.columns:
        raise ValueError("Coluna 'periodo' não encontrada no DataFrame. Impossível remover outliers.")
    if col not in df.columns:
        raise ValueError(f"Coluna '{col}' não encontrada no DataFrame. Impossível remover outliers.")

    # Conversões e limpeza mínima
    df['periodo'] = pd.to_datetime(df['periodo'], errors='coerce')
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['periodo', col]).sort_values('periodo')

    # cálculo IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    df_filtrado = df[(df[col] >= lower) & (df[col] <= upper)].copy()
    # mantém coluna periodo como coluna (não transforma em índice) para consistência
    df_filtrado.reset_index(drop=True, inplace=True)
    return df_filtrado


if __name__ == "__main__":
    file_path = "archive/new_table.csv"
    while os.path.exists(file_path) == False:
        create_archive()
    dados = pd.read_csv(file_path)

    # Garante que as colunas mínimas existam (periodo / total_vendido)
    # se pd.read_csv trouxe o 'periodo' como índice string, ele estará como coluna
    if 'periodo' not in dados.columns:
        # se o arquivo foi salvo com índice, pandas cria coluna "Unnamed: 0" ou similar
        possible_index_cols = [c for c in dados.columns if 'Unnamed' in c or 'index' in c.lower()]
        if possible_index_cols:
            dados = dados.rename(columns={possible_index_cols[0]: 'periodo'})
        else:
            # última opção: descobrir se primeira coluna corresponde a datas
            first_col = dados.columns[0]
            try:
                pd.to_datetime(dados[first_col])
                dados = dados.rename(columns={first_col: 'periodo'})
            except Exception:
                pass

    # quick-check
    if 'periodo' not in dados.columns or 'total_vendido' not in dados.columns:
        raise ValueError("O CSV precisa ter as colunas 'periodo' e 'total_vendido' após leitura.")

    # Mostra tendência original
    tendencia_show(dados)

    # Remoção de outliers (filtragem)
    dados_suave = remove_outliers_iqr(dados)

    # Visualizações simples para checagem
    histograma_show(dados_suave)
    try:
        boxplof_mediana_outline(dados_suave)
    except Exception as e:
        print("Aviso ao desenhar boxplot:", e)

    print("Colunas após limpeza:", dados_suave.columns.tolist())

    # exemplos de uso (descomente conforme necessidade)
    # verificação de dataframe estacionaria < 0.05
    dsa_testa_estacionaridade(dados_suave.set_index('periodo')['total_vendido'])
    teste_modelo, best_order, best_seasonal = testes_sarimax(dados_suave)
    forecast_naive = modelagem_naive(dados_suave, horizon=12)
    modelo = modelo_sarimax(dados_suave) 
