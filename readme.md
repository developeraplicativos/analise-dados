# Analise de dados

pesquisa para apresentação do curso #treinarecife #analisededados

## Bibliotecas necessárias

pandas  
os  
numpy  
create_engine, exc  
matplotlib  
statsmodels  
seasonal_decompose  
SimpleExpSmoothing  
mean_absolute_error, mean_squared_error  
adfuller  
SARIMAX  
seaborn  
pymysql  
warnings  

## resultado
**1- já foram removido os outlines**
**2- a maior quantidade de meses alcançaram entre 1.0 e 1.1**
![alt text](https://github.com/developeraplicativos/analise-dados/blob/main/archive/histograma.png?raw=true)

**3- informa que está havendo um crescimento em sua tendencia com distribuições moderadas e estaveis**
![alt text](https://github.com/developeraplicativos/analise-dados/blob/main/archive/boxplot.png?raw=true)

**4- informa que está havendo um crescimento em sua tendencia com distribuições moderadas e estaveis** 
Resultado do Teste Dickey-Fuller:  
Estatística do Teste            -4.575596  
Valor-p                          0.000143  
Número de Lags Consideradas      5.000000  
Número de Observações Usadas    52.000000  
Valor Crítico (1%)              -3.562879  
Valor Crítico (5%)              -2.918973  
Valor Crítico (10%)             -2.597393  

**5- com a analise de dickey fuller é possível visualizar que valor-p é menor que 0.05 e que a serie é estacionária** 
![alt text](https://github.com/developeraplicativos/analise-dados/blob/main/archive/sarimax.png?raw=true) 
