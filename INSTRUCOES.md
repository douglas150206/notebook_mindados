# Notebook de Mineração de Dados: eficiência das cotações de apostas esportivas

Construa, a partir da pasta atual, um projeto Python com um notebook Jupyter para
a disciplina de Mineração de Dados (ADS, IFSP Câmpus Jacareí). O notebook precisa
gerar os números de um resumo que será submetido à JECET 2026 e implementar cada
etapa metodológica descrita nele. Textos, células markdown e comentários em
português. Leia este arquivo inteiro antes de começar.

## 1. O que o resumo afirma

O notebook precisa sustentar (ou refutar, se os dados mandarem) estes trechos.

**Metodologia declarada:**

> A amostra reúne registros públicos de cerca de quarenta mil partidas das
> principais ligas europeias, disputadas entre as temporadas de 2005 e 2025,
> contendo resultados, estatísticas de jogo e cotações de diferentes operadores.
> O tratamento compreende limpeza da base, padronização dos nomes das equipes,
> construção de atributos históricos calculados apenas com informações anteriores
> a cada partida e conversão das cotações em probabilidades normalizadas. A
> modelagem emprega árvores de decisão, florestas aleatórias e regressão
> logística, com separação temporal entre os conjuntos de treino e teste.

**Resultados a verificar e preencher:**

> Os resultados iniciais indicam que a margem embutida nas cotações varia de
> maneira consistente entre operadores e divisões e que as probabilidades
> atribuídas aos desfechos menos prováveis tendem a ser superestimadas. Na
> previsão do resultado das partidas, a árvore de decisão alcançou [XX]% de
> acerto, a regressão logística [XX]% e a floresta aleatória [XX]%, contra [XX]%
> obtidos ao se adotar apenas o favorito apontado pelas cotações.

**Conclusão provisória, válida só se os dados confirmarem:**

> Conclui-se que os modelos supervisionados aproximam-se do desempenho do
> mercado, mas não o superam de forma consistente.

Se alguma afirmação não se confirmar, o notebook deve dizer isso com clareza.
Nunca ajuste a análise para ela concordar com o texto.

## 2. Estrutura do projeto

```
./
├── INSTRUCOES.md            (este arquivo)
├── README.md                (como instalar e rodar)
├── requirements.txt         (versões fixadas)
├── mineracao_bets.ipynb     (notebook principal, executado, com saídas salvas)
├── src/utils.py             (opcional, se o notebook ficar longo)
├── data/raw/                (CSVs baixados; é cache, não baixar de novo se já existir)
├── data/processed/          (bases intermediárias)
└── outputs/
    ├── figuras/             (PNG, 150 dpi, títulos e eixos em português)
    ├── tabelas/             (CSVs com métricas e agregados)
    └── resultados_resumo.txt
```

Ambiente: Python 3.10+ em um `.venv`. Bibliotecas: pandas, numpy, scikit-learn,
matplotlib, statsmodels, requests, jupyter (com nbconvert e ipykernel). Nada além
disso sem necessidade real. Caminhos relativos com `pathlib`, para rodar tanto no
VSCode quanto no JupyterHub.

## 3. Etapas do notebook

Organize pelas fases do CRISP-DM, com células markdown explicando cada passo para
alguém da turma entender. Código em funções, sem células gigantes.

### 3.1 Coleta

* Fonte: Football-Data.co.uk. Padrão de URL:
  `https://www.football-data.co.uk/mmz4281/{temporada}/{divisao}.csv`, com a
  temporada no formato `0506` para 2005/06 e `2425` para 2024/25. Se o padrão
  falhar, confirme nas páginas de cada liga no site (ex.:
  `https://www.football-data.co.uk/englandm.php`).
* Dicionário de colunas: `https://www.football-data.co.uk/notes.txt`. Os nomes
  variam entre temporadas: a média de mercado aparece como `BbAvH` nas antigas e
  `AvgH` nas recentes; Pinnacle aparece como `PSH` ou `PH`; cotações de fechamento
  levam `C` (ex.: `B365CH`). Confirme em vez de supor.
* Ligas principais (modelagem e análises): `E0` Inglaterra, `SP1` Espanha,
  `I1` Itália, `D1` Alemanha, `F1` França.
* Segundas divisões (somente na análise de margem por divisão, 2005/06 a
  2024/25): `E1`, `SP2`, `I2`, `D2`, `F2`.
* Temporadas das ligas principais:
  * 2000/01 a 2004/05: apenas aquecimento do Elo e das médias móveis, fora da
    amostra analisada.
  * 2005/06 a 2024/25: amostra analisada.
* Leitura robusta: tente UTF-8 e caia para latin-1; descarte colunas e linhas
  totalmente vazias; conte e registre linhas malformadas em vez de sumir com elas
  em silêncio.
* Registre o que foi baixado com sucesso e o que falhou.

### 3.2 Limpeza e padronização

* Datas: aceitar `dd/mm/yy` e `dd/mm/yyyy`. Ordenar por data e, quando houver,
  horário.
* Padronização dos nomes das equipes em seção própria (está no resumo): remover
  espaços extras, procurar grafias diferentes do mesmo time entre temporadas e
  montar um dicionário de mapeamento explícito, exibido no notebook.
* Tabela de cobertura: percentual de valores não nulos das colunas relevantes
  (gols, finalizações, escanteios, cartões, cotações de cada operador) por liga e
  temporada. Salvar em `outputs/tabelas/cobertura.csv`.
* Remover partidas sem resultado ou sem cotações Bet365, registrando quantas.
* Imprimir o N final de partidas das ligas principais em 2005/06 a 2024/25. Esse
  número substitui o "cerca de quarenta mil" do resumo.

### 3.3 Cotações para probabilidades

* Operador principal: Bet365 (`B365H`, `B365D`, `B365A`), porque cobre o período
  todo. Confirmar a cobertura.
* Probabilidade implícita bruta: `1 / cotação`. Margem (overround): soma das três
  implícitas menos 1.
* Probabilidade normalizada: cada implícita dividida pela soma das três.
* Favorito do mercado: desfecho com maior probabilidade normalizada.

### 3.4 Análise do mercado

Esta seção sustenta a primeira frase de resultados.

a) Margem por operador: todos os operadores com cobertura razoável (Bet365,
   Bet&Win, Interwetten, Pinnacle, William Hill, VC Bet etc.), por temporada.
   Gráfico de linhas com a margem média por temporada, uma linha por operador.
b) Margem por divisão: primeira contra segunda divisão de cada país, com Bet365.
   Gráfico de barras.
c) Calibração: empilhar os três desfechos de cada partida (probabilidade
   normalizada Bet365; ocorreu ou não). Faixas de 10 em 10 pontos percentuais.
   Para cada faixa: número de observações, probabilidade média prevista,
   frequência observada e IC 95% de Wilson. Gráfico com a diagonal de referência.
d) Viés favorito-azarão: nas faixas de baixa probabilidade, a frequência
   observada fica abaixo da prevista? Reportar com os ICs. Complementar com
   regressão logística (statsmodels) do resultado sobre o logit da probabilidade;
   inclinação maior que 1 indica o viés. Se o viés não aparecer com as
   probabilidades normalizadas, dizer isso.
e) Retorno médio por faixa de cotação: apostar 1 unidade em cada desfecho de cada
   partida (Bet365); retorno igual a cotação − 1 se ganhou e −1 se perdeu. Faixas
   sugeridas: até 1,5; 1,5 a 2; 2 a 3; 3 a 5; 5 a 10; acima de 10. Mostrar com IC.
   É o gráfico mais didático do trabalho.
f) Pré-jogo contra fechamento, só nas temporadas em que existem as colunas com
   `C`: comparar o log loss das probabilidades normalizadas de cada conjunto.

### 3.5 Atributos históricos sem vazamento

* Transformar a base em formato longo (duas linhas por partida, uma na
  perspectiva de cada time) com gols feitos e sofridos, pontos, finalizações e
  finalizações no alvo feitas e sofridas, escanteios feitos e sofridos.
* Para cada time, ordenado por data, médias móveis das últimas 5 e 10 partidas,
  aplicando `shift(1)` antes do `rolling` para que a partida atual nunca entre no
  próprio cálculo. O histórico é contínuo entre temporadas e inclui o aquecimento.
* Forma por mando: média das últimas 5 partidas em casa do mandante e das últimas
  5 fora do visitante (gols e pontos).
* Elo pré-jogo, calculado por liga: inicial 1500; K = 20; vantagem de mando de 60
  pontos na expectativa (pode ajustar usando apenas o treino); regressão de 1/3 em
  direção à média na virada de temporada; promovidos entram com a média do Elo dos
  rebaixados da mesma liga na temporada anterior. Guardar o valor de antes da
  partida.
* Indicador de time promovido (não disputou a liga na temporada anterior).
* Diferença mandante menos visitante para cada métrica; liga em one-hot.
* Descartar partidas em que algum time tenha menos de 5 jogos anteriores na base,
  registrando quantas.

Colunas proibidas como atributo, porque são do próprio jogo:
`FTHG, FTAG, FTR, HTHG, HTAG, HTR, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR`
e equivalentes. Estatísticas de jogo só entram como médias de partidas anteriores.

Testes obrigatórios dentro do notebook:

* `assert` de que nenhuma coluna proibida está na matriz de atributos.
* Para uma amostra aleatória de partidas, recalcular à mão um atributo móvel
  usando só os jogos anteriores e comparar com o valor da base.

### 3.6 Modelagem

* Alvo: `FTR`, com a ordem de classes `[H, D, A]` fixa em todas as matrizes de
  probabilidade.
* Divisão temporal: treino de 2005/06 a 2021/22; teste de 2022/23 a 2024/25.
  Nunca embaralhar.
* Ajuste de hiperparâmetros só dentro do treino, com validação por temporada
  (treina até T−1 e valida em T, para T = 2019/20, 2020/21 e 2021/22), otimizando
  log loss. O teste é usado uma única vez, no final.
* Pipelines do scikit-learn com imputação pela mediana (ajustada no treino) e
  padronização apenas para a regressão logística.
* Modelos, todos com `random_state=42`:
  * Árvore de decisão, com grade pequena de `max_depth` e `min_samples_leaf`.
  * Regressão logística multinomial, com grade de `C`.
  * Floresta aleatória, com grade pequena de `max_depth` e `min_samples_leaf`,
    `n_estimators` perto de 300 e `n_jobs=-1`.
* Duas variantes:
  * Variante A (principal, é a que vai para o resumo): só atributos de desempenho,
    nenhuma cotação.
  * Variante B (complementar): atributos da variante A mais as probabilidades
    normalizadas da Bet365, para ver se o modelo agrega algo além do mercado.
* Tempo total de execução do notebook: mire em menos de 15 minutos numa máquina
  comum. Reduza as grades se precisar.

### 3.7 Avaliação

* Todos os modelos e baselines avaliados exatamente nas mesmas partidas de teste.
* Baselines: favorito do mercado (Bet365 normalizada), sempre mandante e
  frequência das classes no treino.
* Métricas: acurácia, log loss e RPS (ranked probability score, implementado à mão
  na ordem H, D, A). Precisão, revocação e F1 por classe, com atenção ao empate, e
  percentual de empates previstos.
* Matriz de confusão em gráfico para cada modelo e para o mercado.
* Comparação com o mercado: IC 95% por bootstrap pareado (2000 reamostragens) da
  diferença de acurácia modelo − mercado, e teste de McNemar
  (`statsmodels.stats.contingency_tables.mcnemar`).
* Consistência: acurácia e log loss por temporada de teste e por liga, contando em
  quantas combinações cada modelo supera o mercado.
* Importância de atributos: permutação no teste para a floresta e coeficientes
  para a regressão logística. Top 15 em gráfico.
* Tabela final com as métricas das variantes A e B em `outputs/tabelas/metricas.csv`.

### 3.8 Seção final: números para o resumo

A última seção gera `outputs/resultados_resumo.txt` e mostra na tela:

1. N total de partidas analisadas, N de treino e N de teste.
2. A frase pronta, com a variante A e números em formato brasileiro (vírgula
   decimal, uma casa): "Na previsão do resultado das partidas, a árvore de decisão
   alcançou {dt}% de acerto, a regressão logística {rl}% e a floresta aleatória
   {fa}%, contra {mercado}% obtidos ao se adotar apenas o favorito apontado pelas
   cotações."
3. As duas afirmações de resultados iniciais (margem varia entre operadores e
   divisões; desfechos improváveis superestimados), cada uma marcada como
   CONFIRMADA ou NÃO CONFIRMADA, com os números que embasam.
4. Sugestão de frase de conclusão, escolhida pelos dados considerando acurácia e
   log loss:
   * Nenhum modelo supera o mercado com significância e a distância é de até 3
     pontos percentuais: manter "aproximam-se do desempenho do mercado, mas não o
     superam de forma consistente".
   * Distância maior que 3 pontos percentuais: sugerir "ficam abaixo do desempenho
     do mercado".
   * Algum modelo supera o mercado com significância: informar e emitir alerta
     pedindo revisão de vazamento antes de aceitar.
5. Alarme de vazamento: se qualquer modelo passar de 60% de acurácia, imprimir
   aviso em destaque e não gerar a frase.

Feche o notebook com uma seção markdown "Decisões e limitações" listando as
escolhas feitas (tratamento de promovidos, colunas indisponíveis em certas ligas,
partidas descartadas etc.).

## 4. Regras inegociáveis

* Nenhuma informação do próprio jogo nos atributos.
* Nada ajustado com dados de teste: imputação, padronização, hiperparâmetros,
  limiares.
* Nenhum resultado digitado à mão; todo número exibido ou exportado vem da
  execução.
* Sementes fixas.
* Afirmação do resumo que não se confirmar deve ser relatada, nunca contornada.
* Situação não coberta aqui: escolha a opção mais simples e defensável e registre
  em "Decisões e limitações".

## 5. Como finalizar

1. Executar o notebook de ponta a ponta:
   `jupyter nbconvert --to notebook --execute --inplace mineracao_bets.ipynb --ExecutePreprocessor.timeout=1800`
2. Se falhar, corrigir e executar de novo até rodar limpo.
3. Gerar `requirements.txt` com as versões instaladas e o `README.md`.
4. Mostrar no terminal o conteúdo de `outputs/resultados_resumo.txt`, a lista de
   arquivos gerados e qualquer ressalva importante sobre os dados.
