# Eficiência das cotações de apostas esportivas

Projeto da disciplina de **Mineração de Dados** — Tecnologia em Análise e
Desenvolvimento de Sistemas, **IFSP Câmpus Jacareí**.

O notebook `mineracao_bets.ipynb` produz os números do resumo submetido à
**JECET 2026** e implementa cada etapa metodológica descrita nele, organizado
pelas fases do **CRISP-DM**.

## O que o trabalho investiga

1. A **margem embutida nas cotações** (*overround*) varia entre operadores e
   entre divisões?
2. As probabilidades atribuídas aos **desfechos menos prováveis** são
   superestimadas (viés favorito-azarão)?
3. Modelos supervisionados treinados apenas com **desempenho histórico**
   (árvore de decisão, regressão logística e floresta aleatória) preveem o
   resultado das partidas melhor do que o mercado?

Regra de ouro do projeto: **nenhum número é digitado à mão**. Tudo o que aparece
no relatório final vem da execução. Afirmação do resumo que não se confirma é
relatada como tal, nunca contornada.

## Instalação

Requer **Python 3.10 ou superior**.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Como rodar

### No Jupyter (VSCode ou JupyterHub)

Abra `mineracao_bets.ipynb` e execute todas as células. Os caminhos são
relativos (via `pathlib`), então o notebook funciona nos dois ambientes sem
ajuste.

### Pela linha de comando

```bash
jupyter nbconvert --to notebook --execute --inplace mineracao_bets.ipynb \
  --ExecutePreprocessor.timeout=1800
```

A execução completa leva **menos de 15 minutos** numa máquina comum. Na primeira
vez são baixados cerca de 28 MB de CSVs; nas seguintes, a pasta `data/raw/`
funciona como cache e nada é baixado de novo.

## Estrutura

```
.
├── INSTRUCOES.md              enunciado do trabalho
├── README.md                  este arquivo
├── requirements.txt           versões fixadas
├── mineracao_bets.ipynb       notebook principal (executado, com saídas salvas)
├── brasileirao_2026.ipynb     notebook complementar do Brasileirão 2026
├── src/
│   ├── utils.py               download, limpeza, Elo, médias móveis, métricas
│   └── utils_brasileirao.py   modelo de gols (Poisson/Dixon-Coles) e cálculo de green
├── data/
│   ├── raw/                   CSVs baixados (cache; 10 divisões x 25 temporadas)
│   │   └── brasileirao/       arquivo único do Brasileirão (cache)
│   └── processed/             bases intermediárias
└── outputs/
    ├── figuras/               PNG a 150 dpi, títulos e eixos em português
    ├── tabelas/               CSVs com métricas e agregados
    ├── resultados_resumo.txt  números prontos para o resumo
    └── brasileirao/           saídas do notebook do Brasileirão
        ├── figuras/
        ├── tabelas/
        └── resumo_brasileirao_2026.txt
```

## Dados

**Fonte:** [Football-Data.co.uk](https://www.football-data.co.uk/), que publica
por temporada e divisão um CSV com resultado, estatísticas de jogo e cotações de
vários operadores. O dicionário de colunas está em
<https://www.football-data.co.uk/notes.txt>.

| | |
|---|---|
| **Ligas principais** | `E0` Inglaterra, `SP1` Espanha, `I1` Itália, `D1` Alemanha, `F1` França |
| **Segundas divisões** | `E1`, `SP2`, `I2`, `D2`, `F2` — só na análise de margem por divisão |
| **Amostra analisada** | 2005/06 a 2024/25 |
| **Aquecimento** | 2000/01 a 2004/05 — só para o Elo e as médias móveis, fora da amostra |
| **Operador principal** | Bet365 (`B365H`, `B365D`, `B365A`), o único com cobertura em todo o período |

> **Aviso sobre a origem dos arquivos.** O notebook tenta primeiro a fonte
> oficial. No ambiente em que esta versão foi executada, a política de rede
> bloqueia o domínio `football-data.co.uk`, e a coleta caiu automaticamente para
> um **espelho público** que republica os mesmos CSVs com as colunas originais
> preservadas. A origem de cada arquivo fica registrada em
> `data/raw/_origens.json` e é reexibida a cada execução, inclusive quando tudo
> vem do cache — assim o relatório nunca afirma "acesso direto" sem ter de fato
> falado com a fonte oficial. **Numa rede sem esse bloqueio, a fonte oficial é
> usada automaticamente, sem alterar nenhuma linha de código.** A integridade da
> base foi conferida contra fatos conhecidos das competições (ver
> "Decisões e limitações", ao fim do notebook).
>
> Para forçar uma recoleta do zero, apague `data/raw/` e execute o notebook de
> novo.

## Metodologia, em resumo

* **Limpeza** — datas em `dd/mm/yy` e `dd/mm/yyyy`, codificação UTF-8 com queda
  para latin-1, padronização dos nomes das equipes, tabela de cobertura por liga
  e temporada, remoção de partidas sem resultado ou sem cotação.
* **Cotações → probabilidades** — implícita bruta (`1/cotação`), margem
  (soma das três menos 1) e probabilidade normalizada (implícita dividida pela
  soma).
* **Atributos sem vazamento** — médias móveis de 5 e 10 partidas com `shift(1)`
  **antes** do `rolling`, forma por mando, Elo pré-jogo por liga (K = 20,
  vantagem de mando de 60, regressão de 1/3 na virada de temporada), indicador
  de promovido e diferenças mandante − visitante. Nenhuma informação do próprio
  jogo entra como atributo, e isso é verificado por dois `assert` dentro do
  notebook (colunas proibidas e recálculo manual das médias móveis).
* **Modelagem** — divisão **temporal** (treino 2005/06–2021/22, teste
  2022/23–2024/25), hiperparâmetros ajustados só no treino com validação por
  temporada, `Pipeline`s com imputação pela mediana e padronização apenas na
  regressão logística. Duas variantes: **A** (só desempenho, é a que vai para o
  resumo) e **B** (desempenho + probabilidades da Bet365).
* **Avaliação** — acurácia, log loss e RPS, precisão/revocação/F1 por classe,
  matrizes de confusão, comparação com o mercado por bootstrap pareado (2000
  reamostragens) e teste de McNemar, consistência por temporada e por liga, e
  importância de atributos por permutação e por coeficientes.

## Saídas geradas

Todas em `outputs/`, criadas pela execução:

* **`resultados_resumo.txt`** — tamanhos da amostra, a frase de resultados já no
  formato brasileiro, cada afirmação do resumo marcada como **CONFIRMADA** ou
  **NÃO CONFIRMADA** com os números que a embasam, a sugestão de frase de
  conclusão escolhida pelos dados e a verificação de vazamento.
* **`figuras/`** — margem por operador e por divisão, calibração, retorno por
  faixa de cotação, matrizes de confusão, importância de atributos e acurácia
  por temporada.
* **`tabelas/`** — cobertura, margens, calibração, retorno, métricas dos
  modelos, comparação com o mercado, consistência e verificações de vazamento.

## Notebook complementar: Brasileirão Série A 2026

`brasileirao_2026.ipynb` aplica a mesma disciplina metodológica ao campeonato
brasileiro e responde a duas perguntas práticas. Roda em **menos de 1 minuto**.

```bash
jupyter nbconvert --to notebook --execute --inplace brasileirao_2026.ipynb \
  --ExecutePreprocessor.timeout=1800
```

**1. O que deve acontecer nas partidas que faltam.** Um modelo de **Poisson com
correção de Dixon-Coles**, ponderado no tempo, estima ataque e defesa de cada
clube e devolve, para cada confronto: gols esperados dos dois lados, placar mais
provável, probabilidade de vitória/empate/derrota, mais de 2,5 gols e ambas as
equipes marcarem. As **partidas restantes são deduzidas do formato do
campeonato** (turno e returno com 20 clubes = 380 confrontos), sem precisar de
tabela externa. Uma simulação de Monte Carlo com 10 mil repetições projeta a
classificação final, com chances de título, G4 e Z4.

**2. A chance de voltar *green* em cada cotação.** A análise separa duas
perguntas que costumam ser confundidas: quanto voltou green historicamente em
cada faixa de cotação, e o que a probabilidade do modelo diz sobre uma aposta
específica. A conclusão é que **a cotação não muda a chance de green** — ela
muda o preço pago por essa chance.

> **Escopo.** A base do Brasileirão traz apenas gols, resultado e cotações de
> fechamento do mercado 1X2. **Não há finalizações, escanteios nem cartões**,
> então esses não são previstos. Também não há cotações de over/under nem de
> ambas marcam para confrontar com as previsões desses mercados.

> **Aviso.** O notebook mede eficiência de mercado; não é um sistema de apostas.
> Os resultados mostram retorno esperado negativo em todas as faixas de cotação,
> inclusive apostando apenas onde o modelo enxerga vantagem — filtrar por valor
> esperado **piora** o retorno, porque quando o modelo discorda muito do mercado
> em geral quem está errado é o modelo.

## Reprodutibilidade

Sementes fixas (`random_state=42` em todos os modelos, gerador do bootstrap
semeado), divisão temporal sem embaralhamento e nenhum ajuste feito com dados de
teste — imputação, padronização e hiperparâmetros vêm só do treino, e o teste é
usado uma única vez, no fim.
