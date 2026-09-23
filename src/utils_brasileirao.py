"""
Funções auxiliares do notebook do Campeonato Brasileiro Série A.

Disciplina de Mineração de Dados - ADS, IFSP Câmpus Jacareí.

Complementa ``utils.py`` (das ligas europeias) com o que é específico do
Brasileirão: a base do Football-Data vem em outro formato, sem estatísticas de
jogo, e a previsão é de partidas que ainda não aconteceram. Os utilitários
genéricos (IC de Wilson, RPS, log loss, bootstrap, estilo dos gráficos) são
reaproveitados de ``utils``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import utils  # noqa: F401  (usa a semente, o estilo e as métricas genéricas)

# ---------------------------------------------------------------------------
# Caminhos e configuração
# ---------------------------------------------------------------------------

RAIZ = utils.RAIZ
DIR_BRUTO = RAIZ / "data" / "raw" / "brasileirao"
DIR_SAIDAS = RAIZ / "outputs" / "brasileirao"
DIR_FIGURAS = DIR_SAIDAS / "figuras"
DIR_TABELAS = DIR_SAIDAS / "tabelas"

for _d in (DIR_BRUTO, DIR_FIGURAS, DIR_TABELAS):
    _d.mkdir(parents=True, exist_ok=True)

SEMENTE = utils.SEMENTE
CLASSES = utils.CLASSES          # ordem fixa H, D, A

TEMPORADA_ALVO = 2026
TIMES_POR_TEMPORADA = 20
# Pontos corridos em turno e returno: cada time enfrenta todos os outros duas
# vezes, uma em casa e uma fora. Logo o calendário tem n*(n-1) partidas e é
# inteiramente determinado pelo conjunto de times.
PARTIDAS_POR_TEMPORADA = TIMES_POR_TEMPORADA * (TIMES_POR_TEMPORADA - 1)

# Fonte oficial e espelho, na mesma lógica do notebook das ligas europeias.
URL_OFICIAL = "https://www.football-data.co.uk/new/BRA.csv"
URL_ESPELHO = ("https://raw.githubusercontent.com/huhao930422-debug/"
               "football-odds-mirror/main/data/brazil/all-seasons.csv")

ARQUIVO_BASE = DIR_BRUTO / "brasileirao_todas_temporadas.csv"
ARQUIVO_ORIGEM = DIR_BRUTO / "_origem.json"

# Operadores disponíveis nesta base. Só há cotações de FECHAMENTO (sufixo C) e
# só do mercado 1X2 - não há over/under nem ambas marcam.
OPERADORES = {
    "Média de mercado": "AvgC",
    "Máxima do mercado": "MaxC",
    "Pinnacle": "PSC",
    "Betfair Exchange": "BFEC",
    "Bet365": "B365C",
}
# Operador de referência: é o único com 100% de cobertura em todas as temporadas.
OPERADOR_PADRAO = "AvgC"


def salvar_figura(figura, nome: str) -> Path:
    caminho = DIR_FIGURAS / f"{nome}.png"
    figura.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=utils.SUPERFICIE)
    return caminho


def salvar_tabela(dados: pd.DataFrame, nome: str) -> Path:
    caminho = DIR_TABELAS / f"{nome}.csv"
    dados.to_csv(caminho, index=False, encoding="utf-8")
    return caminho


# ---------------------------------------------------------------------------
# Coleta
# ---------------------------------------------------------------------------

def baixar_base(forcar: bool = False) -> dict:
    """
    Baixa (ou reaproveita do cache) o arquivo único do Brasileirão.

    Igual ao outro notebook: tenta a fonte oficial e cai para o espelho quando
    ela está inacessível, gravando a origem em disco para que o relatório nunca
    afirme "fonte oficial" sem ter de fato falado com ela.
    """
    import requests

    if ARQUIVO_BASE.exists() and ARQUIVO_BASE.stat().st_size > 1000 and not forcar:
        origem = "desconhecida"
        if ARQUIVO_ORIGEM.exists():
            try:
                origem = json.loads(ARQUIVO_ORIGEM.read_text(encoding="utf-8")).get(
                    "origem", "desconhecida")
            except (ValueError, OSError):
                pass
        return {"situacao": "cache", "origem": origem,
                "bytes": ARQUIVO_BASE.stat().st_size, "detalhe": ""}

    erros = []
    for origem, url in (("oficial", URL_OFICIAL), ("espelho", URL_ESPELHO)):
        try:
            resposta = requests.get(url, timeout=90)
            resposta.raise_for_status()
            if len(resposta.content) < 1000:
                raise ValueError(f"resposta muito curta ({len(resposta.content)} bytes)")
            ARQUIVO_BASE.write_bytes(resposta.content)
            ARQUIVO_ORIGEM.write_text(
                json.dumps({"origem": origem, "url": url}, indent=2, ensure_ascii=False),
                encoding="utf-8")
            return {"situacao": "baixado", "origem": origem,
                    "bytes": len(resposta.content), "detalhe": ""}
        except Exception as erro:
            erros.append(f"{origem}: {type(erro).__name__}: {str(erro)[:90]}")
    return {"situacao": "falhou", "origem": "nenhuma", "bytes": 0,
            "detalhe": "; ".join(erros)}


# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------

def carregar_base() -> pd.DataFrame:
    """
    Lê o CSV, converte datas e tipos e devolve a base ordenada no tempo.

    Colunas da origem: Country, League, Season, Date, Time, Home, Away,
    HG (gols do mandante), AG (gols do visitante), Res (H/D/A) e as cotações
    de fechamento de cada operador.
    """
    bruto = ARQUIVO_BASE.read_bytes()
    for codificacao in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            texto = bruto.decode(codificacao)
            break
        except UnicodeDecodeError:
            continue

    from io import StringIO
    base = pd.read_csv(StringIO(texto))
    base.columns = [c.replace("﻿", "").strip() for c in base.columns]

    base["data"] = pd.to_datetime(base["Date"], format="%d/%m/%Y", errors="coerce")
    faltando = base["data"].isna()
    if faltando.any():
        base.loc[faltando, "data"] = pd.to_datetime(
            base.loc[faltando, "Date"], dayfirst=True, errors="coerce")
    base["horario"] = base.get("Time", pd.Series("", index=base.index)).fillna("").astype(str)

    for coluna in ("Home", "Away"):
        base[coluna] = (base[coluna].astype(str)
                        .str.replace(r"\s+", " ", regex=True).str.strip())

    for coluna in base.columns:
        if coluna.endswith(("H", "D", "A")) and coluna not in ("Home", "Away"):
            base[coluna] = pd.to_numeric(base[coluna], errors="coerce")
    for coluna in ("HG", "AG", "Season"):
        base[coluna] = pd.to_numeric(base[coluna], errors="coerce")

    base = base.sort_values(["data", "horario", "Home"], kind="mergesort").reset_index(drop=True)
    base["partida_id"] = np.arange(len(base))
    base["jogada"] = base["Res"].isin(CLASSES) & base["HG"].notna() & base["AG"].notna()
    return base


def partidas_restantes(base: pd.DataFrame, temporada: int = TEMPORADA_ALVO) -> pd.DataFrame:
    """
    Deduz as partidas que ainda faltam na temporada, sem precisar de tabela
    externa de jogos.

    O Brasileirão é disputado em pontos corridos, turno e returno: com 20 times,
    o calendário tem exatamente 20 x 19 = 380 confrontos, um para cada par
    ordenado (mandante, visitante). O que falta é, portanto, o conjunto de pares
    ordenados que ainda não apareceram.
    """
    do_ano = base[(base["Season"] == temporada) & base["jogada"]]
    times = sorted(set(do_ano["Home"]) | set(do_ano["Away"]))
    disputados = set(zip(do_ano["Home"], do_ano["Away"]))

    restantes = [{"Season": temporada, "Home": mandante, "Away": visitante}
                 for mandante in times for visitante in times
                 if mandante != visitante and (mandante, visitante) not in disputados]

    return pd.DataFrame(restantes), times, len(disputados)


# ---------------------------------------------------------------------------
# Cotações -> probabilidades
# ---------------------------------------------------------------------------

def colunas_operador(prefixo: str) -> list[str]:
    return [f"{prefixo}{sufixo}" for sufixo in ("H", "D", "A")]


def probabilidades_mercado(base: pd.DataFrame, prefixo: str = OPERADOR_PADRAO):
    """Probabilidades normalizadas (sem margem) e a margem de cada partida."""
    colunas = colunas_operador(prefixo)
    cotacoes = base[colunas].where(base[colunas] > 1.0)
    implicitas = 1.0 / cotacoes
    soma = implicitas.sum(axis=1, min_count=3)
    normalizadas = implicitas.div(soma, axis=0)
    normalizadas.columns = ["prob_H", "prob_D", "prob_A"]
    return normalizadas, soma - 1.0


# ---------------------------------------------------------------------------
# Atributos históricos sem vazamento
# ---------------------------------------------------------------------------

# Colunas do próprio jogo: nunca podem virar atributo.
COLUNAS_PROIBIDAS = ["HG", "AG", "Res"]

METRICAS = ["gols_feitos", "gols_sofridos", "pontos"]
JANELAS = (5, 10)


def formato_longo(base: pd.DataFrame) -> pd.DataFrame:
    """Duas linhas por partida, uma na perspectiva de cada time."""
    jogadas = base[base["jogada"]]
    comum = {"partida_id": jogadas["partida_id"], "Season": jogadas["Season"],
             "data": jogadas["data"], "horario": jogadas["horario"]}

    casa = pd.DataFrame({
        **comum, "time": jogadas["Home"], "adversario": jogadas["Away"], "mando": "C",
        "gols_feitos": jogadas["HG"], "gols_sofridos": jogadas["AG"],
        "pontos": jogadas["Res"].map({"H": 3, "D": 1, "A": 0})})
    fora = pd.DataFrame({
        **comum, "time": jogadas["Away"], "adversario": jogadas["Home"], "mando": "F",
        "gols_feitos": jogadas["AG"], "gols_sofridos": jogadas["HG"],
        "pontos": jogadas["Res"].map({"H": 0, "D": 1, "A": 3})})

    longo = pd.concat([casa, fora], ignore_index=True)
    return longo.sort_values(["time", "data", "horario", "partida_id"],
                             kind="mergesort").reset_index(drop=True)


def medias_moveis(longo: pd.DataFrame, janelas=JANELAS) -> pd.DataFrame:
    """
    Médias móveis das últimas w partidas de cada time, com ``shift(1)`` ANTES do
    ``rolling`` para que a partida atual nunca entre no próprio atributo.
    """
    longo = longo.copy()
    por_time = longo.groupby("time", sort=False)

    novas = {}
    for metrica in METRICAS:
        deslocada = por_time[metrica].shift(1)
        for janela in janelas:
            novas[f"{metrica}_m{janela}"] = (
                deslocada.groupby(longo["time"], sort=False)
                .rolling(janela, min_periods=1).mean()
                .reset_index(level=0, drop=True))
    novas["jogos_anteriores"] = por_time.cumcount()

    for coluna in ("forma_mando_gols_5", "forma_mando_pontos_5"):
        novas[coluna] = pd.Series(np.nan, index=longo.index)
    longo = pd.concat([longo, pd.DataFrame(novas, index=longo.index)], axis=1)

    for mando in ("C", "F"):
        sub = longo[longo["mando"] == mando]
        por_mando = sub.groupby("time", sort=False)
        for metrica, destino in (("gols_feitos", "forma_mando_gols_5"),
                                 ("pontos", "forma_mando_pontos_5")):
            deslocada = por_mando[metrica].shift(1)
            longo.loc[sub.index, destino] = (
                deslocada.groupby(sub["time"], sort=False)
                .rolling(5, min_periods=1).mean()
                .reset_index(level=0, drop=True))
    return longo


def calcular_elo(base: pd.DataFrame, k: float = 20.0, vantagem_mando: float = 60.0,
                 elo_inicial: float = 1500.0, fator_regressao: float = 1 / 3) -> pd.DataFrame:
    """
    Elo pré-jogo. Mesmas regras do notebook europeu: K = 20, vantagem de mando de
    60 pontos, regressão de 1/3 à média na virada de temporada e promovidos
    entrando com a média do Elo dos rebaixados do ano anterior.
    """
    jogadas = base[base["jogada"]].sort_values(["data", "horario", "partida_id"],
                                               kind="mergesort")
    elo: dict[str, float] = {}
    times_anteriores: set[str] = set()
    registros = []

    for temporada in sorted(jogadas["Season"].unique()):
        do_ano = jogadas[jogadas["Season"] == temporada]
        times_atuais = set(do_ano["Home"]) | set(do_ano["Away"])

        if times_anteriores:
            for time in list(elo):
                elo[time] += (elo_inicial - elo[time]) * fator_regressao
            rebaixados = times_anteriores - times_atuais
            valores = [elo[t] for t in rebaixados if t in elo]
            elo_promovido = float(np.mean(valores)) if valores else elo_inicial
            for time in times_atuais - times_anteriores:
                elo[time] = elo_promovido
        promovidos = (times_atuais - times_anteriores) if times_anteriores else set()
        for time in times_atuais:
            elo.setdefault(time, elo_inicial)

        for partida_id, mandante, visitante, resultado in zip(
                do_ano["partida_id"], do_ano["Home"], do_ano["Away"], do_ano["Res"]):
            elo_mandante, elo_visitante = elo[mandante], elo[visitante]
            registros.append({"partida_id": partida_id,
                              "elo_mandante": elo_mandante,
                              "elo_visitante": elo_visitante,
                              "promovido_mandante": int(mandante in promovidos),
                              "promovido_visitante": int(visitante in promovidos)})
            esperado = 1.0 / (1.0 + 10 ** (
                (elo_visitante - elo_mandante - vantagem_mando) / 400.0))
            obtido = {"H": 1.0, "D": 0.5, "A": 0.0}[resultado]
            ajuste = k * (obtido - esperado)
            elo[mandante] = elo_mandante + ajuste
            elo[visitante] = elo_visitante - ajuste

        times_anteriores = times_atuais

    return pd.DataFrame(registros), elo


# ---------------------------------------------------------------------------
# Modelo de gols: Poisson com correção de Dixon-Coles
# ---------------------------------------------------------------------------

def _pesos_temporais(datas: pd.Series, referencia: pd.Timestamp,
                     meia_vida_dias: float) -> np.ndarray:
    """
    Peso exponencial: uma partida de ``meia_vida_dias`` atrás vale metade de uma
    de hoje. É o que faz o modelo acompanhar elenco novo, troca de técnico e
    mudança de fase sem jogar fora o histórico antigo.
    """
    dias = (referencia - datas).dt.total_seconds() / 86400.0
    return np.power(0.5, np.maximum(dias, 0) / meia_vida_dias)


def _tau_dixon_coles(placar_casa, placar_fora, lambda_casa, lambda_fora, rho):
    """
    Correção de Dixon-Coles para os placares baixos.

    O Poisson independente supõe que os dois times marcam sem se influenciar e,
    com isso, subestima 0-0 e 1-1 — justamente os placares mais comuns em jogo
    travado. A correção mexe só nessas quatro células de placar baixo; ``rho``
    mede o tamanho do ajuste e é estimado nos dados de treino.
    """
    placar_casa = np.asarray(placar_casa)
    placar_fora = np.asarray(placar_fora)
    tau = np.ones(np.broadcast(placar_casa, placar_fora,
                               np.asarray(lambda_casa), np.asarray(lambda_fora)).shape)
    tau = np.where((placar_casa == 0) & (placar_fora == 0),
                   1 - lambda_casa * lambda_fora * rho, tau)
    tau = np.where((placar_casa == 0) & (placar_fora == 1), 1 + lambda_casa * rho, tau)
    tau = np.where((placar_casa == 1) & (placar_fora == 0), 1 + lambda_fora * rho, tau)
    tau = np.where((placar_casa == 1) & (placar_fora == 1), 1 - rho, tau)
    return tau


def _times_promovidos(base_treino: pd.DataFrame) -> set[str]:
    """Times que, em alguma temporada da janela, não haviam disputado a anterior."""
    jogadas = base_treino[base_treino["jogada"]]
    temporadas = sorted(jogadas["Season"].unique())
    promovidos: set[str] = set()
    anteriores: set[str] = set()
    for i, temporada in enumerate(temporadas):
        do_ano = jogadas[jogadas["Season"] == temporada]
        atuais = set(do_ano["Home"]) | set(do_ano["Away"])
        if i > 0:
            promovidos |= (atuais - anteriores)
        anteriores = atuais
    return promovidos


class ModeloGols:
    """
    Modelo de gols de Poisson com correção de Dixon-Coles.

        log(lambda) = intercepto + ataque[time] + defesa[adversário] + mando

    Os coeficientes ficam guardados em dicionários em vez de a previsão passar
    de novo pela fórmula. Isso resolve um problema concreto: um time recém
    promovido (o Remo, em 2026) não existe no histórico e quebraria a previsão.
    Aqui ele recebe explicitamente o **perfil médio dos promovidos** da janela de
    treino, que é a informação honesta disponível sobre quem acabou de subir.
    """

    def __init__(self, intercepto, ataque, defesa, mando, ataque_novo, defesa_novo,
                 rho=0.0, meia_vida_dias=None, n_partidas=0, max_gols=10):
        self.intercepto = float(intercepto)
        self.ataque = dict(ataque)
        self.defesa = dict(defesa)
        self.mando = float(mando)
        self.ataque_novo = float(ataque_novo)
        self.defesa_novo = float(defesa_novo)
        self.rho = float(rho)
        self.meia_vida_dias = meia_vida_dias
        self.n_partidas = int(n_partidas)
        self.max_gols = int(max_gols)

    @property
    def times_conhecidos(self) -> set[str]:
        return set(self.ataque)

    def sem_historico(self, times) -> list[str]:
        return sorted({t for t in times if t not in self.ataque})

    def lambdas(self, mandante: str, visitante: str) -> tuple[float, float]:
        """Gols esperados de cada lado. Time sem histórico usa o perfil de promovido."""
        ataque_mandante = self.ataque.get(mandante, self.ataque_novo)
        defesa_mandante = self.defesa.get(mandante, self.defesa_novo)
        ataque_visitante = self.ataque.get(visitante, self.ataque_novo)
        defesa_visitante = self.defesa.get(visitante, self.defesa_novo)
        lambda_casa = np.exp(self.intercepto + ataque_mandante + defesa_visitante
                             + self.mando)
        lambda_fora = np.exp(self.intercepto + ataque_visitante + defesa_mandante)
        return float(lambda_casa), float(lambda_fora)

    def prever(self, confrontos: pd.DataFrame) -> pd.DataFrame:
        """Aplica o modelo a uma tabela com as colunas Home e Away."""
        linhas = []
        for mandante, visitante in zip(confrontos["Home"], confrontos["Away"]):
            lambda_casa, lambda_fora = self.lambdas(mandante, visitante)
            matriz = matriz_placares(lambda_casa, lambda_fora, self.rho, self.max_gols)
            linhas.append({"Home": mandante, "Away": visitante,
                           "lambda_mandante": lambda_casa,
                           "lambda_visitante": lambda_fora,
                           **previsoes_do_placar(matriz)})
        return pd.DataFrame(linhas)

    def forca_dos_times(self) -> pd.DataFrame:
        """Ataque e defesa de cada time, em escala multiplicativa (1 = média)."""
        tabela = pd.DataFrame({
            "time": sorted(self.ataque),
            "ataque": [np.exp(self.ataque[t]) for t in sorted(self.ataque)],
            "defesa": [np.exp(self.defesa[t]) for t in sorted(self.ataque)]})
        # Ataque acima de 1 marca mais que a média; defesa abaixo de 1 leva menos.
        tabela["saldo"] = tabela["ataque"] / tabela["defesa"]
        return tabela.sort_values("saldo", ascending=False).reset_index(drop=True)


def ajustar_modelo_gols(base_treino: pd.DataFrame, referencia: pd.Timestamp,
                        meia_vida_dias: float = 365.0,
                        estimar_correcao: bool = True,
                        max_gols: int = 10) -> ModeloGols:
    """
    Ajusta o modelo de gols com ponderação temporal e correção de Dixon-Coles.

    Cada partida entra como duas observações (uma por time). As partidas antigas
    pesam menos: uma de ``meia_vida_dias`` atrás vale metade de uma de hoje.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    jogadas = base_treino[base_treino["jogada"]]
    longo = pd.concat([
        pd.DataFrame({"gols": jogadas["HG"], "ataque": jogadas["Home"],
                      "defesa": jogadas["Away"], "mando": 1.0, "data": jogadas["data"]}),
        pd.DataFrame({"gols": jogadas["AG"], "ataque": jogadas["Away"],
                      "defesa": jogadas["Home"], "mando": 0.0, "data": jogadas["data"]}),
    ], ignore_index=True).dropna(subset=["gols"])

    pesos = _pesos_temporais(longo["data"], referencia, meia_vida_dias)
    glm = smf.glm("gols ~ C(ataque) + C(defesa) + mando", data=longo,
                  family=sm.families.Poisson(), freq_weights=pesos).fit()

    # Extrai os coeficientes; o time de referência do patsy fica com zero.
    times = sorted(set(longo["ataque"]))
    ataque = {t: 0.0 for t in times}
    defesa = {t: 0.0 for t in times}
    padrao = re.compile(r"C\((ataque|defesa)\)\[T\.(.+)\]")
    for nome, valor in glm.params.items():
        achado = padrao.fullmatch(nome)
        if achado:
            alvo = ataque if achado.group(1) == "ataque" else defesa
            alvo[achado.group(2)] = float(valor)

    # Perfil do time sem histórico: a média dos promovidos da janela de treino.
    promovidos = [t for t in _times_promovidos(base_treino) if t in ataque]
    if promovidos:
        ataque_novo = float(np.mean([ataque[t] for t in promovidos]))
        defesa_novo = float(np.mean([defesa[t] for t in promovidos]))
    else:
        ataque_novo = float(np.mean(list(ataque.values())))
        defesa_novo = float(np.mean(list(defesa.values())))

    modelo = ModeloGols(glm.params["Intercept"], ataque, defesa, glm.params["mando"],
                        ataque_novo, defesa_novo, 0.0, meia_vida_dias,
                        len(jogadas), max_gols)
    if estimar_correcao:
        modelo.rho = estimar_rho(base_treino, modelo)
    return modelo


def estimar_rho(base_treino: pd.DataFrame, modelo: "ModeloGols",
                limites=(-0.25, 0.25)) -> float:
    """Estima rho maximizando a verossimilhança de Dixon-Coles NO TREINO."""
    from scipy.optimize import minimize_scalar
    from scipy.stats import poisson

    jogadas = base_treino[base_treino["jogada"]]
    lambdas = np.array([modelo.lambdas(m, v)
                        for m, v in zip(jogadas["Home"], jogadas["Away"])])
    lambda_casa, lambda_fora = lambdas[:, 0], lambdas[:, 1]
    gols_casa = jogadas["HG"].to_numpy()
    gols_fora = jogadas["AG"].to_numpy()
    base_log = (poisson.logpmf(gols_casa, lambda_casa)
                + poisson.logpmf(gols_fora, lambda_fora))

    def log_verossimilhanca_negativa(rho):
        tau = _tau_dixon_coles(gols_casa, gols_fora, lambda_casa, lambda_fora, rho)
        if np.any(tau <= 0):
            return 1e12
        return -float(np.sum(base_log + np.log(tau)))

    resultado = minimize_scalar(log_verossimilhanca_negativa, bounds=limites,
                                method="bounded")
    return float(resultado.x)


def matriz_placares(lambda_casa: float, lambda_fora: float, rho: float = 0.0,
                    max_gols: int = 10) -> np.ndarray:
    """
    Probabilidade de cada placar até ``max_gols`` x ``max_gols``.

    Dessa matriz saem todas as previsões do notebook: resultado, total de gols,
    ambas marcam e o placar mais provável.
    """
    from scipy.stats import poisson

    faixa = np.arange(max_gols + 1)
    p_casa = poisson.pmf(faixa, lambda_casa)
    p_fora = poisson.pmf(faixa, lambda_fora)
    matriz = np.outer(p_casa, p_fora)

    if rho != 0.0:
        grade_casa, grade_fora = np.meshgrid(faixa, faixa, indexing="ij")
        matriz = matriz * _tau_dixon_coles(grade_casa, grade_fora,
                                           lambda_casa, lambda_fora, rho)
        matriz = np.clip(matriz, 0, None)
    return matriz / matriz.sum()


def previsoes_do_placar(matriz: np.ndarray) -> dict:
    """Extrai da matriz de placares todas as estatísticas previstas."""
    n = matriz.shape[0]
    grade_casa, grade_fora = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    total = grade_casa + grade_fora
    indice = np.unravel_index(matriz.argmax(), matriz.shape)

    return {
        "prob_H": float(matriz[grade_casa > grade_fora].sum()),
        "prob_D": float(matriz[grade_casa == grade_fora].sum()),
        "prob_A": float(matriz[grade_casa < grade_fora].sum()),
        "gols_esperados_mandante": float((matriz * grade_casa).sum()),
        "gols_esperados_visitante": float((matriz * grade_fora).sum()),
        "gols_esperados_total": float((matriz * total).sum()),
        "prob_over_25": float(matriz[total > 2.5].sum()),
        "prob_under_25": float(matriz[total < 2.5].sum()),
        "prob_ambas_marcam": float(matriz[(grade_casa > 0) & (grade_fora > 0)].sum()),
        "prob_mandante_marca": float(matriz[grade_casa > 0].sum()),
        "prob_visitante_marca": float(matriz[grade_fora > 0].sum()),
        "placar_mais_provavel": f"{indice[0]}-{indice[1]}",
        "prob_placar_mais_provavel": float(matriz[indice]),
    }


# ---------------------------------------------------------------------------
# "Green" por cotação
# ---------------------------------------------------------------------------
#
# Vocabulário: uma aposta "volta green" quando ganha. Apostando 1 unidade numa
# cotação `c`, o green devolve `c - 1` de lucro; o red perde a unidade.
#
# Duas perguntas diferentes se escondem em "qual a chance de green nessa
# cotação", e o notebook responde as duas separadamente:
#
#   1. HISTÓRICA - entre as apostas historicamente oferecidas a uma dada faixa
#      de cotação, que percentual voltou green? (função `green_por_faixa`)
#   2. PROSPECTIVA - dada a probabilidade p que o modelo atribui a um desfecho,
#      a chance de green é o próprio p, e o que a cotação determina é se esse p
#      paga o preço cobrado. (função `avaliar_aposta`)

FAIXAS_COTACAO = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]
ROTULOS_COTACAO = ["até 1,5", "1,5 a 2", "2 a 3", "3 a 5", "5 a 10", "acima de 10"]


def empilhar_desfechos(base: pd.DataFrame, prefixo: str = OPERADOR_PADRAO) -> pd.DataFrame:
    """Três linhas por partida - uma por desfecho - com cotação e se ocorreu."""
    jogadas = base[base["jogada"]]
    probabilidades, _ = probabilidades_mercado(jogadas, prefixo)
    partes = []
    for classe, sufixo in zip(CLASSES, ("H", "D", "A")):
        partes.append(pd.DataFrame({
            "partida_id": jogadas["partida_id"].to_numpy(),
            "Season": jogadas["Season"].to_numpy(),
            "desfecho": classe,
            "cotacao": jogadas[f"{prefixo}{sufixo}"].to_numpy(),
            "prob_mercado": probabilidades[f"prob_{classe}"].to_numpy(),
            "green": (jogadas["Res"].to_numpy() == classe).astype(int)}))
    return pd.concat(partes, ignore_index=True).dropna(subset=["cotacao"])


def green_por_faixa(apostas: pd.DataFrame) -> pd.DataFrame:
    """
    Taxa histórica de green por faixa de cotação, com IC 95% de Wilson, e o
    retorno médio de apostar 1 unidade em cada desfecho.

    A comparação que interessa é entre a **taxa de green observada** e a
    **probabilidade implícita** na cotação: se o green fica sistematicamente
    abaixo da implícita, o preço cobrado é maior que o risco real.
    """
    apostas = apostas.copy()
    apostas["retorno"] = np.where(apostas["green"] == 1, apostas["cotacao"] - 1.0, -1.0)
    apostas["implicita"] = 1.0 / apostas["cotacao"]
    apostas["faixa"] = pd.cut(apostas["cotacao"], bins=FAIXAS_COTACAO,
                              labels=ROTULOS_COTACAO, right=False)

    linhas = []
    for faixa, grupo in apostas.groupby("faixa", observed=True):
        greens = int(grupo["green"].sum())
        n = len(grupo)
        inferior, superior = utils.ic_wilson(np.array([greens]), np.array([n]))
        # Erro padrão do retorno agrupado por partida (3 apostas por jogo).
        por_partida = grupo.groupby("partida_id")["retorno"].mean()
        erro = (por_partida.std(ddof=1) / np.sqrt(len(por_partida))
                if len(por_partida) > 1 else np.nan)
        retorno_medio = grupo["retorno"].mean()
        linhas.append({
            "faixa_cotacao": str(faixa), "n_apostas": n, "greens": greens,
            "cotacao_media": grupo["cotacao"].mean(),
            "taxa_green": greens / n,
            "green_ic95_inferior": float(inferior[0]),
            "green_ic95_superior": float(superior[0]),
            "prob_implicita_media": grupo["implicita"].mean(),
            "prob_normalizada_media": grupo["prob_mercado"].mean(),
            "retorno_medio": retorno_medio,
            "retorno_ic95_inferior": retorno_medio - 1.96 * erro,
            "retorno_ic95_superior": retorno_medio + 1.96 * erro})

    tabela = pd.DataFrame(linhas)
    tabela["faixa_cotacao"] = pd.Categorical(tabela["faixa_cotacao"],
                                             categories=ROTULOS_COTACAO, ordered=True)
    return tabela.sort_values("faixa_cotacao").reset_index(drop=True)


def avaliar_aposta(probabilidade, cotacao):
    """
    Avalia uma aposta a partir da probabilidade do modelo e da cotação oferecida.

    * ``chance_green``      - é a própria probabilidade do desfecho. A cotação
      NÃO muda a chance de acertar: muda quanto se recebe por acertar.
    * ``cotacao_minima``    - 1/p, a cotação de equilíbrio. Abaixo dela a aposta
      tem valor esperado negativo por construção.
    * ``valor_esperado``    - p * cotação - 1, o lucro médio por unidade.
    * ``vantagem``          - p - 1/cotação, a diferença entre o que o modelo
      acha e o que o preço embute.
    """
    probabilidade = np.asarray(probabilidade, dtype=float)
    cotacao = np.asarray(cotacao, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame({
            "chance_green": probabilidade,
            "cotacao_oferecida": cotacao,
            "cotacao_minima": 1.0 / probabilidade,
            "prob_implicita": 1.0 / cotacao,
            "valor_esperado": probabilidade * cotacao - 1.0,
            "vantagem": probabilidade - 1.0 / cotacao,
            "retorno_se_green": cotacao - 1.0})


def tabela_green_por_cotacao(probabilidade: float,
                             cotacoes=(1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0)
                             ) -> pd.DataFrame:
    """
    Para uma probabilidade fixa, mostra o que muda quando a cotação muda.

    Serve para deixar explícito o ponto que mais confunde quem aposta: a chance
    de green é a mesma em toda a coluna; o que muda é o valor esperado.
    """
    tabela = avaliar_aposta(np.full(len(cotacoes), probabilidade), np.array(cotacoes))
    tabela["vale_a_pena"] = tabela["valor_esperado"] > 0
    return tabela


# ---------------------------------------------------------------------------
# Classificação e simulação do restante da temporada
# ---------------------------------------------------------------------------

def classificacao(base: pd.DataFrame, temporada: int = TEMPORADA_ALVO) -> pd.DataFrame:
    """Tabela de classificação com as partidas já disputadas na temporada."""
    do_ano = base[(base["Season"] == temporada) & base["jogada"]]
    longo = pd.concat([
        pd.DataFrame({"time": do_ano["Home"], "gols_pro": do_ano["HG"],
                      "gols_contra": do_ano["AG"],
                      "pontos": do_ano["Res"].map({"H": 3, "D": 1, "A": 0}),
                      "vitoria": (do_ano["Res"] == "H").astype(int)}),
        pd.DataFrame({"time": do_ano["Away"], "gols_pro": do_ano["AG"],
                      "gols_contra": do_ano["HG"],
                      "pontos": do_ano["Res"].map({"H": 0, "D": 1, "A": 3}),
                      "vitoria": (do_ano["Res"] == "A").astype(int)}),
    ], ignore_index=True)

    tabela = (longo.groupby("time")
              .agg(jogos=("pontos", "size"), pontos=("pontos", "sum"),
                   vitorias=("vitoria", "sum"), gols_pro=("gols_pro", "sum"),
                   gols_contra=("gols_contra", "sum"))
              .reset_index())
    tabela["saldo"] = tabela["gols_pro"] - tabela["gols_contra"]
    tabela = tabela.sort_values(["pontos", "vitorias", "saldo", "gols_pro"],
                                ascending=False).reset_index(drop=True)
    tabela.insert(0, "posicao", np.arange(1, len(tabela) + 1))
    return tabela


def simular_temporada(modelo: ModeloGols, restantes: pd.DataFrame,
                      tabela_atual: pd.DataFrame, n_simulacoes: int = 10_000,
                      semente: int = SEMENTE) -> pd.DataFrame:
    """
    Simula as partidas que faltam e devolve as chances de cada time.

    Para cada partida, o placar é sorteado da própria matriz de placares do
    modelo — não do resultado mais provável. Assim a simulação carrega a
    incerteza real: um jogo 55/25/20 termina em vitória do mandante em 55% das
    simulações, não em todas.

    Critérios de desempate aplicados: pontos, vitórias, saldo de gols e gols
    pró. O confronto direto e os critérios disciplinares do regulamento não são
    reproduzidos (ver "Decisões e limitações").
    """
    gerador = np.random.default_rng(semente)
    times = list(tabela_atual["time"])
    indice = {time: i for i, time in enumerate(times)}
    n_times = len(times)

    pontos = np.tile(tabela_atual["pontos"].to_numpy(dtype=float), (n_simulacoes, 1))
    vitorias = np.tile(tabela_atual["vitorias"].to_numpy(dtype=float), (n_simulacoes, 1))
    gols_pro = np.tile(tabela_atual["gols_pro"].to_numpy(dtype=float), (n_simulacoes, 1))
    gols_contra = np.tile(tabela_atual["gols_contra"].to_numpy(dtype=float),
                          (n_simulacoes, 1))

    for mandante, visitante in zip(restantes["Home"], restantes["Away"]):
        lambda_casa, lambda_fora = modelo.lambdas(mandante, visitante)
        matriz = matriz_placares(lambda_casa, lambda_fora, modelo.rho, modelo.max_gols)
        lado = matriz.shape[0]
        # Sorteia um placar por simulação a partir da distribuição acumulada.
        acumulada = np.cumsum(matriz.ravel())
        sorteio = gerador.random(n_simulacoes)
        celulas = np.searchsorted(acumulada, sorteio)
        gols_casa, gols_fora = np.divmod(celulas, lado)

        i, j = indice[mandante], indice[visitante]
        gols_pro[:, i] += gols_casa
        gols_contra[:, i] += gols_fora
        gols_pro[:, j] += gols_fora
        gols_contra[:, j] += gols_casa
        casa_venceu = gols_casa > gols_fora
        fora_venceu = gols_fora > gols_casa
        empate = ~casa_venceu & ~fora_venceu
        pontos[:, i] += np.where(casa_venceu, 3.0, np.where(empate, 1.0, 0.0))
        pontos[:, j] += np.where(fora_venceu, 3.0, np.where(empate, 1.0, 0.0))
        vitorias[:, i] += casa_venceu
        vitorias[:, j] += fora_venceu

    saldo = gols_pro - gols_contra
    # Chave de ordenação combinando os critérios de desempate em um único número.
    chave = (pontos * 1e9 + vitorias * 1e6 + (saldo + 500) * 1e3 + gols_pro)
    ordem = np.argsort(-chave, axis=1, kind="stable")
    posicoes = np.empty_like(ordem)
    linhas = np.arange(n_simulacoes)[:, None]
    posicoes[linhas, ordem] = np.arange(1, n_times + 1)[None, :]

    return pd.DataFrame({
        "time": times,
        "pontos_atuais": tabela_atual["pontos"].to_numpy(),
        "pontos_esperados": pontos.mean(axis=0),
        "posicao_media": posicoes.mean(axis=0),
        "prob_titulo": (posicoes == 1).mean(axis=0),
        "prob_g4": (posicoes <= 4).mean(axis=0),
        "prob_g6": (posicoes <= 6).mean(axis=0),
        "prob_z4": (posicoes >= n_times - 3).mean(axis=0),
    }).sort_values("posicao_media").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-gols (faixas de total de gols)
# ---------------------------------------------------------------------------

# Faixas usuais do mercado brasileiro de "multi-gols".
FAIXAS_MULTIGOLS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4),
                    (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6)]

# Linhas de gols para mais/menos.
LINHAS_GOLS = [0.5, 1.5, 2.5, 3.5, 4.5]


def previsoes_multigols(matriz: np.ndarray) -> dict:
    """Probabilidade de o total de gols cair em cada faixa e acima de cada linha."""
    n = matriz.shape[0]
    grade_casa, grade_fora = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    total = grade_casa + grade_fora

    saida = {}
    for minimo, maximo in FAIXAS_MULTIGOLS:
        saida[f"multigols_{minimo}_{maximo}"] = float(
            matriz[(total >= minimo) & (total <= maximo)].sum())
    for linha in LINHAS_GOLS:
        saida[f"mais_de_{linha}".replace(".", "_")] = float(matriz[total > linha].sum())
    return saida


# ---------------------------------------------------------------------------
# Estatísticas de jogo: escanteios e cartões
# ---------------------------------------------------------------------------
#
# ATENÇÃO — estas duas famílias de mercado vêm de uma base DIFERENTE e mais
# fraca que a dos gols:
#
#   * a fonte das cotações (Football-Data) não traz escanteios nem cartões;
#   * a base pública que traz (adaoduque/Brasileirao_Dataset) só tem estatística
#     preenchida de 2015 a 2023 — em 2024 as colunas estão zeradas e não há
#     2025 nem 2026;
#   * portanto as previsões de escanteios e cartões se apoiam em dados com
#     alguns anos de defasagem, e dois clubes de 2026 (Mirassol e Remo) não têm
#     nenhum histórico nessa base, caindo na média da liga.
#
# Por isso tudo o que sai daqui é rotulado com confiança BAIXA, em oposição aos
# mercados derivados de gols, que usam dados até a rodada mais recente.

# Fonte: camada "silver" do datalake público leeofernandes1980/brasileirao-dataset,
# que consolida o histórico de adaoduque (2003-2023) com importação da API do
# Sofascore para 2024 em diante. É o que permite usar a TEMPORADA CORRENTE:
# os arquivos CSV da raiz daquele repositório param em 2023, mas os parquet da
# silver chegam a 2026, e trazem também a rodada e a data de cada partida.
#
# A base foi conferida contra a fonte de gols e cotações: os 277 resultados já
# disputados de 2026 batem exatamente, e as 103 partidas que faltam coincidem
# com as deduzidas pelo formato de turno e returno.
URL_DATALAKE = ("https://raw.githubusercontent.com/leeofernandes1980/"
                "brasileirao-dataset/master/datalake/silver/{arquivo}.parquet")

ARQUIVO_DL_PARTIDAS = DIR_BRUTO / "datalake_partidas.parquet"
ARQUIVO_DL_ESTATISTICAS = DIR_BRUTO / "datalake_estatisticas.parquet"

ANO_INICIO_ESTATISTICAS = 2015   # antes disso as colunas vêm zeradas
ANO_FIM_ESTATISTICAS = 2026      # temporada corrente

# Nomes divergem entre as duas bases; a base de gols (Football-Data) manda.
MAPA_NOMES_ESTATISTICAS = {
    "Flamengo": "Flamengo RJ",
    "Botafogo-RJ": "Botafogo RJ",
    "Chapecoense": "Chapecoense-SC",
}

LINHAS_ESCANTEIOS = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
LINHAS_CARTOES = [2.5, 3.5, 4.5, 5.5, 6.5]


def baixar_estatisticas() -> dict:
    """Baixa (com cache) os parquet da camada silver do datalake."""
    import requests

    resultado = {}
    for chave, destino in (("partidas", ARQUIVO_DL_PARTIDAS),
                           ("estatisticas", ARQUIVO_DL_ESTATISTICAS)):
        if destino.exists() and destino.stat().st_size > 10_000:
            resultado[chave] = "cache"
            continue
        resposta = requests.get(URL_DATALAKE.format(arquivo=chave), timeout=180)
        resposta.raise_for_status()
        destino.write_bytes(resposta.content)
        resultado[chave] = "baixado"
    return resultado


def _partidas_datalake() -> pd.DataFrame:
    partidas = pd.read_parquet(ARQUIVO_DL_PARTIDAS)
    partidas = partidas.rename(columns={"mandante": "Home", "visitante": "Away",
                                        "temporada": "ano"})
    for coluna in ("Home", "Away"):
        partidas[coluna] = (partidas[coluna].astype(str).str.strip()
                            .replace(MAPA_NOMES_ESTATISTICAS))
    partidas["data"] = pd.to_datetime(partidas["data"], errors="coerce")
    partidas["jogada"] = partidas["gols_mandante"].notna()
    return partidas


def carregar_estatisticas() -> pd.DataFrame:
    """
    Uma linha por partida com escanteios e cartões dos dois lados, de 2015 até
    a temporada corrente.

    Regras de qualidade aplicadas:

    * só entram partidas com o dado dos DOIS times (meia partida distorceria
      tanto o efeito do time quanto o total);
    * ``cartao_vermelho`` nulo com amarelo preenchido é lido como zero — na
      importação recente o campo só é gravado quando houve expulsão;
    * cada mercado usa o seu próprio subconjunto válido, porque escanteios e
      cartões faltam em partidas diferentes.
    """
    estatisticas = pd.read_parquet(ARQUIVO_DL_ESTATISTICAS)
    estatisticas = estatisticas.rename(columns={"temporada": "ano"})
    estatisticas["clube"] = (estatisticas["clube"].astype(str).str.strip()
                             .replace(MAPA_NOMES_ESTATISTICAS))
    for coluna in ("escanteios", "cartao_amarelo", "cartao_vermelho"):
        estatisticas[coluna] = pd.to_numeric(estatisticas[coluna], errors="coerce")
    estatisticas["cartoes"] = (estatisticas["cartao_amarelo"]
                               + estatisticas["cartao_vermelho"].fillna(0))
    estatisticas.loc[estatisticas["cartao_amarelo"].isna(), "cartoes"] = np.nan

    partidas = _partidas_datalake()
    juntado = estatisticas.merge(
        partidas[["partida_id", "ano", "data", "rodada", "Home", "Away", "jogada"]],
        on="partida_id", how="inner", suffixes=("", "_partida"))
    juntado = juntado[juntado["jogada"]
                      & juntado["ano"].between(ANO_INICIO_ESTATISTICAS,
                                               ANO_FIM_ESTATISTICAS)]

    # A coluna "clube" tem DUAS convenções na base, indicadas pela coluna
    # "fonte": as linhas do histórico em CSV (até 2023) trazem o nome do clube,
    # e as importadas do Sofascore (2024 em diante) trazem literalmente
    # "mandante" ou "visitante". Ignorar isso descartaria silenciosamente todas
    # as temporadas recentes - justamente as que mais importam.
    posicional = juntado["clube"].isin(["mandante", "visitante"])
    juntado = juntado.assign(lado=np.where(
        posicional, juntado["clube"],
        np.where(juntado["clube"] == juntado["Home"], "mandante", "visitante")))

    casa = juntado[juntado["lado"] == "mandante"]
    fora = juntado[juntado["lado"] == "visitante"]
    jogos = casa.merge(fora[["partida_id", "escanteios", "cartoes"]],
                       on="partida_id", suffixes=("_mandante", "_visitante"))
    jogos = jogos[["partida_id", "ano", "data", "rodada", "Home", "Away",
                   "escanteios_mandante", "escanteios_visitante",
                   "cartoes_mandante", "cartoes_visitante"]]
    return jogos.sort_values(["data", "partida_id"]).reset_index(drop=True)


def validas_para(estatisticas: pd.DataFrame, mercado: str) -> pd.DataFrame:
    """Subconjunto com o dado dos dois times para o mercado pedido."""
    colunas = ([f"escanteios_{lado}" for lado in ("mandante", "visitante")]
               if mercado == "escanteios"
               else [f"cartoes_{lado}" for lado in ("mandante", "visitante")])
    return estatisticas.dropna(subset=colunas).reset_index(drop=True)


def carregar_calendario(temporada: int = TEMPORADA_ALVO) -> pd.DataFrame:
    """
    Calendário oficial da temporada: rodada, data, mandante e visitante.

    É daqui que sai a aba de rodadas. As partidas ainda não disputadas trazem a
    rodada e a data previstas; jogos adiados aparecem na rodada original, que
    pode ser anterior à rodada corrente.
    """
    partidas = _partidas_datalake()
    do_ano = partidas[partidas["ano"] == temporada].copy()
    do_ano["rodada"] = pd.to_numeric(do_ano["rodada"], errors="coerce").astype("Int64")
    return do_ano[["partida_id", "rodada", "data", "Home", "Away", "jogada",
                   "gols_mandante", "gols_visitante"]].sort_values(
        ["rodada", "data", "Home"]).reset_index(drop=True)


class ModeloContagem:
    """
    Modelo de contagem por partida (escanteios ou cartões).

        log(lambda) = intercepto + efeito_feito[time] + efeito_sofrido[adversário]
                      + mando

    O total da partida é a soma dos dois lados. Como a contagem observada é
    **sobredispersa** em relação a Poisson (a variância supera a média), as
    probabilidades de mais/menos usam uma **binomial negativa** com a dispersão
    estimada dos resíduos — Poisson puro subestimaria as caudas, que é
    justamente onde ficam as linhas de aposta.
    """

    def __init__(self, nome, intercepto, feito, sofrido, mando, dispersao,
                 media_liga, n_partidas, anos):
        self.nome = nome
        self.intercepto = float(intercepto)
        self.feito = dict(feito)
        self.sofrido = dict(sofrido)
        self.mando = float(mando)
        self.dispersao = float(dispersao)     # parâmetro k da binomial negativa
        self.media_liga = float(media_liga)
        self.n_partidas = int(n_partidas)
        self.anos = anos

    def sem_historico(self, times) -> list[str]:
        return sorted({t for t in times if t not in self.feito})

    def lambdas(self, mandante: str, visitante: str) -> tuple[float, float]:
        """Contagem esperada de cada lado; time sem histórico fica na média."""
        lambda_casa = np.exp(self.intercepto
                             + self.feito.get(mandante, 0.0)
                             + self.sofrido.get(visitante, 0.0)
                             + self.mando)
        lambda_fora = np.exp(self.intercepto
                             + self.feito.get(visitante, 0.0)
                             + self.sofrido.get(mandante, 0.0))
        return float(lambda_casa), float(lambda_fora)

    def prob_acima(self, mandante: str, visitante: str, linha: float) -> float:
        """P(total da partida > linha), pela binomial negativa."""
        from scipy.stats import nbinom

        media = sum(self.lambdas(mandante, visitante))
        k = self.dispersao
        p = k / (k + media)
        # P(X > linha) = 1 - P(X <= floor(linha))
        return float(1.0 - nbinom.cdf(np.floor(linha), k, p))

    def total_esperado(self, mandante: str, visitante: str) -> float:
        return sum(self.lambdas(mandante, visitante))


def ajustar_modelo_contagem(dados: pd.DataFrame, nome: str,
                            coluna_mandante: str, coluna_visitante: str,
                            meia_vida_dias: float | None = None
                            ) -> ModeloContagem:
    """
    Ajusta o modelo de contagem e estima a dispersão dos resíduos.

    ``meia_vida_dias`` pondera as partidas no tempo. Isso importa muito para
    cartões: o número médio por jogo subiu ao longo dos anos (arbitragem mais
    rigorosa), e um ajuste sem peso fica preso à média antiga e subestima as
    linhas de mais/menos.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    longo = pd.concat([
        pd.DataFrame({"valor": dados[coluna_mandante], "feito": dados["Home"],
                      "sofrido": dados["Away"], "mando": 1.0,
                      "data": dados["data"]}),
        pd.DataFrame({"valor": dados[coluna_visitante], "feito": dados["Away"],
                      "sofrido": dados["Home"], "mando": 0.0,
                      "data": dados["data"]}),
    ], ignore_index=True).dropna(subset=["valor"])

    if meia_vida_dias:
        pesos = _pesos_temporais(pd.to_datetime(longo["data"]),
                                 pd.to_datetime(dados["data"]).max(),
                                 meia_vida_dias)
        glm = smf.glm("valor ~ C(feito) + C(sofrido) + mando", data=longo,
                      family=sm.families.Poisson(), freq_weights=pesos).fit()
    else:
        glm = smf.glm("valor ~ C(feito) + C(sofrido) + mando", data=longo,
                      family=sm.families.Poisson()).fit()

    times = sorted(set(longo["feito"]))
    feito = {t: 0.0 for t in times}
    sofrido = {t: 0.0 for t in times}
    padrao = re.compile(r"C\((feito|sofrido)\)\[T\.(.+)\]")
    for parametro, valor in glm.params.items():
        achado = padrao.fullmatch(parametro)
        if achado:
            alvo = feito if achado.group(1) == "feito" else sofrido
            alvo[achado.group(2)] = float(valor)

    # Recentragem: o patsy deixa um time como referência com coeficiente zero,
    # e os demais saem relativos a ELE. Sem corrigir isso, um time sem
    # histórico (que cai no valor 0,0) seria tratado como aquele time de
    # referência específico, e não como um time médio — no Brasileirão isso
    # inflava os escanteios do adversário do Remo em mais de 10%.
    # Deslocando a média dos coeficientes para o intercepto, o zero passa a
    # significar "média da liga" e as previsões dos times conhecidos não mudam.
    media_feito = float(np.mean(list(feito.values())))
    media_sofrido = float(np.mean(list(sofrido.values())))
    feito = {t: v - media_feito for t, v in feito.items()}
    sofrido = {t: v - media_sofrido for t, v in sofrido.items()}
    intercepto = float(glm.params["Intercept"]) + media_feito + media_sofrido

    modelo = ModeloContagem(nome, intercepto, feito, sofrido,
                            glm.params["mando"], np.inf,
                            float(longo["valor"].mean()), len(dados),
                            (int(dados["ano"].min()), int(dados["ano"].max())))

    # Dispersão do TOTAL da partida: V = mu + mu^2 / k  =>  k = mu^2 / (V - mu).
    totais = (dados[coluna_mandante] + dados[coluna_visitante]).to_numpy(dtype=float)
    esperados = np.array([modelo.total_esperado(m, v)
                          for m, v in zip(dados["Home"], dados["Away"])])
    excesso = np.mean((totais - esperados) ** 2 - esperados)
    modelo.dispersao = (float(np.mean(esperados ** 2) / excesso)
                        if excesso > 0 else 1e6)   # 1e6 ~ Poisson
    return modelo


def ancorar_modelo_contagem(modelo: ModeloContagem, dados_recentes: pd.DataFrame,
                            coluna_mandante: str, coluna_visitante: str) -> ModeloContagem:
    """
    Desloca o intercepto para que a média prevista bata com a média observada
    nas temporadas mais recentes.

    Serve para um problema medido: o número de cartões por partida subiu ao
    longo dos anos e um modelo ajustado na janela inteira fica preso ao nível
    antigo. Ancorar na última temporada disponível reduziu o viés de
    "mais de 4,5 cartões" de -4,8 para -2,9 pontos percentuais na validação
    fora da amostra.
    """
    observado = float((dados_recentes[coluna_mandante]
                       + dados_recentes[coluna_visitante]).mean())
    previsto = float(np.mean([modelo.total_esperado(m, v)
                              for m, v in zip(dados_recentes["Home"],
                                              dados_recentes["Away"])]))
    if previsto > 0 and observado > 0:
        modelo.intercepto += float(np.log(observado / previsto))
    return modelo


def ajustar_modelos_estatisticas(estatisticas: pd.DataFrame,
                                 meia_vida_dias: float = 365.0,
                                 temporadas_ancora: int = 1) -> dict:
    """
    Ajusta os modelos de escanteios e cartões na configuração validada:
    ponderação temporal de 365 dias e âncora na temporada mais recente.
    """
    modelos = {}
    for nome, coluna_mandante, coluna_visitante in (
            ("escanteios", "escanteios_mandante", "escanteios_visitante"),
            ("cartoes", "cartoes_mandante", "cartoes_visitante")):
        # Cada mercado usa o seu próprio subconjunto: escanteios e cartões
        # faltam em partidas diferentes.
        dados = validas_para(estatisticas, nome)
        ultimo_ano = int(dados["ano"].max())
        recentes = dados[dados["ano"] > ultimo_ano - temporadas_ancora]
        modelo = ajustar_modelo_contagem(dados, nome, coluna_mandante,
                                         coluna_visitante, meia_vida_dias)
        modelo = ancorar_modelo_contagem(modelo, recentes, coluna_mandante,
                                         coluna_visitante)
        modelos[nome] = modelo
    return modelos
