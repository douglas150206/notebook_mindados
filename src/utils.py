"""
Funções auxiliares do notebook de Mineração de Dados.

Disciplina de Mineração de Dados - ADS, IFSP Câmpus Jacareí.
Tema: eficiência das cotações de apostas esportivas.

O módulo concentra o código mecânico (download, leitura robusta, cálculo de
Elo, médias móveis, métricas) para que o notebook fique legível e as células
curtas. Todas as decisões metodológicas ficam explicadas no notebook.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos do projeto (relativos, via pathlib, para rodar no VSCode e no Jupyter)
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "data"
DIR_BRUTO = DIR_DADOS / "raw"
DIR_PROCESSADO = DIR_DADOS / "processed"
DIR_SAIDAS = RAIZ / "outputs"
DIR_FIGURAS = DIR_SAIDAS / "figuras"
DIR_TABELAS = DIR_SAIDAS / "tabelas"

for _d in (DIR_BRUTO, DIR_PROCESSADO, DIR_FIGURAS, DIR_TABELAS):
    _d.mkdir(parents=True, exist_ok=True)

SEMENTE = 42

# ---------------------------------------------------------------------------
# Configuração da amostra
# ---------------------------------------------------------------------------

LIGAS_PRINCIPAIS = ["E0", "SP1", "I1", "D1", "F1"]
LIGAS_SEGUNDAS = ["E1", "SP2", "I2", "D2", "F2"]

NOME_LIGA = {
    "E0": "Inglaterra - Premier League",
    "E1": "Inglaterra - Championship",
    "SP1": "Espanha - LaLiga",
    "SP2": "Espanha - LaLiga 2",
    "I1": "Itália - Serie A",
    "I2": "Itália - Serie B",
    "D1": "Alemanha - Bundesliga",
    "D2": "Alemanha - 2. Bundesliga",
    "F1": "França - Ligue 1",
    "F2": "França - Ligue 2",
}

PAIS_LIGA = {
    "E0": "Inglaterra", "E1": "Inglaterra",
    "SP1": "Espanha", "SP2": "Espanha",
    "I1": "Itália", "I2": "Itália",
    "D1": "Alemanha", "D2": "Alemanha",
    "F1": "França", "F2": "França",
}

DIVISAO_LIGA = {
    "E0": "1ª divisão", "SP1": "1ª divisão", "I1": "1ª divisão",
    "D1": "1ª divisão", "F1": "1ª divisão",
    "E1": "2ª divisão", "SP2": "2ª divisão", "I2": "2ª divisão",
    "D2": "2ª divisão", "F2": "2ª divisão",
}


def codigo_temporada(ano_inicial: int) -> str:
    """2005 -> '0506'; 2024 -> '2425' (padrão de URL do Football-Data)."""
    return f"{ano_inicial % 100:02d}{(ano_inicial + 1) % 100:02d}"


def rotulo_temporada(ano_inicial: int) -> str:
    """2005 -> '2005/06'."""
    return f"{ano_inicial}/{(ano_inicial + 1) % 100:02d}"


# Aquecimento do Elo e das médias móveis (fora da amostra analisada).
ANOS_AQUECIMENTO = list(range(2000, 2005))
# Amostra efetivamente analisada.
ANOS_AMOSTRA = list(range(2005, 2025))
ANOS_TODOS = ANOS_AQUECIMENTO + ANOS_AMOSTRA

TEMPORADAS_AQUECIMENTO = [codigo_temporada(a) for a in ANOS_AQUECIMENTO]
TEMPORADAS_AMOSTRA = [codigo_temporada(a) for a in ANOS_AMOSTRA]
TEMPORADAS_TODAS = [codigo_temporada(a) for a in ANOS_TODOS]

# Divisão temporal da modelagem.
ANOS_TREINO = list(range(2005, 2022))   # 2005/06 a 2021/22
ANOS_TESTE = list(range(2022, 2025))    # 2022/23 a 2024/25
ANOS_VALIDACAO = [2019, 2020, 2021]     # validação por temporada, dentro do treino

# ---------------------------------------------------------------------------
# Fontes de dados
# ---------------------------------------------------------------------------

# Fonte canônica declarada na metodologia.
URL_OFICIAL = "https://www.football-data.co.uk/mmz4281/{temporada}/{divisao}.csv"

# Espelho público dos mesmos arquivos, usado apenas quando a fonte oficial está
# inacessível a partir do ambiente de execução (ver "Decisões e limitações").
URL_ESPELHO = (
    "https://raw.githubusercontent.com/huhao930422-debug/football-odds-mirror"
    "/main/data/{pasta}/season-{temporada}.csv"
)

PASTA_ESPELHO = {
    "E0": "premier-league", "E1": "championship",
    "SP1": "la-liga", "SP2": "la-liga-2",
    "I1": "serie-a", "I2": "serie-b",
    "D1": "bundesliga", "D2": "bundesliga-2",
    "F1": "ligue-1", "F2": "ligue-2",
}


@dataclass
class RegistroDownload:
    """Uma linha do relatório de coleta."""
    divisao: str
    temporada: str
    situacao: str          # 'cache', 'oficial', 'espelho' ou 'falhou'
    bytes_arquivo: int = 0
    detalhe: str = ""


def _baixar_url(url: str, tempo_limite: int = 60) -> bytes:
    import requests
    resposta = requests.get(url, timeout=tempo_limite)
    resposta.raise_for_status()
    conteudo = resposta.content
    if len(conteudo) < 200 or conteudo[:15].lower().startswith(b"404:"):
        raise ValueError(f"resposta vazia ou inválida ({len(conteudo)} bytes)")
    return conteudo


def baixar_temporadas(
    divisoes: list[str],
    temporadas: list[str],
    usar_oficial: bool = True,
    max_threads: int = 12,
) -> pd.DataFrame:
    """
    Baixa os CSVs para ``data/raw`` e devolve o relatório da coleta.

    O diretório funciona como cache: arquivos já presentes não são baixados de
    novo. Tenta primeiro a fonte oficial (Football-Data.co.uk); se ela estiver
    inacessível a partir da máquina, passa a usar o espelho público para os
    arquivos restantes, registrando a origem de cada um.
    """
    import concurrent.futures as cf

    estado = {"oficial_ok": usar_oficial, "motivo_oficial": ""}
    pendentes: list[tuple[str, str]] = []
    registros: list[RegistroDownload] = []

    for divisao in divisoes:
        for temporada in temporadas:
            destino = DIR_BRUTO / f"{divisao}_{temporada}.csv"
            if destino.exists() and destino.stat().st_size > 200:
                registros.append(RegistroDownload(
                    divisao, temporada, "cache", destino.stat().st_size))
            else:
                pendentes.append((divisao, temporada))

    # Sonda única da fonte oficial: evita repetir 250 tentativas quando o
    # ambiente bloqueia o domínio.
    if pendentes and estado["oficial_ok"]:
        divisao, temporada = pendentes[0]
        try:
            _baixar_url(URL_OFICIAL.format(temporada=temporada, divisao=divisao))
        except Exception as erro:
            estado["oficial_ok"] = False
            estado["motivo_oficial"] = f"{type(erro).__name__}: {str(erro)[:150]}"

    def tarefa(par: tuple[str, str]) -> RegistroDownload:
        divisao, temporada = par
        destino = DIR_BRUTO / f"{divisao}_{temporada}.csv"
        erros = []
        if estado["oficial_ok"]:
            try:
                dados = _baixar_url(
                    URL_OFICIAL.format(temporada=temporada, divisao=divisao))
                destino.write_bytes(dados)
                return RegistroDownload(divisao, temporada, "oficial", len(dados))
            except Exception as erro:
                erros.append(f"oficial: {type(erro).__name__}")
        try:
            dados = _baixar_url(URL_ESPELHO.format(
                pasta=PASTA_ESPELHO[divisao], temporada=temporada))
            destino.write_bytes(dados)
            return RegistroDownload(divisao, temporada, "espelho", len(dados))
        except Exception as erro:
            erros.append(f"espelho: {type(erro).__name__}: {str(erro)[:80]}")
        return RegistroDownload(divisao, temporada, "falhou", 0, "; ".join(erros))

    if pendentes:
        with cf.ThreadPoolExecutor(max_threads) as executor:
            registros.extend(executor.map(tarefa, pendentes))

    relatorio = pd.DataFrame([r.__dict__ for r in registros])
    relatorio = relatorio.sort_values(["divisao", "temporada"]).reset_index(drop=True)
    relatorio.attrs["oficial_ok"] = estado["oficial_ok"]
    relatorio.attrs["motivo_oficial"] = estado["motivo_oficial"]
    return relatorio


# ---------------------------------------------------------------------------
# Leitura robusta dos CSVs
# ---------------------------------------------------------------------------

@dataclass
class ResultadoLeitura:
    dados: pd.DataFrame
    codificacao: str
    linhas_vazias: int = 0
    linhas_malformadas: int = 0
    colunas_descartadas: int = 0


def ler_csv_robusto(caminho: Path) -> ResultadoLeitura:
    """
    Lê um CSV do Football-Data tolerando os problemas típicos da base:

    * codificação UTF-8 ou latin-1 (as temporadas antigas vêm em Windows-1252);
    * vírgulas sobrando no fim das linhas (colunas fantasma);
    * linhas e colunas completamente vazias;
    * linhas com número de campos diferente do cabeçalho.

    Linhas realmente malformadas são contadas e descartadas, nunca ignoradas
    em silêncio.
    """
    bruto = caminho.read_bytes()
    for codificacao in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            texto = bruto.decode(codificacao)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 aceita qualquer byte
        texto = bruto.decode("latin-1", errors="replace")
        codificacao = "latin-1 (com substituições)"

    leitor = csv.reader(io.StringIO(texto))
    try:
        cabecalho = next(leitor)
    except StopIteration:
        return ResultadoLeitura(pd.DataFrame(), codificacao)

    cabecalho = [c.strip() for c in cabecalho]
    # Descarta as colunas sem nome do fim da linha (vírgulas sobrando).
    while cabecalho and cabecalho[-1] == "":
        cabecalho.pop()
    n_colunas = len(cabecalho)

    linhas, vazias, malformadas = [], 0, 0
    for linha in leitor:
        if not any(campo.strip() for campo in linha):
            vazias += 1
            continue
        if len(linha) > n_colunas:
            excedente = linha[n_colunas:]
            if any(campo.strip() for campo in excedente):
                malformadas += 1
                continue
            linha = linha[:n_colunas]
        elif len(linha) < n_colunas:
            linha = linha + [""] * (n_colunas - len(linha))
        linhas.append(linha)

    dados = pd.DataFrame(linhas, columns=cabecalho)
    dados = dados.replace("", np.nan)

    antes = dados.shape[1]
    dados = dados.dropna(axis=1, how="all")          # colunas totalmente vazias
    dados = dados.dropna(axis=0, how="all")          # linhas totalmente vazias
    # Nomes de coluna duplicados aparecem em algumas temporadas antigas.
    dados = dados.loc[:, ~dados.columns.duplicated()]
    descartadas = antes - dados.shape[1]

    return ResultadoLeitura(
        dados.reset_index(drop=True), codificacao, vazias, malformadas, descartadas)


# Colunas numéricas conhecidas do Football-Data que devem virar float.
_PREFIXOS_NUMERICOS = (
    "FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF",
    "HC", "AC", "HY", "AY", "HR", "AR",
)


def _converter_numericos(dados: pd.DataFrame) -> pd.DataFrame:
    """Converte estatísticas e cotações para número, mantendo texto onde é texto."""
    texto = {"Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR", "HTR", "Referee"}
    for coluna in dados.columns:
        if coluna in texto:
            continue
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    return dados


def carregar_base_bruta(
    divisoes: list[str], temporadas: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Empilha todos os CSVs baixados em uma única base e devolve também o
    relatório de leitura (codificação e linhas problemáticas por arquivo).
    """
    partes, diagnostico = [], []
    for divisao in divisoes:
        for temporada in temporadas:
            caminho = DIR_BRUTO / f"{divisao}_{temporada}.csv"
            if not caminho.exists():
                diagnostico.append({
                    "divisao": divisao, "temporada": temporada,
                    "situacao": "arquivo ausente", "linhas": 0,
                    "codificacao": "", "linhas_vazias": 0,
                    "linhas_malformadas": 0, "colunas_descartadas": 0})
                continue
            leitura = ler_csv_robusto(caminho)
            dados = leitura.dados
            if dados.empty:
                diagnostico.append({
                    "divisao": divisao, "temporada": temporada,
                    "situacao": "vazio", "linhas": 0,
                    "codificacao": leitura.codificacao, "linhas_vazias": 0,
                    "linhas_malformadas": 0, "colunas_descartadas": 0})
                continue
            dados = _converter_numericos(dados)
            dois_digitos = int(temporada[:2])
            ano = (2000 if dois_digitos < 90 else 1900) + dois_digitos
            dados = dados.assign(Div=divisao, temporada=temporada, ano_temporada=ano).copy()
            partes.append(dados)
            diagnostico.append({
                "divisao": divisao, "temporada": temporada, "situacao": "ok",
                "linhas": len(dados), "codificacao": leitura.codificacao,
                "linhas_vazias": leitura.linhas_vazias,
                "linhas_malformadas": leitura.linhas_malformadas,
                "colunas_descartadas": leitura.colunas_descartadas})

    base = pd.concat(partes, ignore_index=True, sort=False) if partes else pd.DataFrame()
    return base, pd.DataFrame(diagnostico)


# ---------------------------------------------------------------------------
# Limpeza: datas, horários e ordenação
# ---------------------------------------------------------------------------

def preparar_datas(base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Converte ``Date`` aceitando ``dd/mm/yy`` e ``dd/mm/yyyy`` e ordena a base por
    data e, quando a coluna existe, por horário.
    """
    base = base.copy()
    texto = base["Date"].astype(str).str.strip()

    datas = pd.to_datetime(texto, format="%d/%m/%Y", errors="coerce")
    faltantes = datas.isna()
    datas.loc[faltantes] = pd.to_datetime(
        texto[faltantes], format="%d/%m/%y", errors="coerce")
    # Rede de segurança para qualquer formato residual (sempre dia antes do mês).
    faltantes = datas.isna()
    if faltantes.any():
        datas.loc[faltantes] = pd.to_datetime(
            texto[faltantes], dayfirst=True, errors="coerce")

    base["data"] = datas
    if "Time" in base.columns:
        base["horario"] = base["Time"].astype(str).str.strip().replace("nan", "")
    else:
        base["horario"] = ""

    relatorio = {
        "total": len(base),
        "formato_longo_dd_mm_yyyy": int((texto.str.len() == 10).sum()),
        "formato_curto_dd_mm_yy": int((texto.str.len() == 8).sum()),
        "datas_nao_convertidas": int(base["data"].isna().sum()),
        "com_horario": int((base["horario"] != "").sum()),
    }

    base = base.sort_values(["data", "horario", "Div", "HomeTeam"], kind="mergesort")
    base = base.reset_index(drop=True)
    base["partida_id"] = np.arange(len(base))
    return base, relatorio


# ---------------------------------------------------------------------------
# Padronização dos nomes das equipes
# ---------------------------------------------------------------------------

def normalizar_texto(nome: str) -> str:
    """Chave de comparação: sem acentos, sem pontuação, minúsculas."""
    sem_acento = (unicodedata.normalize("NFKD", str(nome))
                  .encode("ascii", "ignore").decode())
    return re.sub(r"[^a-z0-9]", "", sem_acento.lower())


def limpar_espacos(serie: pd.Series) -> pd.Series:
    """Remove espaços nas pontas e colapsa espaços internos repetidos."""
    return (serie.astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip())


def diagnosticar_equipes(base: pd.DataFrame, limiar: float = 0.80) -> dict:
    """
    Procura grafias diferentes da mesma equipe.

    Devolve, por liga, dois diagnósticos:

    * ``colisoes``  - nomes distintos que colapsam na mesma chave normalizada
      (acento, pontuação ou caixa diferentes). São renomeações certas.
    * ``candidatos`` - pares de nomes com similaridade alta, para inspeção
      manual. Times homônimos legítimos aparecem aqui e NÃO devem ser unidos.
    """
    import difflib

    resultado = {}
    for liga, grupo in base.groupby("Div"):
        nomes = sorted(set(grupo["HomeTeam"].dropna()) | set(grupo["AwayTeam"].dropna()))

        chaves: dict[str, list[str]] = {}
        for nome in nomes:
            chaves.setdefault(normalizar_texto(nome), []).append(nome)
        colisoes = {k: v for k, v in chaves.items() if len(v) > 1}

        candidatos = []
        for i, a in enumerate(nomes):
            for b in nomes[i + 1:]:
                razao = difflib.SequenceMatcher(
                    None, normalizar_texto(a), normalizar_texto(b)).ratio()
                if razao >= limiar:
                    candidatos.append((a, b, round(razao, 3)))
        resultado[liga] = {"n_nomes": len(nomes), "colisoes": colisoes,
                           "candidatos": candidatos}
    return resultado


def mapeamento_por_colisao(diagnostico: dict, base: pd.DataFrame) -> dict[str, str]:
    """
    Monta o dicionário de mapeamento a partir das colisões de normalização:
    todas as grafias viram a variante mais frequente na base.
    """
    contagem = pd.concat([base["HomeTeam"], base["AwayTeam"]]).value_counts()
    mapeamento: dict[str, str] = {}
    for info in diagnostico.values():
        for variantes in info["colisoes"].values():
            preferido = max(variantes, key=lambda n: contagem.get(n, 0))
            for variante in variantes:
                if variante != preferido:
                    mapeamento[variante] = preferido
    return mapeamento


def aplicar_mapeamento_equipes(
    base: pd.DataFrame, mapeamento: dict[str, str]
) -> pd.DataFrame:
    """Aplica limpeza de espaços e o dicionário explícito de renomeações."""
    base = base.copy()
    for coluna in ("HomeTeam", "AwayTeam"):
        base[coluna] = limpar_espacos(base[coluna])
        if mapeamento:
            base[coluna] = base[coluna].replace(mapeamento)
    return base


# ---------------------------------------------------------------------------
# Cobertura das colunas relevantes
# ---------------------------------------------------------------------------

COLUNAS_RESULTADO = ["FTHG", "FTAG", "FTR"]
COLUNAS_ESTATISTICAS = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY"]


def tabela_cobertura(base: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Percentual de valores não nulos por liga e temporada (formato longo)."""
    linhas = []
    for (liga, temporada), grupo in base.groupby(["Div", "temporada"]):
        linha = {"liga": liga, "temporada": temporada, "n_partidas": len(grupo)}
        for coluna in colunas:
            if coluna in grupo.columns:
                linha[coluna] = round(100 * grupo[coluna].notna().mean(), 2)
            else:
                linha[coluna] = np.nan   # coluna inexistente na temporada
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values(["liga", "temporada"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cotações -> probabilidades
# ---------------------------------------------------------------------------

# Cada operador tem uma lista de prefixos candidatos, porque os nomes mudam
# entre temporadas (ver notes.txt do Football-Data).
OPERADORES = {
    "Bet365": ["B365"],
    "Bet&Win": ["BW"],
    "Interwetten": ["IW"],
    "Pinnacle": ["PS", "P"],
    "William Hill": ["WH"],
    "VC Bet": ["VC"],
    "Média de mercado": ["BbAv", "Avg"],
}


def colunas_operador(base: pd.DataFrame, prefixos: list[str]) -> list[str] | None:
    """Primeiro trio (H, D, A) existente na base para um operador."""
    for prefixo in prefixos:
        colunas = [f"{prefixo}{sufixo}" for sufixo in ("H", "D", "A")]
        if all(coluna in base.columns for coluna in colunas):
            return colunas
    return None


def implicitas_e_margem(
    base: pd.DataFrame, colunas: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Probabilidades implícitas brutas (1 / cotação) e margem (overround):
    soma das três implícitas menos 1.
    """
    cotacoes = base[colunas].apply(pd.to_numeric, errors="coerce")
    cotacoes = cotacoes.where(cotacoes > 1.0)      # cotação <= 1 é inválida
    implicitas = 1.0 / cotacoes
    soma = implicitas.sum(axis=1, min_count=3)
    return implicitas, soma - 1.0


def probabilidades_normalizadas(
    base: pd.DataFrame, colunas: list[str]
) -> pd.DataFrame:
    """Cada implícita dividida pela soma das três (remove a margem)."""
    implicitas, margem = implicitas_e_margem(base, colunas)
    soma = margem + 1.0
    normalizadas = implicitas.div(soma, axis=0)
    normalizadas.columns = ["prob_H", "prob_D", "prob_A"]
    return normalizadas


# ---------------------------------------------------------------------------
# Estatística de apoio
# ---------------------------------------------------------------------------

def ic_wilson(sucessos: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Intervalo de confiança de Wilson para uma proporção (95% com z=1,96)."""
    sucessos = np.asarray(sucessos, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, sucessos / n, np.nan)
        denominador = 1 + z**2 / n
        centro = (p + z**2 / (2 * n)) / denominador
        raiz = np.sqrt(np.maximum(p * (1 - p) / n + z**2 / (4 * n**2), 0))
        margem = z * raiz / denominador
    return centro - margem, centro + margem


# ---------------------------------------------------------------------------
# Atributos históricos sem vazamento
# ---------------------------------------------------------------------------

# Colunas do próprio jogo: nunca podem virar atributo.
COLUNAS_PROIBIDAS = [
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]

# Métricas acompanhadas na perspectiva de cada time.
METRICAS_LONGO = [
    "gols_feitos", "gols_sofridos", "pontos",
    "fin_feitas", "fin_sofridas",
    "fin_alvo_feitas", "fin_alvo_sofridas",
    "esc_feitos", "esc_sofridos",
]

JANELAS = (5, 10)


def formato_longo(base: pd.DataFrame) -> pd.DataFrame:
    """
    Duas linhas por partida, uma na perspectiva de cada time.

    Usa os dados do próprio jogo apenas para *construir o histórico*; nenhuma
    dessas colunas entra na matriz de atributos sem passar por ``shift(1)``.
    """
    def coluna(nome: str) -> pd.Series:
        if nome in base.columns:
            return pd.to_numeric(base[nome], errors="coerce")
        return pd.Series(np.nan, index=base.index)

    pontos_mandante = base["FTR"].map({"H": 3, "D": 1, "A": 0})
    pontos_visitante = base["FTR"].map({"H": 0, "D": 1, "A": 3})

    comum = {
        "partida_id": base["partida_id"], "Div": base["Div"],
        "data": base["data"], "horario": base["horario"],
        "ano_temporada": base["ano_temporada"],
    }

    casa = pd.DataFrame({
        **comum, "time": base["HomeTeam"], "adversario": base["AwayTeam"],
        "mando": "C",
        "gols_feitos": coluna("FTHG"), "gols_sofridos": coluna("FTAG"),
        "pontos": pontos_mandante,
        "fin_feitas": coluna("HS"), "fin_sofridas": coluna("AS"),
        "fin_alvo_feitas": coluna("HST"), "fin_alvo_sofridas": coluna("AST"),
        "esc_feitos": coluna("HC"), "esc_sofridos": coluna("AC"),
    })
    fora = pd.DataFrame({
        **comum, "time": base["AwayTeam"], "adversario": base["HomeTeam"],
        "mando": "F",
        "gols_feitos": coluna("FTAG"), "gols_sofridos": coluna("FTHG"),
        "pontos": pontos_visitante,
        "fin_feitas": coluna("AS"), "fin_sofridas": coluna("HS"),
        "fin_alvo_feitas": coluna("AST"), "fin_alvo_sofridas": coluna("HST"),
        "esc_feitos": coluna("AC"), "esc_sofridos": coluna("HC"),
    })

    longo = pd.concat([casa, fora], ignore_index=True)
    return longo.sort_values(["time", "data", "horario", "partida_id"],
                             kind="mergesort").reset_index(drop=True)


def medias_moveis(longo: pd.DataFrame, janelas=JANELAS) -> pd.DataFrame:
    """
    Médias móveis das últimas ``w`` partidas de cada time.

    O ``shift(1)`` vem ANTES do ``rolling``: a partida atual nunca entra no seu
    próprio atributo. O histórico é contínuo entre temporadas e inclui o
    aquecimento (2000/01 a 2004/05).
    """
    longo = longo.copy()
    por_time = longo.groupby("time", sort=False)

    novas = {}
    for metrica in METRICAS_LONGO:
        deslocada = por_time[metrica].shift(1)
        for janela in janelas:
            novas[f"{metrica}_m{janela}"] = (
                deslocada.groupby(longo["time"], sort=False)
                .rolling(janela, min_periods=1).mean()
                .reset_index(level=0, drop=True))

    # Número de partidas anteriores do time na base (inclui o aquecimento).
    novas["jogos_anteriores"] = por_time.cumcount()

    # Forma por mando: últimas 5 partidas do time NO MESMO mando.
    for coluna in ("forma_mando_gols_5", "forma_mando_pontos_5"):
        novas[coluna] = pd.Series(np.nan, index=longo.index)
    longo = pd.concat([longo, pd.DataFrame(novas, index=longo.index)], axis=1)

    for mando in ("C", "F"):
        sub = longo[longo["mando"] == mando]
        por_time_mando = sub.groupby("time", sort=False)
        for metrica, destino in (("gols_feitos", "forma_mando_gols_5"),
                                 ("pontos", "forma_mando_pontos_5")):
            deslocada = por_time_mando[metrica].shift(1)
            longo.loc[sub.index, destino] = (
                deslocada.groupby(sub["time"], sort=False)
                .rolling(5, min_periods=1).mean()
                .reset_index(level=0, drop=True))

    return longo


def calcular_elo(
    base: pd.DataFrame,
    k: float = 20.0,
    vantagem_mando: float = 60.0,
    elo_inicial: float = 1500.0,
    fator_regressao: float = 1 / 3,
) -> pd.DataFrame:
    """
    Elo pré-jogo, calculado separadamente por liga.

    Regras (as do enunciado):

    * início em 1500, K = 20, vantagem de mando de 60 pontos na expectativa;
    * na virada de temporada, cada Elo regride 1/3 em direção à média (1500);
    * promovidos (times que não disputaram a liga na temporada anterior) entram
      com a média do Elo dos rebaixados da mesma liga na temporada anterior.

    Devolve ``partida_id`` com o Elo de ANTES da partida e o indicador de
    promovido para cada lado.
    """
    registros = []

    for liga, jogos_liga in base.groupby("Div", sort=False):
        elo: dict[str, float] = {}
        times_anteriores: set[str] = set()

        for ano in sorted(jogos_liga["ano_temporada"].unique()):
            jogos = jogos_liga[jogos_liga["ano_temporada"] == ano].sort_values(
                ["data", "horario", "partida_id"], kind="mergesort")
            times_atuais = set(jogos["HomeTeam"]) | set(jogos["AwayTeam"])

            if times_anteriores:
                # 1) regressão à média de todos os times conhecidos
                for time in list(elo):
                    elo[time] += (elo_inicial - elo[time]) * fator_regressao
                # 2) promovidos herdam a média (já regredida) dos rebaixados
                rebaixados = times_anteriores - times_atuais
                valores = [elo[t] for t in rebaixados if t in elo]
                elo_promovido = float(np.mean(valores)) if valores else elo_inicial
                for time in times_atuais - times_anteriores:
                    elo[time] = elo_promovido
            promovidos = (times_atuais - times_anteriores) if times_anteriores else set()

            for time in times_atuais:
                elo.setdefault(time, elo_inicial)

            for partida_id, mandante, visitante, resultado in zip(
                    jogos["partida_id"], jogos["HomeTeam"],
                    jogos["AwayTeam"], jogos["FTR"]):
                elo_mandante, elo_visitante = elo[mandante], elo[visitante]
                registros.append({
                    "partida_id": partida_id,
                    "elo_mandante": elo_mandante,
                    "elo_visitante": elo_visitante,
                    "promovido_mandante": int(mandante in promovidos),
                    "promovido_visitante": int(visitante in promovidos),
                })
                esperado = 1.0 / (1.0 + 10 ** (
                    (elo_visitante - elo_mandante - vantagem_mando) / 400.0))
                obtido = {"H": 1.0, "D": 0.5, "A": 0.0}.get(resultado)
                if obtido is None:      # partida sem resultado não atualiza o Elo
                    continue
                ajuste = k * (obtido - esperado)
                elo[mandante] = elo_mandante + ajuste
                elo[visitante] = elo_visitante - ajuste

            times_anteriores = times_atuais

    return pd.DataFrame(registros)


def montar_matriz_atributos(
    base: pd.DataFrame, longo: pd.DataFrame, elo: pd.DataFrame
) -> pd.DataFrame:
    """
    Junta, por partida, os atributos do mandante e do visitante, as diferenças
    (mandante - visitante) e a liga em one-hot.
    """
    colunas_moveis = [f"{m}_m{j}" for m in METRICAS_LONGO for j in JANELAS]
    colunas_time = colunas_moveis + [
        "forma_mando_gols_5", "forma_mando_pontos_5", "jogos_anteriores"]

    casa = (longo[longo["mando"] == "C"]
            .set_index("partida_id")[colunas_time]
            .add_suffix("_mandante"))
    fora = (longo[longo["mando"] == "F"]
            .set_index("partida_id")[colunas_time]
            .add_suffix("_visitante"))

    atributos = (base.set_index("partida_id")[["Div", "ano_temporada", "data"]]
                 .join(casa).join(fora)
                 .join(elo.set_index("partida_id")))

    for coluna in colunas_moveis:
        atributos[f"dif_{coluna}"] = (
            atributos[f"{coluna}_mandante"] - atributos[f"{coluna}_visitante"])
    atributos["dif_elo"] = atributos["elo_mandante"] - atributos["elo_visitante"]

    for liga in sorted(base["Div"].unique()):
        atributos[f"liga_{liga}"] = (atributos["Div"] == liga).astype(int)

    return atributos.reset_index()


def colunas_de_atributos(atributos: pd.DataFrame) -> list[str]:
    """Colunas que entram no modelo (variante A: só desempenho, sem cotações)."""
    excluir = {"partida_id", "Div", "ano_temporada", "data",
               "jogos_anteriores_mandante", "jogos_anteriores_visitante"}
    return [c for c in atributos.columns if c not in excluir]


# ---------------------------------------------------------------------------
# Métricas de avaliação
# ---------------------------------------------------------------------------

CLASSES = ["H", "D", "A"]      # ordem fixa em todas as matrizes de probabilidade


def rps(y_verdadeiro, probabilidades) -> float:
    """
    Ranked Probability Score na ordem H, D, A (implementado à mão).

    RPS = 1/(r-1) * soma_i ( soma_{j<=i} (p_j - e_j) )^2, com r = 3 desfechos.
    Quanto menor, melhor.
    """
    probabilidades = np.asarray(probabilidades, dtype=float)
    indice = {classe: i for i, classe in enumerate(CLASSES)}
    real = np.zeros_like(probabilidades)
    real[np.arange(len(y_verdadeiro)),
         [indice[v] for v in y_verdadeiro]] = 1.0
    acumulado = np.cumsum(probabilidades, axis=1) - np.cumsum(real, axis=1)
    return float(np.mean(np.sum(acumulado[:, :-1] ** 2, axis=1) / (len(CLASSES) - 1)))


def bootstrap_pareado_acuracia(
    acertos_modelo: np.ndarray,
    acertos_referencia: np.ndarray,
    n_reamostras: int = 2000,
    semente: int = SEMENTE,
):
    """
    IC 95% por bootstrap pareado da diferença de acurácia (modelo - referência).

    Pareado: cada reamostragem sorteia as MESMAS partidas para os dois lados.
    """
    acertos_modelo = np.asarray(acertos_modelo, dtype=float)
    acertos_referencia = np.asarray(acertos_referencia, dtype=float)
    diferenca = acertos_modelo - acertos_referencia
    gerador = np.random.default_rng(semente)
    n = len(diferenca)
    indices = gerador.integers(0, n, size=(n_reamostras, n))
    distribuicao = diferenca[indices].mean(axis=1)
    return (float(diferenca.mean()),
            float(np.percentile(distribuicao, 2.5)),
            float(np.percentile(distribuicao, 97.5)))


def teste_mcnemar(acertos_modelo: np.ndarray, acertos_referencia: np.ndarray):
    """Teste de McNemar sobre acertos pareados. Devolve (estatística, p-valor)."""
    from statsmodels.stats.contingency_tables import mcnemar

    acertos_modelo = np.asarray(acertos_modelo).astype(bool)
    acertos_referencia = np.asarray(acertos_referencia).astype(bool)
    tabela = [
        [int(np.sum(acertos_modelo & acertos_referencia)),
         int(np.sum(acertos_modelo & ~acertos_referencia))],
        [int(np.sum(~acertos_modelo & acertos_referencia)),
         int(np.sum(~acertos_modelo & ~acertos_referencia))],
    ]
    discordantes = tabela[0][1] + tabela[1][0]
    resultado = mcnemar(tabela, exact=discordantes < 25, correction=True)
    return float(resultado.statistic), float(resultado.pvalue), tabela


def formatar_br(valor: float, casas: int = 1) -> str:
    """Número no formato brasileiro: vírgula decimal."""
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "-"
    return f"{valor:.{casas}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Padrão visual das figuras
# ---------------------------------------------------------------------------

# Paleta categórica validada para daltonismo (ΔE mínimo entre vizinhos = 9,1 em
# OKLab x100, acima do alvo de 8). A ordem dos slots é fixa: a cor acompanha a
# entidade (o operador), nunca a sua posição no ranking.
CORES_SERIES = [
    "#2a78d6",  # azul
    "#eb6834",  # laranja
    "#1baf7a",  # água
    "#eda100",  # amarelo
    "#e87ba4",  # magenta
    "#008300",  # verde
    "#4a3aa7",  # violeta
]

SUPERFICIE = "#fcfcfb"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_SUAVE = "#8a8984"
GRADE = "#e6e5e1"

# Rampa sequencial de um único tom (claro -> escuro), para magnitude.
RAMPA_SEQUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                    "#256abf", "#184f95", "#0d366b"]


def estilo_padrao() -> None:
    """Aplica o padrão visual: marcas finas, grade discreta, rótulos em português."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
        "font.size": 10,
        "text.color": TINTA_PRIMARIA,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#c9c8c3",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",          # grade tracejada vira ruído visual
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        "errorbar.capsize": 3,
        # Sinal de menos ASCII, igual ao de formatar_br, para que os
        # rótulos de dados e os do eixo não usem glifos diferentes.
        "axes.unicode_minus": False,
    })
    plt.close("all")


def mapa_sequencial():
    """Colormap de um único tom para matrizes de confusão e mapas de calor."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("azul_sequencial", RAMPA_SEQUENCIAL)


def salvar_figura(figura, nome: str) -> Path:
    """Salva em ``outputs/figuras`` a 150 dpi e devolve o caminho."""
    caminho = DIR_FIGURAS / f"{nome}.png"
    figura.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=SUPERFICIE)
    return caminho


def salvar_tabela(dados: pd.DataFrame, nome: str) -> Path:
    """Salva em ``outputs/tabelas`` e devolve o caminho."""
    caminho = DIR_TABELAS / f"{nome}.csv"
    dados.to_csv(caminho, index=False, encoding="utf-8")
    return caminho


# ---------------------------------------------------------------------------
# Verificação independente das médias móveis (teste obrigatório)
# ---------------------------------------------------------------------------

def verificar_media_movel(
    longo: pd.DataFrame,
    metrica: str = "gols_feitos",
    janela: int = 5,
    n_amostras: int = 200,
    semente: int = SEMENTE,
) -> pd.DataFrame:
    """
    Recalcula "à mão" um atributo móvel para uma amostra aleatória de partidas,
    usando apenas os jogos anteriores do time, e compara com o valor da base.

    É o teste de vazamento: se a partida atual entrasse no próprio cálculo, os
    dois valores divergiriam.
    """
    coluna = f"{metrica}_m{janela}"
    ordenado = longo.sort_values(
        ["time", "data", "horario", "partida_id"], kind="mergesort").reset_index(drop=True)
    ordenado["posicao"] = ordenado.groupby("time", sort=False).cumcount()

    gerador = np.random.default_rng(semente)
    candidatas = ordenado.index.to_numpy()
    escolhidas = gerador.choice(candidatas, size=min(n_amostras, len(candidatas)),
                                replace=False)

    linhas = []
    valores_por_time = {t: g for t, g in ordenado.groupby("time", sort=False)}
    for indice in escolhidas:
        linha = ordenado.loc[indice]
        historico = valores_por_time[linha["time"]]
        anteriores = historico[historico["posicao"] < linha["posicao"]]
        janela_valores = anteriores[metrica].to_numpy()[-janela:]
        if len(janela_valores) == 0 or np.all(np.isnan(janela_valores)):
            esperado = np.nan
        else:
            esperado = float(np.nanmean(janela_valores))
        obtido = linha[coluna]
        iguais = (np.isnan(esperado) and pd.isna(obtido)) or (
            not np.isnan(esperado) and not pd.isna(obtido)
            and abs(esperado - obtido) < 1e-9)
        linhas.append({
            "partida_id": int(linha["partida_id"]), "time": linha["time"],
            "data": linha["data"], "jogos_anteriores": int(linha["posicao"]),
            "recalculado": esperado, "na_base": obtido, "confere": bool(iguais)})
    return pd.DataFrame(linhas)


def log_loss_ordenado(y_verdadeiro, probabilidades, classes=None,
                      epsilon: float = 1e-15) -> float:
    """
    Log loss com a ordem das classes EXPLÍCITA (H, D, A).

    Atenção — é por isso que esta função existe em vez de
    ``sklearn.metrics.log_loss``: o scikit-learn ordena o argumento ``labels``
    alfabeticamente e pressupõe que as colunas de ``y_pred`` sigam essa ordem
    alfabética (A, D, H). Como todo o notebook trabalha na ordem fixa H, D, A,
    chamar a função do scikit-learn trocaria silenciosamente as colunas de
    vitória do mandante e do visitante, produzindo um log loss errado.
    """
    if classes is None:
        classes = CLASSES
    probabilidades = np.asarray(probabilidades, dtype=float)
    probabilidades = np.clip(probabilidades, epsilon, 1 - epsilon)
    probabilidades = probabilidades / probabilidades.sum(axis=1, keepdims=True)
    indice = {classe: i for i, classe in enumerate(classes)}
    linhas = np.arange(len(y_verdadeiro))
    colunas = np.array([indice[valor] for valor in y_verdadeiro])
    return float(-np.mean(np.log(probabilidades[linhas, colunas])))
