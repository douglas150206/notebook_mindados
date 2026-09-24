"""
Gera a planilha de palpites do Brasileirão Série A 2026.

Saída: outputs/brasileirao/palpites_brasileirao_2026.xlsx

Executar:  python gerar_planilha_brasileirao.py

A planilha reúne, para cada uma das partidas que faltam, os mercados de
resultado (1X2 e dupla chance), gols (mais/menos e multi-gols), ambas as
equipes marcam, escanteios e cartões — além de uma calculadora de valor
esperado com fórmulas vivas.

Os modelos e a validação estão em src/utils_brasileirao.py e no notebook
brasileirao_2026.ipynb.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

import utils  # noqa: E402
import utils_brasileirao as ub  # noqa: E402

ARQUIVO_SAIDA = ub.DIR_SAIDAS / "palpites_brasileirao_2026.xlsx"

# Paleta validada (a mesma dos gráficos do projeto).
AZUL = "2A78D6"
AZUL_CLARO = "CDE2FB"
AZUL_ESCURO = "184F95"
LARANJA = "EB6834"
VERDE = "1BAF7A"
VERMELHO = "E34948"
AMARELO = "EDA100"
CINZA_TEXTO = "52514E"
CINZA_CLARO = "F0EFEC"
BRANCO = "FFFFFF"

FONTE = "Arial"


# ---------------------------------------------------------------------------
# 1. Modelos e previsões
# ---------------------------------------------------------------------------

def montar_previsoes():
    """Ajusta todos os modelos e devolve a tabela de previsões das partidas."""
    print("Carregando base de gols e cotações...")
    ub.baixar_base()
    base = ub.carregar_base()
    disputadas = base[base["jogada"]]
    restantes, times_2026, n_disputadas = ub.partidas_restantes(base, ub.TEMPORADA_ALVO)

    print(f"  {n_disputadas} partidas disputadas, {len(restantes)} restantes")

    print("Ajustando modelo de gols (Poisson/Dixon-Coles)...")
    modelo_gols = ub.ajustar_modelo_gols(disputadas, disputadas["data"].max(),
                                         meia_vida_dias=730)

    print("Carregando estatísticas de escanteios e cartões...")
    ub.baixar_estatisticas()
    estatisticas = ub.carregar_estatisticas()
    print(f"  {len(estatisticas)} partidas com estatística "
          f"({estatisticas['ano'].min()}-{estatisticas['ano'].max()})")

    print("Ajustando modelos de escanteios e cartões...")
    modelos_contagem = ub.ajustar_modelos_estatisticas(estatisticas)
    modelo_escanteios = modelos_contagem["escanteios"]
    modelo_cartoes = modelos_contagem["cartoes"]

    sem_historico = sorted(set(modelo_escanteios.sem_historico(times_2026))
                           | set(modelo_cartoes.sem_historico(times_2026)))
    print(f"  Times sem histórico de escanteios/cartões: {sem_historico or 'nenhum'}")

    print("Carregando o calendário oficial (rodada e data)...")
    calendario = ub.carregar_calendario(ub.TEMPORADA_ALVO)
    futuras = calendario[~calendario["jogada"]]
    restantes = restantes.merge(futuras[["rodada", "data", "Home", "Away"]],
                                on=["Home", "Away"], how="left")
    faltando_rodada = int(restantes["rodada"].isna().sum())
    if faltando_rodada:
        print(f"  AVISO: {faltando_rodada} partidas sem rodada no calendário")
    restantes = restantes.sort_values(["rodada", "data", "Home"]).reset_index(drop=True)
    print(f"  {restantes['rodada'].nunique()} rodadas restantes: "
          f"{sorted(restantes['rodada'].dropna().unique().tolist())}")

    print("Calculando previsões das partidas restantes...")
    linhas = []
    for numero, (mandante, visitante) in enumerate(
            zip(restantes["Home"], restantes["Away"]), start=1):
        lambda_casa, lambda_fora = modelo_gols.lambdas(mandante, visitante)
        matriz = ub.matriz_placares(lambda_casa, lambda_fora, modelo_gols.rho,
                                    modelo_gols.max_gols)
        registro_calendario = restantes.iloc[numero - 1]
        rodada = registro_calendario["rodada"]
        data = registro_calendario["data"]
        linha = {"n": numero,
                 "Rodada": int(rodada) if pd.notna(rodada) else None,
                 "Data": data.date() if pd.notna(data) else None,
                 "Mandante": mandante, "Visitante": visitante,
                 **ub.previsoes_do_placar(matriz),
                 **ub.previsoes_multigols(matriz)}

        esc_casa, esc_fora = modelo_escanteios.lambdas(mandante, visitante)
        linha["escanteios_mandante"] = esc_casa
        linha["escanteios_visitante"] = esc_fora
        linha["escanteios_total"] = esc_casa + esc_fora
        for valor in ub.LINHAS_ESCANTEIOS:
            chave = f"escanteios_mais_{valor}".replace(".", "_")
            linha[chave] = modelo_escanteios.prob_acima(mandante, visitante, valor)

        car_casa, car_fora = modelo_cartoes.lambdas(mandante, visitante)
        linha["cartoes_mandante"] = car_casa
        linha["cartoes_visitante"] = car_fora
        linha["cartoes_total"] = car_casa + car_fora
        for valor in ub.LINHAS_CARTOES:
            chave = f"cartoes_mais_{valor}".replace(".", "_")
            linha[chave] = modelo_cartoes.prob_acima(mandante, visitante, valor)

        linha["sem_historico_stats"] = (
            "sim" if (mandante in sem_historico or visitante in sem_historico) else "não")
        # Partida sem resultado cuja data já passou é jogo adiado: a data do
        # calendário não vale mais, só a rodada.
        linha["situacao"] = ("adiado" if (pd.notna(data)
                                          and data < disputadas["data"].max())
                             else "programado")
        linhas.append(linha)

    previsoes = pd.DataFrame(linhas)

    # Palpite de 1X2 e o palpite mais seguro entre os mercados principais.
    rotulos_1x2 = {"prob_H": "Casa", "prob_D": "Empate", "prob_A": "Fora"}
    previsoes["palpite_1x2"] = [
        rotulos_1x2[c] for c in previsoes[["prob_H", "prob_D", "prob_A"]].idxmax(axis=1)]

    candidatos = {
        "Casa ou empate (1X)": previsoes["prob_H"] + previsoes["prob_D"],
        "Casa ou fora (12)": previsoes["prob_H"] + previsoes["prob_A"],
        "Empate ou fora (X2)": previsoes["prob_D"] + previsoes["prob_A"],
        "Mais de 1,5 gols": previsoes["mais_de_1_5"],
        "Menos de 3,5 gols": 1 - previsoes["mais_de_3_5"],
        "Multi-gols 1-4": previsoes["multigols_1_4"],
    }
    tabela_candidatos = pd.DataFrame(candidatos)
    previsoes["palpite_seguro"] = tabela_candidatos.idxmax(axis=1)
    previsoes["palpite_seguro_prob"] = tabela_candidatos.max(axis=1)

    contexto = {
        "base": base, "restantes": restantes, "times_2026": times_2026,
        "n_disputadas": n_disputadas, "modelo_gols": modelo_gols,
        "modelo_escanteios": modelo_escanteios, "modelo_cartoes": modelo_cartoes,
        "estatisticas": estatisticas, "sem_historico": sem_historico,
        "data_corte": disputadas["data"].max(),
        "calendario": calendario,
        "classificacao": ub.classificacao(base, ub.TEMPORADA_ALVO),
    }
    return previsoes, contexto


# ---------------------------------------------------------------------------
# 2. Confiabilidade: viés medido fora da amostra
# ---------------------------------------------------------------------------

def medir_confiabilidade(contexto) -> pd.DataFrame:
    """
    Mede, fora da amostra, o quanto cada mercado erra.

    Para os mercados de gols, treina até a temporada anterior e prevê a
    seguinte, de 2016 a 2026. Para escanteios e cartões, o mesmo de 2019 a 2023
    (limite da base de estatísticas). O viés é previsto menos observado: valor
    negativo significa que o modelo SUBESTIMA aquele mercado.
    """
    base = contexto["base"]
    linhas = []

    print("Medindo confiabilidade dos mercados de gols (walk-forward)...")
    partes = []
    for temporada in range(2016, ub.TEMPORADA_ALVO + 1):
        treino = base[(base["Season"] < temporada) & base["jogada"]]
        alvo = base[(base["Season"] == temporada) & base["jogada"]]
        if alvo.empty:
            continue
        modelo = ub.ajustar_modelo_gols(treino, treino["data"].max(), 730)
        previsto = modelo.prever(alvo[["Home", "Away"]])
        matrizes = [ub.matriz_placares(*modelo.lambdas(m, v), modelo.rho,
                                       modelo.max_gols)
                    for m, v in zip(alvo["Home"], alvo["Away"])]
        extras = pd.DataFrame([ub.previsoes_multigols(mz) for mz in matrizes])
        previsto = pd.concat([previsto.reset_index(drop=True), extras], axis=1)
        previsto["Res"] = alvo["Res"].to_numpy()
        previsto["total_gols"] = (alvo["HG"] + alvo["AG"]).to_numpy()
        previsto["ambas_real"] = ((alvo["HG"] > 0) & (alvo["AG"] > 0)).astype(int).to_numpy()
        partes.append(previsto)
    walk = pd.concat(partes, ignore_index=True)

    def registrar(mercado, previsto, observado, n, confianca, obs=""):
        linhas.append({"Mercado": mercado, "Previsto médio": float(previsto),
                       "Observado": float(observado),
                       "Viés (p.p.)": 100 * (float(previsto) - float(observado)),
                       "Partidas testadas": int(n), "Confiança": confianca,
                       "Observação": obs})

    acertos = (np.array(utils.CLASSES)[
        walk[["prob_H", "prob_D", "prob_A"]].to_numpy().argmax(axis=1)]
        == walk["Res"].to_numpy())
    registrar("Acerto do 1X2 (acurácia)", acertos.mean(), acertos.mean(), len(walk),
              "Alta", "acurácia do palpite de resultado; o mercado acerta ~51%")
    registrar("Vitória do mandante", walk["prob_H"].mean(),
              (walk["Res"] == "H").mean(), len(walk), "Alta")
    registrar("Empate", walk["prob_D"].mean(), (walk["Res"] == "D").mean(),
              len(walk), "Alta")
    registrar("Vitória do visitante", walk["prob_A"].mean(),
              (walk["Res"] == "A").mean(), len(walk), "Alta")
    for valor in ub.LINHAS_GOLS:
        chave = f"mais_de_{valor}".replace(".", "_")
        registrar(f"Mais de {valor:.1f} gols".replace(".", ","),
                  walk[chave].mean(), (walk["total_gols"] > valor).mean(),
                  len(walk), "Alta")
    for minimo, maximo in ub.FAIXAS_MULTIGOLS:
        registrar(f"Multi-gols {minimo}-{maximo}",
                  walk[f"multigols_{minimo}_{maximo}"].mean(),
                  ((walk["total_gols"] >= minimo) & (walk["total_gols"] <= maximo)).mean(),
                  len(walk), "Alta")
    registrar("Ambas as equipes marcam", walk["prob_ambas_marcam"].mean(),
              walk["ambas_real"].mean(), len(walk), "Média",
              "modelo supõe gols independentes e subestima este mercado")

    print("Medindo confiabilidade de escanteios e cartões...")
    estatisticas = contexto["estatisticas"]
    acumulado = {}
    n_testadas = 0
    for ano in range(2019, ub.ANO_FIM_ESTATISTICAS + 1):
        treino = estatisticas[estatisticas["ano"] < ano]
        alvo = estatisticas[estatisticas["ano"] == ano]
        if alvo.empty:
            continue
        n_testadas += len(alvo)
        modelos = ub.ajustar_modelos_estatisticas(treino)
        total_esc = (alvo["escanteios_mandante"] + alvo["escanteios_visitante"]).to_numpy()
        total_car = (alvo["cartoes_mandante"] + alvo["cartoes_visitante"]).to_numpy()
        for rotulo, modelo, valores, linhas_mercado in (
                ("escanteios", modelos["escanteios"], total_esc, ub.LINHAS_ESCANTEIOS),
                ("cartoes", modelos["cartoes"], total_car, ub.LINHAS_CARTOES)):
            for valor in linhas_mercado:
                previsto = np.mean([modelo.prob_acima(m, v, valor)
                                    for m, v in zip(alvo["Home"], alvo["Away"])])
                observado = float((valores > valor).mean())
                acumulado.setdefault((rotulo, valor), []).append((previsto, observado))

    for (rotulo, valor), pares in acumulado.items():
        previstos = np.mean([p for p, _ in pares])
        observados = np.mean([o for _, o in pares])
        desvio = np.std([100 * (p - o) for p, o in pares], ddof=1)
        nome = "escanteios" if rotulo == "escanteios" else "cartões"
        registrar(f"Mais de {valor:.1f} {nome}".replace(".", ","),
                  previstos, observados, n_testadas, "Média",
                  f"base vai até {ub.ANO_FIM_ESTATISTICAS}; oscilação de "
                  f"±{desvio:.1f} p.p. entre temporadas")

    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 3. Sistema visual da planilha
# ---------------------------------------------------------------------------
#
# Regras de desenho, aplicadas em todas as abas:
#   * nada de linhas de grade nem de bordas em toda célula — o que separa as
#     linhas é uma faixa alternada bem clara, e o que separa os blocos é a cor
#     do cabeçalho de grupo;
#   * cada família de mercado tem a sua cor no cabeçalho, para a vista achar a
#     região certa numa tabela larga sem ler os rótulos;
#   * probabilidade recebe escala de cor (quanto mais provável, mais azul); a
#     odd justa ao lado fica sem preenchimento, em cinza, porque é informação
#     derivada e não deve competir com a probabilidade;
#   * linhas altas e colunas largas: a planilha é para ser lida, não para caber.

from openpyxl import Workbook  # noqa: E402
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule, DataBarRule  # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.workbook.defined_name import DefinedName  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

FONTE = "Arial"

# Tintas
TINTA_TITULO = "0F2942"
TINTA_PRIMARIA = "1F2937"
TINTA_SECUNDARIA = "6B7280"
TINTA_CLARA = "9CA3AF"
VERDE = "12805C"
VERMELHO = "B3261E"
AMBAR = "B45309"

# Superfícies
BRANCO = "FFFFFF"
BANDA = "F5F8FC"
CARTAO = "EEF4FC"
DESTAQUE_SUAVE = "FFF8E6"

# Cores de grupo (cabeçalho de cada família de mercado)
COR_JOGO = "42566E"
COR_RESULTADO = "1E4E8C"
COR_DUPLA = "4A3AA7"
COR_GOLS = "12805C"
COR_AMBAS = "C2501F"
COR_ESCANTEIOS = "0E7490"
COR_CARTOES = "B45309"
COR_DESTAQUE = "7A1F6B"

# Escala das probabilidades: branco -> azul claro (texto preto continua legível)
PROB_MIN = "FFFFFF"
PROB_MAX = "9EC5F4"

PCT = "0.0%"
ODD = "0.00"
NUM = "0.00"
DATA_BR = "DD/MM/YYYY"

F_TITULO = Font(name=FONTE, size=18, bold=True, color=TINTA_TITULO)
F_SUBTITULO = Font(name=FONTE, size=10, color=TINTA_SECUNDARIA)
F_SECAO = Font(name=FONTE, size=12, bold=True, color=TINTA_TITULO)
F_GRUPO = Font(name=FONTE, size=10, bold=True, color=BRANCO)
F_COLUNA = Font(name=FONTE, size=9, bold=True, color=TINTA_PRIMARIA)
F_CORPO = Font(name=FONTE, size=10, color=TINTA_PRIMARIA)
F_FORTE = Font(name=FONTE, size=10, bold=True, color=TINTA_PRIMARIA)
F_ODD = Font(name=FONTE, size=10, color=TINTA_SECUNDARIA)
F_NOTA = Font(name=FONTE, size=9, color=TINTA_SECUNDARIA)

CENTRO = Alignment(horizontal="center", vertical="center")
CENTRO_QUEBRA = Alignment(horizontal="center", vertical="center", wrap_text=True)
ESQUERDA = Alignment(horizontal="left", vertical="center")
DIREITA = Alignment(horizontal="right", vertical="center")

REGUA_COLUNA = Border(bottom=Side(style="thin", color="D5DEE9"))


def escurecer(hex_cor, fator=0.78):
    """Tom mais escuro da mesma cor, para alternar grupos vizinhos."""
    r, g, b = (int(hex_cor[i:i + 2], 16) for i in (0, 2, 4))
    return "".join(f"{int(c * fator):02X}" for c in (r, g, b))


def preparar_aba(livro, titulo, cor_aba):
    planilha = livro.create_sheet(titulo)
    planilha.sheet_view.showGridLines = False
    planilha.sheet_properties.tabColor = cor_aba
    return planilha


def bloco_titulo(planilha, titulo, subtitulo, ultima_coluna):
    planilha["A1"] = titulo
    planilha["A1"].font = F_TITULO
    planilha["A1"].alignment = ESQUERDA
    planilha.row_dimensions[1].height = 30
    planilha["A2"] = subtitulo
    planilha["A2"].font = F_SUBTITULO
    planilha["A2"].alignment = ESQUERDA
    planilha.merge_cells(start_row=2, start_column=1, end_row=2,
                         end_column=max(2, ultima_coluna))
    planilha.row_dimensions[2].height = 16


def cabecalho_grupos(planilha, linha, grupos):
    """grupos = [(rótulo, primeira_coluna, n_colunas, cor)]"""
    for rotulo, primeira, quantas, cor in grupos:
        if quantas > 1:
            planilha.merge_cells(start_row=linha, start_column=primeira,
                                 end_row=linha, end_column=primeira + quantas - 1)
        for deslocamento in range(quantas):
            celula = planilha.cell(row=linha, column=primeira + deslocamento)
            celula.fill = PatternFill("solid", fgColor=cor)
        celula = planilha.cell(row=linha, column=primeira, value=rotulo)
        celula.font = F_GRUPO
        celula.alignment = CENTRO
    planilha.row_dimensions[linha].height = 20


def cabecalho_colunas(planilha, linha, rotulos):
    for deslocamento, rotulo in enumerate(rotulos, start=1):
        celula = planilha.cell(row=linha, column=deslocamento, value=rotulo)
        celula.font = F_COLUNA
        celula.alignment = CENTRO_QUEBRA
        celula.border = REGUA_COLUNA
    planilha.row_dimensions[linha].height = 28


def larguras(planilha, valores):
    for indice, largura in enumerate(valores, start=1):
        planilha.column_dimensions[get_column_letter(indice)].width = largura


def bandas(planilha, primeira, ultima, n_colunas, altura=19):
    """Faixa alternada bem clara — substitui as bordas em toda célula."""
    for linha in range(primeira, ultima + 1):
        planilha.row_dimensions[linha].height = altura
        if (linha - primeira) % 2 == 1:
            for coluna in range(1, n_colunas + 1):
                planilha.cell(row=linha, column=coluna).fill = PatternFill(
                    "solid", fgColor=BANDA)


def escala_probabilidade(planilha, primeira, ultima, colunas):
    for coluna in colunas:
        letra = get_column_letter(coluna)
        for linha in range(primeira, ultima + 1):
            celula = planilha.cell(row=linha, column=coluna)
            celula.number_format = PCT
            celula.font = F_CORPO
            celula.alignment = CENTRO
        planilha.conditional_formatting.add(
            f"{letra}{primeira}:{letra}{ultima}",
            ColorScaleRule(start_type="num", start_value=0, start_color=PROB_MIN,
                           end_type="num", end_value=1, end_color=PROB_MAX))


def coluna_odd(planilha, primeira, ultima, colunas):
    """Odd justa: discreta de propósito, para não competir com a probabilidade."""
    for coluna in colunas:
        for linha in range(primeira, ultima + 1):
            celula = planilha.cell(row=linha, column=coluna)
            celula.number_format = ODD
            celula.font = F_ODD
            celula.alignment = CENTRO


def coluna_numero(planilha, primeira, ultima, colunas, formato=NUM):
    for coluna in colunas:
        for linha in range(primeira, ultima + 1):
            celula = planilha.cell(row=linha, column=coluna)
            celula.number_format = formato
            celula.font = F_CORPO
            celula.alignment = CENTRO


def nota_rodape(planilha, linha, texto, ultima_coluna):
    celula = planilha.cell(row=linha, column=1, value=texto)
    celula.font = F_NOTA
    celula.alignment = ESQUERDA
    planilha.merge_cells(start_row=linha, start_column=1, end_row=linha,
                         end_column=max(2, ultima_coluna))


# ---------------------------------------------------------------------------
# Aba de entrada
# ---------------------------------------------------------------------------

def aba_inicio(livro, previsoes, contexto, confiabilidade):
    planilha = preparar_aba(livro, "Início", COR_RESULTADO)
    larguras(planilha, [2, 24, 24, 24, 24, 24, 24, 4])

    planilha["B2"] = "BRASILEIRÃO SÉRIE A 2026"
    planilha["B2"].font = Font(name=FONTE, size=24, bold=True, color=TINTA_TITULO)
    planilha.row_dimensions[2].height = 34
    planilha["B3"] = "Previsões de gols, resultados, escanteios e cartões — com a odd justa de cada mercado"
    planilha["B3"].font = Font(name=FONTE, size=11, color=TINTA_SECUNDARIA)
    planilha.merge_cells("B3:G3")

    # --- cartões com os números-chave ---------------------------------------
    rodadas = sorted(previsoes["Rodada"].dropna().unique())
    cartoes = [
        ("PARTIDAS RESTANTES", f"{len(previsoes)}", "de 380 no campeonato"),
        ("RODADAS RESTANTES", f"{len(rodadas)}",
         f"da {int(min(rodadas))} à {int(max(rodadas))}"),
        ("LÍDER", contexto["classificacao"].iloc[0]["time"],
         f"{int(contexto['classificacao'].iloc[0]['pontos'])} pontos"),
        ("DADOS ATÉ", contexto["data_corte"].strftime("%d/%m/%Y"),
         f"{contexto['n_disputadas']} partidas disputadas"),
    ]
    linha = 5
    for indice, (rotulo, valor, apoio) in enumerate(cartoes):
        coluna = 2 + indice
        planilha.cell(row=linha, column=coluna, value=rotulo).font = Font(
            name=FONTE, size=8, bold=True, color=TINTA_SECUNDARIA)
        planilha.cell(row=linha + 1, column=coluna, value=valor).font = Font(
            name=FONTE, size=16, bold=True, color=COR_RESULTADO)
        planilha.cell(row=linha + 2, column=coluna, value=apoio).font = Font(
            name=FONTE, size=9, color=TINTA_CLARA)
        for deslocamento in range(3):
            celula = planilha.cell(row=linha + deslocamento, column=coluna)
            celula.fill = PatternFill("solid", fgColor=CARTAO)
            celula.alignment = ESQUERDA
    planilha.row_dimensions[linha].height = 14
    planilha.row_dimensions[linha + 1].height = 24
    planilha.row_dimensions[linha + 2].height = 16

    # --- por onde começar ---------------------------------------------------
    linha = 10
    planilha.cell(row=linha, column=2, value="POR ONDE COMEÇAR").font = F_SECAO
    linha += 1
    abas = [
        ("Palpites", "Todas as 103 partidas numa tela, com probabilidade e odd justa.", COR_RESULTADO),
        ("Rodadas", "As mesmas previsões no calendário oficial: resumo por rodada e jogo a jogo.", COR_JOGO),
        ("Melhores Palpites", "Os palpites mais prováveis de cada rodada, um por partida.", COR_DESTAQUE),
        ("Bilhetes", "Múltiplas pré-montadas, do mais seguro ao de odd mais alta — com o retorno esperado de cada uma.", COR_AMBAS),
        ("Calculadora", "Escolha a partida e o mercado, digite a cotação da casa e veja se compensa.", COR_DESTAQUE),
        ("Resultado 1X2", "Vitória, empate, derrota e dupla chance, com as odds justas.", COR_RESULTADO),
        ("Gols", "Mais/menos de 0,5 a 4,5 gols e o placar mais provável.", COR_GOLS),
        ("Multi-Gols", "Chance de o total de gols cair em cada faixa.", COR_GOLS),
        ("Ambas Marcam", "Ambas marcam sim/não e a chance de cada equipe marcar.", COR_AMBAS),
        ("Escanteios", "Escanteios esperados e linhas de mais/menos.", COR_ESCANTEIOS),
        ("Cartões", "Cartões esperados e linhas de mais/menos.", COR_CARTOES),
        ("Confiabilidade", "Quanto cada mercado errou em teste fora da amostra.", COR_JOGO),
        ("Times", "Força de ataque e defesa de cada clube.", COR_JOGO),
    ]
    for nome, descricao, cor in abas:
        celula = planilha.cell(row=linha, column=2, value=nome)
        celula.font = Font(name=FONTE, size=10, bold=True, color=BRANCO)
        celula.fill = PatternFill("solid", fgColor=cor)
        celula.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        descricao_celula = planilha.cell(row=linha, column=3, value=descricao)
        descricao_celula.font = F_CORPO
        descricao_celula.alignment = ESQUERDA
        planilha.merge_cells(start_row=linha, start_column=3, end_row=linha, end_column=7)
        planilha.row_dimensions[linha].height = 18
        linha += 1

    # --- como ler ------------------------------------------------------------
    linha += 1
    planilha.cell(row=linha, column=2, value="COMO LER A ODD JUSTA").font = F_SECAO
    linha += 1
    for texto in [
        "A odd justa é 1 dividido pela probabilidade: o preço de equilíbrio daquele palpite.",
        "Casa paga ACIMA da odd justa  →  a aposta tem valor esperado positivo.",
        "Casa paga ABAIXO da odd justa  →  valor esperado negativo, mesmo que o palpite acerte muito.",
        "A cotação não muda a chance de acertar. Muda só o quanto se recebe por acertar.",
    ]:
        celula = planilha.cell(row=linha, column=2, value="•   " + texto)
        celula.font = F_CORPO
        planilha.merge_cells(start_row=linha, start_column=2, end_row=linha, end_column=7)
        planilha.row_dimensions[linha].height = 16
        linha += 1

    # --- confiança -----------------------------------------------------------
    linha += 1
    planilha.cell(row=linha, column=2, value="CONFIANÇA DE CADA FAMÍLIA").font = F_SECAO
    linha += 1
    niveis = [
        ("ALTA", VERDE, "Resultado, gols e multi-gols",
         "Poisson com Dixon-Coles, ajustado com todas as partidas até a rodada mais "
         "recente de 2026 e testado em 4.076 partidas fora da amostra."),
        ("MÉDIA", AMBAR, "Escanteios e cartões",
         "Base própria de estatísticas cobrindo 2015 a 2026, com os 20 clubes. Viés "
         "fora da amostra de 1 a 2 p.p., mas oscila cerca de ±5 p.p. entre temporadas."),
        ("MÉDIA", AMBAR, "Ambas as equipes marcam",
         "O modelo supõe gols independentes e subestima este mercado em cerca de 4 "
         "pontos percentuais. Some o viés antes de usar."),
    ]
    for nivel, cor, familia, explicacao in niveis:
        celula = planilha.cell(row=linha, column=2, value=nivel)
        celula.font = Font(name=FONTE, size=10, bold=True, color=BRANCO)
        celula.fill = PatternFill("solid", fgColor=cor)
        celula.alignment = CENTRO
        planilha.cell(row=linha, column=3, value=familia).font = F_FORTE
        explicacao_celula = planilha.cell(row=linha, column=4, value=explicacao)
        explicacao_celula.font = F_NOTA
        explicacao_celula.alignment = Alignment(wrap_text=True, vertical="center")
        planilha.merge_cells(start_row=linha, start_column=4, end_row=linha, end_column=7)
        planilha.row_dimensions[linha].height = 22
        linha += 1

    # --- aviso ---------------------------------------------------------------
    linha += 1
    for indice, texto in enumerate([
        "AVISO",
        "Isto é um trabalho de análise de dados, não um sistema de apostas. A própria "
        "análise deste projeto mostrou que o retorno esperado é negativo em todas as "
        "faixas de cotação do Brasileirão, e que apostar só onde o modelo enxerga "
        "vantagem piora o resultado. O mercado acerta mais que este modelo: 51,3% "
        "contra 48,9% no 1X2.",
    ]):
        celula = planilha.cell(row=linha, column=2, value=texto)
        celula.font = (Font(name=FONTE, size=10, bold=True, color=AMBAR) if indice == 0
                       else Font(name=FONTE, size=9, color=TINTA_PRIMARIA))
        celula.alignment = Alignment(wrap_text=True, vertical="top")
        planilha.merge_cells(start_row=linha, start_column=2, end_row=linha, end_column=7)
        for coluna in range(2, 8):
            planilha.cell(row=linha, column=coluna).fill = PatternFill(
                "solid", fgColor=DESTAQUE_SUAVE)
        planilha.row_dimensions[linha].height = 14 if indice == 0 else 46
        linha += 1

    linha += 1
    nota_rodape(planilha, linha,
                "Gols e cotações: Football-Data.co.uk. Escanteios, cartões e calendário: "
                "datalake leeofernandes1980/brasileirao-dataset (2015-2026). Os 277 "
                "resultados já disputados de 2026 batem entre as duas bases.", 7)
    planilha.sheet_view.zoomScale = 100
    return planilha


# ---------------------------------------------------------------------------
# Aba principal
# ---------------------------------------------------------------------------

def aba_palpites(livro, previsoes):
    """
    Uma linha por partida. Cada mercado aparece em par: a probabilidade
    (colorida) e, ao lado, a odd justa (discreta), calculada por fórmula.
    """
    planilha = preparar_aba(livro, "Palpites", COR_RESULTADO)
    bloco_titulo(planilha, "PALPITES — TODAS AS PARTIDAS",
                 f"{len(previsoes)} jogos restantes, na ordem do calendário. "
                 "Use os filtros do cabeçalho. Ao lado de cada probabilidade está "
                 "a odd justa: se a casa paga mais que ela, a aposta tem valor.", 22)

    grupos = [
        ("JOGO", 1, 4, COR_JOGO),
        ("RESULTADO", 5, 9, COR_RESULTADO),
        ("GOLS", 14, 3, COR_GOLS),
        ("AMBAS", 17, 2, COR_AMBAS),
        ("ESC.", 19, 1, COR_ESCANTEIOS),
        ("CART.", 20, 1, COR_CARTOES),
        ("PALPITE MAIS SEGURO", 21, 3, COR_DESTAQUE),
    ]
    linha_grupo = 4
    cabecalho_grupos(planilha, linha_grupo, grupos)
    rotulos = ["Rodada", "Data", "Mandante", "Visitante",
               "Palpite", "Placar", "Placar %",
               "Casa", "Odd", "Empate", "Odd", "Fora", "Odd",
               "Gols esp.", "+2,5", "Odd",
               "Sim", "Odd",
               "Escanteios", "Cartões",
               "Aposta", "Chance", "Odd"]
    cabecalho_colunas(planilha, linha_grupo + 1, rotulos)

    primeira = linha_grupo + 2
    ordenado = previsoes.sort_values(["Rodada", "Data", "Mandante"])
    for deslocamento, registro in enumerate(ordenado.itertuples(index=False)):
        linha = primeira + deslocamento
        planilha.cell(row=linha, column=1, value=registro.Rodada)
        planilha.cell(row=linha, column=2, value=registro.Data)
        planilha.cell(row=linha, column=3, value=registro.Mandante)
        planilha.cell(row=linha, column=4, value=registro.Visitante)
        planilha.cell(row=linha, column=5, value=registro.palpite_1x2)
        planilha.cell(row=linha, column=6, value=registro.placar_mais_provavel)
        planilha.cell(row=linha, column=7, value=registro.prob_placar_mais_provavel)
        planilha.cell(row=linha, column=8, value=registro.prob_H)
        planilha.cell(row=linha, column=9, value=f'=IFERROR(1/H{linha},"")')
        planilha.cell(row=linha, column=10, value=registro.prob_D)
        planilha.cell(row=linha, column=11, value=f'=IFERROR(1/J{linha},"")')
        planilha.cell(row=linha, column=12, value=registro.prob_A)
        planilha.cell(row=linha, column=13, value=f'=IFERROR(1/L{linha},"")')
        planilha.cell(row=linha, column=14, value=registro.gols_esperados_total)
        planilha.cell(row=linha, column=15, value=registro.mais_de_2_5)
        planilha.cell(row=linha, column=16, value=f'=IFERROR(1/O{linha},"")')
        planilha.cell(row=linha, column=17, value=registro.prob_ambas_marcam)
        planilha.cell(row=linha, column=18, value=f'=IFERROR(1/Q{linha},"")')
        planilha.cell(row=linha, column=19, value=registro.escanteios_total)
        planilha.cell(row=linha, column=20, value=registro.cartoes_total)
        planilha.cell(row=linha, column=21, value=registro.palpite_seguro)
        planilha.cell(row=linha, column=22, value=registro.palpite_seguro_prob)
        planilha.cell(row=linha, column=23, value=f'=IFERROR(1/V{linha},"")')

    ultima = primeira + len(ordenado) - 1
    bandas(planilha, primeira, ultima, len(rotulos))
    # A coluna "Chance" (22) também é probabilidade: entra na formatação para
    # não sair como 0,786 em vez de 78,6%.
    escala_probabilidade(planilha, primeira, ultima, [7, 8, 10, 12, 15, 17, 22])
    coluna_odd(planilha, primeira, ultima, [9, 11, 13, 16, 18, 23])
    coluna_numero(planilha, primeira, ultima, [14, 19, 20])

    for linha in range(primeira, ultima + 1):
        planilha.cell(row=linha, column=1).font = F_CORPO
        planilha.cell(row=linha, column=1).alignment = CENTRO
        planilha.cell(row=linha, column=2).number_format = DATA_BR
        planilha.cell(row=linha, column=2).font = F_CORPO
        planilha.cell(row=linha, column=2).alignment = CENTRO
        for coluna in (3, 4):
            planilha.cell(row=linha, column=coluna).font = F_CORPO
            planilha.cell(row=linha, column=coluna).alignment = ESQUERDA
        planilha.cell(row=linha, column=5).font = F_FORTE
        planilha.cell(row=linha, column=5).alignment = CENTRO
        planilha.cell(row=linha, column=6).font = F_CORPO
        planilha.cell(row=linha, column=6).alignment = CENTRO
        planilha.cell(row=linha, column=21).font = F_CORPO
        planilha.cell(row=linha, column=21).alignment = ESQUERDA

    # Barra de dados na chance do palpite mais seguro: leitura instantânea.
    planilha.conditional_formatting.add(
        f"V{primeira}:V{ultima}",
        DataBarRule(start_type="num", start_value=0, end_type="num", end_value=1,
                    color=COR_DESTAQUE, showValue=True))

    larguras(planilha, [8, 11, 18, 18, 9, 8, 9, 8, 7, 9, 7, 8, 7, 10, 8, 7,
                        8, 7, 11, 9, 23, 9, 7])
    planilha.freeze_panes = planilha.cell(row=primeira, column=5).coordinate
    planilha.auto_filter.ref = (f"A{linha_grupo + 1}:"
                                f"{get_column_letter(len(rotulos))}{ultima}")
    nota_rodape(planilha, ultima + 2,
                "Odd justa = 1 / probabilidade. 'Placar %' é a chance do placar exato "
                "mostrado ao lado. Escanteios e cartões são o total esperado da partida "
                "(confiança média — ver aba Início).", len(rotulos))
    return planilha


def aba_rodadas(livro, previsoes):
    """Resumo de cada rodada e, abaixo, os jogos na ordem do calendário."""
    planilha = preparar_aba(livro, "Rodadas", COR_JOGO)
    com_rodada = previsoes.dropna(subset=["Rodada"]).copy()
    com_rodada["Rodada"] = com_rodada["Rodada"].astype(int)

    bloco_titulo(planilha, "PREVISÕES RODADA A RODADA",
                 f"{len(com_rodada)} partidas em {com_rodada['Rodada'].nunique()} "
                 "rodadas. Jogos marcados como ADIADO estão na rodada original e a "
                 "data já passou: vale a rodada, não a data.", 16)

    resumo = (com_rodada.groupby("Rodada")
              .agg(jogos=("n", "size"), primeira_data=("Data", "min"),
                   ultima_data=("Data", "max"),
                   gols=("gols_esperados_total", "mean"),
                   casa=("prob_H", "mean"), empate=("prob_D", "mean"),
                   over=("mais_de_2_5", "mean"), ambas=("prob_ambas_marcam", "mean"),
                   escanteios=("escanteios_total", "mean"),
                   cartoes=("cartoes_total", "mean"))
              .reset_index())

    linha = 4
    planilha.cell(row=linha, column=1, value="RESUMO POR RODADA").font = F_SECAO
    linha += 1
    cabecalho_grupos(planilha, linha, [
        ("RODADA", 1, 4, COR_JOGO), ("MÉDIAS DA RODADA", 5, 5, COR_RESULTADO),
        ("ESC.", 10, 1, COR_ESCANTEIOS), ("CART.", 11, 1, COR_CARTOES)])
    cabecalho_colunas(planilha, linha + 1,
                      ["Rodada", "Jogos", "De", "Até", "Gols esp.", "Mandante",
                       "Empate", "+2,5", "Ambas", "Escanteios", "Cartões"])
    primeira_resumo = linha + 2
    for deslocamento, registro in enumerate(resumo.itertuples(index=False)):
        atual = primeira_resumo + deslocamento
        for coluna, valor in enumerate(
                [registro.Rodada, registro.jogos, registro.primeira_data,
                 registro.ultima_data, registro.gols, registro.casa, registro.empate,
                 registro.over, registro.ambas, registro.escanteios,
                 registro.cartoes], start=1):
            planilha.cell(row=atual, column=coluna, value=valor)
    ultima_resumo = primeira_resumo + len(resumo) - 1
    bandas(planilha, primeira_resumo, ultima_resumo, 11)
    escala_probabilidade(planilha, primeira_resumo, ultima_resumo, [6, 7, 8, 9])
    coluna_numero(planilha, primeira_resumo, ultima_resumo, [5, 10, 11])
    for atual in range(primeira_resumo, ultima_resumo + 1):
        for coluna in (1, 2):
            planilha.cell(row=atual, column=coluna).font = F_FORTE
            planilha.cell(row=atual, column=coluna).alignment = CENTRO
        for coluna in (3, 4):
            planilha.cell(row=atual, column=coluna).number_format = DATA_BR
            planilha.cell(row=atual, column=coluna).font = F_CORPO
            planilha.cell(row=atual, column=coluna).alignment = CENTRO

    total = ultima_resumo + 1
    planilha.cell(row=total, column=1, value="TODAS").font = F_FORTE
    planilha.cell(row=total, column=2,
                  value=f"=SUM(B{primeira_resumo}:B{ultima_resumo})")
    for coluna in range(5, 12):
        letra = get_column_letter(coluna)
        planilha.cell(row=total, column=coluna,
                      value=f"=AVERAGE({letra}{primeira_resumo}:{letra}{ultima_resumo})")
    for coluna in range(1, 12):
        celula = planilha.cell(row=total, column=coluna)
        celula.fill = PatternFill("solid", fgColor=CARTAO)
        celula.font = F_FORTE
        celula.alignment = CENTRO
        if coluna in (6, 7, 8, 9):
            celula.number_format = PCT
        elif coluna in (5, 10, 11):
            celula.number_format = NUM
    planilha.row_dimensions[total].height = 20

    # --- jogo a jogo ---------------------------------------------------------
    linha = total + 3
    planilha.cell(row=linha, column=1, value="JOGO A JOGO").font = F_SECAO
    linha += 1
    cabecalho_grupos(planilha, linha, [
        ("JOGO", 1, 5, COR_JOGO), ("RESULTADO", 6, 6, COR_RESULTADO),
        ("GOLS", 12, 3, COR_GOLS), ("AMBAS", 15, 1, COR_AMBAS),
        ("ESC.", 16, 1, COR_ESCANTEIOS), ("CART.", 17, 1, COR_CARTOES),
        ("PALPITE MAIS SEGURO", 18, 3, COR_DESTAQUE)])
    rotulos = ["Rodada", "Data", "Situação", "Mandante", "Visitante",
               "Palpite", "Placar", "Casa", "Empate", "Fora", "Odd fav.",
               "Gols esp.", "+2,5", "Odd", "Ambas", "Escanteios", "Cartões",
               "Aposta", "Chance", "Odd"]
    cabecalho_colunas(planilha, linha + 1, rotulos)
    primeira_detalhe = linha + 2

    atual = primeira_detalhe
    for registro in com_rodada.sort_values(["Rodada", "Data", "Mandante"]).itertuples(
            index=False):
        planilha.cell(row=atual, column=1, value=registro.Rodada)
        planilha.cell(row=atual, column=2, value=registro.Data)
        planilha.cell(row=atual, column=3, value=registro.situacao)
        planilha.cell(row=atual, column=4, value=registro.Mandante)
        planilha.cell(row=atual, column=5, value=registro.Visitante)
        planilha.cell(row=atual, column=6, value=registro.palpite_1x2)
        planilha.cell(row=atual, column=7, value=registro.placar_mais_provavel)
        planilha.cell(row=atual, column=8, value=registro.prob_H)
        planilha.cell(row=atual, column=9, value=registro.prob_D)
        planilha.cell(row=atual, column=10, value=registro.prob_A)
        planilha.cell(row=atual, column=11,
                      value=f'=IFERROR(1/MAX(H{atual}:J{atual}),"")')
        planilha.cell(row=atual, column=12, value=registro.gols_esperados_total)
        planilha.cell(row=atual, column=13, value=registro.mais_de_2_5)
        planilha.cell(row=atual, column=14, value=f'=IFERROR(1/M{atual},"")')
        planilha.cell(row=atual, column=15, value=registro.prob_ambas_marcam)
        planilha.cell(row=atual, column=16, value=registro.escanteios_total)
        planilha.cell(row=atual, column=17, value=registro.cartoes_total)
        planilha.cell(row=atual, column=18, value=registro.palpite_seguro)
        planilha.cell(row=atual, column=19, value=registro.palpite_seguro_prob)
        planilha.cell(row=atual, column=20, value=f'=IFERROR(1/S{atual},"")')
        atual += 1
    ultima_detalhe = atual - 1

    bandas(planilha, primeira_detalhe, ultima_detalhe, len(rotulos))
    escala_probabilidade(planilha, primeira_detalhe, ultima_detalhe,
                         [8, 9, 10, 13, 15, 19])  # 19 = Chance, também percentual
    coluna_odd(planilha, primeira_detalhe, ultima_detalhe, [11, 14, 20])
    coluna_numero(planilha, primeira_detalhe, ultima_detalhe, [12, 16, 17])
    for atual in range(primeira_detalhe, ultima_detalhe + 1):
        planilha.cell(row=atual, column=1).font = F_FORTE
        planilha.cell(row=atual, column=1).alignment = CENTRO
        planilha.cell(row=atual, column=2).number_format = DATA_BR
        planilha.cell(row=atual, column=2).font = F_CORPO
        planilha.cell(row=atual, column=2).alignment = CENTRO
        planilha.cell(row=atual, column=3).font = F_NOTA
        planilha.cell(row=atual, column=3).alignment = CENTRO
        for coluna in (4, 5, 18):
            planilha.cell(row=atual, column=coluna).font = F_CORPO
            planilha.cell(row=atual, column=coluna).alignment = ESQUERDA
        planilha.cell(row=atual, column=6).font = F_FORTE
        planilha.cell(row=atual, column=6).alignment = CENTRO
        planilha.cell(row=atual, column=7).font = F_CORPO
        planilha.cell(row=atual, column=7).alignment = CENTRO

    planilha.conditional_formatting.add(
        f"C{primeira_detalhe}:C{ultima_detalhe}",
        CellIsRule(operator="equal", formula=['"adiado"'],
                   fill=PatternFill("solid", fgColor=DESTAQUE_SUAVE),
                   font=Font(name=FONTE, size=9, bold=True, color=AMBAR)))

    larguras(planilha, [8, 11, 10, 18, 18, 9, 8, 8, 8, 8, 8, 10, 8, 7, 8, 11, 9,
                        23, 8, 7])
    planilha.freeze_panes = planilha.cell(row=primeira_detalhe, column=4).coordinate
    planilha.auto_filter.ref = (f"A{primeira_detalhe - 1}:"
                                f"{get_column_letter(len(rotulos))}{ultima_detalhe}")
    nota_rodape(planilha, ultima_detalhe + 2,
                "'Odd fav.' é a odd justa do desfecho mais provável da partida.",
                len(rotulos))
    return planilha


# ---------------------------------------------------------------------------
# Abas de mercado: cada linha de aposta vira um par probabilidade + odd justa
# ---------------------------------------------------------------------------

def aba_mercado(livro, previsoes, titulo, cor, subtitulo, mercados,
                numericos=None, aviso=None):
    """
    ``mercados`` é uma lista de (rótulo, campo). Cada um ocupa duas colunas:
    a probabilidade e a odd justa, esta última por fórmula.
    ``numericos`` são campos mostrados como número puro (gols, escanteios),
    sem odd ao lado.
    """
    planilha = preparar_aba(livro, titulo, cor)
    numericos = numericos or []
    n_fixas = 4 + len(numericos)
    total_colunas = n_fixas + 2 * len(mercados)
    bloco_titulo(planilha, titulo.upper(), subtitulo, total_colunas)

    linha_grupo = 4
    grupos = [("JOGO", 1, 4, COR_JOGO)]
    if numericos:
        grupos.append(("ESPERADO", 5, len(numericos), COR_JOGO))
    cor_alternada = escurecer(cor)
    for indice, (rotulo, _) in enumerate(mercados):
        # Vizinhos em tons diferentes: sem isso os pares Prob./Odd viram uma
        # barra contínua e a vista perde onde um mercado acaba e o outro começa.
        grupos.append((rotulo, n_fixas + 1 + 2 * indice, 2,
                       cor if indice % 2 == 0 else cor_alternada))
    cabecalho_grupos(planilha, linha_grupo, grupos)

    rotulos = ["Rodada", "Data", "Mandante", "Visitante"]
    rotulos += [rotulo for rotulo, _ in numericos]
    for _ in mercados:
        rotulos += ["Prob.", "Odd justa"]
    cabecalho_colunas(planilha, linha_grupo + 1, rotulos)

    primeira = linha_grupo + 2
    ordenado = previsoes.sort_values(["Rodada", "Data", "Mandante"])
    for deslocamento, registro in enumerate(ordenado.itertuples(index=False)):
        linha = primeira + deslocamento
        planilha.cell(row=linha, column=1, value=registro.Rodada)
        planilha.cell(row=linha, column=2, value=registro.Data)
        planilha.cell(row=linha, column=3, value=registro.Mandante)
        planilha.cell(row=linha, column=4, value=registro.Visitante)
        for indice, (_, campo) in enumerate(numericos):
            planilha.cell(row=linha, column=5 + indice, value=getattr(registro, campo))
        for indice, (_, campo) in enumerate(mercados):
            coluna_prob = n_fixas + 1 + 2 * indice
            planilha.cell(row=linha, column=coluna_prob, value=getattr(registro, campo))
            letra = get_column_letter(coluna_prob)
            planilha.cell(row=linha, column=coluna_prob + 1,
                          value=f'=IFERROR(1/{letra}{linha},"")')

    ultima = primeira + len(ordenado) - 1
    bandas(planilha, primeira, ultima, total_colunas)
    colunas_prob = [n_fixas + 1 + 2 * i for i in range(len(mercados))]
    escala_probabilidade(planilha, primeira, ultima, colunas_prob)
    coluna_odd(planilha, primeira, ultima, [c + 1 for c in colunas_prob])
    if numericos:
        coluna_numero(planilha, primeira, ultima,
                      list(range(5, 5 + len(numericos))))
    for linha in range(primeira, ultima + 1):
        planilha.cell(row=linha, column=1).font = F_FORTE
        planilha.cell(row=linha, column=1).alignment = CENTRO
        planilha.cell(row=linha, column=2).number_format = DATA_BR
        planilha.cell(row=linha, column=2).font = F_CORPO
        planilha.cell(row=linha, column=2).alignment = CENTRO
        for coluna in (3, 4):
            planilha.cell(row=linha, column=coluna).font = F_CORPO
            planilha.cell(row=linha, column=coluna).alignment = ESQUERDA

    larguras(planilha, [8, 11, 18, 18] + [11] * len(numericos)
             + [8, 9] * len(mercados))
    planilha.freeze_panes = planilha.cell(row=primeira, column=5).coordinate
    planilha.auto_filter.ref = (f"A{linha_grupo + 1}:"
                                f"{get_column_letter(total_colunas)}{ultima}")
    if aviso:
        nota_rodape(planilha, ultima + 2, aviso, total_colunas)
    return planilha


# Mercados oferecidos na calculadora: rótulo -> coluna da tabela de previsões.
MERCADOS_CALCULADORA = [
    ("Vitória do mandante", "prob_H"),
    ("Empate", "prob_D"),
    ("Vitória do visitante", "prob_A"),
    ("Dupla chance 1X (casa ou empate)", "dupla_1x"),
    ("Dupla chance 12 (casa ou fora)", "dupla_12"),
    ("Dupla chance X2 (empate ou fora)", "dupla_x2"),
    ("Mais de 0,5 gol", "mais_de_0_5"),
    ("Mais de 1,5 gols", "mais_de_1_5"),
    ("Mais de 2,5 gols", "mais_de_2_5"),
    ("Mais de 3,5 gols", "mais_de_3_5"),
    ("Menos de 2,5 gols", "menos_de_2_5"),
    ("Ambas as equipes marcam - SIM", "prob_ambas_marcam"),
    ("Ambas as equipes marcam - NÃO", "ambas_nao"),
    ("Multi-gols 1-2", "multigols_1_2"),
    ("Multi-gols 1-3", "multigols_1_3"),
    ("Multi-gols 1-4", "multigols_1_4"),
    ("Multi-gols 2-3", "multigols_2_3"),
    ("Multi-gols 2-4", "multigols_2_4"),
    ("Mais de 8,5 escanteios", "escanteios_mais_8_5"),
    ("Mais de 9,5 escanteios", "escanteios_mais_9_5"),
    ("Mais de 10,5 escanteios", "escanteios_mais_10_5"),
    ("Mais de 3,5 cartões", "cartoes_mais_3_5"),
    ("Mais de 4,5 cartões", "cartoes_mais_4_5"),
    ("Mais de 5,5 cartões", "cartoes_mais_5_5"),
]


def aba_dados(livro, previsoes):
    """Matriz partida x mercado que alimenta a calculadora (aba de apoio)."""
    planilha = livro.create_sheet("Dados")
    planilha["A1"] = "Partida"
    for coluna, (rotulo, _) in enumerate(MERCADOS_CALCULADORA, start=2):
        planilha.cell(row=1, column=coluna, value=rotulo)
    ordenado = previsoes.sort_values(["Rodada", "Data", "Mandante"])
    for linha, registro in enumerate(ordenado.itertuples(index=False), start=2):
        planilha.cell(row=linha, column=1,
                      value=f"{registro.Mandante} x {registro.Visitante}")
        for coluna, (_, campo) in enumerate(MERCADOS_CALCULADORA, start=2):
            planilha.cell(row=linha, column=coluna, value=getattr(registro, campo))
            planilha.cell(row=linha, column=coluna).number_format = PCT
    larguras(planilha, [34] + [16] * len(MERCADOS_CALCULADORA))
    planilha.sheet_state = "hidden"
    return planilha


def aba_calculadora(livro, previsoes):
    """Calculadora de valor: o usuário só escolhe nas listas e digita a cotação."""
    planilha = preparar_aba(livro, "Calculadora", COR_DESTAQUE)
    larguras(planilha, [2, 30, 26, 3, 13, 13, 13, 15, 10])

    n = len(previsoes)
    ultima_coluna = get_column_letter(1 + len(MERCADOS_CALCULADORA))

    planilha["B2"] = "CALCULADORA DE VALOR"
    planilha["B2"].font = Font(name=FONTE, size=20, bold=True, color=TINTA_TITULO)
    planilha.row_dimensions[2].height = 28
    planilha["B3"] = ("Escolha a partida e o mercado nas listas, digite a cotação "
                      "da casa no campo amarelo e leia o veredicto.")
    planilha["B3"].font = F_SUBTITULO
    planilha.merge_cells("B3:I3")

    planilha["B5"] = "1. O QUE VOCÊ VAI APOSTAR"
    planilha["B5"].font = F_SECAO
    planilha["B9"] = "2. QUANTO A CASA PAGA"
    planilha["B9"].font = F_SECAO
    planilha["B12"] = "3. VEREDICTO"
    planilha["B12"].font = F_SECAO

    campos = [
        (6, "Partida", None, True),
        (7, "Mercado", None, True),
        (8, "Probabilidade do modelo",
         f"=IFERROR(INDEX(Dados!$B$2:${ultima_coluna}${n + 1},"
         f"MATCH($C$6,Dados!$A$2:$A${n + 1},0),"
         f"MATCH($C$7,Dados!$B$1:${ultima_coluna}$1,0)),\"\")", False),
        (10, "Odd justa (equilíbrio)", '=IFERROR(1/$C$8,"")', False),
        (11, "Cotação oferecida", None, True),
        (13, "Valor esperado por real", '=IFERROR($C$8*$C$11-1,"")', False),
        (14, "Vantagem sobre o preço", '=IFERROR($C$8-1/$C$11,"")', False),
        (15, "Lucro se vier green", '=IFERROR($C$11-1,"")', False),
        (16, "Em 100 apostas de R$ 10", '=IFERROR($C$13*1000,"")', False),
    ]
    for linha, rotulo, formula, entrada in campos:
        celula_rotulo = planilha.cell(row=linha, column=2, value=rotulo)
        celula_rotulo.font = F_FORTE if not entrada else Font(
            name=FONTE, size=10, bold=True, color=TINTA_PRIMARIA)
        celula_rotulo.alignment = ESQUERDA
        celula = planilha.cell(row=linha, column=3)
        if formula:
            celula.value = formula
        celula.alignment = CENTRO
        planilha.row_dimensions[linha].height = 20
        if entrada:
            celula.fill = PatternFill("solid", fgColor="FFF3B0")
            celula.font = Font(name=FONTE, size=10, bold=True, color="1A3E7A")
            celula.border = Border(*[Side(style="thin", color="D9A400")] * 4)
        else:
            celula.font = F_CORPO
            celula.fill = PatternFill("solid", fgColor=CARTAO)

    planilha["C6"] = (f"{previsoes.iloc[0]['Mandante']} x "
                      f"{previsoes.iloc[0]['Visitante']}")
    planilha["C7"] = MERCADOS_CALCULADORA[2][0]
    planilha["C11"] = 3.0

    planilha["C8"].number_format = PCT
    for celula in ("C10", "C11", "C15"):
        planilha[celula].number_format = ODD
    for celula in ("C13", "C14"):
        planilha[celula].number_format = "0.0%"
    planilha["C16"].number_format = 'R$ #,##0.00;[Red]-R$ #,##0.00'

    veredicto = planilha.cell(row=17, column=2, value="Resultado")
    veredicto.font = F_FORTE
    celula_veredicto = planilha.cell(row=17, column=3)
    celula_veredicto.value = (
        '=IF($C$8="","preencha os campos",'
        'IF($C$11="","digite a cotação",'
        'IF($C$13>0,"PAGA ACIMA DA ODD JUSTA","paga abaixo da odd justa")))')
    celula_veredicto.font = Font(name=FONTE, size=11, bold=True)
    celula_veredicto.alignment = CENTRO
    planilha.row_dimensions[17].height = 24
    planilha.conditional_formatting.add("C17", CellIsRule(
        operator="equal", formula=['"PAGA ACIMA DA ODD JUSTA"'],
        fill=PatternFill("solid", fgColor="D6F0E0"),
        font=Font(name=FONTE, size=11, bold=True, color=VERDE)))
    planilha.conditional_formatting.add("C17", CellIsRule(
        operator="equal", formula=['"paga abaixo da odd justa"'],
        fill=PatternFill("solid", fgColor="FBE0E0"),
        font=Font(name=FONTE, size=11, bold=True, color=VERMELHO)))
    for celula, cor_boa, cor_ruim in (("C13", VERDE, VERMELHO), ("C14", VERDE, VERMELHO)):
        planilha.conditional_formatting.add(celula, CellIsRule(
            operator="greaterThan", formula=["0"],
            font=Font(name=FONTE, size=10, bold=True, color=cor_boa)))
        planilha.conditional_formatting.add(celula, CellIsRule(
            operator="lessThanOrEqual", formula=["0"],
            font=Font(name=FONTE, size=10, bold=True, color=cor_ruim)))

    livro.defined_names.add(
        DefinedName("Lista_Partidas", attr_text=f"Dados!$A$2:$A${n + 1}"))
    livro.defined_names.add(
        DefinedName("Lista_Mercados", attr_text=f"Dados!$B$1:${ultima_coluna}$1"))
    validacao_partida = DataValidation(type="list", formula1="=Lista_Partidas",
                                       allow_blank=False)
    planilha.add_data_validation(validacao_partida)
    validacao_partida.add(planilha["C6"])
    validacao_mercado = DataValidation(type="list", formula1="=Lista_Mercados",
                                       allow_blank=False)
    planilha.add_data_validation(validacao_mercado)
    validacao_mercado.add(planilha["C7"])

    # --- tabela lateral: a mesma aposta em várias cotações -------------------
    linha = 5
    planilha.cell(row=linha, column=5,
                  value="A MESMA APOSTA EM VÁRIAS COTAÇÕES").font = F_SECAO
    planilha.merge_cells(start_row=linha, start_column=5, end_row=linha, end_column=9)
    linha += 1
    planilha.cell(row=linha, column=5,
                  value="A chance de acertar é a MESMA em todas as linhas. "
                        "O que muda é o preço.").font = F_SUBTITULO
    planilha.merge_cells(start_row=linha, start_column=5, end_row=linha, end_column=9)
    linha += 2
    cabecalho_grupos(planilha, linha, [("SIMULAÇÃO", 5, 5, COR_DESTAQUE)])
    linha += 1
    for deslocamento, rotulo in enumerate(
            ["Cotação", "Chance", "Valor esp.", "Lucro se green", "Vale?"]):
        celula = planilha.cell(row=linha, column=5 + deslocamento, value=rotulo)
        celula.font = F_COLUNA
        celula.alignment = CENTRO
        celula.border = REGUA_COLUNA
    primeira_simulacao = linha + 1
    for deslocamento, cotacao in enumerate([1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0,
                                            5.0, 7.0, 10.0]):
        atual = primeira_simulacao + deslocamento
        planilha.cell(row=atual, column=5, value=cotacao).number_format = ODD
        planilha.cell(row=atual, column=6, value="=$C$8").number_format = PCT
        planilha.cell(row=atual, column=7,
                      value=f'=IFERROR($C$8*E{atual}-1,"")').number_format = "0.0%"
        planilha.cell(row=atual, column=8, value=f"=E{atual}-1").number_format = ODD
        planilha.cell(row=atual, column=9,
                      value=f'=IF(G{atual}="","",IF(G{atual}>0,"sim","não"))')
        for coluna in range(5, 10):
            planilha.cell(row=atual, column=coluna).font = F_CORPO
            planilha.cell(row=atual, column=coluna).alignment = CENTRO
    ultima_simulacao = primeira_simulacao + 9
    bandas(planilha, primeira_simulacao, ultima_simulacao, 9)
    planilha.conditional_formatting.add(
        f"G{primeira_simulacao}:G{ultima_simulacao}",
        CellIsRule(operator="greaterThan", formula=["0"],
                   fill=PatternFill("solid", fgColor="D6F0E0"),
                   font=Font(name=FONTE, size=10, color=VERDE)))
    planilha.conditional_formatting.add(
        f"G{primeira_simulacao}:G{ultima_simulacao}",
        CellIsRule(operator="lessThanOrEqual", formula=["0"],
                   fill=PatternFill("solid", fgColor="FBE0E0"),
                   font=Font(name=FONTE, size=10, color=VERMELHO)))

    nota_rodape(planilha, ultima_simulacao + 2,
                "Lembrete: valor esperado positivo segundo o modelo não garante lucro. "
                "No teste histórico do projeto, apostar só onde o modelo via vantagem "
                "rendeu PIOR do que apostar em tudo.", 9)
    return planilha


# ---------------------------------------------------------------------------
# Abas de apoio
# ---------------------------------------------------------------------------

def aba_confiabilidade(livro, confiabilidade):
    planilha = preparar_aba(livro, "Confiabilidade", COR_JOGO)
    bloco_titulo(planilha, "QUANTO CADA MERCADO ERRA",
                 "Viés = previsto menos observado, medido FORA DA AMOSTRA: o modelo "
                 "treina até a temporada anterior e é conferido na seguinte. "
                 "Negativo significa que o modelo subestima o mercado.", 7)

    linha = 4
    cabecalho_grupos(planilha, linha, [
        ("MERCADO", 1, 1, COR_JOGO), ("MEDIDO FORA DA AMOSTRA", 2, 4, COR_RESULTADO),
        ("LEITURA", 6, 2, COR_DESTAQUE)])
    cabecalho_colunas(planilha, linha + 1,
                      ["Mercado", "Previsto", "Observado", "Viés (p.p.)",
                       "Partidas", "Confiança", "Observação"])
    primeira = linha + 2
    for deslocamento, registro in enumerate(confiabilidade.itertuples(index=False)):
        atual = primeira + deslocamento
        for coluna, valor in enumerate(registro, start=1):
            planilha.cell(row=atual, column=coluna, value=valor)
    ultima = primeira + len(confiabilidade) - 1

    bandas(planilha, primeira, ultima, 7, altura=26)
    escala_probabilidade(planilha, primeira, ultima, [2, 3])
    for atual in range(primeira, ultima + 1):
        planilha.cell(row=atual, column=1).font = F_FORTE
        planilha.cell(row=atual, column=1).alignment = ESQUERDA
        celula = planilha.cell(row=atual, column=4)
        celula.number_format = "+0.00;-0.00"
        celula.font = F_CORPO
        celula.alignment = CENTRO
        planilha.cell(row=atual, column=5).font = F_CORPO
        planilha.cell(row=atual, column=5).alignment = CENTRO
        planilha.cell(row=atual, column=6).alignment = CENTRO
        planilha.cell(row=atual, column=6).font = F_FORTE
        observacao = planilha.cell(row=atual, column=7)
        observacao.font = F_NOTA
        observacao.alignment = Alignment(wrap_text=True, vertical="center")

    planilha.conditional_formatting.add(f"D{primeira}:D{ultima}", CellIsRule(
        operator="lessThan", formula=["-3"],
        fill=PatternFill("solid", fgColor="FBE0E0"),
        font=Font(name=FONTE, size=10, bold=True, color=VERMELHO)))
    planilha.conditional_formatting.add(f"F{primeira}:F{ultima}", CellIsRule(
        operator="equal", formula=['"Alta"'],
        font=Font(name=FONTE, size=10, bold=True, color=VERDE)))
    planilha.conditional_formatting.add(f"F{primeira}:F{ultima}", CellIsRule(
        operator="equal", formula=['"Média"'],
        font=Font(name=FONTE, size=10, bold=True, color=AMBAR)))

    larguras(planilha, [30, 11, 11, 11, 11, 12, 62])
    planilha.freeze_panes = planilha.cell(row=primeira, column=2).coordinate
    planilha.auto_filter.ref = f"A{linha + 1}:G{ultima}"
    return planilha


def aba_times(livro, contexto):
    planilha = preparar_aba(livro, "Times", COR_JOGO)
    modelo_gols = contexto["modelo_gols"]
    modelo_escanteios = contexto["modelo_escanteios"]
    modelo_cartoes = contexto["modelo_cartoes"]
    classificacao = contexto["classificacao"].set_index("time")

    linhas = []
    for time in contexto["times_2026"]:
        linhas.append({
            "Time": time,
            "Pos.": int(classificacao.loc[time, "posicao"]),
            "Pontos": int(classificacao.loc[time, "pontos"]),
            "Ataque": float(np.exp(modelo_gols.ataque.get(time, 0.0))),
            "Defesa": float(np.exp(modelo_gols.defesa.get(time, 0.0))),
            "Esc. a favor": float(np.exp(modelo_escanteios.feito.get(time, 0.0))),
            "Esc. contra": float(np.exp(modelo_escanteios.sofrido.get(time, 0.0))),
            "Cartões": float(np.exp(modelo_cartoes.feito.get(time, 0.0))),
        })
    tabela = pd.DataFrame(linhas).sort_values("Pos.")

    bloco_titulo(planilha, "FORÇA DOS CLUBES",
                 "Valores relativos à média da liga, onde 1,00 é a média. Ataque acima "
                 "de 1,00 marca mais que a média; defesa abaixo de 1,00 sofre menos. "
                 "Escanteios e cartões vêm da base 2015-2026.", 8)

    linha = 4
    cabecalho_grupos(planilha, linha, [
        ("CLUBE", 1, 3, COR_JOGO), ("GOLS", 4, 2, COR_GOLS),
        ("ESCANTEIOS", 6, 2, COR_ESCANTEIOS), ("CARTÕES", 8, 1, COR_CARTOES)])
    cabecalho_colunas(planilha, linha + 1, list(tabela.columns))
    primeira = linha + 2
    for deslocamento, registro in enumerate(tabela.itertuples(index=False)):
        atual = primeira + deslocamento
        for coluna, valor in enumerate(registro, start=1):
            planilha.cell(row=atual, column=coluna, value=valor)
    ultima = primeira + len(tabela) - 1

    bandas(planilha, primeira, ultima, 8, altura=20)
    coluna_numero(planilha, primeira, ultima, [4, 5, 6, 7, 8])
    for atual in range(primeira, ultima + 1):
        planilha.cell(row=atual, column=1).font = F_FORTE
        planilha.cell(row=atual, column=1).alignment = ESQUERDA
        for coluna in (2, 3):
            planilha.cell(row=atual, column=coluna).font = F_CORPO
            planilha.cell(row=atual, column=coluna).alignment = CENTRO

    # Ataque e escanteios a favor: mais é melhor. Defesa e cartões: menos é melhor.
    for coluna, invertido in ((4, False), (5, True), (6, False), (7, True), (8, True)):
        letra = get_column_letter(coluna)
        planilha.conditional_formatting.add(
            f"{letra}{primeira}:{letra}{ultima}",
            ColorScaleRule(start_type="min",
                           start_color=PROB_MAX if invertido else PROB_MIN,
                           end_type="max",
                           end_color=PROB_MIN if invertido else PROB_MAX))

    larguras(planilha, [20, 8, 10, 12, 12, 14, 14, 12])
    planilha.freeze_panes = planilha.cell(row=primeira, column=2).coordinate
    nota_rodape(planilha, ultima + 2,
                "Azul mais forte = melhor naquele quesito. Em defesa e cartões, "
                "melhor é ter o número mais baixo.", 8)
    return planilha


# ---------------------------------------------------------------------------
# Melhores palpites e bilhetes pré-montados
# ---------------------------------------------------------------------------
#
# Um bilhete múltiplo multiplica as odds das pernas — e multiplica junto a
# margem da casa. Com a margem medida no Brasileirão (cerca de 7% por mercado),
# o retorno esperado cai de -6,5% numa aposta simples para -41,7% numa múltipla
# de oito pernas. Por isso a aba de bilhetes traz essa coluna: o bilhete de odd
# alta é o que mais devolve dinheiro à casa, e isso precisa estar à vista.
#
# Todas as pernas de um bilhete vêm de PARTIDAS DIFERENTES. Duas seleções do
# mesmo jogo são correlacionadas (quem vence costuma marcar), e multiplicar as
# probabilidades nesse caso daria um número errado para mais.

# (rótulo exibido, campo na tabela de previsões, confiança da família)
MERCADOS_PALPITE = [
    ("Vitória do mandante", "prob_H", "Alta"),
    ("Empate", "prob_D", "Alta"),
    ("Vitória do visitante", "prob_A", "Alta"),
    ("Casa ou empate (1X)", "dupla_1x", "Alta"),
    ("Casa ou fora (12)", "dupla_12", "Alta"),
    ("Empate ou fora (X2)", "dupla_x2", "Alta"),
    ("Mais de 0,5 gol", "mais_de_0_5", "Alta"),
    ("Mais de 1,5 gols", "mais_de_1_5", "Alta"),
    ("Mais de 2,5 gols", "mais_de_2_5", "Alta"),
    ("Mais de 3,5 gols", "mais_de_3_5", "Alta"),
    ("Menos de 2,5 gols", "menos_de_2_5", "Alta"),
    ("Ambas marcam - SIM", "prob_ambas_marcam", "Média"),
    ("Ambas marcam - NÃO", "ambas_nao", "Média"),
    ("Mais de 8,5 escanteios", "escanteios_mais_8_5", "Média"),
    ("Mais de 9,5 escanteios", "escanteios_mais_9_5", "Média"),
    ("Mais de 10,5 escanteios", "escanteios_mais_10_5", "Média"),
    ("Mais de 3,5 cartões", "cartoes_mais_3_5", "Média"),
    ("Mais de 4,5 cartões", "cartoes_mais_4_5", "Média"),
    ("Mais de 5,5 cartões", "cartoes_mais_5_5", "Média"),
]
MERCADOS_PALPITE += [(f"Multi-gols {a}-{b}", f"multigols_{a}_{b}", "Alta")
                     for a, b in ub.FAIXAS_MULTIGOLS]

# Escada de bilhetes: alvo de odd -> nome do perfil.
ESCADA_BILHETES = [
    (1.5, "Muito seguro"), (2.0, "Seguro"), (3.0, "Moderado"),
    (5.0, "Equilibrado"), (10.0, "Ousado"), (25.0, "Arriscado"),
    (50.0, "Agressivo"), (100.0, "Muito agressivo"), (250.0, "Risco máximo"),
]
PROB_MINIMA_PERNA = 0.35   # abaixo disso o modelo é menos confiável
MAX_PERNAS = 8


def candidatos_por_partida(previsoes: pd.DataFrame) -> pd.DataFrame:
    """Todas as seleções disponíveis, uma linha por partida x mercado."""
    linhas = []
    for registro in previsoes.itertuples(index=False):
        for rotulo, campo, confianca in MERCADOS_PALPITE:
            probabilidade = getattr(registro, campo, None)
            if probabilidade is None or not np.isfinite(probabilidade):
                continue
            linhas.append({
                "Rodada": registro.Rodada, "Data": registro.Data,
                "Partida": f"{registro.Mandante} x {registro.Visitante}",
                "Mercado": rotulo, "prob": float(probabilidade),
                "Confiança": confianca})
    candidatos = pd.DataFrame(linhas)
    candidatos["odd_justa"] = 1.0 / candidatos["prob"]
    return candidatos


# Faixas de odd para a vitrine de palpites: para cada partida mostramos o
# palpite mais seguro e um para cada nível de risco.
ALVOS_VITRINE = [("Mais seguro", None), ("Odd ~1,5", 1.5),
                 ("Odd ~2,0", 2.0), ("Odd ~3,0", 3.0)]


def melhores_por_partida(candidatos: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada partida, quatro palpites: o mais provável e um em cada faixa de
    odd.

    Ordenar por probabilidade pura não serve de vitrine: como a odd justa é
    exatamente 1/probabilidade, o topo do ranking é sempre o mesmo mercado
    trivial ("mais de 0,5 gol", perto de 94% e odd 1,06) em todas as partidas.
    Separar por faixa de odd é o que dá variedade e deixa o leitor escolher o
    risco que quer correr.
    """
    linhas = []
    for partida, grupo in candidatos.groupby("Partida", sort=False):
        registro = {"Rodada": grupo["Rodada"].iloc[0],
                    "Data": grupo["Data"].iloc[0], "Partida": partida}
        usados = set()
        for rotulo, alvo in ALVOS_VITRINE:
            disponiveis = grupo[~grupo["Mercado"].isin(usados)]
            if disponiveis.empty:
                continue
            if alvo is None:
                escolhido = disponiveis.loc[disponiveis["prob"].idxmax()]
            else:
                distancia = (disponiveis["odd_justa"] - alvo).abs()
                escolhido = disponiveis.loc[distancia.idxmin()]
            usados.add(escolhido["Mercado"])
            registro[f"{rotulo}|mercado"] = escolhido["Mercado"]
            registro[f"{rotulo}|prob"] = float(escolhido["prob"])
            registro[f"{rotulo}|conf"] = escolhido["Confiança"]
        linhas.append(registro)
    tabela = pd.DataFrame(linhas)
    return tabela.sort_values(["Rodada", "Data", "Partida"]).reset_index(drop=True)


def _escolher_pernas(grupo: pd.DataFrame, n_pernas: int,
                     prob_alvo: float) -> pd.DataFrame:
    """n seleções de partidas diferentes, com probabilidade perto do alvo."""
    ordenado = grupo.assign(
        distancia=(grupo["prob"] - prob_alvo).abs()).sort_values(
        ["distancia", "prob"], ascending=[True, False])
    escolhidas, usadas = [], set()
    for registro in ordenado.itertuples(index=False):
        if registro.Partida in usadas:
            continue
        escolhidas.append(registro)
        usadas.add(registro.Partida)
        if len(escolhidas) == n_pernas:
            break
    return pd.DataFrame(escolhidas)


def montar_bilhetes(candidatos: pd.DataFrame, margem: float) -> pd.DataFrame:
    """
    Monta a escada de bilhetes de cada rodada.

    Para um alvo de odd T, a probabilidade combinada é sempre 1/T — isso não
    depende de como o bilhete é montado. O que depende é o NÚMERO DE PERNAS:
    cada perna adiciona a margem da casa. Por isso escolhemos sempre o menor
    número de pernas que atinge o alvo sem que nenhuma perna fique abaixo de
    35% de probabilidade.
    """
    linhas = []
    for rodada, grupo in candidatos.groupby("Rodada"):
        n_partidas = grupo["Partida"].nunique()
        for alvo, perfil in ESCADA_BILHETES:
            n_pernas = None
            for n in range(2, min(MAX_PERNAS, n_partidas) + 1):
                if (1.0 / alvo) ** (1.0 / n) >= PROB_MINIMA_PERNA:
                    n_pernas = n
                    break
            if n_pernas is None:
                continue   # a rodada não tem partidas suficientes para esse alvo
            prob_alvo = (1.0 / alvo) ** (1.0 / n_pernas)
            pernas = _escolher_pernas(grupo, n_pernas, prob_alvo)
            if len(pernas) < n_pernas:
                continue

            prob_combinada = float(pernas["prob"].prod())
            odd_justa = 1.0 / prob_combinada
            # A casa paga a odd justa reduzida pela margem, uma vez por perna.
            odd_casa = odd_justa / ((1.0 + margem) ** n_pernas)
            retorno = prob_combinada * odd_casa - 1.0

            for posicao, perna in enumerate(pernas.itertuples(index=False), start=1):
                linhas.append({
                    "Rodada": int(rodada), "Bilhete": perfil,
                    "Alvo de odd": alvo, "Pernas": n_pernas,
                    "Prob. combinada": prob_combinada,
                    "Odd justa do bilhete": odd_justa,
                    "Odd provável na casa": odd_casa,
                    "Retorno esperado": retorno,
                    "Perna": posicao, "Data": perna.Data,
                    "Partida": perna.Partida, "Mercado": perna.Mercado,
                    "Prob. da perna": perna.prob,
                    "Odd justa da perna": perna.odd_justa,
                    "Confiança": perna.Confiança})
    return pd.DataFrame(linhas)


def aba_melhores(livro, melhores):
    planilha = preparar_aba(livro, "Melhores Palpites", COR_DESTAQUE)
    bloco_titulo(planilha, "MELHORES PALPITES DE CADA RODADA",
                 "Quatro sugestões por partida: a mais provável e uma em cada faixa de "
                 "odd. Como a odd justa é 1/probabilidade, o palpite mais seguro é "
                 "sempre o que menos paga — por isso a vitrine é separada por risco.", 15)

    cores_faixa = [COR_GOLS, COR_RESULTADO, COR_DUPLA, COR_AMBAS]
    linha = 4
    grupos = [("PARTIDA", 1, 3, COR_JOGO)]
    for indice, (rotulo, _) in enumerate(ALVOS_VITRINE):
        grupos.append((rotulo.upper(), 4 + 3 * indice, 3, cores_faixa[indice]))
    cabecalho_grupos(planilha, linha, grupos)
    rotulos = ["Rodada", "Data", "Partida"]
    for _ in ALVOS_VITRINE:
        rotulos += ["Mercado", "Prob.", "Odd justa"]
    cabecalho_colunas(planilha, linha + 1, rotulos)

    primeira = linha + 2
    for deslocamento, registro in enumerate(melhores.to_dict("records")):
        atual = primeira + deslocamento
        planilha.cell(row=atual, column=1, value=registro["Rodada"])
        planilha.cell(row=atual, column=2, value=registro["Data"])
        planilha.cell(row=atual, column=3, value=registro["Partida"])
        for indice, (rotulo, _) in enumerate(ALVOS_VITRINE):
            coluna = 4 + 3 * indice
            planilha.cell(row=atual, column=coluna,
                          value=registro.get(f"{rotulo}|mercado"))
            planilha.cell(row=atual, column=coluna + 1,
                          value=registro.get(f"{rotulo}|prob"))
            letra = get_column_letter(coluna + 1)
            planilha.cell(row=atual, column=coluna + 2,
                          value=f'=IFERROR(1/{letra}{atual},"")')
    ultima = primeira + len(melhores) - 1

    bandas(planilha, primeira, ultima, len(rotulos))
    colunas_prob = [5 + 3 * i for i in range(len(ALVOS_VITRINE))]
    escala_probabilidade(planilha, primeira, ultima, colunas_prob)
    coluna_odd(planilha, primeira, ultima, [c + 1 for c in colunas_prob])
    for atual in range(primeira, ultima + 1):
        planilha.cell(row=atual, column=1).font = F_FORTE
        planilha.cell(row=atual, column=1).alignment = CENTRO
        planilha.cell(row=atual, column=2).number_format = DATA_BR
        planilha.cell(row=atual, column=2).font = F_CORPO
        planilha.cell(row=atual, column=2).alignment = CENTRO
        planilha.cell(row=atual, column=3).font = F_CORPO
        planilha.cell(row=atual, column=3).alignment = ESQUERDA
        for indice in range(len(ALVOS_VITRINE)):
            celula = planilha.cell(row=atual, column=4 + 3 * indice)
            celula.font = F_FORTE
            celula.alignment = ESQUERDA

    larguras(planilha, [8, 11, 32] + [23, 8, 10] * len(ALVOS_VITRINE))
    planilha.freeze_panes = planilha.cell(row=primeira, column=4).coordinate
    planilha.auto_filter.ref = (f"A{linha + 1}:"
                                f"{get_column_letter(len(rotulos))}{ultima}")
    nota_rodape(planilha, ultima + 2,
                "Cada coluna traz um mercado diferente: o mesmo palpite não se repete "
                "entre as faixas da mesma partida. 'Mais seguro' costuma cair em "
                "mercados de odd baixa, que pagam pouco justamente por serem prováveis.",
                len(rotulos))
    return planilha


def aba_bilhetes(livro, bilhetes, margem):
    planilha = preparar_aba(livro, "Bilhetes", COR_AMBAS)
    bloco_titulo(planilha, "BILHETES PRÉ-MONTADOS",
                 "Uma escada por rodada, do bilhete mais seguro ao de odd mais alta. "
                 "Todas as pernas vêm de partidas diferentes. Filtre pela rodada ou "
                 "pelo perfil do bilhete.", 15)

    # --- painel: o que a múltipla faz com o retorno --------------------------
    linha = 4
    planilha.cell(row=linha, column=1,
                  value="O QUE ACONTECE QUANDO SE EMPILHA PERNA").font = F_SECAO
    linha += 1
    planilha.cell(row=linha, column=1,
                  value=f"A margem medida no Brasileirão é de "
                        f"{100 * margem:.1f}% por mercado. Numa múltipla ela incide uma "
                        "vez por perna, então o retorno esperado piora a cada perna "
                        "adicionada — independentemente de quais jogos entrem.").font = F_SUBTITULO
    planilha.merge_cells(start_row=linha, start_column=1, end_row=linha, end_column=15)
    linha += 1
    cabecalho_grupos(planilha, linha, [("PERNAS", 1, 1, COR_JOGO),
                                       ("RETORNO ESPERADO", 2, 1, COR_AMBAS)])
    primeira_painel = linha + 1
    for deslocamento in range(8):
        n = deslocamento + 1
        atual = primeira_painel + deslocamento
        planilha.cell(row=atual, column=1, value=n).font = F_CORPO
        planilha.cell(row=atual, column=1).alignment = CENTRO
        celula = planilha.cell(row=atual, column=2,
                               value=1 / ((1 + margem) ** n) - 1)
        celula.number_format = "0.0%"
        celula.font = Font(name=FONTE, size=10, bold=True, color=VERMELHO)
        celula.alignment = CENTRO
    ultima_painel = primeira_painel + 7
    bandas(planilha, primeira_painel, ultima_painel, 2)

    # --- tabela de bilhetes --------------------------------------------------
    linha = ultima_painel + 2
    planilha.cell(row=linha, column=1, value="OS BILHETES").font = F_SECAO
    linha += 1
    cabecalho_grupos(planilha, linha, [
        ("BILHETE", 1, 3, COR_AMBAS), ("O BILHETE INTEIRO", 4, 5, COR_RESULTADO),
        ("PERNA A PERNA", 9, 6, COR_JOGO)])
    cabecalho_colunas(planilha, linha + 1, [
        "Rodada", "Perfil", "Alvo", "Pernas", "Prob. combinada",
        "Odd justa", "Odd provável na casa", "Retorno esperado",
        "Perna", "Data", "Partida", "Mercado", "Prob.", "Odd justa", "Confiança"])
    primeira = linha + 2

    # Acesso por NOME da coluna: vários rótulos têm espaço e ponto, e os
    # atributos posicionais do itertuples (registro._3) quebrariam em silêncio
    # se a ordem das colunas mudasse.
    COLUNAS_BILHETE = ["Rodada", "Bilhete", "Alvo de odd", "Pernas",
                       "Prob. combinada", "Odd justa do bilhete",
                       "Odd provável na casa", "Retorno esperado", "Perna",
                       "Data", "Partida", "Mercado", "Prob. da perna",
                       "Odd justa da perna", "Confiança"]
    ordenado = bilhetes.sort_values(["Rodada", "Alvo de odd", "Perna"])
    registros = ordenado.to_dict("records")
    for deslocamento, registro in enumerate(registros):
        atual = primeira + deslocamento
        for coluna, nome in enumerate(COLUNAS_BILHETE, start=1):
            planilha.cell(row=atual, column=coluna, value=registro[nome])
    ultima = primeira + len(registros) - 1

    # Faixa por bilhete (e não por linha): mantém o bloco de pernas junto.
    chave_anterior, alternar = None, False
    for deslocamento, registro in enumerate(registros):
        atual = primeira + deslocamento
        chave = (registro["Rodada"], registro["Alvo de odd"])
        if chave != chave_anterior:
            alternar = not alternar
            chave_anterior = chave
        planilha.row_dimensions[atual].height = 19
        if alternar:
            for coluna in range(1, 16):
                planilha.cell(row=atual, column=coluna).fill = PatternFill(
                    "solid", fgColor=BANDA)

    escala_probabilidade(planilha, primeira, ultima, [5, 13])
    coluna_odd(planilha, primeira, ultima, [6, 7, 14])
    for atual in range(primeira, ultima + 1):
        planilha.cell(row=atual, column=1).font = F_FORTE
        planilha.cell(row=atual, column=1).alignment = CENTRO
        planilha.cell(row=atual, column=2).font = F_FORTE
        planilha.cell(row=atual, column=2).alignment = ESQUERDA
        planilha.cell(row=atual, column=3).number_format = ODD
        planilha.cell(row=atual, column=3).font = F_NOTA
        planilha.cell(row=atual, column=3).alignment = CENTRO
        planilha.cell(row=atual, column=4).font = F_CORPO
        planilha.cell(row=atual, column=4).alignment = CENTRO
        retorno = planilha.cell(row=atual, column=8)
        retorno.number_format = "0.0%"
        retorno.font = Font(name=FONTE, size=10, bold=True, color=VERMELHO)
        retorno.alignment = CENTRO
        planilha.cell(row=atual, column=9).font = F_NOTA
        planilha.cell(row=atual, column=9).alignment = CENTRO
        planilha.cell(row=atual, column=10).number_format = DATA_BR
        planilha.cell(row=atual, column=10).font = F_NOTA
        planilha.cell(row=atual, column=10).alignment = CENTRO
        planilha.cell(row=atual, column=11).font = F_CORPO
        planilha.cell(row=atual, column=11).alignment = ESQUERDA
        planilha.cell(row=atual, column=12).font = F_CORPO
        planilha.cell(row=atual, column=12).alignment = ESQUERDA
        planilha.cell(row=atual, column=15).font = F_NOTA
        planilha.cell(row=atual, column=15).alignment = CENTRO
    planilha.conditional_formatting.add(f"O{primeira}:O{ultima}", CellIsRule(
        operator="equal", formula=['"Média"'],
        font=Font(name=FONTE, size=9, bold=True, color=AMBAR)))

    larguras(planilha, [8, 16, 8, 8, 14, 10, 17, 14, 7, 11, 32, 24, 8, 10, 11])
    planilha.freeze_panes = planilha.cell(row=primeira, column=3).coordinate
    planilha.auto_filter.ref = f"A{linha + 1}:O{ultima}"
    nota_rodape(planilha, ultima + 2,
                "'Odd provável na casa' é a odd justa reduzida pela margem medida, uma "
                "vez por perna — é uma estimativa do que a casa ofereceria, não uma "
                "cotação real. O retorno esperado é negativo em todos os bilhetes, e "
                "quanto mais alta a odd, pior: é assim que a conta funciona.", 15)
    return planilha



# ---------------------------------------------------------------------------
# 4. Programa principal
# ---------------------------------------------------------------------------

def main():
    inicio = time.time()
    previsoes, contexto = montar_previsoes()
    confiabilidade = medir_confiabilidade(contexto)

    # Colunas derivadas usadas pela calculadora e pelas abas de mercado.
    previsoes["dupla_1x"] = previsoes["prob_H"] + previsoes["prob_D"]
    previsoes["dupla_12"] = previsoes["prob_H"] + previsoes["prob_A"]
    previsoes["dupla_x2"] = previsoes["prob_D"] + previsoes["prob_A"]
    previsoes["menos_de_2_5"] = 1 - previsoes["mais_de_2_5"]
    previsoes["ambas_nao"] = 1 - previsoes["prob_ambas_marcam"]

    print("Montando a planilha...")
    livro = Workbook()
    livro.remove(livro.active)

    # Margem real do mercado, medida nas temporadas recentes: é ela que define
    # o quanto uma múltipla devolve à casa a cada perna adicionada.
    recentes = contexto["base"]
    recentes = recentes[recentes["jogada"] & (recentes["Season"] >= 2025)]
    margem_mercado = float(ub.probabilidades_mercado(recentes)[1].mean())
    print(f"Margem média do mercado (2025-2026): {100 * margem_mercado:.2f}%")

    candidatos = candidatos_por_partida(previsoes)
    melhores = melhores_por_partida(candidatos)
    bilhetes = montar_bilhetes(candidatos, margem_mercado)
    print(f"  {len(melhores)} partidas na vitrine e "
          f"{bilhetes['Bilhete'].groupby([bilhetes['Rodada'], bilhetes['Bilhete']]).ngroups} "
          f"bilhetes montados")

    aba_inicio(livro, previsoes, contexto, confiabilidade)
    aba_palpites(livro, previsoes)
    aba_rodadas(livro, previsoes)
    aba_melhores(livro, melhores)
    aba_bilhetes(livro, bilhetes, margem_mercado)

    aba_mercado(livro, previsoes, "Resultado 1X2", COR_RESULTADO,
                "Cada desfecho com a sua probabilidade e a odd justa correspondente. "
                "Dupla chance cobre dois dos três resultados.",
                [("Casa", "prob_H"), ("Empate", "prob_D"), ("Fora", "prob_A"),
                 ("1X", "dupla_1x"), ("12", "dupla_12"), ("X2", "dupla_x2")])

    aba_mercado(livro, previsoes, "Gols", COR_GOLS,
                "Linhas de mais/menos gols. 'Menos de 2,5' é o complemento de "
                "'mais de 2,5'.",
                [("+0,5", "mais_de_0_5"), ("+1,5", "mais_de_1_5"),
                 ("+2,5", "mais_de_2_5"), ("+3,5", "mais_de_3_5"),
                 ("+4,5", "mais_de_4_5"), ("-2,5", "menos_de_2_5")],
                numericos=[("Gols mandante", "gols_esperados_mandante"),
                           ("Gols visitante", "gols_esperados_visitante"),
                           ("Gols total", "gols_esperados_total")])

    aba_mercado(livro, previsoes, "Multi-Gols", COR_GOLS,
                "Chance de o total de gols da partida cair dentro de cada faixa.",
                [(f"{a}-{b}", f"multigols_{a}_{b}") for a, b in ub.FAIXAS_MULTIGOLS])

    aba_mercado(livro, previsoes, "Ambas Marcam", COR_AMBAS,
                "Ambas as equipes marcam, e a chance de cada lado marcar pelo menos "
                "um gol.",
                [("Sim", "prob_ambas_marcam"), ("Não", "ambas_nao"),
                 ("Mandante marca", "prob_mandante_marca"),
                 ("Visitante marca", "prob_visitante_marca")],
                aviso="ATENÇÃO: o modelo supõe que os gols dos dois times são "
                      "independentes e por isso SUBESTIMA este mercado em cerca de 4 "
                      "pontos percentuais. Some o viés antes de usar — ver aba "
                      "Confiabilidade.")

    aba_mercado(livro, previsoes, "Escanteios", COR_ESCANTEIOS,
                "Total de escanteios da partida (soma dos dois times) e as linhas de "
                "mais/menos.",
                [(f"+{v:.1f}".replace(".", ","),
                  f"escanteios_mais_{str(v).replace('.', '_')}")
                 for v in ub.LINHAS_ESCANTEIOS],
                numericos=[("Esc. mandante", "escanteios_mandante"),
                           ("Esc. visitante", "escanteios_visitante"),
                           ("Esc. total", "escanteios_total")],
                aviso="Confiança média: base de estatísticas cobre 2015 a 2026, com os "
                      "20 clubes. O viés fora da amostra é pequeno, mas oscila cerca de "
                      "±5 p.p. entre temporadas.")

    aba_mercado(livro, previsoes, "Cartões", COR_CARTOES,
                "Total de cartões da partida (amarelos mais vermelhos, somando os dois "
                "times) e as linhas de mais/menos.",
                [(f"+{v:.1f}".replace(".", ","),
                  f"cartoes_mais_{str(v).replace('.', '_')}")
                 for v in ub.LINHAS_CARTOES],
                numericos=[("Cart. mandante", "cartoes_mandante"),
                           ("Cart. visitante", "cartoes_visitante"),
                           ("Cart. total", "cartoes_total")],
                aviso="Confiança média: incluir as temporadas de 2024 a 2026 derrubou o "
                      "viés de 'mais de 4,5 cartões' de -4,3 para -1,8 ponto percentual "
                      "na validação fora da amostra.")

    aba_dados(livro, previsoes)
    aba_calculadora(livro, previsoes)
    aba_confiabilidade(livro, confiabilidade)
    aba_times(livro, contexto)

    # Impressão: paisagem ajustada à largura, cabeçalho repetido nas tabelas.
    for planilha in livro.worksheets:
        planilha.page_setup.orientation = "landscape"
        planilha.page_setup.fitToWidth = 1
        planilha.page_setup.fitToHeight = 0
        planilha.sheet_properties.pageSetUpPr.fitToPage = True
        planilha.print_options.horizontalCentered = True

    livro.active = 0
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    livro.save(ARQUIVO_SAIDA)
    print(f"\nPlanilha salva em {ARQUIVO_SAIDA.relative_to(ub.RAIZ)}")
    print(f"Abas: {', '.join(livro.sheetnames)}")
    print(f"Tempo total: {time.time() - inicio:.1f}s")

    previsoes.to_csv(ub.DIR_TABELAS / "palpites_completos.csv", index=False,
                     encoding="utf-8")
    confiabilidade.to_csv(ub.DIR_TABELAS / "confiabilidade_mercados.csv", index=False,
                          encoding="utf-8")
    melhores.to_csv(ub.DIR_TABELAS / "melhores_palpites.csv", index=False,
                    encoding="utf-8")
    bilhetes.to_csv(ub.DIR_TABELAS / "bilhetes.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
