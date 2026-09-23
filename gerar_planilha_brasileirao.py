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

    print("Calculando previsões das partidas restantes...")
    linhas = []
    for numero, (mandante, visitante) in enumerate(
            zip(restantes["Home"], restantes["Away"]), start=1):
        lambda_casa, lambda_fora = modelo_gols.lambdas(mandante, visitante)
        matriz = ub.matriz_placares(lambda_casa, lambda_fora, modelo_gols.rho,
                                    modelo_gols.max_gols)
        linha = {"n": numero, "Mandante": mandante, "Visitante": visitante,
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
                  previstos, observados, n_testadas, "Baixa",
                  f"base só vai até 2023; oscilação de ±{desvio:.1f} p.p. entre temporadas")

    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 3. Construção da planilha
# ---------------------------------------------------------------------------

from openpyxl import Workbook  # noqa: E402
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule  # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.workbook.defined_name import DefinedName  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402
from openpyxl.worksheet.table import Table, TableStyleInfo  # noqa: E402

PERCENTUAL = "0.0%"
DECIMAL = "0.00"
COTACAO = "0.00"

FONTE_TITULO = Font(name=FONTE, size=16, bold=True, color=AZUL_ESCURO)
FONTE_SUBTITULO = Font(name=FONTE, size=11, color=CINZA_TEXTO)
FONTE_CABECALHO = Font(name=FONTE, size=10, bold=True, color=BRANCO)
FONTE_NORMAL = Font(name=FONTE, size=10)
FONTE_NEGRITO = Font(name=FONTE, size=10, bold=True)
FONTE_AVISO = Font(name=FONTE, size=10, bold=True, color="9C3A00")

FUNDO_CABECALHO = PatternFill("solid", fgColor=AZUL)
FUNDO_AVISO = PatternFill("solid", fgColor="FFF3CD")
FUNDO_ENTRADA = PatternFill("solid", fgColor="FFFF00")
FUNDO_SECAO = PatternFill("solid", fgColor=CINZA_CLARO)

BORDA_FINA = Border(*[Side(style="thin", color="D9D9D9")] * 4)


def escrever_cabecalho(planilha, cabecalhos, linha=1):
    for coluna, texto in enumerate(cabecalhos, start=1):
        celula = planilha.cell(row=linha, column=coluna, value=texto)
        celula.font = FONTE_CABECALHO
        celula.fill = FUNDO_CABECALHO
        celula.alignment = Alignment(horizontal="center", vertical="center",
                                     wrap_text=True)
        celula.border = BORDA_FINA
    planilha.row_dimensions[linha].height = 34


def ajustar_larguras(planilha, larguras):
    for indice, largura in enumerate(larguras, start=1):
        planilha.column_dimensions[get_column_letter(indice)].width = largura


def faixa_percentual(planilha, primeira_linha, ultima_linha, colunas):
    """Formata como percentual e aplica escala de cor (claro -> escuro)."""
    for coluna in colunas:
        letra = get_column_letter(coluna)
        for linha in range(primeira_linha, ultima_linha + 1):
            planilha.cell(row=linha, column=coluna).number_format = PERCENTUAL
        planilha.conditional_formatting.add(
            f"{letra}{primeira_linha}:{letra}{ultima_linha}",
            ColorScaleRule(start_type="num", start_value=0, start_color=BRANCO,
                           end_type="num", end_value=1, end_color=AZUL_CLARO))


def criar_tabela(planilha, nome, primeira_linha, ultima_linha, ultima_coluna):
    referencia = f"A{primeira_linha}:{get_column_letter(ultima_coluna)}{ultima_linha}"
    tabela = Table(displayName=nome, ref=referencia)
    tabela.tableStyleInfo = TableStyleInfo(
        name="TableStyleLight9", showRowStripes=True, showColumnStripes=False)
    planilha.add_table(tabela)


def aba_leiame(livro, previsoes, contexto, confiabilidade):
    planilha = livro.create_sheet("Leia-me")
    planilha.sheet_view.showGridLines = False
    ajustar_larguras(planilha, [3, 46, 30, 30, 22, 22])

    planilha["B2"] = "PALPITES DO BRASILEIRÃO SÉRIE A 2026"
    planilha["B2"].font = FONTE_TITULO
    planilha["B3"] = (f"Gerado em {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')} · "
                      f"dados até {contexto['data_corte'].date().strftime('%d/%m/%Y')} · "
                      f"{len(previsoes)} partidas restantes")
    planilha["B3"].font = FONTE_SUBTITULO

    linha = 5
    planilha[f"B{linha}"] = "O QUE TEM AQUI"
    planilha[f"B{linha}"].font = Font(name=FONTE, size=12, bold=True)
    linha += 1
    abas = [
        ("Palpites", "Todas as partidas com os principais mercados, uma linha por jogo. Comece por aqui."),
        ("Resultado 1X2", "Vitória, empate, derrota, dupla chance e a cotação mínima de cada um."),
        ("Gols", "Mais/menos de 0,5 a 4,5 gols, gols esperados e placar mais provável."),
        ("Multi-Gols", "Probabilidade de o total de gols cair em cada faixa (0-1, 1-2, 1-3...)."),
        ("Ambas Marcam", "Ambas marcam sim/não e a chance de cada equipe marcar."),
        ("Escanteios", "Escanteios esperados e linhas de mais/menos. CONFIANÇA BAIXA."),
        ("Cartões", "Cartões esperados e linhas de mais/menos. CONFIANÇA BAIXA."),
        ("Calculadora", "Escolha a partida e o mercado, digite a cotação e veja se vale a pena."),
        ("Confiabilidade", "Quanto cada mercado errou em testes fora da amostra."),
        ("Times", "Força de ataque e defesa de cada clube, em gols, escanteios e cartões."),
    ]
    planilha.cell(row=linha, column=2, value="Aba").font = FONTE_CABECALHO
    planilha.cell(row=linha, column=2).fill = FUNDO_CABECALHO
    planilha.cell(row=linha, column=3, value="Para que serve").font = FONTE_CABECALHO
    planilha.cell(row=linha, column=3).fill = FUNDO_CABECALHO
    planilha.merge_cells(start_row=linha, start_column=3, end_row=linha, end_column=6)
    linha += 1
    for nome, descricao in abas:
        planilha.cell(row=linha, column=2, value=nome).font = FONTE_NEGRITO
        planilha.cell(row=linha, column=3, value=descricao).font = FONTE_NORMAL
        planilha.merge_cells(start_row=linha, start_column=3, end_row=linha, end_column=6)
        linha += 1

    linha += 1
    planilha[f"B{linha}"] = "CONFIANÇA DE CADA FAMÍLIA DE MERCADO"
    planilha[f"B{linha}"].font = Font(name=FONTE, size=12, bold=True)
    linha += 1
    niveis = [
        ("Resultado, gols, multi-gols", "ALTA",
         "Modelo de Poisson com correção de Dixon-Coles, ajustado com TODAS as partidas "
         "até a rodada mais recente de 2026. Testado em 4.076 partidas fora da amostra."),
        ("Ambas as equipes marcam", "MÉDIA",
         "Mesmo modelo, mas ele supõe que os gols dos dois times são independentes e por "
         "isso SUBESTIMA este mercado em cerca de 4 pontos percentuais (valor medido)."),
        ("Escanteios e cartões", "BAIXA",
         "Vêm de outra base pública, que só tem estatística preenchida de 2015 a 2023 — "
         "não há 2024, 2025 nem 2026. Mirassol e Remo não têm nenhum histórico nela e "
         "caem na média da liga. Trate como ordem de grandeza, não como número fino."),
    ]
    for mercado, nivel, explicacao in niveis:
        planilha.cell(row=linha, column=2, value=mercado).font = FONTE_NEGRITO
        celula = planilha.cell(row=linha, column=3, value=nivel)
        celula.font = Font(name=FONTE, size=10, bold=True,
                           color={"ALTA": "1B7A4A", "MÉDIA": "9C6500",
                                  "BAIXA": "A32020"}[nivel])
        planilha.cell(row=linha, column=4, value=explicacao).font = FONTE_NORMAL
        planilha.merge_cells(start_row=linha, start_column=4, end_row=linha, end_column=6)
        planilha.row_dimensions[linha].height = 42
        planilha.cell(row=linha, column=4).alignment = Alignment(wrap_text=True,
                                                                 vertical="top")
        linha += 1

    linha += 1
    planilha[f"B{linha}"] = "COMO LER AS PROBABILIDADES"
    planilha[f"B{linha}"].font = Font(name=FONTE, size=12, bold=True)
    linha += 1
    for texto in [
        "A coluna de probabilidade é a chance de o evento acontecer segundo o modelo.",
        "A 'cotação mínima' é 1 dividido por essa probabilidade: é o preço de equilíbrio.",
        "Se a casa paga ACIMA da cotação mínima, a aposta tem valor esperado positivo.",
        "Se paga ABAIXO, tem valor esperado negativo — mesmo que o palpite acerte com frequência.",
        "A cotação NÃO muda a chance de acertar. Ela muda o quanto se recebe por acertar.",
    ]:
        planilha.cell(row=linha, column=2, value="•  " + texto).font = FONTE_NORMAL
        planilha.merge_cells(start_row=linha, start_column=2, end_row=linha, end_column=6)
        linha += 1

    linha += 1
    aviso = planilha.cell(row=linha, column=2)
    aviso.value = "AVISO IMPORTANTE"
    aviso.font = Font(name=FONTE, size=12, bold=True, color="9C3A00")
    linha += 1
    for texto in [
        "Esta planilha é um trabalho de análise de dados, não um sistema de apostas.",
        "A análise do próprio projeto mostrou que o retorno esperado é NEGATIVO em todas as",
        "faixas de cotação do Brasileirão (de -1,7% nas cotações baixas a -31,8% acima de 10),",
        "e que apostar só onde o modelo enxerga vantagem PIORA o resultado, não melhora.",
        "O mercado acerta mais que este modelo: 51,3% contra 48,9% de acurácia no 1X2.",
        "Aposta é entretenimento com custo esperado. Nenhuma linha aqui é recomendação.",
    ]:
        celula = planilha.cell(row=linha, column=2, value=texto)
        celula.font = FONTE_AVISO if texto.startswith("Esta") else FONTE_NORMAL
        celula.fill = FUNDO_AVISO
        planilha.merge_cells(start_row=linha, start_column=2, end_row=linha, end_column=6)
        for coluna in range(2, 7):
            planilha.cell(row=linha, column=coluna).fill = FUNDO_AVISO
        linha += 1

    linha += 1
    planilha.cell(row=linha, column=2,
                  value="Fonte dos gols e cotações: Football-Data.co.uk (via espelho público). "
                        "Fonte de escanteios e cartões: adaoduque/Brasileirao_Dataset (2015-2023).")
    planilha.cell(row=linha, column=2).font = FONTE_SUBTITULO
    planilha.merge_cells(start_row=linha, start_column=2, end_row=linha, end_column=6)
    return planilha


def aba_palpites(livro, previsoes):
    """Aba principal: uma linha por partida com os mercados mais usados."""
    planilha = livro.create_sheet("Palpites")
    cabecalhos = [
        "Nº", "Mandante", "Visitante", "Palpite 1X2", "Placar provável",
        "Casa", "Empate", "Fora", "1X", "12", "X2",
        "Gols esperados", "+1,5", "+2,5", "+3,5",
        "Ambas marcam", "Multi 1-3", "Multi 2-4",
        "Escanteios", "Esc +8,5", "Esc +9,5", "Cartões", "Cart +3,5", "Cart +4,5",
        "Palpite mais seguro", "Chance",
    ]
    escrever_cabecalho(planilha, cabecalhos)

    for indice, registro in enumerate(previsoes.itertuples(index=False), start=2):
        planilha.cell(row=indice, column=1, value=registro.n)
        planilha.cell(row=indice, column=2, value=registro.Mandante)
        planilha.cell(row=indice, column=3, value=registro.Visitante)
        planilha.cell(row=indice, column=4, value=registro.palpite_1x2)
        planilha.cell(row=indice, column=5, value=registro.placar_mais_provavel)
        planilha.cell(row=indice, column=6, value=registro.prob_H)
        planilha.cell(row=indice, column=7, value=registro.prob_D)
        planilha.cell(row=indice, column=8, value=registro.prob_A)
        # Dupla chance como fórmula: recalcula se a probabilidade mudar.
        planilha.cell(row=indice, column=9, value=f"=F{indice}+G{indice}")
        planilha.cell(row=indice, column=10, value=f"=F{indice}+H{indice}")
        planilha.cell(row=indice, column=11, value=f"=G{indice}+H{indice}")
        planilha.cell(row=indice, column=12, value=registro.gols_esperados_total)
        planilha.cell(row=indice, column=13, value=registro.mais_de_1_5)
        planilha.cell(row=indice, column=14, value=registro.mais_de_2_5)
        planilha.cell(row=indice, column=15, value=registro.mais_de_3_5)
        planilha.cell(row=indice, column=16, value=registro.prob_ambas_marcam)
        planilha.cell(row=indice, column=17, value=registro.multigols_1_3)
        planilha.cell(row=indice, column=18, value=registro.multigols_2_4)
        planilha.cell(row=indice, column=19, value=registro.escanteios_total)
        planilha.cell(row=indice, column=20, value=getattr(registro, "escanteios_mais_8_5"))
        planilha.cell(row=indice, column=21, value=getattr(registro, "escanteios_mais_9_5"))
        planilha.cell(row=indice, column=22, value=registro.cartoes_total)
        planilha.cell(row=indice, column=23, value=getattr(registro, "cartoes_mais_3_5"))
        planilha.cell(row=indice, column=24, value=getattr(registro, "cartoes_mais_4_5"))
        planilha.cell(row=indice, column=25, value=registro.palpite_seguro)
        planilha.cell(row=indice, column=26, value=registro.palpite_seguro_prob)

    ultima = len(previsoes) + 1
    faixa_percentual(planilha, 2, ultima,
                     [6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26])
    for coluna in (12, 19, 22):
        for linha in range(2, ultima + 1):
            planilha.cell(row=linha, column=coluna).number_format = DECIMAL

    for linha in range(2, ultima + 1):
        for coluna in range(1, len(cabecalhos) + 1):
            celula = planilha.cell(row=linha, column=coluna)
            celula.font = FONTE_NORMAL
            celula.border = BORDA_FINA
        planilha.cell(row=linha, column=4).font = FONTE_NEGRITO
        planilha.cell(row=linha, column=25).font = FONTE_NEGRITO

    ajustar_larguras(planilha, [5, 17, 17, 12, 13, 8, 8, 8, 8, 8, 8, 13, 8, 8, 8,
                                13, 10, 10, 11, 9, 9, 10, 9, 9, 24, 9])
    planilha.freeze_panes = "D2"
    planilha.auto_filter.ref = f"A1:{get_column_letter(len(cabecalhos))}{ultima}"

    # Destaca em laranja as colunas de confiança baixa (escanteios e cartões).
    for coluna in range(19, 25):
        celula = planilha.cell(row=1, column=coluna)
        celula.fill = PatternFill("solid", fgColor=LARANJA)

    nota = planilha.cell(row=ultima + 2, column=1)
    nota.value = ("Colunas em laranja (escanteios e cartões) têm CONFIANÇA BAIXA: "
                  "vêm de base que só cobre 2015-2023. Ver aba Leia-me.")
    nota.font = FONTE_AVISO
    return planilha


def aba_mercado(livro, previsoes, titulo, colunas, aviso=None, formatos=None):
    """Monta uma aba de detalhe de um mercado a partir de (cabeçalho, coluna)."""
    planilha = livro.create_sheet(titulo)
    primeira = 1
    if aviso:
        planilha["A1"] = aviso
        planilha["A1"].font = FONTE_AVISO
        planilha["A1"].fill = FUNDO_AVISO
        planilha.merge_cells(start_row=1, start_column=1, end_row=1,
                             end_column=len(colunas) + 3)
        for coluna in range(1, len(colunas) + 4):
            planilha.cell(row=1, column=coluna).fill = FUNDO_AVISO
        primeira = 3

    cabecalhos = ["Nº", "Mandante", "Visitante"] + [c for c, _ in colunas]
    escrever_cabecalho(planilha, cabecalhos, linha=primeira)

    for indice, registro in enumerate(previsoes.itertuples(index=False),
                                      start=primeira + 1):
        planilha.cell(row=indice, column=1, value=registro.n)
        planilha.cell(row=indice, column=2, value=registro.Mandante)
        planilha.cell(row=indice, column=3, value=registro.Visitante)
        for deslocamento, (_, campo) in enumerate(colunas, start=4):
            planilha.cell(row=indice, column=deslocamento,
                          value=getattr(registro, campo))

    ultima = primeira + len(previsoes)
    formatos = formatos or {}
    colunas_percentuais = [4 + i for i, (_, campo) in enumerate(colunas)
                           if formatos.get(campo, "pct") == "pct"]
    faixa_percentual(planilha, primeira + 1, ultima, colunas_percentuais)
    for i, (_, campo) in enumerate(colunas):
        if formatos.get(campo) == "num":
            for linha in range(primeira + 1, ultima + 1):
                planilha.cell(row=linha, column=4 + i).number_format = DECIMAL

    for linha in range(primeira + 1, ultima + 1):
        for coluna in range(1, len(cabecalhos) + 1):
            planilha.cell(row=linha, column=coluna).font = FONTE_NORMAL
            planilha.cell(row=linha, column=coluna).border = BORDA_FINA

    ajustar_larguras(planilha, [5, 17, 17] + [11] * len(colunas))
    planilha.freeze_panes = planilha.cell(row=primeira + 1, column=4).coordinate
    planilha.auto_filter.ref = (f"A{primeira}:"
                                f"{get_column_letter(len(cabecalhos))}{ultima}")
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


def aba_base_calculadora(livro, previsoes):
    """Matriz partida x mercado que alimenta a calculadora (aba de apoio)."""
    planilha = livro.create_sheet("Dados")
    planilha["A1"] = "Partida"
    for coluna, (rotulo, _) in enumerate(MERCADOS_CALCULADORA, start=2):
        planilha.cell(row=1, column=coluna, value=rotulo)

    for linha, registro in enumerate(previsoes.itertuples(index=False), start=2):
        planilha.cell(row=linha, column=1,
                      value=f"{registro.Mandante} x {registro.Visitante}")
        for coluna, (_, campo) in enumerate(MERCADOS_CALCULADORA, start=2):
            planilha.cell(row=linha, column=coluna, value=getattr(registro, campo))
            planilha.cell(row=linha, column=coluna).number_format = PERCENTUAL

    ajustar_larguras(planilha, [34] + [16] * len(MERCADOS_CALCULADORA))
    planilha.sheet_state = "hidden"
    return planilha


def aba_calculadora(livro, previsoes):
    """Calculadora de valor: fórmulas vivas, o usuário só digita a cotação."""
    planilha = livro.create_sheet("Calculadora")
    planilha.sheet_view.showGridLines = False
    ajustar_larguras(planilha, [3, 38, 20, 16, 16, 16, 16, 16])

    n = len(previsoes)
    ultima_coluna = get_column_letter(1 + len(MERCADOS_CALCULADORA))

    planilha["B2"] = "CALCULADORA DE VALOR DA APOSTA"
    planilha["B2"].font = FONTE_TITULO
    planilha["B3"] = ("Escolha a partida e o mercado nas listas, digite a cotação "
                      "que a casa oferece na célula amarela e leia o veredicto.")
    planilha["B3"].font = FONTE_SUBTITULO

    rotulos = [
        ("B5", "Partida", None),
        ("B6", "Mercado", None),
        ("B7", "Probabilidade do modelo", f"=IFERROR(INDEX(Dados!$B$2:${ultima_coluna}${n+1},"
                                          f"MATCH($C$5,Dados!$A$2:$A${n+1},0),"
                                          f"MATCH($C$6,Dados!$B$1:${ultima_coluna}$1,0)),\"\")"),
        ("B8", "Cotação mínima (equilíbrio)", '=IFERROR(1/$C$7,"")'),
        ("B9", "Cotação oferecida pela casa", None),
        ("B10", "Valor esperado por real apostado", '=IFERROR($C$7*$C$9-1,"")'),
        ("B11", "Vantagem sobre o preço", '=IFERROR($C$7-1/$C$9,"")'),
        ("B12", "Lucro se vier green (por R$ 1)", '=IFERROR($C$9-1,"")'),
        ("B13", "Veredicto",
         '=IF($C$7="","preencha os campos",'
         'IF($C$9="","digite a cotação",'
         'IF($C$10>0,"COTAÇÃO ACIMA DO EQUILÍBRIO","cotação abaixo do equilíbrio")))'),
        ("B14", "Resultado esperado em 100 apostas de R$ 10",
         '=IFERROR($C$10*1000,"")'),
    ]
    for celula, rotulo, formula in rotulos:
        planilha[celula] = rotulo
        planilha[celula].font = FONTE_NEGRITO
        destino = celula.replace("B", "C")
        if formula:
            planilha[destino] = formula
        planilha[destino].font = FONTE_NORMAL

    planilha["C5"] = f"{previsoes.iloc[0]['Mandante']} x {previsoes.iloc[0]['Visitante']}"
    planilha["C6"] = MERCADOS_CALCULADORA[2][0]
    planilha["C9"] = 3.0
    for entrada in ("C5", "C6", "C9"):
        planilha[entrada].fill = FUNDO_ENTRADA
        planilha[entrada].font = Font(name=FONTE, size=10, bold=True, color="0000FF")
        planilha[entrada].border = BORDA_FINA

    planilha["C7"].number_format = PERCENTUAL
    for celula in ("C8", "C9", "C12"):
        planilha[celula].number_format = COTACAO
    for celula in ("C10", "C11"):
        planilha[celula].number_format = "0.0%"
    planilha["C14"].number_format = 'R$ #,##0.00;[Red]-R$ #,##0.00'
    planilha["C13"].font = FONTE_NEGRITO

    planilha.conditional_formatting.add("C10", CellIsRule(
        operator="greaterThan", formula=["0"],
        font=Font(name=FONTE, size=10, bold=True, color="1B7A4A")))
    planilha.conditional_formatting.add("C10", CellIsRule(
        operator="lessThanOrEqual", formula=["0"],
        font=Font(name=FONTE, size=10, bold=True, color="A32020")))

    # As listas apontam para INTERVALOS NOMEADOS em vez de referenciar a aba
    # oculta diretamente: algumas versões do Excel recusam validação de dados
    # que aponta para outra planilha sem passar por um nome definido.
    livro_atual = planilha.parent
    livro_atual.defined_names.add(
        DefinedName("Lista_Partidas", attr_text=f"Dados!$A$2:$A${n+1}"))
    livro_atual.defined_names.add(
        DefinedName("Lista_Mercados", attr_text=f"Dados!$B$1:${ultima_coluna}$1"))

    validacao_partida = DataValidation(type="list", formula1="=Lista_Partidas",
                                       allow_blank=False)
    planilha.add_data_validation(validacao_partida)
    validacao_partida.add(planilha["C5"])

    validacao_mercado = DataValidation(type="list", formula1="=Lista_Mercados",
                                       allow_blank=False)
    planilha.add_data_validation(validacao_mercado)
    validacao_mercado.add(planilha["C6"])

    linha = 17
    planilha.cell(row=linha, column=2,
                  value="O QUE MUDA QUANDO A COTAÇÃO MUDA (mesma partida e mercado)")
    planilha.cell(row=linha, column=2).font = Font(name=FONTE, size=12, bold=True)
    linha += 1
    planilha.cell(row=linha, column=2,
                  value="Repare que a chance de acertar é a MESMA em todas as linhas. "
                        "O que muda é o preço.")
    planilha.cell(row=linha, column=2).font = FONTE_SUBTITULO
    linha += 2

    escrever_cabecalho_simulacao = ["Cotação", "Chance de green", "Valor esperado",
                                    "Lucro se green", "Vale a pena?"]
    for coluna, texto in enumerate(escrever_cabecalho_simulacao, start=2):
        celula = planilha.cell(row=linha, column=coluna, value=texto)
        celula.font = FONTE_CABECALHO
        celula.fill = FUNDO_CABECALHO
        celula.alignment = Alignment(horizontal="center")
    primeira_simulacao = linha + 1

    for deslocamento, cotacao in enumerate([1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0,
                                            5.0, 7.0, 10.0]):
        atual = primeira_simulacao + deslocamento
        planilha.cell(row=atual, column=2, value=cotacao).number_format = COTACAO
        planilha.cell(row=atual, column=3, value="=$C$7").number_format = PERCENTUAL
        planilha.cell(row=atual, column=4,
                      value=f"=IFERROR($C$7*B{atual}-1,\"\")").number_format = "0.0%"
        planilha.cell(row=atual, column=5,
                      value=f"=B{atual}-1").number_format = COTACAO
        planilha.cell(row=atual, column=6,
                      value=f'=IF(D{atual}="","",IF(D{atual}>0,"sim","não"))')
        for coluna in range(2, 7):
            planilha.cell(row=atual, column=coluna).font = FONTE_NORMAL
            planilha.cell(row=atual, column=coluna).border = BORDA_FINA

    ultima_simulacao = primeira_simulacao + 9
    planilha.conditional_formatting.add(
        f"D{primeira_simulacao}:D{ultima_simulacao}",
        CellIsRule(operator="greaterThan", formula=["0"],
                   fill=PatternFill("solid", fgColor="D6F0E0")))
    planilha.conditional_formatting.add(
        f"D{primeira_simulacao}:D{ultima_simulacao}",
        CellIsRule(operator="lessThanOrEqual", formula=["0"],
                   fill=PatternFill("solid", fgColor="FBE0E0")))

    aviso = planilha.cell(row=ultima_simulacao + 2, column=2)
    aviso.value = ("Lembrete: valor esperado positivo segundo o modelo não garante lucro. "
                   "No teste histórico do projeto, apostar só onde o modelo via vantagem "
                   "deu retorno PIOR do que apostar em tudo.")
    aviso.font = FONTE_AVISO
    planilha.merge_cells(start_row=ultima_simulacao + 2, start_column=2,
                         end_row=ultima_simulacao + 2, end_column=8)
    return planilha


def aba_confiabilidade(livro, confiabilidade):
    planilha = livro.create_sheet("Confiabilidade")
    planilha.sheet_view.showGridLines = False
    ajustar_larguras(planilha, [30, 14, 12, 12, 14, 12, 58])

    planilha["A1"] = "QUANTO CADA MERCADO ERRA"
    planilha["A1"].font = FONTE_TITULO
    planilha["A2"] = ("Viés = previsto menos observado, medido FORA DA AMOSTRA: o modelo "
                      "treina até a temporada anterior e é conferido na seguinte. "
                      "Negativo significa que o modelo subestima o mercado.")
    planilha["A2"].font = FONTE_SUBTITULO
    planilha.merge_cells("A2:G2")

    cabecalhos = list(confiabilidade.columns)
    escrever_cabecalho(planilha, cabecalhos, linha=4)
    for linha, registro in enumerate(confiabilidade.itertuples(index=False), start=5):
        for coluna, valor in enumerate(registro, start=1):
            celula = planilha.cell(row=linha, column=coluna, value=valor)
            celula.font = FONTE_NORMAL
            celula.border = BORDA_FINA
        planilha.cell(row=linha, column=2).number_format = PERCENTUAL
        planilha.cell(row=linha, column=3).number_format = PERCENTUAL
        planilha.cell(row=linha, column=4).number_format = "+0.00;-0.00"
        planilha.cell(row=linha, column=7).alignment = Alignment(wrap_text=True,
                                                                 vertical="top")
    ultima = 4 + len(confiabilidade)
    planilha.conditional_formatting.add(f"D5:D{ultima}", CellIsRule(
        operator="lessThan", formula=["-3"],
        fill=PatternFill("solid", fgColor="FBE0E0")))
    planilha.conditional_formatting.add(f"F5:F{ultima}", CellIsRule(
        operator="equal", formula=['"Baixa"'],
        fill=PatternFill("solid", fgColor="FFF3CD")))
    planilha.freeze_panes = "A5"
    planilha.auto_filter.ref = f"A4:G{ultima}"
    return planilha


def aba_times(livro, contexto):
    planilha = livro.create_sheet("Times")
    planilha.sheet_view.showGridLines = False

    modelo_gols = contexto["modelo_gols"]
    modelo_escanteios = contexto["modelo_escanteios"]
    modelo_cartoes = contexto["modelo_cartoes"]
    times = contexto["times_2026"]
    classificacao = contexto["classificacao"].set_index("time")

    linhas = []
    for time in times:
        tem_stats = time in modelo_escanteios.feito
        linhas.append({
            "Time": time,
            "Posição": int(classificacao.loc[time, "posicao"]),
            "Pontos": int(classificacao.loc[time, "pontos"]),
            "Ataque (gols)": float(np.exp(modelo_gols.ataque.get(time, 0.0))),
            "Defesa (gols)": float(np.exp(modelo_gols.defesa.get(time, 0.0))),
            "Escanteios a favor": float(np.exp(modelo_escanteios.feito.get(time, 0.0))),
            "Escanteios contra": float(np.exp(modelo_escanteios.sofrido.get(time, 0.0))),
            "Cartões tomados": float(np.exp(modelo_cartoes.feito.get(time, 0.0))),
            "Histórico de estatísticas?": "sim" if tem_stats else "NÃO (usa média da liga)",
        })
    tabela = pd.DataFrame(linhas).sort_values("Posição")

    planilha["A1"] = "FORÇA DOS CLUBES"
    planilha["A1"].font = FONTE_TITULO
    planilha["A2"] = ("Valores relativos à média da liga: 1,00 é a média. Ataque acima de "
                      "1,00 marca mais que a média; defesa abaixo de 1,00 sofre menos. "
                      "Escanteios e cartões vêm da base 2015-2023.")
    planilha["A2"].font = FONTE_SUBTITULO
    planilha.merge_cells("A2:I2")

    escrever_cabecalho(planilha, list(tabela.columns), linha=4)
    for linha, registro in enumerate(tabela.itertuples(index=False), start=5):
        for coluna, valor in enumerate(registro, start=1):
            celula = planilha.cell(row=linha, column=coluna, value=valor)
            celula.font = FONTE_NORMAL
            celula.border = BORDA_FINA
            if coluna in (4, 5, 6, 7, 8):
                celula.number_format = DECIMAL
    ultima = 4 + len(tabela)
    for coluna in (4, 6, 8):
        letra = get_column_letter(coluna)
        planilha.conditional_formatting.add(
            f"{letra}5:{letra}{ultima}",
            ColorScaleRule(start_type="min", start_color=BRANCO,
                           end_type="max", end_color=AZUL_CLARO))
    planilha.conditional_formatting.add(f"I5:I{ultima}", CellIsRule(
        operator="notEqual", formula=['"sim"'],
        fill=PatternFill("solid", fgColor="FFF3CD")))
    ajustar_larguras(planilha, [18, 10, 9, 14, 14, 18, 18, 16, 26])
    planilha.freeze_panes = "A5"
    return planilha


# ---------------------------------------------------------------------------
# 4. Programa principal
# ---------------------------------------------------------------------------

def main():
    inicio = time.time()
    previsoes, contexto = montar_previsoes()
    confiabilidade = medir_confiabilidade(contexto)

    # Colunas derivadas usadas pela calculadora.
    previsoes["dupla_1x"] = previsoes["prob_H"] + previsoes["prob_D"]
    previsoes["dupla_12"] = previsoes["prob_H"] + previsoes["prob_A"]
    previsoes["dupla_x2"] = previsoes["prob_D"] + previsoes["prob_A"]
    previsoes["menos_de_2_5"] = 1 - previsoes["mais_de_2_5"]
    previsoes["ambas_nao"] = 1 - previsoes["prob_ambas_marcam"]

    print("Montando a planilha...")
    livro = Workbook()
    livro.remove(livro.active)

    aba_leiame(livro, previsoes, contexto, confiabilidade)
    aba_palpites(livro, previsoes)

    aba_mercado(livro, previsoes, "Resultado 1X2", [
        ("Casa", "prob_H"), ("Empate", "prob_D"), ("Fora", "prob_A"),
        ("1X", "dupla_1x"), ("12", "dupla_12"), ("X2", "dupla_x2"),
        ("Placar provável", "placar_mais_provavel"),
        ("Chance do placar", "prob_placar_mais_provavel"),
    ], formatos={"placar_mais_provavel": "texto"})

    aba_mercado(livro, previsoes, "Gols", [
        ("Gols esp. mandante", "gols_esperados_mandante"),
        ("Gols esp. visitante", "gols_esperados_visitante"),
        ("Gols esp. total", "gols_esperados_total"),
        ("+0,5", "mais_de_0_5"), ("+1,5", "mais_de_1_5"), ("+2,5", "mais_de_2_5"),
        ("+3,5", "mais_de_3_5"), ("+4,5", "mais_de_4_5"),
        ("-2,5", "menos_de_2_5"),
    ], formatos={"gols_esperados_mandante": "num", "gols_esperados_visitante": "num",
                 "gols_esperados_total": "num"})

    aba_mercado(livro, previsoes, "Multi-Gols",
                [(f"{a}-{b}", f"multigols_{a}_{b}") for a, b in ub.FAIXAS_MULTIGOLS])

    aba_mercado(livro, previsoes, "Ambas Marcam", [
        ("Ambas SIM", "prob_ambas_marcam"), ("Ambas NÃO", "ambas_nao"),
        ("Mandante marca", "prob_mandante_marca"),
        ("Visitante marca", "prob_visitante_marca"),
    ], aviso="ATENÇÃO: este mercado é SUBESTIMADO pelo modelo em cerca de 4 pontos "
             "percentuais (viés medido fora da amostra). Ver aba Confiabilidade.")

    aba_mercado(livro, previsoes, "Escanteios",
                [("Esc. mandante", "escanteios_mandante"),
                 ("Esc. visitante", "escanteios_visitante"),
                 ("Esc. total", "escanteios_total")]
                + [(f"+{v:.1f}".replace(".", ","),
                    f"escanteios_mais_{str(v).replace('.', '_')}")
                   for v in ub.LINHAS_ESCANTEIOS],
                aviso="CONFIANÇA BAIXA: escanteios vêm de base que só cobre 2015-2023. "
                      "Mirassol e Remo não têm histórico e usam a média da liga.",
                formatos={"escanteios_mandante": "num", "escanteios_visitante": "num",
                          "escanteios_total": "num"})

    aba_mercado(livro, previsoes, "Cartões",
                [("Cart. mandante", "cartoes_mandante"),
                 ("Cart. visitante", "cartoes_visitante"),
                 ("Cart. total", "cartoes_total")]
                + [(f"+{v:.1f}".replace(".", ","),
                    f"cartoes_mais_{str(v).replace('.', '_')}")
                   for v in ub.LINHAS_CARTOES],
                aviso="CONFIANÇA BAIXA: cartões vêm de base que só cobre 2015-2023 e o "
                      "modelo SUBESTIMA as linhas em cerca de 4 pontos percentuais. "
                      "Mirassol e Remo usam a média da liga.",
                formatos={"cartoes_mandante": "num", "cartoes_visitante": "num",
                          "cartoes_total": "num"})

    aba_base_calculadora(livro, previsoes)
    aba_calculadora(livro, previsoes)
    aba_confiabilidade(livro, confiabilidade)
    aba_times(livro, contexto)

    # Impressão: paisagem, ajustada à largura, com o cabeçalho repetindo em
    # todas as páginas nas abas que são tabelas longas.
    abas_tabela = {"Palpites", "Resultado 1X2", "Gols", "Multi-Gols",
                   "Ambas Marcam", "Escanteios", "Cartões", "Confiabilidade",
                   "Times"}
    for planilha in livro.worksheets:
        planilha.page_setup.orientation = "landscape"
        planilha.page_setup.fitToWidth = 1
        planilha.page_setup.fitToHeight = 0
        planilha.sheet_properties.pageSetUpPr.fitToPage = True
        planilha.print_options.horizontalCentered = True
        if planilha.title in abas_tabela:
            linha_cabecalho = 3 if planilha.title in ("Ambas Marcam", "Escanteios",
                                                      "Cartões") else 1
            if planilha.title in ("Confiabilidade", "Times"):
                linha_cabecalho = 4
            planilha.print_title_rows = f"{linha_cabecalho}:{linha_cabecalho}"

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    livro.save(ARQUIVO_SAIDA)
    print(f"\nPlanilha salva em {ARQUIVO_SAIDA.relative_to(ub.RAIZ)}")
    print(f"Abas: {', '.join(livro.sheetnames)}")
    print(f"Tempo total: {time.time() - inicio:.1f}s")

    previsoes.to_csv(ub.DIR_TABELAS / "palpites_completos.csv", index=False,
                     encoding="utf-8")
    confiabilidade.to_csv(ub.DIR_TABELAS / "confiabilidade_mercados.csv", index=False,
                          encoding="utf-8")


if __name__ == "__main__":
    main()
