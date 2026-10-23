"""Gera o relatório em PDF da atividade avaliativa da Aula 7 (KNN com a base Iris).

Todos os números do relatório são lidos de ``resultados/`` e as figuras de
``figuras/``, ambos produzidos pela execução do notebook
``Aula7_KNN_resolvido.ipynb``. Execute o notebook antes deste script.

Uso:
    python gerar_relatorio.py [--aluno "Nome Sobrenome"] [--saida relatorio.pdf]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

PASTA = Path(__file__).resolve().parent
RESULTADOS = PASTA / "resultados"
FIGURAS = PASTA / "figuras"

# ----------------------------------------------------------------------------
# Fonte: uma TTF com acentos e o sinal de menos; sem ela, Helvetica.
# ----------------------------------------------------------------------------
CANDIDATAS = [
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
     "C:/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/arialbi.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
     "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"),
]


def registrar_fonte():
    for arquivos in CANDIDATAS:
        if all(Path(a).exists() for a in arquivos):
            nomes = ["Texto", "Texto-Negrito", "Texto-Italico", "Texto-NegritoItalico"]
            for nome, arquivo in zip(nomes, arquivos):
                pdfmetrics.registerFont(TTFont(nome, arquivo))
            pdfmetrics.registerFontFamily("Texto", normal=nomes[0], bold=nomes[1],
                                          italic=nomes[2], boldItalic=nomes[3])
            return "Texto", "Texto-Negrito", "Texto-Italico", "\u2212"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "-"


FONTE, FONTE_NEGRITO, FONTE_ITALICO, MENOS = registrar_fonte()

TINTA = colors.HexColor("#1f1e1c")
CINZA = colors.HexColor("#6b6a66")
AZUL = colors.HexColor("#1d5fae")
LINHA = colors.HexColor("#d9d8d4")
FUNDO_CABECALHO = colors.HexColor("#eef3fa")
FUNDO_DESTAQUE = colors.HexColor("#fdf1ea")

ESTILOS = {
    "titulo": ParagraphStyle("titulo", fontName=FONTE_NEGRITO, fontSize=17, leading=21,
                             textColor=TINTA, spaceAfter=4),
    "subtitulo": ParagraphStyle("subtitulo", fontName=FONTE, fontSize=9.5, leading=13,
                                textColor=CINZA),
    "h1": ParagraphStyle("h1", fontName=FONTE_NEGRITO, fontSize=12.5, leading=16,
                         textColor=AZUL, spaceBefore=12, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName=FONTE_NEGRITO, fontSize=10.5, leading=14,
                         textColor=TINTA, spaceBefore=8, spaceAfter=3),
    "corpo": ParagraphStyle("corpo", fontName=FONTE, fontSize=9.6, leading=13.6,
                            textColor=TINTA, spaceAfter=5),
    "item": ParagraphStyle("item", fontName=FONTE, fontSize=9.6, leading=13.6,
                           textColor=TINTA, leftIndent=12, bulletIndent=2, spaceAfter=2),
    "legenda": ParagraphStyle("legenda", fontName=FONTE_ITALICO, fontSize=8.3, leading=11,
                              textColor=CINZA, alignment=TA_CENTER, spaceBefore=2, spaceAfter=8),
    "celula": ParagraphStyle("celula", fontName=FONTE, fontSize=8.3, leading=10.5, textColor=TINTA),
}

LARGURA_UTIL = A4[0] - 4 * cm


# ----------------------------------------------------------------------------
# Formatação de números no padrão brasileiro
# ----------------------------------------------------------------------------
def num(valor, casas=3):
    if pd.isna(valor):
        return "\u2014"
    texto = f"{abs(valor):.{casas}f}".replace(".", ",")
    return (MENOS + texto) if valor < 0 and round(abs(valor), casas) != 0 else texto


def pontos(valor, casas=1, sinal=True):
    texto = num(valor, casas)
    return ("+" + texto) if sinal and valor > 0 and round(valor, casas) != 0 else texto


def lista_pt(itens):
    itens = [str(i) for i in itens]
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


# ----------------------------------------------------------------------------
# Blocos de página
# ----------------------------------------------------------------------------
def p(texto, estilo="corpo"):
    return Paragraph(texto, ESTILOS[estilo])


def figura(nome, legenda, largura=LARGURA_UTIL):
    caminho = FIGURAS / f"{nome}.png"
    w, h = ImageReader(str(caminho)).getSize()
    return KeepTogether([Image(str(caminho), width=largura, height=largura * h / w),
                         p(legenda, "legenda")])


def tabela(cabecalho, linhas, larguras, alinhar_direita=(), destacar_linhas=(), legenda=None):
    direita = ParagraphStyle("celula_direita", parent=ESTILOS["celula"], alignment=2)

    def celula(texto, coluna):
        return Paragraph(str(texto), direita if coluna in alinhar_direita else ESTILOS["celula"])

    dados = [[celula(f"<b>{c}</b>", j) for j, c in enumerate(cabecalho)]]
    dados += [[celula(c, j) for j, c in enumerate(linha)] for linha in linhas]
    t = Table(dados, colWidths=larguras, repeatRows=1, hAlign="LEFT")
    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), FUNDO_CABECALHO),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, AZUL),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINHA),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in destacar_linhas:
        estilo.append(("BACKGROUND", (0, i + 1), (-1, i + 1), FUNDO_DESTAQUE))
    t.setStyle(TableStyle(estilo))
    bloco = [t]
    if legenda:
        bloco.append(p(legenda, "legenda"))
    return KeepTogether(bloco)


def rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONTE, 7.5)
    canvas.setFillColor(CINZA)
    canvas.drawString(2 * cm, 1.2 * cm, "Mineração de Dados \u2014 Aula 7 (KNN) \u2014 Atividade avaliativa")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"página {doc.page}")
    canvas.restoreState()


def so_versicolor_virginica(matriz):
    """True se todos os erros da matriz forem trocas entre versicolor (1) e virginica (2)."""
    return all(matriz[i][j] == 0 for i in range(3) for j in range(3) if i != j and {i, j} != {1, 2})


def descrever_erros(matriz, nomes):
    erros = [f"{matriz[i][j]} {nomes[i]} {'prevista' if matriz[i][j] == 1 else 'previstas'} como {nomes[j]}"
             for i in range(len(nomes)) for j in range(len(nomes)) if i != j and matriz[i][j]]
    return lista_pt(erros) if erros else "nenhum erro"


# ----------------------------------------------------------------------------
# Relatório
# ----------------------------------------------------------------------------
def montar(aluno, saida):
    R = json.loads((RESULTADOS / "avaliativa_resumo.json").read_text(encoding="utf-8"))
    tabela_k = pd.read_csv(RESULTADOS / "avaliativa_tabela_k.csv")
    metricas = pd.read_csv(RESULTADOS / "avaliativa_metricas_teste.csv", index_col=0)
    grade = pd.read_csv(RESULTADOS / "avaliativa_grade_cv.csv", index_col=0)
    consolidado = pd.read_csv(RESULTADOS / "avaliativa_resultados.csv")
    normalizacao = pd.read_csv(RESULTADOS / "avaliativa_normalizacao.csv", index_col=0)
    repeticoes = pd.read_csv(RESULTADOS / "avaliativa_repeticoes.csv")

    iris = load_iris(as_frame=True)
    especies = list(iris.target_names)
    medias = iris.data.groupby(iris.target).mean()

    n_teste = R["n_teste"]
    pp_flor_teste = 100 / n_teste
    kp = R["knn_parametros"]
    nome_peso = {"uniform": "uniforme", "distance": "proporcional à distância"}[kp["weights"]]
    nome_metrica = {1: "Manhattan", 2: "euclidiana"}[kp["p"]]
    acertos_arvore = sum(R["arvore_matriz_confusao"][i][i] for i in range(3))
    diferenca_flores = R["knn_teste_acertos"] - acertos_arvore
    k_pior_cv = tabela_k.loc[tabela_k["acurácia CV (média)"].idxmin()]
    faixa_cv = tabela_k[tabela_k["K"] > 1]["acurácia CV (média)"]
    k_escolhidos = {int(k): v for k, v in R["repeticoes_k_escolhido"].items()}
    k_mais_comum = max(k_escolhidos, key=k_escolhidos.get)
    outros_k = [f"K={k}" for k in sorted(k_escolhidos) if k != k_mais_comum]
    sem_melhor = normalizacao.index[normalizacao["diferença (pontos)"] < -1e-9].tolist()
    empates_norm = normalizacao.index[normalizacao["diferença (pontos)"].abs() <= 1e-9].tolist()

    doc = SimpleDocTemplate(str(saida), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                            title="Relatório — KNN com a base Iris",
                            author=aluno or "", subject="Mineração de Dados — Aula 7")
    h = []

    # Cabeçalho
    h.append(p("Classificação de espécies de Iris com KNN", "titulo"))
    h.append(p("Relatório da atividade avaliativa \u2014 Aula 7: K-Vizinhos Mais Próximos<br/>"
               "Mineração de Dados \u2014 Análise e Desenvolvimento de Sistemas \u2014 Profa. Ana Paula<br/>"
               f"Aluno(a): {aluno or '_______________________________________'}"
               f" &nbsp;&nbsp;|&nbsp;&nbsp; Gerado em {date.today():%d/%m/%Y}", "subtitulo"))
    h.append(Spacer(1, 6))

    # 1. Objetivo
    h.append(p("1. Objetivo", "h1"))
    h.append(p("Classificar a espécie de flores de Iris a partir de quatro medidas, usando o algoritmo "
               "K-Vizinhos Mais Próximos (KNN), escolhendo seus hiperparâmetros de forma reproduzível e "
               "sem usar o conjunto de teste, e comparando o resultado com uma Árvore de Decisão. "
               "O relatório resume o experimento do notebook <i>Aula7_KNN_resolvido.ipynb</i> "
               "(Atividade 11); todos os números abaixo são lidos dos arquivos gerados por ele."))

    # 2. Dados
    h.append(p("2. Dados", "h1"))
    h.append(p(f"A base Iris tem <b>{len(iris.data)} flores</b> e <b>3 espécies</b> "
               f"({lista_pt(especies)}), com 50 flores de cada \u2014 uma base balanceada. "
               "Os quatro atributos são medidas em centímetros da <b>sépala</b> (folhas externas, que "
               "protegem o botão) e da <b>pétala</b> (folhas internas, coloridas). A classe-alvo é a espécie."))
    significados = ["comprimento da sépala", "largura da sépala", "comprimento da pétala", "largura da pétala"]
    h.append(tabela(["Atributo", "Significado"] + [f"Média {e}" for e in especies],
                    [[c, s] + [num(medias.loc[i, c], 2) for i in range(3)]
                     for c, s in zip(iris.data.columns, significados)],
                    [3.4 * cm, 3.8 * cm, 3.2 * cm, 3.3 * cm, 3.3 * cm], alinhar_direita=(2, 3, 4),
                    legenda="Tabela 1. Atributos e média por espécie (cm). As medidas da pétala "
                            "separam bem as espécies; as da sépala se sobrepõem."))

    # 3. Método
    h.append(p("3. Método", "h1"))
    for texto in [
        f"<b>Divisão:</b> 75% treino ({R['n_treino']} flores) e 25% teste ({n_teste} flores), "
        "com <font face='Courier'>random_state=123</font> e <font face='Courier'>stratify=y</font>.",
        "<b>Modelo:</b> <font face='Courier'>Pipeline(StandardScaler, KNeighborsClassifier)</font>. "
        "O padronizador fica dentro do pipeline, então só aprende média e desvio com o treino.",
        "<b>Escolha dos parâmetros:</b> <font face='Courier'>GridSearchCV</font> com validação cruzada "
        "estratificada de 5 folds <b>apenas no treino</b>, testando K = 1, 3, 5, 7, 9, 11 e 15, pesos "
        f"uniformes ou por distância e distância euclidiana ou Manhattan ({R['knn_total_combinacoes']} combinações).",
        "<b>Avaliação:</b> o modelo escolhido é avaliado <b>uma única vez</b> no teste "
        "(acurácia, precisão, recall, F1 e matriz de confusão).",
        "<b>Comparação:</b> Árvore de Decisão com <font face='Courier'>max_depth</font> escolhido pela "
        "mesma validação cruzada, no mesmo particionamento.",
        f"<b>Robustez:</b> o procedimento inteiro (escolha + teste) repetido em {R['repeticoes_n']} "
        "sorteios estratificados de treino e teste.",
    ]:
        h.append(Paragraph(texto, ESTILOS["item"], bulletText="\u2022"))

    # 4. Resultados
    h.append(p("4. Resultados", "h1"))

    h.append(p("4.1 Sensibilidade a K", "h2"))
    h.append(tabela(["K", "Acurácia de treino", "Validação cruzada (média)", "Validação cruzada (desvio)"],
                    [[int(r["K"]), num(r["acurácia treino"]), num(r["acurácia CV (média)"]),
                      num(r["acurácia CV (desvio)"])] for _, r in tabela_k.iterrows()],
                    [2 * cm, 4 * cm, 5 * cm, 5 * cm], alinhar_direita=(0, 1, 2, 3),
                    legenda="Tabela 2. KNN com pesos uniformes e distância euclidiana. "
                            "A validação usa só o treino."))
    h.append(figura("avaliativa_curva_k", "Figura 1. Acurácia de treino e de validação cruzada por K. "
                                          "A faixa clara é \u00b11 desvio-padrão entre os 5 folds.",
                    largura=12.5 * cm))
    h.append(p(f"Com K=1 a acurácia de treino é {num(tabela_k.iloc[0]['acurácia treino'])}: cada flor é a "
               "vizinha mais próxima de si mesma, então o modelo apenas memoriza. É também o K com a "
               f"pior validação ({num(k_pior_cv['acurácia CV (média)'])}). Para K de 3 a 15 a validação "
               f"fica entre {num(faixa_cv.min())} e {num(faixa_cv.max())}, uma variação menor que o "
               f"desvio entre os folds ({num(tabela_k['acurácia CV (desvio)'].min(), 2)} a "
               f"{num(tabela_k['acurácia CV (desvio)'].max(), 2)}): a curva é praticamente plana."))

    h.append(p("4.2 Modelo escolhido e avaliação no teste", "h2"))
    h.append(p(f"A validação cruzada escolheu <b>K={kp['n_neighbors']}, peso {nome_peso} e distância "
               f"{nome_metrica}</b>, com acurácia média de <b>{num(R['knn_cv_media'])}</b> "
               f"(desvio {num(R['knn_cv_desvio'])}). A escolha é apertada: "
               f"{R['knn_combinacoes_a_0001_do_melhor']} das {R['knn_total_combinacoes']} combinações "
               "ficam a menos de 0,001 da melhor. No teste, o modelo acertou "
               f"<b>{R['knn_teste_acertos']} de {n_teste} flores (acurácia {num(R['knn_teste_acuracia'])})</b>, "
               f"com precisão, recall e F1 macro de {num(R['knn_teste_precisao_macro'])}, "
               f"{num(R['knn_teste_recall_macro'])} e {num(R['knn_teste_f1_macro'])}. "
               f"Erros: {descrever_erros(R['knn_matriz_confusao'], especies)} "
               "(matriz de confusão na Figura 3)."))
    linhas_metricas = [[nome, num(metricas.loc[chave, "precision"]), num(metricas.loc[chave, "recall"]),
                        num(metricas.loc[chave, "f1-score"]), int(metricas.loc[chave, "support"])]
                       for chave, nome in [(e, e) for e in especies]
                       + [("macro avg", "média macro"), ("weighted avg", "média ponderada")]]
    h.append(tabela(["Classe", "Precisão", "Recall", "F1-score", "Flores"], linhas_metricas,
                    [4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm], alinhar_direita=(1, 2, 3, 4),
                    legenda=f"Tabela 3. Métricas do KNN escolhido no teste ({n_teste} flores)."))
    if R["knn_treino"] >= 0.9999 and kp["weights"] == "distance":
        h.append(p(f"A acurácia de treino desse modelo é {num(R['knn_treino'])}, mas isso <b>não</b> indica "
                   "memorização: com pesos por distância, cada flor de treino está a distância zero de si "
                   "mesma e recebe peso máximo, então a acurácia de treino é sempre 1. Nesse caso, a "
                   "validação cruzada é o indicador que importa."))

    h.append(p("4.3 Pesos e métrica de distância", "h2"))
    h.append(tabela(["K"] + list(grade.columns),
                    [[int(k)] + [num(v) for v in linha] for k, linha in grade.iterrows()],
                    [1.6 * cm] + [3.85 * cm] * 4, alinhar_direita=(0, 1, 2, 3, 4),
                    legenda="Tabela 4. Acurácia média de validação cruzada para cada combinação."))
    h.append(figura("avaliativa_pesos_metricas", "Figura 2. As quatro combinações de peso e métrica por K "
                                                 "(onde as linhas se sobrepõem, os resultados são iguais).",
                    largura=12.5 * cm))
    peso_txt = ("nunca piorou a validação" if R["peso_dif_min_pontos"] >= 0
                else f"variou de {pontos(R['peso_dif_min_pontos'])} a {pontos(R['peso_dif_max_pontos'])} pontos")
    h.append(p(f"Em média, o peso por distância ficou <b>{pontos(R['peso_dif_media_pontos'])} ponto</b> "
               f"em relação ao uniforme e {peso_txt}, com o maior ganho "
               f"({pontos(R['peso_dif_max_pontos'])} pontos) com K grande, quando vizinhos distantes "
               "passariam a votar com o mesmo peso dos próximos. Entre euclidiana e Manhattan a diferença "
               f"média foi de {pontos(R['metrica_dif_media_pontos'])} ponto, variando de "
               f"{pontos(R['metrica_dif_min_pontos'])} a {pontos(R['metrica_dif_max_pontos'])} conforme K, "
               "sem vencedor fixo. Na média dos folds, cada flor vale cerca de "
               f"{num(100 / R['n_treino'], 1)} ponto, então essas diferenças correspondem a poucas flores."))

    h.append(p("4.4 Comparação com a Árvore de Decisão", "h2"))
    linhas_cons = []
    destaques = []
    for i, r in consolidado.iterrows():
        cv = "\u2014" if pd.isna(r["acurácia CV (média)"]) else (
            num(r["acurácia CV (média)"]) + ("" if pd.isna(r["acurácia CV (desvio)"])
                                            else f" \u00b1 {num(r['acurácia CV (desvio)'])}"))
        linhas_cons.append([r["modelo"], r["configuração"], r["papel"].lower(), cv,
                            num(r["acurácia treino"]), num(r["acurácia teste"])])
        if r["papel"].startswith("ESCOLHID"):
            destaques.append(i)
    h.append(tabela(["Modelo", "Configuração", "Papel", "Validação (média \u00b1 desvio)", "Treino", "Teste"],
                    linhas_cons, [2.6 * cm, 4.6 * cm, 2.9 * cm, 3.1 * cm, 1.5 * cm, 1.5 * cm],
                    alinhar_direita=(3, 4, 5), destacar_linhas=destaques,
                    legenda="Tabela 5. Resultados reunidos. Linhas destacadas: modelos escolhidos pela "
                            "validação. Só os modelos escolhidos e as referências foram avaliados no teste."))
    h.append(figura("avaliativa_knn_vs_arvore", "Figura 3. Matrizes de confusão no teste: KNN escolhido "
                                                "e Árvore de Decisão escolhida.", largura=13.5 * cm))
    importancias = R["arvore_importancias"]
    atributo_top = max(importancias, key=importancias.get)
    h.append(p(f"A árvore escolhida pela validação (<font face='Courier'>max_depth={R['arvore_max_depth']}</font>, "
               f"validação {num(R['arvore_cv_media'])}) acertou <b>{acertos_arvore} de {n_teste}</b> flores "
               f"(acurácia {num(R['arvore_teste_acuracia'])})"
               + (", com erros também na fronteira versicolor \u00d7 virginica"
                  if so_versicolor_virginica(R["arvore_matriz_confusao"]) else "")
               + f", e usa quase só o atributo <i>{atributo_top}</i> "
               f"(importância {num(importancias[atributo_top])}). A diferença para o KNN é de "
               f"{diferenca_flores} {'flor' if abs(diferenca_flores) == 1 else 'flores'}, ou "
               f"{num(pp_flor_teste * diferenca_flores, 1)} pontos. "
               + ("Ela não é robusta: a árvore sem limite de profundidade, que a validação não escolheu, "
                  f"teria acurácia de teste {num(R['arvore_livre_teste_acuracia'])}."
                  if R["arvore_livre_teste_acuracia"] >= R["knn_teste_acuracia"] else
                  "A árvore sem limite de profundidade, que a validação não escolheu, teria acurácia de "
                  f"teste {num(R['arvore_livre_teste_acuracia'])}.")))

    h.append(p("4.5 Generalização em outros sorteios", "h2"))
    h.append(figura("avaliativa_repeticoes", f"Figura 4. Acurácia de teste do procedimento completo em "
                                             f"{R['repeticoes_n']} sorteios de treino/teste.",
                    largura=12 * cm))
    h.append(p(f"Repetindo o procedimento completo em {R['repeticoes_n']} sorteios, o KNN teve acurácia média "
               f"de <b>{num(R['repeticoes_knn_media'])}</b> (desvio {num(R['repeticoes_knn_desvio'])}) e a árvore "
               f"<b>{num(R['repeticoes_arvore_media'])}</b> (desvio {num(R['repeticoes_arvore_desvio'])}). "
               f"O KNN foi melhor em {R['repeticoes_knn_melhor']} sorteios, a árvore em "
               f"{R['repeticoes_arvore_melhor']} e houve empate em {R['repeticoes_empates']}. "
               f"O K escolhido também mudou de um sorteio para outro: K={k_mais_comum} em "
               f"{k_escolhidos[k_mais_comum]} deles"
               + (f", e os demais espalhados por {lista_pt(outros_k)}." if outros_k else ".")))

    # 5. Conclusão
    h.append(p("5. Conclusão", "h1"))
    norm_k = kp["n_neighbors"]
    conclusoes = [
        ("Desempenho", f"O KNN classificou a Iris muito bem: validação de {num(R['knn_cv_media'])} e "
                       f"{R['knn_teste_acertos']} de {n_teste} flores certas no teste. "
                       + ("A setosa é trivial; todos os erros ficam na fronteira versicolor \u00d7 virginica, "
                          "que se sobrepõem nas medidas." if so_versicolor_virginica(R["knn_matriz_confusao"])
                          else "Os erros estão detalhados na matriz de confusão (Figura 3).")),
        ("Sensibilidade a K", f"K=1 memoriza o treino e valida pior; de K=3 em diante a validação é "
                              "praticamente plana, com diferenças menores que a variação entre folds. "
                              "Não existe um \u201cK certo\u201d para a Iris, e sim uma faixa de valores razoáveis "
                              "\u2014 tanto que a validação escolheu Ks diferentes em sorteios diferentes."),
        ("Normalização", f"Na Iris a padronização não foi decisiva, porque os quatro atributos estão em "
                         f"centímetros e em escalas parecidas. No K escolhido ela "
                         + ("ajudou" if R["normalizacao_com_no_k_escolhido"] > R["normalizacao_sem_no_k_escolhido"]
                            else "não ajudou")
                         + f" ({num(R['normalizacao_com_no_k_escolhido'])} com \u00d7 "
                         f"{num(R['normalizacao_sem_no_k_escolhido'])} sem)"
                         + (f", mas com K={lista_pt(sem_melhor)} a versão sem padronização foi melhor"
                            if sem_melhor else "")
                         + (f" e com K={lista_pt(empates_norm)} houve empate" if empates_norm else "")
                         + ". Ainda assim, padronizar dentro do pipeline é a escolha segura: quando as escalas "
                           "são muito diferentes (como no exemplo sintético da aula, com um atributo multiplicado "
                           "por 1000), a falta dela derruba a acurácia."),
        ("Limitações", f"O teste tem só {n_teste} flores, então cada flor vale "
                       f"{num(pp_flor_teste, 1)} pontos e diferenças de uma ou duas flores não são conclusivas. "
                       "A Iris é pequena, limpa, balanceada e quase separável \u2014 o cenário mais favorável ao "
                       "KNN, que sofre com muitos atributos, atributos irrelevantes e ruído. O KNN também guarda "
                       "toda a base de treino e calcula distâncias a cada previsão, o que fica caro em bases "
                       "grandes, e não produz regras legíveis como a árvore."),
        ("Generalização", f"A proximidade entre validação ({num(R['knn_cv_media'])}) e teste "
                          f"({num(R['knn_teste_acuracia'])}), e a média de {num(R['repeticoes_knn_media'])} "
                          f"em {R['repeticoes_n']} sorteios, indicam que o modelo generaliza bem para flores da "
                          "mesma população. O número mais honesto a relatar não é a acurácia de um único "
                          f"sorteio, e sim algo em torno de {num(R['repeticoes_knn_media'], 2)}, com variação "
                          f"de cerca de \u00b1{num(100 * R['repeticoes_knn_desvio'], 0)} pontos. "
                          + ("A vantagem média sobre a árvore existe, mas é menor que a variação entre sorteios."
                             if 0 < R["repeticoes_knn_media"] - R["repeticoes_arvore_media"] < R["repeticoes_knn_desvio"]
                             else "A comparação com a árvore deve ser lida junto com essa variação.")),
    ]
    for titulo, texto in conclusoes:
        h.append(Paragraph(f"<b>{titulo}.</b> {texto}", ESTILOS["item"], bulletText="\u2022"))

    # 6. Reprodutibilidade
    h.append(p("6. Reprodutibilidade", "h1"))
    h.append(p("Sementes fixas (<font face='Courier'>random_state=123</font> na divisão, na validação "
               "cruzada e na árvore; sorteios de robustez com <font face='Courier'>random_state</font> de 0 a "
               f"{R['repeticoes_n'] - 1}). Para refazer: execute <i>Aula7_KNN_resolvido.ipynb</i> do início ao "
               "fim (ele grava <i>figuras/</i> e <i>resultados/</i>) e depois "
               "<font face='Courier'>python gerar_relatorio.py</font>. Referências: CASTRO e FERRARI, "
               "<i>Introdução à mineração de dados</i> (Saraiva, 2016), cap. 5, seção 5.3.1; "
               "documentação do Scikit-learn (Nearest Neighbors, KNeighborsClassifier, Pipeline)."))

    doc.build(h, onFirstPage=rodape, onLaterPages=rodape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aluno", default="", help="nome que aparece no cabeçalho do relatório")
    parser.add_argument("--saida", default=str(PASTA / "relatorio_aula7_knn.pdf"))
    args = parser.parse_args()
    montar(args.aluno, Path(args.saida))
    print("Relatório gerado em", args.saida)
