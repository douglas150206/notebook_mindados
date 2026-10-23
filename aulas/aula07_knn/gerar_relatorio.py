"""Gera o relatório em PDF da atividade avaliativa da Aula 7 (KNN com a base Iris).

Refaz o mesmo experimento do notebook ``Aula7_KNN_atividade_avaliativa.ipynb``
(mesma divisão, mesmas sementes), salva os gráficos em ``figuras/`` e monta o
PDF. Nenhum número do relatório é digitado à mão.

Uso:
    python gerar_relatorio.py [--aluno "Nome Sobrenome"] [--saida relatorio.pdf]
"""
import argparse
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PASTA = Path(__file__).resolve().parent
FIGURAS = PASTA / "figuras"
VALORES_K = [1, 3, 5, 7, 9, 11, 15]
NOME_PESO = {"uniform": "uniforme", "distance": "distância"}
NOME_METRICA = {1: "Manhattan", 2: "euclidiana"}
ESTILOS_LINHA = {"uniforme + euclidiana": ("#2a78d6", "-"), "distância + euclidiana": ("#2a78d6", "--"),
                 "uniforme + Manhattan": ("#eb6834", "-"), "distância + Manhattan": ("#eb6834", "--")}


# ----------------------------------------------------------------------------
# Experimento (o mesmo do notebook)
# ----------------------------------------------------------------------------
def experimento():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=123, stratify=y)

    modelo = Pipeline([("escala", StandardScaler()), ("knn", KNeighborsClassifier())])
    tabela_k = pd.DataFrame({"K": VALORES_K, "acurácia treino": [
        clone(modelo).set_params(knn__n_neighbors=k).fit(X_treino, y_treino).score(X_treino, y_treino)
        for k in VALORES_K]})

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    grade = {"knn__n_neighbors": VALORES_K, "knn__weights": ["uniform", "distance"], "knn__p": [1, 2]}
    busca = GridSearchCV(modelo, grade, cv=cv, scoring="accuracy").fit(X_treino, y_treino)
    y_prev = busca.predict(X_teste)

    res = pd.DataFrame(busca.cv_results_)
    res["configuração"] = (res["param_knn__weights"].map(NOME_PESO) + " + "
                           + res["param_knn__p"].map(NOME_METRICA))
    comparacao = res.pivot_table(index="param_knn__n_neighbors", columns="configuração",
                                 values="mean_test_score").rename_axis("K")[list(ESTILOS_LINHA)]

    busca_arvore = GridSearchCV(DecisionTreeClassifier(random_state=123),
                                {"max_depth": [1, 2, 3, 4, 5, None]}, cv=cv, scoring="accuracy")
    busca_arvore.fit(X_treino, y_treino)
    y_prev_arvore = busca_arvore.predict(X_teste)

    return dict(
        iris=iris, X=X, n_treino=len(X_treino), n_teste=len(X_teste), tabela_k=tabela_k,
        melhor=busca.best_params_, cv_knn=busca.best_score_,
        relatorio=classification_report(y_teste, y_prev, target_names=iris.target_names, output_dict=True),
        acc_knn=accuracy_score(y_teste, y_prev), cm_knn=confusion_matrix(y_teste, y_prev),
        comparacao=comparacao, y_teste=y_teste, y_prev=y_prev, y_prev_arvore=y_prev_arvore,
        arvore=busca_arvore.best_params_, cv_arvore=busca_arvore.best_score_,
        acc_arvore=accuracy_score(y_teste, y_prev_arvore), cm_arvore=confusion_matrix(y_teste, y_prev_arvore),
    )


def salvar_figuras(E):
    FIGURAS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nome, (cor, linha) in ESTILOS_LINHA.items():
        ax.plot(E["comparacao"].index, E["comparacao"][nome], color=cor, linestyle=linha,
                marker="o", label=nome)
    ax.set_xticks(VALORES_K)
    ax.set_xlabel("K")
    ax.set_ylabel("Acurácia média na validação cruzada")
    ax.set_title("Validação cruzada por K, peso e distância")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.savefig(FIGURAS / "validacao_por_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    nomes = E["iris"].target_names
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for eixo, titulo, prev, acc in [(eixos[0], "KNN", E["y_prev"], E["acc_knn"]),
                                    (eixos[1], "Árvore de Decisão", E["y_prev_arvore"], E["acc_arvore"])]:
        ConfusionMatrixDisplay.from_predictions(E["y_teste"], prev, display_labels=nomes, cmap="Blues",
                                                colorbar=False, ax=eixo)
        eixo.set_title(f"{titulo}: acurácia {acc:.3f}".replace(".", ","))
        eixo.set_xlabel("Espécie prevista")
        eixo.set_ylabel("Espécie real")
    fig.tight_layout()
    fig.savefig(FIGURAS / "matrizes_confusao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fonte e estilos do PDF
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
            return "Texto", "Texto-Negrito", "Texto-Italico"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONTE, FONTE_NEGRITO, FONTE_ITALICO = registrar_fonte()
TINTA = colors.HexColor("#1f1e1c")
CINZA = colors.HexColor("#6b6a66")
AZUL = colors.HexColor("#1d5fae")
LINHA = colors.HexColor("#d9d8d4")
FUNDO_CABECALHO = colors.HexColor("#eef3fa")

ESTILOS = {
    "titulo": ParagraphStyle("titulo", fontName=FONTE_NEGRITO, fontSize=17, leading=21,
                             textColor=TINTA, spaceAfter=4),
    "subtitulo": ParagraphStyle("subtitulo", fontName=FONTE, fontSize=9.5, leading=13, textColor=CINZA),
    "h1": ParagraphStyle("h1", fontName=FONTE_NEGRITO, fontSize=12.5, leading=16, textColor=AZUL,
                         spaceBefore=11, spaceAfter=4),
    "corpo": ParagraphStyle("corpo", fontName=FONTE, fontSize=9.6, leading=13.6, textColor=TINTA,
                            spaceAfter=5),
    "item": ParagraphStyle("item", fontName=FONTE, fontSize=9.6, leading=13.6, textColor=TINTA,
                           leftIndent=12, bulletIndent=2, spaceAfter=2),
    "legenda": ParagraphStyle("legenda", fontName=FONTE_ITALICO, fontSize=8.3, leading=11, textColor=CINZA,
                              alignment=TA_CENTER, spaceBefore=2, spaceAfter=8),
    "celula": ParagraphStyle("celula", fontName=FONTE, fontSize=8.5, leading=10.5, textColor=TINTA),
}


def num(valor, casas=3):
    return f"{valor:.{casas}f}".replace(".", ",")


def p(texto, estilo="corpo"):
    return Paragraph(texto, ESTILOS[estilo])


def item(texto):
    return Paragraph(texto, ESTILOS["item"], bulletText="\u2022")


def figura(nome, legenda, largura):
    caminho = FIGURAS / nome
    w, h = ImageReader(str(caminho)).getSize()
    return KeepTogether([Image(str(caminho), width=largura, height=largura * h / w), p(legenda, "legenda")])


def tabela(cabecalho, linhas, larguras, legenda, alinhar_direita=()):
    direita = ParagraphStyle("celula_direita", parent=ESTILOS["celula"], alignment=2)

    def celula(texto, coluna):
        return Paragraph(str(texto), direita if coluna in alinhar_direita else ESTILOS["celula"])

    dados = [[celula(f"<b>{c}</b>", j) for j, c in enumerate(cabecalho)]]
    dados += [[celula(c, j) for j, c in enumerate(linha)] for linha in linhas]
    t = Table(dados, colWidths=larguras, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), FUNDO_CABECALHO),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, AZUL),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINHA),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return KeepTogether([t, p(legenda, "legenda")])


def rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONTE, 7.5)
    canvas.setFillColor(CINZA)
    canvas.drawString(2 * cm, 1.2 * cm, "Mineração de Dados \u2014 Aula 7 (KNN) \u2014 Atividade avaliativa")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"página {doc.page}")
    canvas.restoreState()


def descrever_erros(matriz, nomes):
    erros = [f"{matriz[i][j]} {nomes[i]} {'prevista' if matriz[i][j] == 1 else 'previstas'} como {nomes[j]}"
             for i in range(len(nomes)) for j in range(len(nomes)) if i != j and matriz[i][j]]
    if not erros:
        return "nenhum erro"
    return erros[0] if len(erros) == 1 else ", ".join(erros[:-1]) + " e " + erros[-1]


# ----------------------------------------------------------------------------
# Relatório
# ----------------------------------------------------------------------------
def montar(aluno, saida):
    E = experimento()
    salvar_figuras(E)

    especies = list(E["iris"].target_names)
    X = E["X"]
    n_teste = E["n_teste"]
    m = E["melhor"]
    k, peso, metrica = m["knn__n_neighbors"], NOME_PESO[m["knn__weights"]], NOME_METRICA[m["knn__p"]]
    acertos_knn = int(E["cm_knn"].trace())
    acertos_arvore = int(E["cm_arvore"].trace())
    diferenca = abs(acertos_knn - acertos_arvore)
    rel = E["relatorio"]
    comp = E["comparacao"]
    melhor_por_k = comp.max(axis=1)
    melhor_k_3_15 = melhor_por_k[melhor_por_k.index >= 3]
    ganho_distancia = pd.concat([comp["distância + euclidiana"] - comp["uniforme + euclidiana"],
                                 comp["distância + Manhattan"] - comp["uniforme + Manhattan"]])
    euclid_vs_manhattan = pd.concat([comp["uniforme + euclidiana"] - comp["uniforme + Manhattan"],
                                     comp["distância + euclidiana"] - comp["distância + Manhattan"]])
    k_maior = max(VALORES_K)
    treino_k1 = E["tabela_k"].set_index("K").loc[1, "acurácia treino"]
    desvios = X.std()

    doc = SimpleDocTemplate(str(saida), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                            title="Relatório \u2014 KNN com a base Iris", author=aluno or "",
                            subject="Mineração de Dados \u2014 Aula 7")
    h = [
        p("Classificação de espécies de Iris com KNN", "titulo"),
        p("Relatório da atividade avaliativa \u2014 Aula 7: K-Vizinhos Mais Próximos<br/>"
          "Mineração de Dados \u2014 Análise e Desenvolvimento de Sistemas \u2014 Profa. Ana Paula<br/>"
          f"Aluno(a): {aluno or '_______________________________________'}"
          f" &nbsp;&nbsp;|&nbsp;&nbsp; {date.today():%d/%m/%Y}", "subtitulo"),
        Spacer(1, 4),

        p("1. Objetivo e dados", "h1"),
        p(f"Classificar a espécie de flores de Iris com KNN. A base tem {len(X)} flores e 3 espécies "
          f"({', '.join(especies[:-1])} e {especies[-1]}), com 50 de cada. A classe-alvo é a espécie, e os "
          "quatro atributos são medidas em centímetros:"),
        tabela(["Atributo", "Significado", "Média (cm)", "Desvio (cm)"],
               [[c, s, num(X[c].mean(), 2), num(desvios[c], 2)] for c, s in zip(
                   X.columns, ["comprimento da sépala", "largura da sépala",
                               "comprimento da pétala", "largura da pétala"])],
               [4 * cm, 5 * cm, 3 * cm, 3 * cm], "Tabela 1. Atributos da base Iris.", alinhar_direita=(2, 3)),

        p("2. Método", "h1"),
        item(f"Divisão de 75% para treino ({E['n_treino']} flores) e 25% para teste ({n_teste} flores), "
             "com <font face='Courier'>random_state=123</font> e <font face='Courier'>stratify=y</font>."),
        item("<font face='Courier'>Pipeline</font> com <font face='Courier'>StandardScaler</font> e "
             "<font face='Courier'>KNeighborsClassifier</font>."),
        item("Validação cruzada estratificada de 5 folds, <b>só no treino</b>, para escolher K (1, 3, 5, 7, 9, "
             "11 e 15), o peso (uniforme ou por distância) e a distância (euclidiana ou Manhattan)."),
        item("Avaliação do modelo escolhido <b>uma única vez</b> no teste."),
        item("Árvore de Decisão no mesmo particionamento, com <font face='Courier'>max_depth</font> "
             "escolhido pela mesma validação cruzada."),

        p("3. Resultados", "h1"),
        tabela(["K", "Acurácia de treino"],
               [[int(r["K"]), num(r["acurácia treino"])] for _, r in E["tabela_k"].iterrows()],
               [2 * cm, 4 * cm], "Tabela 2. Acurácia de treino por K (pesos uniformes, euclidiana).",
               alinhar_direita=(0, 1)),
        p(f"A validação cruzada escolheu <b>K={k}, peso por {peso} e distância {metrica}</b>, com acurácia "
          f"média de <b>{num(E['cv_knn'])}</b>. No teste, o modelo acertou <b>{acertos_knn} de {n_teste} "
          f"flores (acurácia {num(E['acc_knn'])})</b>. Erros: {descrever_erros(E['cm_knn'], especies)}."),
        tabela(["Classe", "Precisão", "Recall", "F1-score", "Flores"],
               [[nome, num(rel[chave]["precision"]), num(rel[chave]["recall"]), num(rel[chave]["f1-score"]),
                 int(rel[chave]["support"])]
                for chave, nome in [(e, e) for e in especies] + [("macro avg", "média macro")]],
               [4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm], f"Tabela 3. Métricas do KNN no teste ({n_teste} flores).",
               alinhar_direita=(1, 2, 3, 4)),
        tabela(["K"] + list(comp.columns),
               [[int(kk)] + [num(v) for v in linha] for kk, linha in comp.iterrows()],
               [1.6 * cm] + [3.85 * cm] * 4,
               "Tabela 4. Acurácia média na validação cruzada por K, peso e distância.",
               alinhar_direita=(0, 1, 2, 3, 4)),
        figura("validacao_por_k.png", "Figura 1. Acurácia de validação cruzada por K, peso e distância.",
               12.5 * cm),
        tabela(["Modelo", "Parâmetros escolhidos", "Acurácia validação", "Acurácia teste", "Acertos no teste"],
               [["KNN", f"K={k}, peso por {peso}, {metrica}", num(E["cv_knn"]), num(E["acc_knn"]),
                 f"{acertos_knn} de {n_teste}"],
                ["Árvore de Decisão", f"max_depth={E['arvore']['max_depth']}", num(E["cv_arvore"]),
                 num(E["acc_arvore"]), f"{acertos_arvore} de {n_teste}"]],
               [3.2 * cm, 4.6 * cm, 3 * cm, 2.6 * cm, 3.2 * cm], "Tabela 5. KNN × Árvore de Decisão.",
               alinhar_direita=(2, 3, 4)),
        figura("matrizes_confusao.png", "Figura 2. Matrizes de confusão no teste.", 13.5 * cm),

        p("4. Conclusão", "h1"),
        item(f"<b>Desempenho:</b> o KNN escolhido (K={k}, peso por {peso}, {metrica}) obteve {num(E['cv_knn'])} "
             f"na validação e {num(E['acc_knn'])} no teste ({acertos_knn} de {n_teste}), com precisão, recall e "
             f"F1 macro de {num(rel['macro avg']['precision'])}, {num(rel['macro avg']['recall'])} e "
             f"{num(rel['macro avg']['f1-score'])}. Erros: {descrever_erros(E['cm_knn'], especies)}."),
        item(f"<b>Sensibilidade a K:</b> com K=1 a acurácia de treino é {num(treino_k1)}, porque cada flor é "
             "vizinha de si mesma e o modelo memoriza. De K=3 a K=15, a melhor validação de cada K fica entre "
             f"{num(melhor_k_3_15.min())} e {num(melhor_k_3_15.max())}, então vários valores de K são equivalentes."),
        item("<b>Pesos e distância:</b> o peso por distância "
             + ("nunca piorou a validação" if ganho_distancia.min() >= 0 else "não teve efeito de sinal fixo")
             + f" e ajudou mais com K grande (em K={k_maior}, {num(comp.loc[k_maior, 'distância + euclidiana'])} "
             f"contra {num(comp.loc[k_maior, 'uniforme + euclidiana'])} com peso uniforme, na euclidiana). "
             + ("Entre euclidiana e Manhattan não houve vencedor fixo."
                if euclid_vs_manhattan.min() < 0 < euclid_vs_manhattan.max()
                else "Uma das distâncias foi sempre igual ou melhor que a outra.")),
        item(f"<b>Normalização:</b> as quatro medidas estão em cm e têm escalas parecidas (desvio de "
             f"{num(desvios.min(), 2)} a {num(desvios.max(), 2)} cm), então a padronização pesa menos aqui do que "
             "em bases com unidades diferentes. Ela fica dentro do pipeline para que nenhum atributo domine a "
             "distância e para ser ajustada só com o treino."),
        item(f"<b>KNN × Árvore:</b> a árvore (max_depth={E['arvore']['max_depth']}) acertou {acertos_arvore} de "
             f"{n_teste} ({num(E['acc_arvore'])}). A diferença para o KNN é de {diferenca} "
             f"{'flor' if diferenca == 1 else 'flores'}, pequena demais para afirmar que um algoritmo é superior."),
        item(f"<b>Limitações e generalização:</b> o teste tem só {n_teste} flores (cada uma vale "
             f"{num(100 / n_teste, 1)} pontos), e a Iris é pequena, limpa e balanceada, um cenário favorável ao "
             "KNN. O KNN também guarda todo o treino e calcula distâncias a cada previsão, o que fica caro em "
             f"bases grandes. A proximidade entre validação ({num(E['cv_knn'])}) e teste ({num(E['acc_knn'])}) "
             "indica boa generalização para flores da mesma população, mas o número exato depende do sorteio "
             "treino/teste."),
    ]
    doc.build(h, onFirstPage=rodape, onLaterPages=rodape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aluno", default="", help="nome que aparece no cabeçalho do relatório")
    parser.add_argument("--saida", default=str(PASTA / "relatorio_aula7_knn.pdf"))
    args = parser.parse_args()
    montar(args.aluno, Path(args.saida))
    print("Relatório gerado em", args.saida)
