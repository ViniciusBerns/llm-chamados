import json
import random
from pathlib import Path

random.seed(42)

# 7 frases por categoria: as 5 primeiras vão para treino, as 2 últimas só para teste
PROBLEMAS = {
    "Hardware": [
        "Meu notebook não liga, a luz de energia nem acende.",
        "O monitor fica piscando e depois apaga sozinho.",
        "O teclado parou de funcionar, várias teclas não respondem.",
        "O computador reinicia sozinho várias vezes ao dia.",
        "O mouse não é reconhecido quando conecto na porta USB.",
        "A ventoinha do computador faz um barulho muito alto e a máquina esquenta demais.",
        "A tela do notebook trincou e apareceram listras coloridas.",
    ],
    "Rede": [
        "Estou sem acesso à internet no meu computador.",
        "O Wi-Fi do andar fica caindo toda hora.",
        "Não consigo conectar na VPN para trabalhar de casa.",
        "As páginas da intranet demoram muito para abrir.",
        "O cabo de rede está conectado, mas aparece sem conexão.",
        "A videoconferência trava e a conexão oscila bastante.",
        "O roteador da sala está com as luzes piscando e ninguém consegue navegar.",
    ],
    "Software": [
        "O Excel fecha sozinho quando abro a planilha de vendas.",
        "Preciso instalar o Adobe Reader no meu computador.",
        "O sistema ERP mostra uma mensagem de erro ao emitir nota fiscal.",
        "O Word travou e perdi o documento que estava editando.",
        "O antivírus está bloqueando um programa que uso todo dia.",
        "O Teams não abre depois da última atualização do Windows.",
        "O programa de contabilidade fica congelado na tela de carregamento.",
    ],
    "Acesso": [
        "Esqueci minha senha de acesso ao sistema.",
        "Minha conta foi bloqueada depois de algumas tentativas de login.",
        "Preciso de permissão para acessar a pasta do departamento financeiro.",
        "O código de verificação em dois fatores não chega no meu celular.",
        "Sou funcionário novo e ainda não tenho login para os sistemas.",
        "Minha senha da rede expirou e não consigo cadastrar uma nova.",
        "Preciso que revoguem o acesso de um ex-colaborador aos nossos sistemas.",
    ],
    "E-mail": [
        "Não estou recebendo e-mails de clientes desde ontem.",
        "Meus e-mails ficam presos na caixa de saída e não são enviados.",
        "Minha caixa de entrada está cheia e não consigo receber mensagens novas.",
        "Muitos e-mails legítimos estão caindo na pasta de spam.",
        "Preciso configurar minha assinatura de e-mail no Outlook.",
        "Recebi um e-mail de erro dizendo que a mensagem não pôde ser entregue.",
        "O calendário do Outlook não sincroniza com meu celular.",
    ],
}

# 5 frases por prioridade: as 3 primeiras para treino, as 2 últimas só para teste
URGENCIAS = {
    "Alta": [
        "Isso é urgente: tenho uma reunião com um cliente em 15 minutos.",
        "Toda a equipe do setor está parada por causa disso.",
        "Estou impedido de trabalhar e preciso de solução imediata.",
        "É crítico: o fechamento do mês depende disso e o prazo acaba agora.",
        "A diretoria está esperando e não podemos atrasar.",
    ],
    "Média": [
        "Está atrapalhando meu trabalho, mas consigo contornar por enquanto.",
        "Preciso de uma solução até o fim da semana.",
        "Consigo trabalhar, só que mais devagar do que o normal.",
        "Se puderem resolver amanhã, já ajuda bastante.",
        "Isso incomoda, mas não paralisa as minhas tarefas.",
    ],
    "Baixa": [
        "Não é urgente, pode ser resolvido quando houver tempo.",
        "Sem pressa, só estou avisando para constar.",
        "Pode ficar para a próxima semana, não atrapalha em nada.",
        "É só uma melhoria que gostaria de ter, nada crítico.",
        "Não tenho nenhum prazo, resolvam quando der.",
    ],
}

NOMES = ["Ana", "Bruno", "Carla", "Diego", "Elisa", "Fábio", "Gabriela", "Henrique", "Isabela", "João"]
SETORES = ["financeiro", "comercial", "RH", "logística", "jurídico", "marketing", "atendimento", "diretoria"]


def gerar(problemas, urgencias, n):
    linhas = []
    for _ in range(n):
        cat = random.choice(list(problemas))
        prio = random.choice(list(urgencias))
        texto = (
            f"Olá, aqui é {random.choice(NOMES)} do setor {random.choice(SETORES)}. "
            f"{random.choice(problemas[cat])} {random.choice(urgencias[prio])}"
        )
        linhas.append({"texto": texto, "categoria": cat, "prioridade": prio})
    return linhas


def salvar(linhas, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")


Path("data").mkdir(exist_ok=True)
treino = gerar({c: v[:5] for c, v in PROBLEMAS.items()}, {p: v[:3] for p, v in URGENCIAS.items()}, 600)
teste = gerar({c: v[5:] for c, v in PROBLEMAS.items()}, {p: v[3:] for p, v in URGENCIAS.items()}, 100)
salvar(treino, "data/train.jsonl")
salvar(teste, "data/test.jsonl")
print(f"Treino: {len(treino)} exemplos | Teste: {len(teste)} exemplos")