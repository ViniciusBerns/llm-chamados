import json
import torch

# Define as instruções e o formato de resposta esperado do modelo
SYSTEM_PROMPT = (
    "Você é um assistente de suporte de TI. Classifique o chamado do usuário. "
    "Categorias possíveis: Hardware, Rede, Software, Acesso, E-mail. "
    "Prioridades possíveis: Alta, Média, Baixa. "
    'Responda somente com JSON no formato {"categoria": "...", "prioridade": "..."}.'
)


# Monta o prompt seguindo o formato de conversação do modelo
def montar_prompt(tokenizer, texto, system=SYSTEM_PROMPT):
    mensagens = [
        {"role": "system", "content": system},
        {"role": "user", "content": texto},
    ]
    return tokenizer.apply_chat_template(
        mensagens, tokenize=False, add_generation_prompt=True
    )


# Converte as classificações esperadas para o formato JSON
def resposta_json(ex):
    return json.dumps(
        {"categoria": ex["categoria"], "prioridade": ex["prioridade"]},
        ensure_ascii=False,
    )


# Extrai e valida o JSON presente na resposta gerada pelo modelo
def extrair_json(texto):
    ini, fim = texto.find("{"), texto.rfind("}")
    if ini == -1 or fim == -1 or fim < ini:
        return None
    try:
        return json.loads(texto[ini : fim + 1])
    except json.JSONDecodeError:
        return None


# Carrega os dados armazenados em arquivos JSONL
def carregar_jsonl(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]


# Verifica a disponibilidade da GPU para execução do modelo
def escolher_dispositivo():
    if not torch.cuda.is_available():
        raise SystemExit("ERRO: GPU não detectada. Verifique a instalação (Etapa 3).")
    return torch.device("cuda")


# Seleciona a precisão numérica compatível com a GPU
def escolher_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16