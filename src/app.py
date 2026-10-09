import sys

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (SYSTEM_PROMPT, escolher_dispositivo, escolher_dtype,
                    extrair_json, montar_prompt)

# Configuração do modelo base e do adaptador LoRA treinado
MODELO = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "outputs/lr2e-4"

# Carregamento do modelo e preparação para inferência
device = escolher_dispositivo()
tok = AutoTokenizer.from_pretrained(MODELO)
base = AutoModelForCausalLM.from_pretrained(MODELO, torch_dtype=escolher_dtype()).to(device)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()


# Geração determinística de respostas para manter a consistência
def gerar(enc):
    return model.generate(**enc, max_new_tokens=60, do_sample=False,
                          temperature=None, top_p=None, top_k=None,
                          pad_token_id=tok.pad_token_id)


# Processamento do chamado e comparação entre o modelo base e o modelo treinado
def responder(texto, system, usar_lora):
    if not texto.strip():
        return "", {"erro": "Digite o texto do chamado."}
    prompt = montar_prompt(tok, texto, system)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        if usar_lora:
            saida = gerar(enc)
        else:
            with model.disable_adapter(): 
                saida = gerar(enc)

    # Decodificação da resposta e validação do JSON gerado
    resp = tok.decode(
        saida[0, enc["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    dados = extrair_json(resp)
    return resp, (dados if dados else {"erro": "A resposta não é um JSON válido."})


# Construção da interface web para testar e comparar as classificações
with gr.Blocks(title="Classificador de chamados de TI") as demo:
    gr.Markdown("# Classificador de chamados de suporte de TI")
    gr.Markdown(f"GPU em uso: **{torch.cuda.get_device_name(0)}** | Modelo treinado: `{ADAPTER}`")
    texto = gr.Textbox(label="Escreva o chamado (seu prompt)", lines=5,
                       placeholder="Ex.: Olá, meu notebook não liga e tenho reunião em 10 minutos.")
    usar_lora = gr.Checkbox(value=True, label="Usar modelo treinado (desmarque para ver o modelo original)")
    with gr.Accordion("Instrução do sistema (avançado)", open=False):
        system = gr.Textbox(value=SYSTEM_PROMPT, lines=4, label="Prompt do sistema")
    botao = gr.Button("Classificar", variant="primary")
    bruta = gr.Textbox(label="Resposta bruta do modelo")
    saida_json = gr.JSON(label="Resposta interpretada")

    # Vincula a classificação ao botão e apresenta os resultados
    botao.click(responder, inputs=[texto, system, usar_lora], outputs=[bruta, saida_json])


# Inicialização da aplicação em ambiente local
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)