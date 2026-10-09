import argparse
import csv
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (carregar_jsonl, escolher_dispositivo, escolher_dtype,
                    extrair_json, montar_prompt)

# Configuração dos parâmetros para avaliação do modelo
p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--adapter", default=None)
p.add_argument("--nome", default="base_sem_treino")
p.add_argument("--batch", type=int, default=16)
args = p.parse_args()

# Carregamento do modelo base e aplicação opcional do adaptador LoRA
device = escolher_dispositivo()
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=escolher_dtype()).to(device)
if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
model.eval()

# Carregamento dos dados de teste e inicialização das métricas
dados = carregar_jsonl("data/test.jsonl")
ok_json = ok_cat = ok_prio = ok_ambos = 0
predicoes = []
t0 = time.time()

# Avaliação dos chamados em lotes para otimizar o processamento
for i in range(0, len(dados), args.batch):
    lote = dados[i : i + args.batch]
    prompts = [montar_prompt(tok, ex["texto"]) for ex in lote]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        saida = model.generate(**enc, max_new_tokens=40, do_sample=False,
                               temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.pad_token_id)
    textos = tok.batch_decode(saida[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for ex, txt in zip(lote, textos):
        j = extrair_json(txt)
        valido = isinstance(j, dict)
        cat = valido and j.get("categoria") == ex["categoria"]
        prio = valido and j.get("prioridade") == ex["prioridade"]
        ok_json += int(valido)
        ok_cat += int(cat)
        ok_prio += int(prio)
        ok_ambos += int(cat and prio)
        predicoes.append({"texto": ex["texto"], "esperado": [ex["categoria"], ex["prioridade"]], "resposta": txt})

# Cálculo das métricas de desempenho e do tempo total de execução
n = len(dados)
res = {
    "nome": args.nome,
    "json_valido": round(100 * ok_json / n, 1),
    "acerto_categoria": round(100 * ok_cat / n, 1),
    "acerto_prioridade": round(100 * ok_prio / n, 1),
    "acerto_total": round(100 * ok_ambos / n, 1),
    "tempo_s": round(time.time() - t0, 1),
}
print(json.dumps(res, indent=2, ensure_ascii=False))

# Armazenamento das predições individuais para análise posterior
Path("resultados").mkdir(exist_ok=True)
with open(f"resultados/{args.nome}_predicoes.jsonl", "w", encoding="utf-8") as f:
    for r in predicoes:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Registro consolidado das métricas para comparação entre experimentos
resumo = Path("resultados/resumo.csv")
nova = not resumo.exists()
with open(resumo, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if nova:
        w.writerow(list(res.keys()))
    w.writerow(list(res.values()))