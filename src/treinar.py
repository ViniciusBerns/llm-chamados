import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import (carregar_jsonl, escolher_dispositivo, escolher_dtype,
                    montar_prompt, resposta_json)

# Configuração dos hiperparâmetros para os experimentos de treinamento
p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--run_name", default="teste1")
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--accum", type=int, default=2)
p.add_argument("--max_len", type=int, default=384)
args = p.parse_args()

# Inicialização das sementes e configuração do processamento em GPU
random.seed(42)
torch.manual_seed(42)
device = escolher_dispositivo()
dtype = escolher_dtype()
print(f"GPU: {torch.cuda.get_device_name(0)} | tipo numérico: {dtype}")

# Carregamento do modelo base e configuração do fine-tuning com LoRA
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
lora = LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()


# Tokenização dos exemplos, considerando apenas a resposta no cálculo da perda
def tokenizar(ex):
    prompt_ids = tok(montar_prompt(tok, ex["texto"]), add_special_tokens=False).input_ids
    resp_ids = tok(resposta_json(ex) + tok.eos_token, add_special_tokens=False).input_ids
    ids = (prompt_ids + resp_ids)[: args.max_len]
    labels = ([-100] * len(prompt_ids) + resp_ids)[: args.max_len]  # só aprende a resposta
    return {"input_ids": ids, "labels": labels}


# Padronização dos lotes para processamento durante o treinamento
def juntar(lote):
    m = max(len(b["input_ids"]) for b in lote)
    ids = torch.full((len(lote), m), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(lote), m), -100, dtype=torch.long)
    att = torch.zeros((len(lote), m), dtype=torch.long)
    for i, b in enumerate(lote):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return ids, att, lab


# Preparação dos dados e divisão entre treinamento e validação
dados = carregar_jsonl("data/train.jsonl")
random.shuffle(dados)
n_val = max(1, int(0.1 * len(dados)))
val = [tokenizar(e) for e in dados[:n_val]]
treino = [tokenizar(e) for e in dados[n_val:]]
train_loader = DataLoader(treino, batch_size=args.batch, shuffle=True, collate_fn=juntar)
val_loader = DataLoader(val, batch_size=args.batch, collate_fn=juntar)

# Configuração do otimizador e da taxa de aprendizado
params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
total_passos = math.ceil(len(train_loader) / args.accum) * args.epochs
sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_passos), total_passos)


# Avaliação da perda no conjunto de validação
@torch.no_grad()
def loss_validacao():
    model.eval()
    soma, n = 0.0, 0
    for ids, att, lab in val_loader:
        out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        soma += out.loss.item()
        n += 1
    model.train()
    return soma / n


# Execução do treinamento com acumulação de gradientes e monitoramento da GPU
model.train()
historico, passo, inicio = [], 0, time.time()
for epoca in range(1, args.epochs + 1):
    soma, n = 0.0, 0
    for i, (ids, att, lab) in enumerate(train_loader, start=1):
        out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        (out.loss / args.accum).backward()
        soma += out.loss.item()
        n += 1
        if i % args.accum == 0 or i == len(train_loader):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            passo += 1
            if passo % 10 == 0:
                print(f"época {epoca} | passo {passo}/{total_passos} | loss {out.loss.item():.4f} "
                      f"| VRAM em uso {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    lv = loss_validacao()
    historico.append({"epoca": epoca, "loss_treino": soma / n, "loss_validacao": lv})
    print(f"=== Época {epoca}: loss treino {soma / n:.4f} | loss validação {lv:.4f}")

# Salvamento do adaptador treinado e das métricas para comparação entre experimentos
pasta = Path("outputs") / args.run_name
model.save_pretrained(pasta)
metricas = {
    "parametros": vars(args),
    "historico": historico,
    "tempo_segundos": round(time.time() - inicio, 1),
    "pico_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
}
(pasta / "treino_metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Modelo salvo em {pasta} | tempo {metricas['tempo_segundos']} s | pico VRAM {metricas['pico_vram_gb']} GB")