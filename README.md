# Treinamento local de LLM para classificação de chamados de TI

Documentação do projeto llm-chamados desenvolvido para a disciplina Generative AI. 

Integrantes e RMs: preencher com os quatro ou cinco integrantes do grupo.

Repositório no GitHub: inserir a URL definitiva antes da entrega.

## 1 - Objetivo e resultado principal

O projeto adapta uma LLM para classificar chamados de suporte de TI. A entrada é a descrição do problema pelo usuário e a saída é um objeto JSON com categoria e prioridade. Essa estrutura permite encaminhar o chamado para uma área responsável e apoiar a ordenação do atendimento.

Selecionamos o modelo Qwen/Qwen2.5-1.5B-Instruct e realizamos ajuste fino supervisionado com LoRA em uma GPU AMD Radeon RX 7600. Comparamos o modelo original com sete configurações de treinamento, variando a taxa de aprendizado, o rank do adaptador e o número de épocas. Todas as avaliações utilizaram os mesmos 100 exemplos de teste.

A configuração **lr2e-4** obteve o maior acerto conjunto observado: 74%, contra 38% do modelo sem treinamento adicional. O ganho foi de 36 pontos percentuais. O acerto de categoria passou de 63% para 83%, e o de prioridade de 59% para 91%. Todas as configurações produziram objetos JSON extraíveis em 100% dos exemplos.

Este relatório apresenta a implementação, o ambiente, o conjunto de dados, os experimentos, as respostas e as evidências. Os resultados descrevem um experimento acadêmico com dados sintéticos; não representam uma validação para uso em produção.

## 2 - Problema escolhido e saída esperada

A classificação manual de chamados exige identificar o tipo de falha e a urgência do atendimento. Escolhemos uma tarefa delimitada para medir os efeitos do treinamento sem depender de respostas longas ou de uma avaliação subjetiva de texto.

| Campo | Valores definidos no projeto | Interpretação |
| --- | --- | --- |
| categoria | Hardware, Rede, Software, Acesso, E-mail | Tipo de problema apresentado |
| prioridade | Alta, Média, Baixa | Urgência indicada no texto |

Hardware inclui falhas físicas de computadores e periféricos. Rede abrange conectividade, internet e VPN. Software inclui instalação e funcionamento de aplicativos. Acesso corresponde a senhas, autenticação e permissões. E-mail reúne problemas de envio, recebimento e configuração de mensagens, incluindo calendário do Outlook conforme os rótulos do dataset.

As prioridades foram definidas pelas frases de urgência do gerador: Alta para paralisação ou prazo imediato; Média para problemas que prejudicam o trabalho, mas permitem continuidade; Baixa para melhorias e situações sem prazo urgente. Essas convenções são os rótulos do experimento, não uma política formal de SLA de uma empresa.

Exemplo de estrutura esperada:

```json
{"categoria": "Hardware", "prioridade": "Alta"}
```

A LLM gera os dois campos como texto. O programa interpreta o JSON, mas não utiliza uma cabeça de classificação separada nem regras fixas para escolher os rótulos. A aplicação também não resolve o chamado nem executa alterações em equipamentos.

## 3 - Modelo escolhido e técnica de adaptação

O identificador usado em treinamento, avaliação e frontend é Qwen/Qwen2.5-1.5B-Instruct. A versão Instruct permite organizar a entrada como uma conversa com instrução do sistema e mensagem do usuário. O tokenizer aplica o template de chat do próprio modelo.

A escolha priorizou a execução local em uma placa com 8 GB de VRAM e a geração de respostas curtas em português. O arquivo de pesos baixado durante a avaliação inicial tinha aproximadamente 3,09 GB. A viabilidade desta escolha foi demonstrada pela execução dos treinamentos registrados.

O treinamento é uma adaptação de um modelo já pré-treinado. Utilizamos LoRA, disponibilizado pela biblioteca PEFT: os pesos originais ficam congelados e matrizes adicionais aprendem a especialização para a tarefa. Isso reduz o número de parâmetros atualizados pelo otimizador.

No rank 16, o terminal registra 18.464.768 parâmetros treináveis, de um total de 1.562.179.072 parâmetros contando os adaptadores, ou 1,1820%. O rank 4 utiliza 4.616.192 parâmetros treináveis (0,2981%), e o rank 64 utiliza 73.859.072 (4,5660%). Esses totais não devem ser confundidos com o número de parâmetros do modelo base isolado.

Os adaptadores são aplicados aos módulos q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj e down_proj. O dropout LoRA é 0,05. Nas três configurações de rank, alpha equivale a duas vezes o rank: 8, 32 e 128. Assim, a razão alpha/r permanece igual a 2, embora a capacidade do adaptador mude.

## 4 - Ambiente de execução e comprovação da GPU

| Componente | Informação disponível |
| --- | --- |
| Sistema operacional | Windows 11 |
| Processador | AMD Ryzen 7 5700X |
| Memória RAM | 16 GB DDR4 |
| GPU | AMD Radeon RX 7600 |
| VRAM | 8,0 GB |
| PyTorch | 2.12.0+rocm7.14.1 |
| ROCm/HIP | 7.14.60850 |
| Tipo numérico dos treinos | torch.bfloat16 |
| Python | 3.12 recomendado no guia | 
| Editor | VS Code |

A GPU utilizada é AMD. O backend observado é ROCm/HIP, e não o CUDA Toolkit da NVIDIA. Apesar disso, o PyTorch usa a interface torch.cuda no código para disponibilizar o dispositivo, consultar a memória e executar operações nessa GPU. A presença da palavra cuda nos scripts não indica utilização de uma placa NVIDIA.

O script verificar_gpu.py consulta torch.cuda.is_available(), imprime o nome da placa e a VRAM, cria uma matriz de 4096 por 4096 elementos no dispositivo e calcula x @ x. A sincronização com torch.cuda.synchronize() ocorre antes da mensagem de sucesso. O print registra GPU disponível: True e Teste de cálculo na GPU: OK.

A função escolher_dispositivo(), compartilhada pelos scripts, encerra a execução quando não detecta GPU. Não há fallback automático para CPU no treinamento, na avaliação ou no frontend. A preparação dos textos e a leitura dos arquivos continuam sendo tarefas do host, enquanto o modelo e seus tensores de entrada são enviados para a GPU.

![Figura 1 Verificação de PyTorch ROCm e cálculo na GPU](<evidencias/prints/imagem (2).png>)

## 5 - Organização do código

| Arquivo ou diretório | Responsabilidade |
| --- | --- |
| src/common.py | Prompt do sistema, template de chat, leitura de JSONL, extração de JSON, seleção de GPU e dtype |
| src/gerar_dataset.py | Construção determinística dos dados sintéticos |
| src/verificar_gpu.py | Identificação da GPU e teste de multiplicação de matrizes |
| src/treinar.py | Tokenização, divisão de validação, LoRA, treinamento e salvamento |
| src/avaliar.py | Inferência sobre teste, métricas e exportação de predições |
| src/app.py | Frontend Gradio e comparação com o modelo original |
| data/ | train.jsonl e test.jsonl |
| resultados/ | resumo.csv e oito arquivos de predições |
| evidencias/prints/ | Dez imagens, histórico do terminal e conversa com Claude |
| requirements.txt | Dependências de aplicação com faixas de versões |

O requirements.txt inclui transformers>=4.51,<5, peft>=0.14, accelerate>=1.0 e gradio>=5,<6. O PyTorch é instalado separadamente conforme o backend da GPU. As versões exatas de todos esses pacotes não foram registradas em um arquivo de lock ou pip freeze no ZIP.

O código salva cada adaptador em outputs/<nome_da_execucao>/, junto com treino_metricas.json. Essa pasta aparece no explorador do VS Code nas evidências, mas não foi incluída no pacote entregue. Os resultados e o código estão disponíveis; para executar imediatamente o frontend, é necessário recuperar o adaptador ou treinar novamente.

## 6 Construção e divisão do conjunto de dados

O gerador utiliza random.seed(42), nomes e setores fictícios e combina três partes: uma saudação, uma descrição do problema e uma frase de urgência. Os registros têm os campos texto, categoria e prioridade. Não foram usados chamados reais nem dados pessoais de clientes.

Foram gerados 600 registros em train.jsonl e 100 em test.jsonl. Para cada categoria, existem sete frases de problema: cinco destinadas ao arquivo de treino e duas reservadas ao teste. Para cada prioridade, existem cinco frases de urgência: três para treino e duas para teste. Essa separação impede que as frases-base reservadas ao teste sejam utilizadas na construção do treino.

| Categoria | Arquivo de treino | Arquivo de teste |
| --- | --- | --- |
| Hardware | 112 | 23 |
| Rede | 106 | 18 |
| Software | 128 | 19 |
| Acesso | 126 | 20 |
| E-mail | 128 | 20 |
| Total | 600 | 100 |

| Prioridade | Arquivo de treino | Arquivo de teste |
| --- | --- | --- |
| Alta | 202 | 34 |
| Média | 212 | 38 |
| Baixa | 186 | 28 |
| Total | 600 | 100 |

O treinamento embaralha os 600 registros com seed 42 e reserva 10% deles para validação. Portanto, utiliza 540 exemplos para atualizar os adaptadores e 60 para calcular a loss de validação. Os 100 exemplos do arquivo de teste ficam fora dessa divisão.

A contagem dos arquivos identifica 589 textos únicos no arquivo de treino e 99 no teste. Isso corresponde a 11 ocorrências repetidas além da primeira no treino e uma no teste. A validação é sorteada por registro, sem deduplicação prévia; o procedimento pode distribuir textos repetidos entre treino e validação. Além disso, os dois subconjuntos compartilham o mesmo repertório de frases-base.

Essas características explicam por que a validação interna é menos exigente que o teste com outras frases. O número de registros não equivale à diversidade de 700 chamados independentes redigidos por pessoas diferentes.

## 7 - Prompt e processamento das entradas

O prompt do sistema implementado em common.py é:

```text
Você é um assistente de suporte de TI. Classifique o chamado do usuário. Categorias possíveis: Hardware, Rede, Software, Acesso, E-mail. Prioridades possíveis: Alta, Média, Baixa. Responda somente com JSON no formato {"categoria": "...", "prioridade": "..."}.
```

montar_prompt() cria as mensagens system e user e chama apply_chat_template com add_generation_prompt=True. No treino, a resposta supervisionada é o JSON com os rótulos do exemplo, seguido do token de fim de sequência.

Os labels dos tokens do prompt recebem -100. Isso faz com que o cálculo de loss considere os tokens da resposta, em vez de ensinar o modelo a reproduzir a instrução e o chamado. O preenchimento dos lotes também recebe -100 nos labels e zero na máscara de atenção. O comprimento máximo de cada sequência é 384 tokens.

Na avaliação, o tokenizer usa padding à esquerda. O código decodifica apenas os novos tokens gerados e compara os campos extraídos com os rótulos esperados. extrair_json() procura a primeira chave de abertura e a última de fechamento e tenta converter esse trecho por json.loads(). Assim, a métrica JSON válido mede um objeto extraível; ela não comprova que a resposta inteira está livre de texto adicional ou que um schema estrito foi validado.

## 8 - Procedimento de treinamento

O treinamento carrega o modelo em bfloat16 no ambiente observado, aplica LoRA e otimiza somente os parâmetros com requires_grad=True. A função escolher_dtype() prevê float16 como alternativa caso bfloat16 não esteja disponível.

| Parâmetro comum | Valor | Papel |
| --- | --- | --- |
| batch | 4 | Exemplos por microbatch |
| accum | 2 | Microbatches acumulados por atualização |
| Batch efetivo nominal | 8 | Produto batch × accum |
| max_len | 384 | Limite de tokens da sequência |
| Otimizador | AdamW | Atualização dos adaptadores |
| weight_decay | 0,0 | Valor usado no otimizador |
| Warmup | 5% dos passos inteiros | Crescimento inicial da taxa |
| Scheduler | Linear | Redução da taxa ao longo do treino |
| Clipping | 1,0 | Limite da norma dos gradientes |
| Sementes | 42 | random e torch |
| LoRA dropout | 0,05 | Regularização dos adaptadores |

Com 540 registros e batch 4, cada época contém 135 microbatches. A acumulação gera 68 atualizações por época, incluindo uma última atualização com apenas um microbatch. Logo, o batch efetivo é nominalmente 8, mas a última atualização usa 4 exemplos. As execuções com uma, duas e quatro épocas realizam 68, 136 e 272 atualizações, respectivamente.

A cada dez atualizações, o terminal registra loss do microbatch e memória alocada. Ao fim de cada época, o código calcula a média da loss dos lotes de treino e de validação. Não há early stopping nem seleção automática do melhor checkpoint pela validação; é salvo o estado ao fim do número solicitado de épocas.

O arquivo treino_metricas.json é previsto pelo código para guardar argumentos, histórico de loss, tempo de treinamento e pico de memória. Como os arquivos de outputs não estão no ZIP, os números desta documentação foram recuperados do histórico de terminal, e não desses JSONs.

## 9 - Experimentos com diferentes parâmetros

Cada configuração começou com o mesmo modelo base e foi avaliada no mesmo arquivo de teste. Primeiro foi estabelecida a referência sem ajuste fino; em seguida foram comparadas taxas de aprendizado, ranks e épocas. Nos testes de rank, alpha também mudou para manter alpha/r=2.

| Execução | Taxa de aprendizado | Rank | Alpha | Épocas |
| --- | --- | --- | --- | --- |
| base_sem_treino | Não se aplica | Não se aplica | Não se aplica | 0 |
| lr2e-4 | 0,0002 | 16 | 32 | 2 |
| lr5e-5 | 0,00005 | 16 | 32 | 2 |
| lr5e-4 | 0,0005 | 16 | 32 | 2 |
| rank4 | 0,0002 | 4 | 8 | 2 |
| rank64 | 0,0002 | 64 | 128 | 2 |
| epocas1 | 0,0002 | 16 | 32 | 1 |
| epocas4 | 0,0002 | 16 | 32 | 4 |

A taxa de aprendizado controla o tamanho das atualizações. O rank controla a capacidade e o número de parâmetros do adaptador. As épocas determinam quantas passagens completas são feitas sobre os dados de treinamento. Alterar esses valores permite estudar a relação entre capacidade, convergência e desempenho em frases reservadas ao teste.

Não foram comparadas diferentes temperaturas ou estratégias de amostragem. Todas as avaliações utilizaram do_sample=False, com temperature, top_p e top_k definidos como None, e limite de 40 novos tokens. O frontend usa o mesmo modo sem amostragem, com limite de 60 novos tokens.

## 10 - Avaliação quantitativa e escolha da configuração

O avaliador processa os 100 exemplos em lotes de 16. Para um adaptador, aplica merge_and_unload() antes da geração. São calculadas quatro taxas: objeto JSON extraível, acerto de categoria, acerto de prioridade e acerto conjunto. O acerto conjunto exige que os dois rótulos estejam corretos no mesmo chamado, por isso foi escolhido como critério principal.

| Execução | JSON válido | Categoria | Prioridade | Acerto conjunto | Avaliação em s |
| --- | --- | --- | --- | --- | --- |
| base_sem_treino | 100% | 63% | 59% | 38% | 7,5 |
| lr2e-4 | 100% | 83% | 91% | 74% | 8,2 |
| lr5e-5 | 100% | 79% | 91% | 70% | 8,5 |
| lr5e-4 | 100% | 74% | 47% | 31% | 8,2 |
| rank4 | 100% | 83% | 79% | 67% | 8,3 |
| rank64 | 100% | 58% | 61% | 34% | 8,7 |
| epocas1 | 100% | 68% | 83% | 53% | 8,6 |
| epocas4 | 100% | 74% | 79% | 55% | 8,2 |

As contagens foram conferidas diretamente nos oito arquivos de predições, com 100 linhas por arquivo, e coincidem com os percentuais do CSV. O resumo.csv contém uma segunda linha para lr5e-4, com as mesmas taxas e tempo de 8,0 s. Ela não foi tratada como um oitavo treinamento: o código acrescenta uma linha ao CSV a cada avaliação e sobrescreve as predições quando o nome é repetido. A tabela mantém a primeira avaliação, também presente no terminal.

A execução lr2e-4 foi selecionada por acertar os dois campos em 74 dos 100 chamados. O aumento sobre a referência foi de 36 pontos percentuais; os erros conjuntos caíram de 62 para 26. lr5e-5 ficou próximo, com 70 acertos, mas a diferença de quatro exemplos não permite afirmar superioridade estatística com um único treinamento por configuração.

A taxa de 5e-4 obteve 31% de acerto conjunto e reduziu o acerto de prioridade para 47%. O rank 64 obteve 34%, apesar de possuir mais parâmetros treináveis. Quatro épocas atingiram 55%, abaixo das duas épocas padrão. Os resultados mostram que aumentar taxa, capacidade ou duração não garantiu melhor desempenho neste conjunto.

## 11 - Tempo de treinamento e memória

| Execução | Tempo de treino em s | Pico alocado em GB |
| --- | --- | --- |
| lr2e-4 | 363,6 | 7,48 |
| lr5e-5 | 558,0 | 7,48 |
| lr5e-4 | 439,7 | 7,48 |
| rank4 | 459,9 | 7,23 |
| rank64 | 711,6 | 8,29 |
| epocas1 | 228,1 | 7,41 |
| epocas4 | 991,0 | 7,48 |

Os tempos de treinamento são separados do tempo de avaliação. Este último mede o laço de inferência e processamento dos 100 exemplos, depois do carregamento do modelo. Não inclui download, carregamento ou fusão do adaptador e não deve ser interpretado como latência individual do frontend. A medição não usa um benchmark dedicado com warmup e sincronização explícita nas fronteiras do cronômetro.

O pico é obtido por torch.cuda.max_memory_allocated(), dividido por 1024³; portanto, o código usa unidades binárias embora imprima GB. O valor alocado não é igual à memória reservada pelo backend nem ao total de memória dedicada apresentado pelo sistema operacional.

O rank 64 registra 8,29 GB no indicador do PyTorch, superior aos 8,0 GB informados para a placa. O print de monitoramento também apresenta memória compartilhada em uso, mas não identifica essa captura como uma medição específica do rank 64. Não é possível atribuir com segurança o excedente ou a duração desse treino a um mecanismo de memória compartilhada apenas com os registros disponíveis.

## 12 - Loss e limites da validação

Na configuração lr2e-4, a primeira época registra loss média de treino 0,0187 e validação 0,0000; a segunda registra 0,0000 em ambas. Os valores são exibidos com quatro casas decimais. Portanto, 0,0000 significa um valor arredondado, e não necessariamente loss exatamente igual a zero.

Uma loss pequena mede boa previsão dos tokens-alvo nesse conjunto. Ela não comprova 100% de acerto de classificação por geração livre no treino ou na validação. Essa taxa não foi calculada pelo script de treinamento.

A baixa loss combinada com 74% de acerto no teste é compatível com aprendizagem muito específica das frases do gerador. Entretanto, não demonstra sozinha a causa dos erros nem prova overfitting em cada configuração. O repertório restrito, a validação por registros do mesmo gerador e a ausência de múltiplas sementes limitam a interpretação.

O mesmo conjunto de teste foi consultado para comparar e escolher configurações. Assim, os 74% representam o melhor desempenho observado nesse conjunto de seleção. Uma avaliação final mais rigorosa exigiria outro conjunto independente, ainda não utilizado para orientar a escolha dos parâmetros.

## 13 - Análise das respostas e dos erros

A inspeção das predições de lr2e-4 permite identificar quais classes explicam os erros. A tabela apresenta as contagens por categoria esperada, recalculadas a partir do arquivo de resultados.

| Categoria esperada | Exemplos | Acertos | Principais confusões |
| --- | --- | --- | --- |
| Hardware | 23 | 23 | Nenhuma |
| Rede | 18 | 16 | 2 classificados como Hardware |
| Software | 19 | 19 | Nenhuma |
| Acesso | 20 | 8 | 12 classificados como Rede |
| E-mail | 20 | 17 | 3 classificados como Software |

Todos os 12 casos de Acesso classificados como Rede no conjunto correspondem ao problema de senha da rede expirada. Isso sugere dificuldade em distinguir autenticação de conectividade quando o texto menciona rede. A confusão E-mail com Software envolve o calendário do Outlook, cujo enquadramento depende da convenção adotada no dataset.

| Prioridade esperada | Exemplos | Acertos | Confusões |
| --- | --- | --- | --- |
| Alta | 34 | 34 | Nenhuma |
| Média | 38 | 30 | 8 classificados como Baixa |
| Baixa | 28 | 27 | 1 classificado como Média |

No total, existem 17 erros de categoria e 9 de prioridade. Os registros mostram 26 chamados com pelo menos um erro; portanto, neste arquivo não houve exemplo com os dois campos incorretos simultaneamente.

Exemplo de erro de categoria: “Olá, aqui é Fábio do setor logística. Minha senha da rede expirou e não consigo cadastrar uma nova. A diretoria está esperando e não podemos atrasar.” O esperado é Acesso/Alta; a resposta foi Rede/Alta. A urgência foi identificada, mas a área foi confundida.

Exemplo de erro de prioridade: “Olá, aqui é Fábio do setor RH. Recebi um e-mail de erro dizendo que a mensagem não pôde ser entregue. Se puderem resolver amanhã, já ajuda bastante.” O esperado é E-mail/Média; a resposta foi E-mail/Baixa.

## 14 - Frontend e testes manuais

O frontend foi implementado com Gradio e iniciado em http://127.0.0.1:7860. A interface mostra a GPU em uso e o caminho do adaptador, oferece um campo de texto para o chamado, um botão Classificar, a resposta bruta e a interpretação em JSON.

A opção Usar modelo treinado é marcada por padrão. Quando desmarcada, o código desativa temporariamente o adaptador com disable_adapter(), permitindo utilizar os pesos originais na mesma aplicação. As capturas recebidas apresentam a opção marcada; não há uma comparação manual pareada entre as duas opções nos prints.

O campo avançado permite editar a instrução do sistema. Isso possibilita experimentar prompts, mas as métricas quantitativas foram obtidas com a instrução padrão. Alterar esse campo muda as condições do experimento e não garante manter o desempenho reportado.

| Entrada apresentada no frontend | Categoria retornada | Prioridade retornada | Arquivo |
| --- | --- | --- | --- |
| Impressora está quebrada | Hardware | Média | imagem (1).png |
| Perdi acesso ao teams e tenho uma reunião urgente | Software | Alta | imagem (5).png e imagem (7).png |
| Preciso de mais um cabo USB | Hardware | Baixa | imagem (8).png |
| Cabo de rede quebrou | Rede | Média | imagem (9).png |

Os testes manuais comprovam interação com a aplicação e produção de respostas estruturadas. Eles não possuem rótulos formais no dataset de teste. O exemplo de acesso ao Teams admite interpretação como autenticação ou falha de aplicativo e ilustra a necessidade de uma taxonomia mais clara. Nos exemplos sem urgência explícita, a prioridade é inferida sem dados suficientes sobre impacto ou prazo.

A função responder() rejeita texto vazio com uma mensagem de erro. Não há histórico persistente de chamados, autenticação ou validação rígida dos valores de categoria e prioridade no frontend. A configuração de lançamento restringe o servidor ao endereço local.

![Figura 2 Frontend com chamado urgente e resposta estruturada](<evidencias/prints/imagem (5).png>)

## 15 - Reprodução do experimento

Os comandos devem ser executados na raiz do projeto em PowerShell. É utilizado Python 3.12 e ambiente virtual.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1102]==2.12.0+rocm7.14.1" "torchvision[device-gfx1102]==0.27.0+rocm7.14.1" "torchaudio==2.11.0+rocm7.14.1"
python -m pip install -r requirements.txt
python src/verificar_gpu.py
```

Antes de prosseguir, a verificação deve reconhecer a GPU e concluir o teste de cálculo. O guia recomenda instalar o driver AMD e selecionar o ambiente virtual no VS Code.

```powershell
python src/avaliar.py --nome base_sem_treino
python src/treinar.py --run_name lr2e-4
python src/avaliar.py --adapter outputs/lr2e-4 --nome lr2e-4
python src/treinar.py --run_name lr5e-5 --lr 5e-5
python src/avaliar.py --adapter outputs/lr5e-5 --nome lr5e-5
python src/treinar.py --run_name lr5e-4 --lr 5e-4
python src/avaliar.py --adapter outputs/lr5e-4 --nome lr5e-4
python src/treinar.py --run_name rank4 --rank 4 --alpha 8
python src/avaliar.py --adapter outputs/rank4 --nome rank4
python src/treinar.py --run_name rank64 --rank 64 --alpha 128
python src/avaliar.py --adapter outputs/rank64 --nome rank64
python src/treinar.py --run_name epocas1 --epochs 1
python src/avaliar.py --adapter outputs/epocas1 --nome epocas1
python src/treinar.py --run_name epocas4 --epochs 4
python src/avaliar.py --adapter outputs/epocas4 --nome epocas4
python src/app.py outputs/lr2e-4
```

O modelo base é obtido por from_pretrained() e armazenado no cache local. O caminho do adaptador deve existir antes de abrir o app. Reexecutar avaliações acrescenta linhas ao resumo.csv; resultados de novas execuções devem ser separados dos originais para preservar a rastreabilidade.

## 16 - Conclusão

O experimento demonstrou ajuste fino local de uma LLM com GPU AMD, comparação controlada de sete configurações sobre o mesmo conjunto e uma interface para testar chamados em linguagem natural. A configuração selecionada foi taxa de aprendizado 2e-4, rank 16, alpha 32 e duas épocas, com 74% de acerto conjunto e 100% de objetos JSON extraíveis no teste.

O principal ganho observado foi a melhora sobre o modelo original, de 38% para 74% de acerto conjunto. O principal limite foi a generalização em um dataset sintético de repertório pequeno, especialmente em problemas de acesso com menção à rede. Mais parâmetros ou mais épocas não produziram melhor resultado nas condições avaliadas.

## Evidências

Atividade da GPU.  
![Figura 3 GPU em atividade e verificação no terminal do VS Code](<evidencias/prints/imagem (4).png>)

Resultado sem treino. 

![Figura 4 Avaliação do modelo original com 38 por cento de acerto conjunto](<evidencias/prints/imagem.png>) 

Resultado do treino com melhor resultado.  

![Figura 5 Avaliação de lr2e 4 com 74 por cento de acerto conjunto](<evidencias/prints/imagem (3).png>)

Exemplos de classificações de chamados. 

![Figura 6 Chamado sobre impressora e classificação Hardware Média](<evidencias/prints/imagem (1).png>)

![Figura 7 Pedido de cabo USB e classificação Hardware Baixa](<evidencias/prints/imagem (8).png>)

![Figura 8 Cabo de rede quebrado e classificação Rede Média](<evidencias/prints/imagem (9).png>)
