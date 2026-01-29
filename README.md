# Deploying a Scalable ML Pipeline with FastAPI

Este repositório implementa um pipeline de ML para o dataset Census Income, com treinamento, avaliação por slices e uma API FastAPI para inferência. Ele foi criado como parte de um projeto educacional e segue boas práticas de versionamento, testes e documentação.

## Visão geral

- **Treinamento e avaliação** com dados tabulares limpos (`data/census.csv`).
- **Avaliação por slices** para detectar viés e performance por categoria.
- **API FastAPI** com endpoints GET e POST para inferência.
- **Model card** documentando uso e limitações.

## Estrutura do projeto

- `data/` — dataset (inclui `census.csv`).
- `ml/` — pipeline de dados e modelo.
- `model/` — artefatos treinados (`model.pkl`, `encoder.pkl`).
- `main.py` — aplicação FastAPI.
- `train_model.py` — script de treino.
- `test_ml.py` — testes de unidade.
- `model_card.md` — documentação do modelo.

## Pré-requisitos

- Python 3.10+ (ou a versão configurada no seu ambiente).
- Git instalado.
- (Opcional) DVC para versionamento do dataset.

> Em Windows, usar WSL 1/2 facilita integrações com Git/DVC, mas não é obrigatório.

## Configuração do ambiente

Escolha **uma** das opções abaixo:

### Opção 1 — Conda

```bash
conda env create -f environment.yml
conda activate fastapi
```

### Opção 2 — Pip

```bash
python -m venv .venv
.
```

```bash
python -m pip install -r requirements.txt
```

## Treinamento e testes

### Treinar o modelo

```bash
python train_model.py
```

### Rodar testes

```bash
pytest -q
```

## API FastAPI

### Subir a aplicação

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Endpoints

- **GET /** — mensagem de boas-vindas
- **POST /data/** — inferência com payload JSON

### Exemplo de request (POST)

```json
{
  "age": 37,
  "workclass": "Private",
  "fnlgt": 178356,
  "education": "HS-grad",
  "education-num": 10,
  "marital-status": "Married-civ-spouse",
  "occupation": "Prof-specialty",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

> A documentação interativa fica em: `http://127.0.0.1:8000/docs`.

## Boas práticas recomendadas

- Commits frequentes com mensagens claras.
- Versões do modelo versionadas em Git.
- Pipeline de CI rodando **pytest** + **flake8**.
- Atualizar `model_card.md` sempre que o modelo mudar.

## Observações

Este projeto foi estruturado para fins educacionais e pode ser adaptado para produção com melhorias de segurança, monitoramento e versionamento de features.
