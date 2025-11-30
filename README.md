# HeatIA-Recife-Web

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![Django](https://img.shields.io/badge/Django-4.0+-darkgreen?style=for-the-badge&logo=django)

Sistema de previsão de acidentes de trânsito usando Inteligência Artificial. Combina um modelo preditivo baseado em PyTorch com API REST e visualização geoespacial interativa para facilitar a tomada de decisões estratégicas em segurança viária.

**Funcionalidades** • **Tecnologias** • **Instalação** • **Uso** • **API** • **Contribuir**

---

## Índice

- Sobre o Projeto
- Funcionalidades
- Tecnologias
- Arquitetura
- Instalação
- Uso
- Documentação da API
- Modelo de IA
- Contribuindo
- Licença

---

## Sobre o Projeto

HeatIA-Recife-Web é uma aplicação fullstack que utiliza **Machine Learning** para prever o número de vítimas em acidentes de trânsito nos bairros de Recife. O projeto foi desenvolvido com foco em análise preditiva, visualização geoespacial e integração entre backend e frontend, oferecendo uma ferramenta prática para gestores públicos e analistas de segurança viária.

### Principais Características

- **Previsão com IA** - Modelo PyTorch treinado com dados históricos de acidentes
- **API REST** - FastAPI para consumo eficiente dos dados preditivos
- **Visualização Geoespacial** - Mapas de calor interativos com Folium
- **Interface Web** - Frontend Django responsivo com Bootstrap 5
- **Análise por Bairro** - Agrupamento inteligente para priorização de ações
- **Dados Temporais** - Previsões baseadas em variáveis geográficas e temporais
- **Arquitetura Modular** - Separação clara entre backend API e frontend web

---

## Funcionalidades

### Backend (API FastAPI)

#### Previsão de Acidentes

- Gerar previsões por bairro para anos específicos
- Entrada: ano de previsão
- Retorna: lista de bairros com latitude, longitude e número previsto de vítimas
- Validação de dados de entrada
- Normalização automática com scaler treinado

#### Modelo Preditivo

- Modelo neural network com PyTorch
- Utiliza dados históricos de acidentes
- Variáveis: localização geográfica, ano, densidade de ocorrências
- Predição de número de vítimas por bairro
- Modelo pré-treinado (`modelo_preditivo.pth`)

### Frontend (Django)

- **Visualização de Mapas** - Mapas de calor interativos com Folium
- **Interface Responsiva** - Design moderno com Bootstrap 5
- **Requisições Assíncronas** - Integração com API FastAPI
- **Dados Dinâmicos** - Visualização atualizada em tempo real
- **Agrupamento por Bairros** - Marcadores e clusters no mapa
- **Informações Detalhadas** - Tooltips com dados de cada região

---

## Tecnologias

### Backend (API)

| Tecnologia   | Versão | Descrição                       |
| ------------ | ------ | ------------------------------- |
| Python       | 3.9+   | Linguagem de programação        |
| FastAPI      | 0.100+ | Framework API REST              |
| PyTorch      | 2.0+   | Framework de Deep Learning      |
| Scikit-learn | 1.3+   | Preprocessamento e normalização |
| Pandas       | 2.0+   | Manipulação de dados            |
| NumPy        | 1.24+  | Computação numérica             |
| Uvicorn      | 0.23+  | Servidor ASGI                   |

### Frontend (Web)

| Tecnologia | Versão | Descrição                       |
| ---------- | ------ | ------------------------------- |
| Django     | 4.0+   | Framework web                   |
| Bootstrap  | 5.3+   | Framework CSS                   |
| Folium     | 0.14+  | Visualização de mapas           |
| Leaflet.js | 1.9+   | Biblioteca de mapas interativos |
| Requests   | 2.31+  | Cliente HTTP Python             |
| HTML5      | -      | Estrutura web                   |

### Machine Learning

- **PyTorch** - Construção e treinamento do modelo
- **Scikit-learn** - Normalização e preprocessamento
- **Pandas** - Análise de dados históricos
- **NumPy** - Operações matriciais

---

## Arquitetura

### Backend - Estrutura de Diretórios

```
backend/
└── app/
    ├── __init__.py         # Inicialização do módulo
    ├── main.py             # Configuração FastAPI
    ├── routes.py           # Endpoints da API
    ├── modelo_preditivo.pth # Pesos do modelo treinado
    ├── dados_2025.csv      # Dataset de entrada
    └── dados_bairros.csv   # Informações dos bairros
```

### Frontend - Estrutura de Diretórios

```
frontend/
├── manage.py              # CLI Django
├── db.sqlite3            # Banco de dados
├── frontend/
│   ├── settings.py       # Configurações do projeto
│   ├── urls.py           # Roteamento principal
│   └── wsgi.py           # WSGI application
└── mapa/
    ├── views.py          # Lógica de visualização
    ├── urls.py           # Rotas do app
    └── templates/
        └── mapa/
            └── mapa_backend.html  # Template do mapa
```

### Fluxo de Dados

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   Django    │       │  FastAPI    │       │   PyTorch   │
│  Frontend   │──────►│   Backend   │──────►│   Modelo    │
│             │       │             │       │   Neural    │
└─────────────┘       └─────────────┘       └─────────────┘
      │                     │                       │
      │                     │                       │
  Usuário               REST API              Previsão IA
  solicita              processa              retorna
  previsão              requisição            resultados
```

---

## Instalação

### Pré-requisitos

- Python 3.9 ou superior - [Download](https://www.python.org/)
- pip - Gerenciador de pacotes Python
- Git - [Download](https://git-scm.com/)

### 1. Clone o repositório

```bash
git clone https://github.com/nevesmarcos42/HeatIA-Recife-Web.git
cd HeatIA-Recife-Web
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Inicie o backend (FastAPI)

```bash
cd backend
uvicorn app.main:app --reload
```

A API estará disponível em: `http://localhost:8000`

### 5. Inicie o frontend (Django)

Em outro terminal:

```bash
cd frontend
python manage.py runserver
```

A aplicação web estará disponível em: `http://localhost:8000`

---

## Uso

### Primeiro Acesso

1. Acesse a aplicação: `http://localhost:8000`
2. A página principal exibirá o mapa interativo de Recife
3. Visualize as previsões de acidentes por bairro
4. Interaja com o mapa para ver detalhes de cada região

### Visualizar Previsões

1. A aplicação carrega automaticamente as previsões para o ano atual
2. Os marcadores no mapa indicam áreas com maior incidência prevista
3. Clique nos marcadores para ver:
   - Nome do bairro
   - Número previsto de vítimas
   - Coordenadas geográficas

### Fazer Nova Previsão

Para gerar previsões para outros anos, faça uma requisição direta à API:

```bash
curl -X POST http://localhost:8000/previsao \
  -H "Content-Type: application/json" \
  -d '{"ano": 2026}'
```

---

## Documentação da API

A documentação interativa está disponível via Swagger UI após iniciar o backend:

**URL:** `http://localhost:8000/docs`

### Principais Endpoints

#### Previsão de Acidentes

```
POST /previsao
```

**Body:**

```json
{
  "ano": 2025
}
```

**Response:**

```json
{
  "ano": 2025,
  "acidentes": [
    {
      "bairro": "Boa Vista",
      "latitude": -8.0478,
      "longitude": -34.8839,
      "vitimas": 3.27
    },
    {
      "bairro": "Recife",
      "latitude": -8.063,
      "longitude": -34.8711,
      "vitimas": 2.85
    }
  ]
}
```

### Exemplo de Requisição

#### Python (Requests)

```python
import requests

url = "http://localhost:8000/previsao"
data = {"ano": 2025}

response = requests.post(url, json=data)
print(response.json())
```

#### cURL

```bash
curl -X POST http://localhost:8000/previsao \
  -H "Content-Type: application/json" \
  -d '{"ano": 2025}'
```

#### JavaScript (Fetch)

```javascript
fetch("http://localhost:8000/previsao", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ ano: 2025 }),
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

---

## Modelo de IA

### Arquitetura do Modelo

O modelo utiliza uma rede neural feedforward implementada em PyTorch:

- **Camada de Entrada:** Features normalizadas (localização, ano, densidade)
- **Camadas Ocultas:** Múltiplas camadas densas com ReLU
- **Camada de Saída:** Previsão do número de vítimas
- **Otimizador:** Adam
- **Função de Perda:** MSE (Mean Squared Error)

### Treinamento

```python
# Estrutura do modelo
class ModeloAcidentes(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
```

### Variáveis de Entrada

- **Latitude** - Coordenada geográfica
- **Longitude** - Coordenada geográfica
- **Ano** - Ano de previsão
- **Densidade de Acidentes** - Histórico da região

### Métricas de Desempenho

O modelo foi treinado com dados históricos de acidentes de Recife:

- ✅ Dataset normalizado com StandardScaler
- ✅ Validação cruzada para evitar overfitting
- ✅ Predições com erro médio controlado
- ✅ Modelo persistido em `modelo_preditivo.pth`

---

## Variáveis de Ambiente

### Backend (FastAPI)

Por padrão, a API roda em `http://localhost:8000`. Para customizar:

```python
# backend/app/main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Frontend (Django)

Configurações em `frontend/frontend/settings.py`:

```python
# URL da API Backend
BACKEND_API_URL = "http://localhost:8000"

# Debug Mode
DEBUG = True

# Allowed Hosts
ALLOWED_HOSTS = ['localhost', '127.0.0.1']
```

---

## Contribuindo

Contribuições são bem-vindas! Siga os passos:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

### Padrões de Código

#### Backend (FastAPI)

- Seguir convenções PEP 8
- Documentar endpoints com docstrings
- Validar dados de entrada com Pydantic
- Escrever testes para novas funcionalidades

#### Frontend (Django)

- Seguir convenções do Django
- Manter templates organizados
- Usar Bootstrap para estilização
- Documentar views e funcionalidades

#### Machine Learning

- Documentar arquitetura do modelo
- Versionar datasets e modelos treinados
- Reportar métricas de desempenho
- Garantir reprodutibilidade dos experimentos

---

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

**Desenvolvido com ❤️ por pessoas que acreditam que IA pode salvar vidas**

**Versão:** 1.0.0

**Última Atualização:** Novembro 2025
