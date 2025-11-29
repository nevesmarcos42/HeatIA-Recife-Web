#  Previsão de Acidentes com IA + API + Visualização Interativa

Este projeto utiliza **Inteligência Artificial** para prever o número de vítimas em acidentes de trânsito a partir de dados históricos. A solução completa combina um modelo preditivo treinado com **PyTorch**, uma **API FastAPI** eficiente, integração com o **Django** no frontend e visualização com **mapas de calor interativos** para facilitar a tomada de decisões estratégicas.

---

##  Funcionalidades

-  Previsão automatizada de vítimas por bairro com base em variáveis geográficas, temporais e densidade de ocorrências.
-  API REST com FastAPI para consumo e integração dos resultados.
-  Visualização geoespacial com mapas dinâmicos e interativos usando **Folium**.
-  Interface web amigável com Django Templates + Bootstrap.
-  Agrupamento inteligente por bairros para priorização de ações preventivas.

---

##  Tecnologias Utilizadas

| Categoria        | Ferramenta                                     |
|------------------|-------------------------------------------------|
|   IA & ML        | PyTorch, Scikit-learn, Pandas, NumPy            |
|   Backend API    | FastAPI                                         |
|   Frontend       | Django, HTML5, Bootstrap 5                      |
|   Mapa Interativo| Folium, Leaflet.js                              |
|   Requisições    | Requests (Python)                               |
|   Estrutura      | Modular, orientada a microsserviços             |

---

##  Estrutura do Projeto
meu_projeto_api/ ├── backend/ │   └── app/ │       ├── routes.py                
# API de previsão │       ├── main.py                  
# Inicializador da API FastAPI │       ├── modelo_preditivo.pth     
# Pesos do modelo treinado │       ├── scaler.pkl               
# Scaler treinado para normalização │       ├── dados_bairros.csv        
# Dados de entrada ├── frontend/ │   ├── templates/ │   │   └── mapa_backend.html        
# Template com visualização do mapa │   ├── views.py                     
# Lógica do frontend Django │   ├── urls.py                      
# Roteamento da aplicação ├── venv/                            
# Ambiente virtual (não versionado)


---

##  Como rodar o projeto localmente

### 1. Clone o repositório
```bash
git clone https://github.com/seuusuario/previsao-acidentes.git
cd previsao-acidentes

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt

cd backend
uvicorn app.main:app --reload

cd frontend
python manage.py runserver

POST /previsao
{
  "ano": 2025
}

{
  "ano": 2025,
  "acidentes": [
    {
      "bairro": "Boa Vista",
      "latitude": -8.04,
      "longitude": -34.88,
      "vitimas": 3.27
    },
    ...
  ]
}
'''
 Observações
- O modelo foi treinado previamente com dados reais (dataset não público), utilizando a arquitetura ModeloAcidentes.
- Todas as visualizações são renderizadas dinamicamente com base nos dados preditivos gerados no backend.
- O projeto foi modularizado para facilitar reuso e extensão futura (ex: dashboard, painel administrativo, predição diária etc).

 Contribuições
Sinta-se à vontade para sugerir melhorias, criar issues ou enviar um pull request! Toda ajuda é bem-vinda. 

 Licença
Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais informações.

Feito por gente que acredita que IA pode salvar vidas.

---

Se quiser, posso gerar a versão `requirements.txt`, instruções para deploy ou adicionar um gráfico de arquitetura. É só pedir!  
Pronto para colocar isso no GitHub ou em produção?   



---

