from django.shortcuts import render
from folium.plugins import HeatMap
from django.utils.safestring import mark_safe
from datetime import datetime

import folium
import requests


def mapa_previsao(request):
    ano_atual = datetime.now().year

    # Tenta obter o ano do formulário ou usa o ano atual
    ano = request.POST.get("ano", str(ano_atual))
    mapa_html = erro = None

    try:
        ano = int(ano)

        # Checagem: ano informado não pode ser no passado
        if ano < ano_atual:
            erro = f"o ano informado deve ser maior ou igual a {ano_atual}."
        else:
            # Faz POST para a API FastAPI com o ano
            resposta = requests.post("http://localhost:8001/previsao",
                                     json={"ano": ano})
            acidentes = resposta.json().get("acidentes", [])

            # Extrai os valores de vítimas para normalização
            valores = [a["vitimas"] for a in acidentes]
            min_v, max_v = min(valores), max(valores)

            # Aplica normalização e prepara dados do mapa de calor
            heat_data = [
                [a["latitude"], a["longitude"], max((a["vitimas"] - min_v) / (max_v - min_v + 1e-6), 0.05)]
                for a in acidentes
            ]
            # Cria o mapa centrado em Recife
            mapa = folium.Map(location=[-8.05, -34.9], zoom_start=12)

            # Adiciona a camada de calor ao mapa
            HeatMap(heat_data, radius=30, blur=20, gradient={
                0.2: "blue", 0.4: "green", 0.6: "yellow", 0.8: "orange", 1.0: "red"
            }).add_to(mapa)

            # Gera o HTML incorporável do mapa
            mapa_html = mark_safe(mapa._repr_html_())
    except Exception as e:
        erro = f"Ocorreu um erro: {e}"

    # Renderiza o template com contexto
    return render(request, "mapa/mapa_backend.html", {
        "ano": ano,
        "ano_atual": ano_atual,
        "erro": erro,
        "mapa_html": mapa_html
    })
