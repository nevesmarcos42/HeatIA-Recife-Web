from django.urls import path
from .views import mapa_previsao

urlpatterns = [
    path('', mapa_previsao, name='mapa-previsao'),
]
