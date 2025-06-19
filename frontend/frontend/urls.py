from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('mapa.urls')),
    path("admin/", admin.site.urls),
]
