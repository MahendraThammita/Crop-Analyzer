from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('cropAnalyzer/', include('cropAnalyzer.urls')),
    path('admin/', admin.site.urls),
]