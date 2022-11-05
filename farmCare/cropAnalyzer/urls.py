from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict',views.predict,name='predict'),
    path('addnewCrop',views.addnewCrop,name='addnewCrop'),
    path('getSimilarCrops',views.getSimilarCrops,name='getSimilarCrops'),
]