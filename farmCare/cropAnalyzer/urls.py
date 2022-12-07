from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict',views.predict,name='predict'),
    path('addnewCrop',views.addnewCrop,name='addnewCrop'),
    path('getSimilarCrops',views.getSimilarCrops,name='getSimilarCrops'),
    path('addAlert',views.addAlert,name='addAlert'),
    path('getCropsList',views.getCropsList,name='getCropsList'),
    path('getAlertsList',views.getAlertsList,name='getAlertsList'),
    path('userLogin',views.userLogin,name='userLogin'),
    path('addOtherAlert',views.addOtherAlert,name='addOtherAlert'),
    path('getOtherAlertsList',views.getOtherAlertsList,name='getOtherAlertsList'),
]