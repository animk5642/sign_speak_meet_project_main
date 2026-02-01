from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('complete-profile/', views.complete_profile, name='complete_profile'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('phone-login/', views.phone_login, name='phone_login'),
]