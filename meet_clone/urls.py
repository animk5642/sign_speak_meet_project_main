from django.shortcuts import redirect
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from users import views as user_views
from video_app import views as video_views

def redirect_to_video_dashboard(request):
    return redirect('home')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', video_views.dashboard, name='home'),  # Main video dashboard
    path('dashboard/', redirect_to_video_dashboard, name='dashboard'),  # Redirect to home
    
    # Authentication URLs
    path('accounts/signup/', user_views.register, name='signup'),
    path('accounts/login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(), name='logout'),
    
    # User URLs
    path('users/', include('users.urls')),
    path('video/', include('video_app.urls')),
]