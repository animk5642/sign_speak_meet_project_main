from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('create-meeting/', views.create_meeting, name='create_meeting'),
    path('join-meeting/<str:room_id>/', views.join_meeting, name='join_meeting'),
    path('meeting/<str:room_id>/', views.meeting_room, name='meeting_room'),
    path('handle-request/<int:request_id>/', views.handle_join_request, name='handle_join_request'),
    path('get-pending-requests/', views.get_pending_requests, name='get_pending_requests'),
    path('get-pending-requests/<str:room_id>/', views.get_pending_requests, name='get_pending_requests_room'),
    
    path('create_member/', views.create_member),
    path('get_member/', views.get_member),
    path('delete_member/', views.delete_member),
    path('get_token/', views.getToken),
]