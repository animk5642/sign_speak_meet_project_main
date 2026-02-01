from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/meeting/(?P<room_id>\w+)/$', consumers.MeetingConsumer.as_asgi()),
    re_path(r'ws/chat/(?P<room_id>\w+)/$', consumers.ChatConsumer.as_asgi()),
    re_path(r'ws/sign-language/(?P<room_id>\w+)/$', consumers.SignLanguageConsumer.as_asgi()),
]