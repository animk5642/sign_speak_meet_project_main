from django.urls import re_path
from . import consumers
from .subtitle_consumer import SubtitleConsumer

websocket_urlpatterns = [
    re_path(r'ws/meeting/(?P<room_id>\w+)/$', consumers.MeetingConsumer.as_asgi()),
    re_path(r'ws/chat/(?P<room_id>\w+)/$', consumers.ChatConsumer.as_asgi()),
    re_path(r'ws/sign-language/(?P<room_id>\w+)/$', consumers.SignLanguageConsumer.as_asgi()),
    re_path(r'ws/subtitle/(?P<room_id>\w+)/$', SubtitleConsumer.as_asgi()),
]