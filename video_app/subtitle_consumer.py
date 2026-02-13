"""
SubtitleConsumer - WebSocket consumer for real-time multilingual subtitles.

Each user connects and sets their preferred subtitle language.
When any user sends a transcription, it's broadcast to all room participants.
Each receiving client translates to their own language client-side.
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)


class SubtitleConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time subtitle broadcasting in meetings.

    Flow:
    1. User connects → sets speak_lang + subtitle_lang
    2. User's browser sends audio to HPC → gets transcription
    3. Browser sends transcription here → broadcast to all in room
    4. Each user's browser translates to their preferred language
    """

    async def connect(self):
        # Reject unauthenticated users
        user = self.scope.get('user')
        if not user or user.is_anonymous:
            await self.close()
            return

        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'subtitle_{self.room_id}'
        self.user_id = str(user.id)
        self.username = user.email
        self.subtitle_lang = 'eng_Latn'  # NLLB code for display
        self.speak_lang = 'en'            # Whisper language mode

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()
        logger.info(f"Subtitle WS connected: user={self.username} room={self.room_id}")

    async def disconnect(self, close_code):
        # Guard: attributes may not exist if connect() rejected early
        room_group = getattr(self, 'room_group_name', None)
        if room_group:
            await self.channel_layer.group_discard(
                room_group,
                self.channel_name
            )
        username = getattr(self, 'username', 'unknown')
        logger.info(f"Subtitle WS disconnected: user={username}")

    async def receive(self, text_data):
        """Handle incoming messages from the client."""
        try:
            data = json.loads(text_data)
            msg_type = data.get('type', '')

            if msg_type == 'set_language':
                # User changed their language preferences
                self.speak_lang = data.get('speak_lang', 'en')
                self.subtitle_lang = data.get('subtitle_lang', 'eng_Latn')
                logger.info(
                    f"Language set: user={self.username} "
                    f"speak={self.speak_lang} subtitle={self.subtitle_lang}"
                )
                # Acknowledge
                await self.send(text_data=json.dumps({
                    'type': 'language_set',
                    'speak_lang': self.speak_lang,
                    'subtitle_lang': self.subtitle_lang,
                }))

            elif msg_type == 'transcription':
                # User's browser got transcription from HPC → broadcast to room
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'subtitle_broadcast',
                        'speaker_id': self.user_id,
                        'speaker_name': self.username,
                        'original_text': data.get('original_text', ''),
                        'source_nllb': data.get('source_nllb', ''),
                        'detected_language': data.get('detected_language', ''),
                    }
                )

            elif msg_type == 'ping':
                await self.send(text_data=json.dumps({'type': 'pong'}))

        except Exception as e:
            logger.error(f"SubtitleConsumer receive error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def subtitle_broadcast(self, event):
        """
        Send transcription to each connected client.
        Each client knows their own subtitle_lang and will
        request translation client-side if needed.
        """
        await self.send(text_data=json.dumps({
            'type': 'subtitle',
            'speaker_id': event['speaker_id'],
            'speaker_name': event['speaker_name'],
            'original_text': event['original_text'],
            'source_nllb': event['source_nllb'],
            'detected_language': event['detected_language'],
        }))
