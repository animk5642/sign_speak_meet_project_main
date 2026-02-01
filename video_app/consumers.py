import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import MeetingRoom, MeetingParticipant, ChatMessage

class MeetingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'meeting_{self.room_id}'
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'user_joined',
                'user_id': self.scope['user'].id,
                'username': self.scope['user'].email,
            }
        )
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'user_left',
                'user_id': self.scope['user'].id,
                'username': self.scope['user'].email,
            }
        )
    
    async def receive(self, text_data):
        data = json.loads(text_data)
        message_type = data['type']
        
        # Targeted signaling: only send to the specific target_user
        if message_type in ['webrtc_offer', 'webrtc_answer', 'ice_candidate']:
            target_user_id = data.get('target_user')
            if target_user_id:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'signal_message',
                        'message': data,
                        'from_user': self.scope['user'].id,
                        'target_user': target_user_id
                    }
                )
    
    async def signal_message(self, event):
        # Only send to the browser if this user is the intended target
        if str(self.scope['user'].id) == str(event['target_user']):
            message = event['message']
            message['from_user'] = event['from_user']
            await self.send(text_data=json.dumps(message))

    async def user_joined(self, event):
        # Don't send join notice to the person who just joined
        if str(self.scope['user'].id) != str(event['user_id']):
            await self.send(text_data=json.dumps({
                'type': 'user_joined',
                'user_id': event['user_id'],
                'username': event['username'],
            }))
    
    async def user_left(self, event):
        await self.send(text_data=json.dumps({
            'type': 'user_left',
            'user_id': event['user_id'],
            'username': event['username'],
        }))

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'chat_{self.room_id}'
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
    
    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data['message']
        message_type = data.get('message_type', 'text')
        
        await self.save_message(message, message_type)
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message,
                'username': self.scope['user'].email,
                'message_type': message_type,
            }
        )
    
    async def chat_message(self, event):
        await self.send(text_data=json.dumps({
            'type': 'chat_message',
            'message': event['message'],
            'username': event['username'],
            'message_type': event['message_type'],
        }))
    
    @database_sync_to_async
    def save_message(self, message, message_type):
        meeting = MeetingRoom.objects.get(room_id=self.room_id)
        ChatMessage.objects.create(
            meeting=meeting,
            user=self.scope['user'],
            content=message,
            message_type=message_type
        )

class SignLanguageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'sign_language_{self.room_id}'
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
    
    async def receive(self, text_data):
        data = json.loads(text_data)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'sign_language_data',
                'sign_data': data['sign_data'],
                'translated_text': data.get('translated_text', ''),
                'username': self.scope['user'].email,
            }
        )
    
    async def sign_language_data(self, event):
        await self.send(text_data=json.dumps({
            'type': 'sign_language_data',
            'sign_data': event['sign_data'],
            'translated_text': event['translated_text'],
            'username': event['username'],
        }))