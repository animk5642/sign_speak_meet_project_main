import json
import logging
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import MeetingRoom, MeetingParticipant, ChatMessage
from .ml_service.sign_language_detector import SignLanguageDetectorPool
from .ml_service.word3_detector import Word3DetectorPool
from .ml_service.config import MODEL_PATH, TRAIN_CSV_PATH, WORD3_MODEL_PATH, WORD3_LABELS_PATH

logger = logging.getLogger(__name__)

# Global detector pools (shared across all connections)
detector_pool = SignLanguageDetectorPool(str(MODEL_PATH), str(TRAIN_CSV_PATH))
word3_pool = Word3DetectorPool(str(WORD3_MODEL_PATH), str(WORD3_LABELS_PATH))

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
    """
    WebSocket consumer for real-time sign language detection
    Processes video frames and broadcasts predictions to all room participants
    """
    
    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'sign_language_{self.room_id}'
        self.user_id = str(self.scope['user'].id)
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Get or create detector for this user
        try:
            await self.initialize_detector()
            logger.info(f"Sign language detector initialized for user {self.user_id} in room {self.room_id}")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            await self.close()
    
    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        
        # Clean up detector for this user
        try:
            await self.cleanup_detector()
            logger.info(f"Sign language detector cleaned up for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error during detector cleanup: {e}")
    
    async def receive(self, text_data):
        """
        Receive video frame from client, process it, and broadcast results
        """
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'video_frame':
                # Process frame and get prediction
                frame_data = data.get('frame')
                prediction = await self.process_frame(frame_data)
                
                if prediction:
                    # Broadcast prediction to all users in the room
                    await self.channel_layer.group_send(
                        self.room_group_name,
                        {
                            'type': 'sign_prediction',
                            'user_id': self.user_id,
                            'username': self.scope['user'].email,
                            'sign': prediction['sign'],
                            'confidence': prediction['confidence'],
                        }
                    )
            
            elif message_type == 'reset':
                # Reset the detector sequence
                await self.reset_detector()
                
            elif message_type == 'status':
                # Send status update
                await self.send(text_data=json.dumps({
                    'type': 'status',
                    'status': 'active',
                    'user_id': self.user_id
                }))
                
        except Exception as e:
            logger.error(f"Error in receive: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Failed to process frame'
            }))
    
    async def sign_prediction(self, event):
        """
        Send prediction to WebSocket client
        """
        await self.send(text_data=json.dumps({
            'type': 'sign_prediction',
            'user_id': event['user_id'],
            'username': event['username'],
            'sign': event['sign'],
            'confidence': event['confidence'],
        }))
    
    @database_sync_to_async
    def initialize_detector(self):
        """Initialize detector in thread pool"""
        detector_pool.get_detector(self.user_id)
    
    @database_sync_to_async
    def cleanup_detector(self):
        """Clean up detector in thread pool"""
        detector_pool.remove_detector(self.user_id)
    
    @database_sync_to_async
    def process_frame(self, frame_data):
        """Process frame in thread pool to avoid blocking"""
        detector = detector_pool.get_detector(self.user_id)
        return detector.process_frame(frame_data)
    
    @database_sync_to_async
    def reset_detector(self):
        """Reset detector sequence in thread pool"""
        detector_pool.reset_detector(self.user_id)


class Word3Consumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for word3 letter-based sign language detection.
    Processes video frames and broadcasts letter/word/sentence predictions.
    """

    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'word3_{self.room_id}'
        self.user_id = str(self.scope['user'].id)

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

        try:
            await self.initialize_detector()
            logger.info(f"Word3 detector initialized for user {self.user_id} in room {self.room_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Word3 detector: {e}")
            await self.close()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        try:
            await self.cleanup_detector()
            logger.info(f"Word3 detector cleaned up for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error during Word3 detector cleanup: {e}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            if message_type == 'video_frame':
                frame_data = data.get('frame')
                prediction = await self.process_frame(frame_data)

                if prediction:
                    await self.channel_layer.group_send(
                        self.room_group_name,
                        {
                            'type': 'word3_prediction',
                            'user_id': self.user_id,
                            'username': self.scope['user'].email,
                            'letter': prediction['letter'],
                            'hold_progress': prediction['hold_progress'],
                            'current_word': prediction['current_word'],
                            'sentence': prediction['sentence'],
                            'hand_landmarks': prediction['hand_landmarks'],
                            'did_backspace': prediction.get('did_backspace', False),
                            'is_swiping': prediction.get('is_swiping', False),
                        }
                    )

            elif message_type == 'backspace':
                await self.do_backspace()

            elif message_type == 'clear':
                await self.do_clear()

            elif message_type == 'reset':
                await self.reset_detector()

            elif message_type == 'status':
                await self.send(text_data=json.dumps({
                    'type': 'status',
                    'status': 'active',
                    'user_id': self.user_id
                }))

        except Exception as e:
            logger.error(f"Word3 receive error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Failed to process frame'
            }))

    async def word3_prediction(self, event):
        """Send prediction to WebSocket client"""
        await self.send(text_data=json.dumps({
            'type': 'word3_prediction',
            'user_id': event['user_id'],
            'username': event['username'],
            'letter': event['letter'],
            'hold_progress': event['hold_progress'],
            'current_word': event['current_word'],
            'sentence': event['sentence'],
            'hand_landmarks': event['hand_landmarks'],
            'did_backspace': event.get('did_backspace', False),
            'is_swiping': event.get('is_swiping', False),
        }))

    @database_sync_to_async
    def initialize_detector(self):
        word3_pool.get_detector(self.user_id)

    @database_sync_to_async
    def cleanup_detector(self):
        word3_pool.remove_detector(self.user_id)

    @database_sync_to_async
    def process_frame(self, frame_data):
        detector = word3_pool.get_detector(self.user_id)
        return detector.process_frame(frame_data)

    @database_sync_to_async
    def reset_detector(self):
        word3_pool.reset_detector(self.user_id)

    @database_sync_to_async
    def do_backspace(self):
        detector = word3_pool.get_detector(self.user_id)
        detector.backspace()

    @database_sync_to_async
    def do_clear(self):
        detector = word3_pool.get_detector(self.user_id)
        detector.clear()