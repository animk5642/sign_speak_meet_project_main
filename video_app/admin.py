from django.contrib import admin
from .models import (
    MeetingRoom, 
    MeetingParticipant, 
    JoinRequest, 
    SignLanguageTranslation, 
    ChatMessage, 
    RoomMember
)

@admin.register(MeetingRoom)
class MeetingRoomAdmin(admin.ModelAdmin):
    list_display = ('room_id', 'title', 'host', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('room_id', 'title', 'host__email')
    readonly_fields = ('created_at',)

@admin.register(MeetingParticipant)
class MeetingParticipantAdmin(admin.ModelAdmin):
    list_display = ('user', 'meeting', 'status', 'joined_at')
    list_filter = ('status', 'joined_at')
    search_fields = ('user__email', 'meeting__title')

@admin.register(JoinRequest)
class JoinRequestAdmin(admin.ModelAdmin):
    list_display = ('user', 'meeting', 'status', 'requested_at', 'processed_at')
    list_filter = ('status', 'requested_at')
    search_fields = ('user__email', 'meeting__title')
    readonly_fields = ('requested_at',)

@admin.register(SignLanguageTranslation)
class SignLanguageTranslationAdmin(admin.ModelAdmin):
    list_display = ('user', 'meeting', 'confidence_score', 'timestamp')
    list_filter = ('timestamp',)
    search_fields = ('user__email', 'meeting__title', 'translated_text')
    readonly_fields = ('timestamp',)

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('user', 'meeting', 'message_type', 'content_preview', 'timestamp')
    list_filter = ('message_type', 'timestamp')
    search_fields = ('user__email', 'meeting__title', 'content')
    readonly_fields = ('timestamp',)
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Message'

@admin.register(RoomMember)
class RoomMemberAdmin(admin.ModelAdmin):
    list_display = ('name', 'uid', 'room_name')
    search_fields = ('name', 'uid', 'room_name')
