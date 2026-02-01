from django.db import models
from django.utils import timezone
from django.contrib.auth import get_user_model

User = get_user_model()

class MeetingRoom(models.Model):
    room_id = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=200)
    host = models.ForeignKey(User, on_delete=models.CASCADE, related_name='hosted_meetings')
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.title} (Host: {self.host.email})"
    
    def pending_requests(self):
        return self.join_requests.filter(status='pending')

class MeetingParticipant(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending Approval'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]
    meeting = models.ForeignKey(MeetingRoom, on_delete=models.CASCADE, related_name='participants')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    joined_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        unique_together = ['meeting', 'user']
    
    def __str__(self):
        return f"{self.user.email} - {self.meeting.title} ({self.status})"

class JoinRequest(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]
    meeting = models.ForeignKey(MeetingRoom, on_delete=models.CASCADE, related_name='join_requests')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    requested_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        unique_together = ['meeting', 'user']
    
    def __str__(self):
        return f"{self.user.email} -> {self.meeting.title}"

class SignLanguageTranslation(models.Model):
    meeting = models.ForeignKey(MeetingRoom, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    sign_language_input = models.TextField()
    translated_text = models.TextField()
    confidence_score = models.FloatField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"Sign translation for {self.user.email}"

class ChatMessage(models.Model):
    MESSAGE_TYPES = [
        ('text', 'Text'),
        ('system', 'System'),
        ('sign', 'Sign Language'),
    ]
    meeting = models.ForeignKey(MeetingRoom, on_delete=models.CASCADE, related_name='chat_messages')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES, default='text')
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.user.email}: {self.content[:50]}"

class RoomMember(models.Model):
    name = models.CharField(max_length=200)
    uid = models.CharField(max_length=200)
    room_name = models.CharField(max_length=200)

    def __str__(self):
        return self.name