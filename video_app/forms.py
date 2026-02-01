from django import forms
from .models import MeetingRoom

class MeetingRoomForm(forms.ModelForm):
    class Meta:
        model = MeetingRoom
        fields = ['title', 'description']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter meeting title'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Enter meeting description (optional)'
            }),
        }