from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # Remove the username field completely
    username = None
    email = models.EmailField('email address', unique=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []  # Remove 'email' from REQUIRED_FIELDS
    
    def __str__(self):
        return self.email