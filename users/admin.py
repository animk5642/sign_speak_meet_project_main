from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.contrib.auth import get_user_model

User = get_user_model()

@admin.register(User)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('email', 'is_staff', 'is_superuser', 'is_active')
    search_fields = ('email',)
