from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import CustomUserCreationForm
from .models import CustomUser

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful!')
            return redirect('home')  # Redirect to main dashboard
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'users/register.html', {'form': form})

@login_required
def dashboard(request):
    # User profile dashboard
    return render(request, 'users/dashboard.html')

@login_required
def complete_profile(request):
    # Simple profile completion view
    if request.method == 'POST':
        # Handle profile completion logic here
        messages.success(request, 'Profile completed successfully!')
        return redirect('dashboard')
    
    return render(request, 'users/complete_profile.html')

def phone_login(request):
    if request.method == 'POST':
        phone_number = request.POST.get('phone_number')
        # Simple phone login simulation
        if phone_number:
            try:
                user = CustomUser.objects.get(phone_number=phone_number)
                login(request, user)
                messages.success(request, 'Logged in successfully!')
                return redirect('home')
            except CustomUser.DoesNotExist:
                # Create new user with phone
                user = CustomUser.objects.create(
                    email=f"{phone_number}@temp.com",
                    phone_number=phone_number
                )
                user.set_unusable_password()
                user.save()
                login(request, user)
                messages.success(request, 'Account created successfully! Please complete your profile.')
                return redirect('complete_profile')
    
    return render(request, 'users/phone_login.html')