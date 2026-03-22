import uuid
import json
import requests as http_requests
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
from decouple import config
from .models import MeetingRoom, MeetingParticipant, JoinRequest, RoomMember
from .forms import MeetingRoomForm

@login_required
def dashboard(request):
    # Get user's meetings and pending requests
    hosted_meetings = MeetingRoom.objects.filter(host=request.user)
    participant_meetings = MeetingParticipant.objects.filter(
        user=request.user,
        status='approved'
    ).exclude(meeting__host=request.user)  # Exclude meetings hosted by the user to prevent duplicates
    
    # Get pending requests for meetings hosted by current user
    pending_requests = JoinRequest.objects.filter(
        meeting__host=request.user,
        status='pending'
    ).select_related('meeting', 'user')
    
    return render(request, 'video_app/dashboard.html', {
        'hosted_meetings': hosted_meetings,
        'participant_meetings': participant_meetings,
        'pending_requests': pending_requests,
    })

@login_required
def create_meeting(request):
    if request.method == 'POST':
        form = MeetingRoomForm(request.POST)
        if form.is_valid():
            meeting = form.save(commit=False)
            meeting.host = request.user
            meeting.room_id = str(uuid.uuid4())[:8].upper()
            meeting.save()
            
            # Auto-approve host as participant
            MeetingParticipant.objects.create(
                meeting=meeting,
                user=request.user,
                status='approved'
            )
            
            messages.success(request, f'Meeting created! Room ID: {meeting.room_id}')
            return redirect('meeting_room', room_id=meeting.room_id)
    else:
        form = MeetingRoomForm()
    
    return render(request, 'video_app/create_meeting.html', {'form': form})

@login_required
def join_meeting(request, room_id):
    meeting = get_object_or_404(MeetingRoom, room_id=room_id)
    
    # Check if user is the host
    if meeting.host == request.user:
        return redirect('meeting_room', room_id=room_id)
    
    # Check if user already has a participant record
    participant = MeetingParticipant.objects.filter(
        meeting=meeting,
        user=request.user
    ).first()
    
    if participant and participant.status == 'approved':
        return redirect('meeting_room', room_id=room_id)
    
    # Create join request if it doesn't exist
    join_request, created = JoinRequest.objects.get_or_create(
        meeting=meeting,
        user=request.user,
        defaults={'status': 'pending'}
    )
    
    if created:
        messages.info(request, 'Join request sent to host. Waiting for approval.')
    else:
        if join_request.status == 'pending':
            messages.info(request, 'Your request is still pending approval.')
        elif join_request.status == 'rejected':
            messages.warning(request, 'Your join request was rejected.')
    
    return render(request, 'video_app/waiting_room.html', {
        'meeting': meeting,
        'join_request': join_request,
    })

import time
from agora_token_builder import RtcTokenBuilder

AGORA_APP_ID = config('AGORA_APP_ID')
AGORA_APP_CERTIFICATE = config('AGORA_APP_CERTIFICATE')
HPC_SERVER_URL = config('HPC_SERVER_URL', default='http://192.168.200.75:8200')

@login_required
def meeting_room(request, room_id):
    meeting = get_object_or_404(MeetingRoom, room_id=room_id)
    
    # Check if user can access this meeting
    if meeting.host != request.user:
        participant = get_object_or_404(
            MeetingParticipant,
            meeting=meeting,
            user=request.user,
            status='approved'
        )
    
    # Get pending requests for host
    pending_requests = []
    if meeting.host == request.user:
        pending_requests = JoinRequest.objects.filter(
            meeting=meeting,
            status='pending'
        ).select_related('user')
    
    return render(request, 'video_app/meeting_room.html', {
        'meeting': meeting,
        'pending_requests': pending_requests,
        'AGORA_APP_ID': AGORA_APP_ID,
        'HPC_SERVER_URL': HPC_SERVER_URL,
    })


def getToken(request):
    appId = AGORA_APP_ID
    appCertificate = AGORA_APP_CERTIFICATE
    channelName = request.GET.get('channel')
    uid = request.GET.get('uid')
    role = 1 # Host
    expirationTimeInSeconds = 3600 * 24
    currentTimeStamp = int(time.time())
    privilegeExpiredTs = currentTimeStamp + expirationTimeInSeconds


    token = RtcTokenBuilder.buildTokenWithUid(appId, appCertificate, channelName, uid, role, privilegeExpiredTs)

    return JsonResponse({'token': token}, safe=False)

@login_required
@require_POST
def handle_join_request(request, request_id):
    join_request = get_object_or_404(
        JoinRequest, 
        id=request_id,
        meeting__host=request.user  # Only host can handle requests
    )
    
    action = request.POST.get('action')
    
    if action == 'approve':
        # Update join request
        join_request.status = 'approved'
        join_request.save()
        
        # Create or update participant
        participant, created = MeetingParticipant.objects.get_or_create(
            meeting=join_request.meeting,
            user=join_request.user,
            defaults={'status': 'approved'}
        )
        if not created:
            participant.status = 'approved'
            participant.save()
        
        messages.success(request, f'Approved {join_request.user.email}')
    
    elif action == 'reject':
        join_request.status = 'rejected'
        join_request.save()
        
        # Update participant if exists
        participant = MeetingParticipant.objects.filter(
            meeting=join_request.meeting,
            user=join_request.user
        ).first()
        if participant:
            participant.status = 'rejected'
            participant.save()
        
        messages.warning(request, f'Rejected {join_request.user.email}')
    
    return redirect('meeting_room', room_id=join_request.meeting.room_id)

@login_required
@require_GET
def get_pending_requests(request, room_id=None):
    """API endpoint to get pending requests for real-time updates"""
    try:
        if room_id:
            # Get requests for specific meeting
            meeting = get_object_or_404(MeetingRoom, room_id=room_id, host=request.user)
            requests = JoinRequest.objects.filter(
                meeting=meeting,
                status='pending'
            ).select_related('user', 'meeting')
        else:
            # Get all pending requests for user's meetings
            requests = JoinRequest.objects.filter(
                meeting__host=request.user,
                status='pending'
            ).select_related('user', 'meeting')
        
        requests_data = []
        for req in requests:
            requests_data.append({
                'id': req.id,
                'user_email': req.user.email,
                'meeting_title': req.meeting.title,
                'room_id': req.meeting.room_id,
                'requested_at': req.requested_at.isoformat(),
            })
        
        return JsonResponse({
            'success': True,
            'pending_count': len(requests_data),
            'new_requests_count': len(requests_data),  # For demo, all are "new"
            'requests': requests_data
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
def create_member(request):
    data = json.loads(request.body)
    member, created = RoomMember.objects.get_or_create(
        name=data['name'],
        uid=data['UID'],
        room_name=data['room_name']
    )

    return JsonResponse({'name':data['name']}, safe=False)


def get_member(request):
    uid = request.GET.get('UID')
    room_name = request.GET.get('room_name')

    member = RoomMember.objects.get(
        uid=uid,
        room_name=room_name,
    )
    return JsonResponse({'name':member.name}, safe=False)

@csrf_exempt
def delete_member(request):
    try:
        data = json.loads(request.body)
        member = RoomMember.objects.get(
            name=data['name'],
            uid=data['UID'],
            room_name=data['room_name']
        )
        member.delete()
        return JsonResponse('Member deleted', safe=False)
    except RoomMember.DoesNotExist:
        # Member already deleted or never existed - this is fine
        return JsonResponse('Member not found (already deleted)', safe=False)
    except Exception as e:
        # Log other errors but don't crash
        print(f"Error deleting member: {e}")
        return JsonResponse({'error': str(e)}, status=400)


# ─── HPC Proxy Views ──────────────────────────────────────────────
# Browser cannot reach HPC directly, so Django proxies the requests.

@csrf_exempt
@login_required
def proxy_transcribe(request):
    """Proxy audio transcription to HPC server."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        # Forward the uploaded file and form fields to HPC
        files = {}
        if 'file' in request.FILES:
            uploaded = request.FILES['file']
            files['file'] = (uploaded.name, uploaded.read(), uploaded.content_type)

        data = {
            'language_mode': request.POST.get('language_mode', 'en'),
            'target_language': request.POST.get('target_language', ''),
        }

        resp = http_requests.post(
            f'{HPC_SERVER_URL}/transcribe',
            files=files,
            data=data,
            timeout=120, # Increased to 120s for hybrid model
        )

        return JsonResponse(resp.json(), status=resp.status_code)

    except http_requests.exceptions.ConnectionError:
        return JsonResponse({
            'error': f'Cannot connect to HPC server at {HPC_SERVER_URL}',
            'success': False,
        }, status=502)
    except http_requests.exceptions.Timeout:
        return JsonResponse({
            'error': 'HPC server timed out',
            'success': False,
        }, status=504)
    except Exception as e:
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@csrf_exempt
@login_required
def proxy_translate(request):
    """Proxy text translation to HPC server."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        data = {
            'text': request.POST.get('text', ''),
            'source_language': request.POST.get('source_language', ''),
            'target_language': request.POST.get('target_language', ''),
        }

        resp = http_requests.post(
            f'{HPC_SERVER_URL}/translate',
            data=data,
            timeout=15,
        )

        return JsonResponse(resp.json(), status=resp.status_code)

    except http_requests.exceptions.ConnectionError:
        return JsonResponse({
            'error': f'Cannot connect to HPC server at {HPC_SERVER_URL}',
            'success': False,
        }, status=502)
    except http_requests.exceptions.Timeout:
        return JsonResponse({
            'error': 'HPC server timed out',
            'success': False,
        }, status=504)
    except Exception as e:
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@csrf_exempt
@login_required
def proxy_hpc_health(request):
    """Check HPC server health."""
    try:
        resp = http_requests.get(f'{HPC_SERVER_URL}/health', timeout=5)
        return JsonResponse(resp.json(), status=resp.status_code)
    except Exception:
        return JsonResponse({'status': 'unreachable', 'success': False}, status=502)