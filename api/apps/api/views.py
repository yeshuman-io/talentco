from django.http import JsonResponse


def health_check(request):
    """Basic health check endpoint."""
    return JsonResponse({
        'status': 'healthy',
        'service': 'talentco-api',
        'version': '0.1.0'
    }) 