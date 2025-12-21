from django.urls import path
from .views import process_image, process_video, process_realtime, video_feed, realtime_result, start_prediction, stop_prediction, download_video, delete_video
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", process_image, name="process_image"),
    path("video/", process_video, name="process_video"),
    path("realtime/", process_realtime, name="process_realtime"),
    path("video_feed/", video_feed, name="video_feed"),
    path("realtime_result/", realtime_result, name="realtime_result"),
    path("start_prediction/", start_prediction, name="start_prediction"),
    path("stop_prediction/", stop_prediction, name="stop_prediction"),    
    path("download_video/", download_video, name="download_video"),
    path("delete_video/", delete_video, name="delete_video"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)