from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .face_detector import RetinaFaceDetector
from .age_gender_model import load_age_gender_model

import cv2
import torch
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
detector = RetinaFaceDetector("./weights/Resnet50_Final.pth")
age_gender_model = load_age_gender_model("./weights/age_gender_resnet50.pt")

# GLOBAL realtime results
realtime_age = 0
realtime_gender = "None"
is_predicting = False

# =============================
# PREPROCESS
# =============================
def preprocess_face(img):
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).float().permute(2, 0, 1)
    return img.unsqueeze(0).to(device)


# ================================================================
# 1️⃣ IMAGE PROCESSING
# ================================================================
def process_image(request):
    ctx_gender = None
    ctx_age = None
    result = None
    error = None

    if request.method == "POST":
        file = request.FILES.get("image")
        if not file:
            return render(request, "image.html", {"error": "No file uploaded"})

        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)

        img = cv2.imread(file_path)
        if img is None:
            return render(request, "image.html", {"error": "Cannot read image"})

        detections = detector.detect(img)
        if detections is None or len(detections) == 0:
            return render(request, "image.html", {"error": "No face detected"})

        with torch.no_grad():
            x1, y1, x2, y2 = map(int, detections[0][:4])

            # Crop safe region
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)

            face = img[y1:y2, x1:x2]
            if face.size == 0:
                return render(request, "image.html", {"error": "Invalid face region"})

            pred_gender, pred_age = age_gender_model(preprocess_face(face))

            gender = "Male" if pred_gender.item() < 0.5 else "Female"
            age = int(pred_age.item() * 100)

            ctx_gender = gender
            ctx_age = age

            # Draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{gender}, {age}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)

        # Save output
        out_name = "output_" + filename
        out_path = fs.path(out_name)
        cv2.imwrite(out_path, img)

        result = fs.url(out_name)

    return render(request, "image.html", {
        "result": result,
        "gender_text": ctx_gender,
        "age_val": ctx_age,
        "error": error
    })


# ================================================================
# 3️⃣ REALTIME WEBCAM STREAMING
# ================================================================
def gen_frames():
    global realtime_age, realtime_gender, is_predicting

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("❌ Webcam not available")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Only predict when enabled
        if is_predicting:
            detections = detector.detect(frame)

            if detections is not None and len(detections) > 0:
                x1, y1, x2, y2 = map(int, detections[0][:4])

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    with torch.no_grad():
                        pred_gender, pred_age = age_gender_model(preprocess_face(face))

                    gender = "Male" if pred_gender.item() < 0.5 else "Female"
                    age = int(pred_age.item() * 100)

                    # Update global state
                    realtime_age = age
                    realtime_gender = gender

                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{gender}, {age}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 255, 255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cam.release()




def process_realtime(request):
    return render(request, "realtime.html")


def video_feed(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


from django.http import JsonResponse

def realtime_result(request):
    global realtime_age, realtime_gender
    return JsonResponse({
        "age": realtime_age,
        "gender": realtime_gender,
    })



def start_prediction(request):
    global is_predicting
    is_predicting = True
    return JsonResponse({"status": "started"})

def stop_prediction(request):
    global is_predicting, realtime_age, realtime_gender
    is_predicting = False
    realtime_age = None
    realtime_gender = None
    return JsonResponse({"status": "stopped"})



import uuid
from django.http import FileResponse, HttpResponse, JsonResponse


def download_video(request):
    file_url = request.GET.get("file")
    file_path = os.path.join(settings.MEDIA_ROOT, file_url.replace("/media/", ""))

    if os.path.exists(file_path):
        return FileResponse(
            open(file_path, "rb"),
            as_attachment=True,
            filename=os.path.basename(file_path)
        )
    return HttpResponse("File not found.")


def delete_video(request):
    file_url = request.GET.get("file")
    file_path = os.path.join(settings.MEDIA_ROOT, file_url.replace("/media/", ""))

    if os.path.exists(file_path):
        os.remove(file_path)
        return render(request, "video.html", {"msg": "Đã xóa video."})

    return HttpResponse("Không tìm thấy file để xóa.")



# ================= VIDEO STREAM FROM UPLOAD =================


def process_video_file(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Cannot open input video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if not fps or fps <= 1:
        fps = 25

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not out.isOpened():
        raise Exception("Cannot open VideoWriter")

    frame_idx = 0

    # buffer trong 1 giây
    sec_genders = []
    sec_ages = []

    # giá trị đang hiển thị
    current_gender = None
    current_age = None

    # timeline kết quả theo từng giây
    timeline = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        if detections is not None and len(detections) > 0:
            x1, y1, x2, y2 = map(int, detections[0][:4])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                with torch.no_grad():
                    g, a = age_gender_model(preprocess_face(face))

                gender = "Male" if g.item() < 0.5 else "Female"
                age = int(a.item() * 100)

                sec_genders.append(gender)
                sec_ages.append(age)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ===== KẾT THÚC 1 GIÂY =====
        if frame_idx > 0 and frame_idx % fps == 0 and sec_genders:
            current_gender = max(set(sec_genders), key=sec_genders.count)
            current_age = int(sum(sec_ages) / len(sec_ages))

            timeline.append({
                "second": frame_idx // fps,
                "gender": current_gender,
                "age": current_age
            })

            sec_genders.clear()
            sec_ages.clear()

        # ===== VẼ TEXT THEO GIÂY =====
        if current_gender is not None and current_age is not None:
            cv2.putText(
                frame,
                f"{current_gender}, {current_age}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    return timeline


import subprocess

def convert_to_web_mp4(input_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)



from django.core.files.storage import FileSystemStorage
from django.conf import settings
import uuid
import os

def process_video(request):
    if request.method == "POST":
        video_file = request.FILES.get("video")
        if not video_file:
            return render(request, "video.html", {"error": "Chưa chọn video"})

        fs = FileSystemStorage()

        uid = uuid.uuid4().hex[:8]
        input_name = fs.save(f"{uid}_{video_file.name}", video_file)
        input_path = fs.path(input_name)

        raw_output_name = f"raw_{input_name}"
        raw_output_path = os.path.join(settings.MEDIA_ROOT, raw_output_name)

        final_output_name = f"output_{input_name}"
        final_output_path = os.path.join(settings.MEDIA_ROOT, final_output_name)

        # XỬ LÝ VIDEO + LẤY KẾT QUẢ THEO GIÂY
        timeline = process_video_file(input_path, raw_output_path)

        # CHUẨN HÓA VIDEO CHO WEB
        convert_to_web_mp4(raw_output_path, final_output_path)

        if os.path.exists(raw_output_path):
            os.remove(raw_output_path)

        return render(request, "video.html", {
            "input_video": fs.url(input_name),
            "output_video": fs.url(final_output_name),
            "timeline": timeline,
        })

    return render(request, "video.html")
