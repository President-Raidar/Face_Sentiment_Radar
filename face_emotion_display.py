import cv2
from PIL import Image
from transformers import pipeline

def detect_and_crop_face(image_path):
    """Detects the first face in the image and returns the cropped face as a PIL Image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print("No faces detected.")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    x, y, w, h = faces[0]
    cropped_face = image[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))

def detect_and_display_face_with_emotion(image_path, results):
    """Draws rectangles and emotion labels on detected faces and returns the annotated PIL Image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print("No faces detected.")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        top_emotion = results[0]
        emotion_text = f"{top_emotion['label']} ({top_emotion['score'] * 100:.2f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_size, baseline = cv2.getTextSize(emotion_text, font, font_scale, thickness)
        text_width, text_height = text_size
        # 텍스트 바탕 사각형 좌표 계산
        rect_x1 = x
        rect_y1 = y - text_height - baseline - 6 if y - text_height - baseline - 6 > 0 else y + h + 6
        rect_x2 = x + text_width + 8
        rect_y2 = rect_y1 + text_height + baseline + 6
        # 바탕 사각형 그리기 (흰색, 불투명)
        cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        # 텍스트 그리기 (파란색)
        text_org = (x + 4, rect_y2 - baseline - 3)
        cv2.putText(image, emotion_text, text_org, font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

if __name__ == "__main__":
    # 모델 이름과 파이프라인 준비
    model_name = 'dima806/facial_emotions_image_detection'
    emotion_analyzer = pipeline('image-classification', model=model_name, device=0)

    # 이미지 경로 지정
    image_path = 'test_trump.jpeg'  # 분석할 이미지 파일 경로

    # 얼굴 감지 및 크롭
    cropped_image = detect_and_crop_face(image_path)

    # 감정 분석
    results = emotion_analyzer(cropped_image)

    # 원본 이미지에 결과 표시
    result_image = detect_and_display_face_with_emotion(image_path, results)

    # 결과 이미지 표시
    result_image.show()