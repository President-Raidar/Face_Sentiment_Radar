import sys
import os
import cv2
from PIL import Image
import PIL.ImageQt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QFileDialog, QLabel, QWidget, QSplitter)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from transformers import pipeline

# 기존 함수 가져오기
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

class FaceEmotionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("얼굴 감정 분석기")
        self.setGeometry(100, 100, 1000, 600)
        
        # 감정 분석 모델 초기화
        self.model_name = 'dima806/facial_emotions_image_detection'
        self.emotion_analyzer = None
        self.image_path = None
        
        # UI 초기화
        self.init_ui()
        
    def init_ui(self):
        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 파일 선택 버튼
        self.file_button = QPushButton("이미지 파일 선택")
        self.file_button.clicked.connect(self.select_image)
        button_layout.addWidget(self.file_button)
        
        # 분석 버튼
        self.analyze_button = QPushButton("감정 분석하기")
        self.analyze_button.clicked.connect(self.analyze_emotion)
        self.analyze_button.setEnabled(False)
        button_layout.addWidget(self.analyze_button)
        
        main_layout.addLayout(button_layout)
        
        # 이미지 표시 영역 구성
        self.splitter = QSplitter(Qt.Horizontal)
        
        # 원본 이미지 표시 영역
        self.original_image_label = QLabel("원본 이미지")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("border: 1px solid black;")
        self.original_image_label.setMinimumSize(300, 300)
        
        # 분석 결과 이미지 표시 영역
        self.result_image_label = QLabel("분석 결과 이미지")
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setStyleSheet("border: 1px solid black;")
        self.result_image_label.setMinimumSize(300, 300)
        
        # 스플리터에 이미지 영역 추가
        self.splitter.addWidget(self.original_image_label)
        self.splitter.addWidget(self.result_image_label)
        
        main_layout.addWidget(self.splitter)
        
        # 상태 표시 레이블
        self.status_label = QLabel("파일을 선택해 주세요.")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
    def select_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "이미지 파일 선택", "", 
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)", 
            options=options
        )
        
        if file_name:
            self.image_path = file_name
            self.status_label.setText(f"선택된 파일: {os.path.basename(file_name)}")
            
            # 원본 이미지 표시
            self.display_original_image()
            
            # 분석 버튼 활성화
            self.analyze_button.setEnabled(True)
    
    def display_original_image(self):
        # 이미지 읽기
        image = cv2.imread(self.image_path)
        if image is None:
            self.status_label.setText(f"이미지를 불러올 수 없습니다: {self.image_path}")
            return
            
        # OpenCV 이미지를 QPixmap으로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        q_img = QImage(image_rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 라벨 크기에 맞게 이미지 조정
        pixmap = pixmap.scaled(self.original_image_label.width(), 
                              self.original_image_label.height(),
                              Qt.KeepAspectRatio, 
                              Qt.SmoothTransformation)
        
        # 이미지 표시
        self.original_image_label.setPixmap(pixmap)
    
    def analyze_emotion(self):
        if not self.image_path:
            self.status_label.setText("이미지를 먼저 선택해 주세요.")
            return
            
        self.status_label.setText("감정 분석 중...")
        
        try:
            # 감정 분석 모델이 아직 로드되지 않았다면 로드
            if self.emotion_analyzer is None:
                self.status_label.setText("모델 로드 중...")
                QApplication.processEvents()  # UI 업데이트
                try:
                    self.emotion_analyzer = pipeline('image-classification', 
                                                   model=self.model_name, 
                                                   device=0 if torch.cuda.is_available() else -1)
                except:
                    # GPU 사용할 수 없을 경우 CPU로 대체
                    self.emotion_analyzer = pipeline('image-classification', 
                                                   model=self.model_name)
                
            # 얼굴 감지 및 크롭
            cropped_image = detect_and_crop_face(self.image_path)
            
            # 감정 분석
            results = self.emotion_analyzer(cropped_image)
            
            # 결과 표시
            result_image = detect_and_display_face_with_emotion(self.image_path, results)
            
            # 분석 결과 이미지 표시
            # PIL 이미지를 QImage로 변환
            # 다른 방식으로 PIL에서 QImage로 변환
            img_data = result_image.convert("RGBA").tobytes("raw", "RGBA")
            qim = QImage(img_data, result_image.width, result_image.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qim)
            pixmap = pixmap.scaled(self.result_image_label.width(), 
                                 self.result_image_label.height(),
                                 Qt.KeepAspectRatio, 
                                 Qt.SmoothTransformation)
            self.result_image_label.setPixmap(pixmap)
            
            # 분석 결과 표시
            emotion = results[0]['label']
            score = results[0]['score']
            self.status_label.setText(f"감정 분석 결과: {emotion} ({score*100:.2f}%)")
            
        except Exception as e:
            self.status_label.setText(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    # PyQt5 앱 실행
    app = QApplication(sys.argv)
    
    # PyTorch import (GPU 사용 가능 여부 확인용)
    try:
        import torch
    except ImportError:
        print("PyTorch를 찾을 수 없습니다. GPU 가속 없이 실행됩니다.")
    
    # 메인 윈도우 생성 및 표시
    window = FaceEmotionApp()
    window.show()
    
    # 앱 실행
    sys.exit(app.exec_())