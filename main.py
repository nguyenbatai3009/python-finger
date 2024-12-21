import cv2
import mediapipe as mp
import math

class NhanDienCuChi:
    def __init__(self):
        # Khởi tạo MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        
    def phat_hien_cu_chi(self, img):
        # Chuyển đổi màu BGR sang RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Xử lý ảnh để phát hiện bàn tay
        results = self.hands.process(imgRGB)
        
        # Danh sách lưu các cử chỉ được phát hiện
        cu_chi = []
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Vẽ các điểm mốc trên bàn tay
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
                # Phân tích cử chỉ
                cu_chi.append(self.phan_tich_cu_chi(handLms))
                
        return img, cu_chi
    
    def phan_tich_cu_chi(self, handLms):
        # Lấy tọa độ các điểm mốc
        landmarks = []
        for lm in handLms.landmark:
            landmarks.append([lm.x, lm.y])
        
        # Phân tích các ngón tay
        ngon_cai = self.kiem_tra_ngon_cai(landmarks)
        ngon_tro = self.kiem_tra_ngon(landmarks, 8, 6)
        ngon_giua = self.kiem_tra_ngon(landmarks, 12, 10)
        ngon_ap = self.kiem_tra_ngon(landmarks, 16, 14)
        ngon_ut = self.kiem_tra_ngon(landmarks, 20, 18)
        
        # Xác định cử chỉ
        if ngon_tro and not ngon_giua and not ngon_ap and not ngon_ut:
            return "Chỉ"
        elif ngon_tro and ngon_giua and not ngon_ap and not ngon_ut:
            return "Chữ V"
        elif ngon_tro and ngon_giua and ngon_ap and ngon_ut:
            return "Bàn tay mở"
        elif not ngon_tro and not ngon_giua and not ngon_ap and not ngon_ut:
            return "Nắm tay"
        else:
            return "Không xác định"
    
    def kiem_tra_ngon_cai(self, landmarks):
        return landmarks[4][0] > landmarks[3][0]
    
    def kiem_tra_ngon(self, landmarks, tip_idx, pip_idx):
        return landmarks[tip_idx][1] < landmarks[pip_idx][1]

def main():
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    detector = NhanDienCuChi()
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        # Phát hiện cử chỉ
        img, cu_chi = detector.phat_hien_cu_chi(img)
        
        # Hiển thị kết quả
        for i, gesture in enumerate(cu_chi):
            cv2.putText(img, f"Cu chi {i+1}: {gesture}", (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị hình ảnh
        cv2.imshow("Nhan Dien Cu Chi Tay", img)
        
        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
