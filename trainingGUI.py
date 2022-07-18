import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Projeto BECN - Gravação")
        self.VBL = QVBoxLayout()
        
        self.InfoLabel = QLabel(self)
        self.InfoLabel.setText(" ")
        self.VBL.addWidget(self.InfoLabel)
        
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)
        
        self.ActionLabel = QLabel(self)
        self.ActionLabel.setText("Sinal:")
        self.VBL.addWidget(self.ActionLabel)
        
        self.InputAction = QLineEdit(self)
        self.VBL.addWidget(self.InputAction)
        
        
        self.StartBTN = QPushButton("Começar Gravação")
        self.StartBTN.clicked.connect(self.StartRec)
        self.VBL.addWidget(self.StartBTN)
        
        self.CancelBTN = QPushButton("Cancelar")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)
        
        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        
           
        self.setLayout(self.VBL)
    
    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))
        self.InfoLabel.setText(self.Worker1.ChangeLabel())

    def CancelFeed(self):
        self.Worker1.stop()
    
    def StartRec(self):
        self.Worker1.SetAction(self.InputAction.text())
        self.Worker1.start()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    
    #MediaPipe solutions - reconhecimento e desenho dos pontos na mão
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    # Caminho que guarda o numpy array com os pontos extraidos
    DATA_PATH = os.path.join('MP_Data')

    # Sinal
    action = ''
    info = ''

    # Representa a quantidade de sequências de frames que tem os dados
    no_sequences = 30

    # Representa a quantidade de frames que cada sequência possui
    sequence_lenght = 30
    
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        
        self.make_action_dirs(self.action)
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Extrai os frames de cada video para cada ação
            for sequence in range(self.no_sequences):
                for frame_num in range(self.sequence_lenght + 1):
                    
                    ret, frame = Capture.read()
                    if ret:
                        #Processa previsoes
                        image, results = self.mediapipe_detection(frame, holistic)

                        #Desenha os pontos
                        self.draw_landmarks(image, results)
                        
                        image = cv2.flip(image, 1)
                        
                        if frame_num == 1:
                           # Tempo de espera entre um vídeo e outro
                           cv2.waitKey(2000) 
     
                        # Interface e espera ao começar gravações
                        if frame_num == 0:
                           #Textos para indicar inicio de gravação
                            image[:] = (0,0,0)
                            cv2.putText(image, "INICIANDO GRAVACAO DE SINAL", 
                           (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, "Gravando para {} Video: {}".format(self.action, sequence), 
                           (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                        else:
                            cv2.putText(image, "Gravando para {} Video: {} Frame: {}".format(self.action, sequence, frame_num), 
                            (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                            # Extrai os pontos de um frame e salva como um arquivo .npy (numpy array)
                            keypoints = self.extract_keypoints(results)
                            npy_path = os.path.join(self.DATA_PATH, self.action, str(sequence), str(frame_num))
                            np.save(npy_path, keypoints)
                            

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        ConvertToQtFormat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                        Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                        self.ImageUpdate.emit(Pic)
                        if (self.ThreadActive == False):
                            image[:] = (0,0,0)
                            ConvertToQtFormat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                            Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                            self.ImageUpdate.emit(Pic)
                            break
            
            Capture.release()
            
        
    def stop(self):
        self.ThreadActive = False
        self.quit()
        
    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACE_CONNECTIONS,
                                 self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                 self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                 self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                 self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                 self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks            else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else              np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def make_action_dirs(self, action):
        for sequence in range(self.no_sequences):
            try: # Cria pastas e subpastas
                os.makedirs(os.path.join(self.DATA_PATH, action, str(sequence)))
            except: # Caso já exista passa para próxima pasta
                pass
            
    def SetAction(self, action):
        self.action = action
    
    def ChangeLabel(self):
        return self.info
    
if __name__ ==  "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())