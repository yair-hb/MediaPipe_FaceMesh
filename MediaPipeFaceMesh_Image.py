import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

index_list = [70,63,105,66,107,336,296,334,293,300,122,196,3,51,281,248,419,351,37,0,267]

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces = 3,
    min_detection_confidence =0.5) as face_mesh:

    imagen = cv2.imread('imagen.jpg')
    height, width, _ = imagen.shape
    imagenRGB = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(imagenRGB)

    print ('face landmarks:', resultados.multi_face_landmarks)

    if resultados.multi_face_landmarks is not None:
        for face_landmarks in resultados.multi_face_landmarks:
            for index in index_list:
                x = int (face_landmarks.landmark[index].x*width)
                y = int (face_landmarks.landmark[index].y*height)
                cv2.circle(imagen, (x,y),2,(255,0,255),2)
            cv2.imshow('imagen', imagen)
            cv2.waitKey(0)

    cv2.imshow('imagen', imagen)
    cv2.waitKey(0)
cv2.destroyAllWindows()
