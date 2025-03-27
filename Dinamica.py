import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from utils import *
from scipy.signal import savgol_filter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Read input video
video_path = '/Users/valen/Downloads/Fisica/salto2.MP4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
tiempo_por_frame = 1/fps

print(frame_width, frame_height, fps)

longitud_brazo_x = 0.65
longitud_pierna_y = 0.94

# Define output video resolution (e.g., half of original)
output_width = frame_width
output_height = frame_height

# Defino cuales son las articulaciones que me interesa estudiar
articulaciones = [
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.LEFT_INDEX,
    mp_pose.PoseLandmark.LEFT_SHOULDER
]

columns = ['frame_number']

for landmark in articulaciones:
    columns.append(landmark.name + '_X')
    columns.append(landmark.name + '_Y')

columns.append("VelocidadAngular")
columns.append("AceleracionAngular")
columns.append("FuerzaGemelo")

columns.append("Velocidad(Rodilla)_Y")
columns.append("Velocidad(Rodilla)_X")
columns.append("Velocidad(Cadera)_Y")
columns.append("Velocidad(Cadera)_X")

columns.append("Aceleracion(Rodilla)_Y")
columns.append("Aceleracion(Rodilla)_X")
columns.append("Aceleracion(Cadera)_Y")
columns.append("Aceleracion(Cadera)_X")

# Prepare output video
out = cv2.VideoWriter('/Users/valen/Downloads/Fisica/tracked_salto.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

# Prepare CSV file for landmark data
csv_file_path = '/Users/valen/Downloads/Fisica/landmarks.csv'
df = pd.DataFrame(columns=columns)

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect landmarks
    result = pose.process(rgb_frame)

    pose_row = {'frame_number': frame_index}
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Extract landmark positions
        for landmark in articulaciones:
            pos = landmarks[landmark]
            pose_row[landmark.name + '_X'] = pos.x * (longitud_brazo_x / 0.3214562238)
            pose_row[landmark.name + '_Y'] = (1-pos.y) * (longitud_pierna_y / 0.2489993274)

        # Draw landmarks
        mp_drawing.draw_landmarks(
            rgb_frame, 
            result.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS)

    df = pd.concat([df, pd.DataFrame([pose_row])], ignore_index=True)
    if(frame_index>0):
        pos_prev_left_knee, pos_prev_left_ankle, pos_prev_left_heel = extraer_posiciones(df, frame_index-1, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL')
        pos_actual_left_knee, pos_actual_left_ankle, pos_actual_left_heel, pos_actual_left_foot_index, pos_actual_left_knee = extraer_posiciones(df, frame_index, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX', 'LEFT_KNEE')
        
        # VELOCIDAD ANGULAR
        angulo_anterior = calculate_angle((pos_prev_left_knee[0], pos_prev_left_knee[1]), (pos_prev_left_ankle[0], pos_prev_left_ankle[1]), (pos_prev_left_heel[0], pos_prev_left_heel[1]))
        angulo_actual = calculate_angle((pos_actual_left_knee[0], pos_actual_left_knee[1]), (pos_actual_left_ankle[0], pos_actual_left_ankle[1]), (pos_actual_left_heel[0], pos_actual_left_heel[1]))
        vel_angular = velocidad_angular(angulo_anterior, angulo_actual, tiempo_por_frame)
        df.loc[df["frame_number"] == frame_index, "VelocidadAngular"] = vel_angular
    # Convert back to BGR for video writing
    output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Write the frame to the output video
    out.write(output_frame)
    
    frame_index += 1

# create new dataframe with smoothed data
df_nuevo = pd.DataFrame(columns=columns)
df_nuevo['frame_number'] = df['frame_number']

columnas_a_suavizar = [
    'LEFT_ANKLE_X', 'LEFT_ANKLE_Y',
    'LEFT_HEEL_X', 'LEFT_HEEL_Y',
    'LEFT_FOOT_INDEX_X', 'LEFT_FOOT_INDEX_Y',
    'LEFT_KNEE_X', 'LEFT_KNEE_Y',
    'LEFT_HIP_X', 'LEFT_HIP_Y','LEFT_INDEX_X','LEFT_INDEX_Y',
    'LEFT_SHOULDER_X','LEFT_SHOULDER_Y'
]

# Aplicar el filtro Savitzky-Golay a cada columna
for columna in columnas_a_suavizar:
    df_nuevo[columna] = savgol_filter(df[columna], window_length=20, polyorder=2)

for i in range(0, frame_index-1):
    pos_prev_left_knee, pos_prev_left_ankle, pos_prev_left_heel, pos_prev_left_hip = extraer_posiciones(df_nuevo, i, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_HIP')
    pos_actual_left_knee, pos_actual_left_ankle, pos_actual_heel, pos_actual_left_hip = extraer_posiciones(df_nuevo, i+1, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_HIP')

    # VELOCIDAD
    velocidad_cadera = velocidad_instantanea(pos_prev_left_hip, pos_actual_left_hip, tiempo_por_frame)
    velocidad_rodilla = velocidad_instantanea(pos_prev_left_knee, pos_actual_left_knee, tiempo_por_frame)
    df_nuevo.loc[df_nuevo["frame_number"] == i, "Velocidad(Cadera)_X"] = velocidad_cadera[0]
    df_nuevo.loc[df_nuevo["frame_number"] == i, "Velocidad(Cadera)_Y"] = velocidad_cadera[1]
    df_nuevo.loc[df_nuevo["frame_number"] == i, "Velocidad(Rodilla)_X"] = velocidad_rodilla[0]
    df_nuevo.loc[df_nuevo["frame_number"] == i, "Velocidad(Rodilla)_Y"] = velocidad_rodilla[1]
    
    # VELOCIDAD ANGULAR
    angulo_anterior = calculate_angle((pos_prev_left_knee[0], pos_prev_left_knee[1]), (pos_prev_left_ankle[0], pos_prev_left_ankle[1]), (pos_prev_left_heel[0], pos_prev_left_heel[1]))
    angulo_actual = calculate_angle((pos_actual_left_knee[0], pos_actual_left_knee[1]), (pos_actual_left_ankle[0], pos_actual_left_ankle[1]), (pos_actual_heel[0], pos_actual_heel[1]))
    vel_angular = velocidad_angular(angulo_anterior, angulo_actual, tiempo_por_frame)
    df_nuevo.loc[df_nuevo["frame_number"] == i, "VelocidadAngular"] = vel_angular

df_nuevo['VelocidadAngular'] = savgol_filter(df_nuevo['VelocidadAngular'], window_length=11, polyorder=2)

# Suavizo la velocidad de la cadera en Y
df_nuevo['Velocidad(Cadera)_Y'] = savgol_filter(df_nuevo['Velocidad(Cadera)_Y'], window_length=20, polyorder=2)

for i in range(0, frame_index-1):
    
    # ACELERACION
    aceleracion_actual_cadera = aceleracion_instantanea(
        df_nuevo.loc[df_nuevo["frame_number"] == i+1, "Velocidad(Cadera)_X"].iloc[0],
        df_nuevo.loc[df_nuevo["frame_number"] == i, "Velocidad(Cadera)_X"].iloc[0],
        df_nuevo.loc[df_nuevo["frame_number"] == i+1, "Velocidad(Cadera)_Y"].iloc[0],
        df_nuevo.loc[df_nuevo["frame_number"] == i, "Velocidad(Cadera)_Y"].iloc[0], tiempo_por_frame)
    
    aceleracion_actual_rodilla = aceleracion_instantanea(
        df_nuevo.loc[df_nuevo["frame_number"] == i+1, "Velocidad(Rodilla)_X"].iloc[0],
        df_nuevo.loc[df_nuevo["frame_number"] == i, "Velocidad(Rodilla)_X"].iloc[0],
        df_nuevo.loc[df_nuevo["frame_number"] == i+1, "Velocidad(Rodilla)_Y"].iloc[0],
        df_nuevo.loc[df_nuevo["frame_number"] == i, "Velocidad(Rodilla)_Y"].iloc[0], tiempo_por_frame)
    
    df_nuevo.loc[df_nuevo["frame_number"] == i, "Aceleracion(Rodilla)_X"] = aceleracion_actual_rodilla[0]
    df_nuevo.loc[df_nuevo["frame_number"] == i, "Aceleracion(Rodilla)_Y"] = aceleracion_actual_rodilla[1]
    df_nuevo.loc[df_nuevo["frame_number"] == i, "Aceleracion(Cadera)_X"] = aceleracion_actual_cadera[0]
    df_nuevo.loc[df_nuevo["frame_number"] == i, "Aceleracion(Cadera)_Y"] = aceleracion_actual_cadera[1]
    
    vel_prev, vel_actual = extraer_velocidad_angular(df_nuevo, i), extraer_velocidad_angular(df_nuevo, i+1)
    acel_angular = (vel_actual - vel_prev) / tiempo_por_frame
    df_nuevo.loc[df_nuevo["frame_number"] == i, "AceleracionAngular"] = acel_angular
    
df_nuevo['Aceleracion(Cadera)_Y'] = savgol_filter(df_nuevo['Aceleracion(Cadera)_Y'], window_length=20, polyorder=2)
df_nuevo['AceleracionAngular'] = savgol_filter(df_nuevo['AceleracionAngular'], window_length=11, polyorder=2)
df_nuevo.interpolate(method='linear',inplace=True)

for i in range(0, frame_index-1):
    pos_left_knee, pos_left_ankle, pos_left_heel, pos_left_foot_index = extraer_posiciones(df_nuevo, i, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX')
    # Posiciones normalizadas para graficar en el video
    pos_left_knee_normalizada, pos_left_ankle_normalizada, pos_left_heel_normalizada, pos_left_foot_index_normalizada = extraer_posiciones(df, i, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX')
    magnitud_fuerza_gemelo = calcular_fuerza_gemelo(df_nuevo, i, pos_left_knee, pos_left_ankle, pos_left_heel, pos_left_foot_index)
    df_nuevo.loc[df_nuevo["frame_number"] == i, "FuerzaGemelo"] = magnitud_fuerza_gemelo

df_nuevo['FuerzaGemelo'] = savgol_filter(df_nuevo['FuerzaGemelo'], window_length=11, polyorder=2)
df_nuevo.interpolate(method='linear',inplace=True)
df_nuevo.to_csv(csv_file_path, index=False)

cap.release()
out.release()

# Read input video
video_path = '/Users/valen/Downloads/Fisica/salto2.MP4'
cap2 = cv2.VideoCapture(video_path)
# Prepare output video
out2 = cv2.VideoWriter('/Users/valen/Downloads/Fisica/tracked_salto2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

# Recorrer el video para dibujar el vector fuerza gemelo
frame_index = 0
while cap2.isOpened():
    ret, frame = cap2.read()
    if not ret:
        break

    # Resize the frame for faster processing
    resized_frame = cv2.resize(frame, (output_width, output_height))
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    tiempo = round(frame_index,2)
    cv2.putText(rgb_frame, "Tiempo:"+ str(tiempo),(20,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    pos_left_knee, pos_left_ankle, pos_left_heel = extraer_posiciones(df_nuevo, frame_index, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL')
    magnitud_fuerza_gemelo = df_nuevo.loc[df_nuevo["frame_number"] == frame_index, "FuerzaGemelo"].iloc[0]
    graficar_vector_fuerza(rgb_frame,magnitud_fuerza_gemelo,pos_left_ankle,pos_left_knee,pos_left_heel,output_width,output_height)
    
    # Convert back to BGR for video writing
    output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Write the frame to the output video
    out2.write(output_frame)
    
    frame_index += 1

# Release resources
cap2.release()
pose.close()
out2.release()

cv2.destroyAllWindows()
