import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils import *
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# # Rutas
video_paths = [
    '/Users/valen/Downloads/Fisica/tincho-sentadilla.MP4',
    '/Users/valen/Downloads/Fisica/aitor_sentadilla.MP4',
    #'/Users/camia/Desktop/proyecto/test.mp4',
]
output_csv_paths = [
    '/Users/valen/Downloads/Fisica/pose_data_tincho.csv',
    '/Users/valen/Downloads/Fisica/pose_data_aitor.csv'
    #'/Users/camia/Desktop/proyecto/pose_data3.csv'
]
output_video_paths = [
    '/Users/valen/Downloads/Fisica/tracked_video_tincho.mp4',
    '/Users/valen/Downloads/Fisica/tracked_video_aitor.mp4'
    #'/Users/camia/Desktop/proyecto/tracked_video3.mp4'
]

# # Input usuario
peso_persona = 65 #kg
altura_persona = 176 #cm
peso_pesa = 140 #kg

cadera_a_rodilla = 0.52 #m
cadera_a_tobillo = 1.04 #m

# # Crear columnas del dataframe
# Procesa el video y almacena los datos de las poses. Define las columnas para un DataFrame donde se guardarán las coordenadas de las articulaciones detectadas en cada cuadro del video.

# %%
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Defino cuales son las articulaciones que me interesa estudiar
articulaciones = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

columns = ['frame_number']

for landmark in articulaciones:
    columns.append(landmark.name + '_X')
    columns.append(landmark.name + '_Y')
    columns.append(landmark.name + '_Z')

columns.append("Tiempo")
columns.append("VelocidadAngular")
columns.append("Velocidad(Rodilla)_X")
columns.append("Velocidad(Rodilla)_Y")
columns.append("Velocidad(Cadera)_X")
columns.append("Velocidad(Cadera)_Y")
columns.append("Aceleracion(Rodilla)_X")
columns.append("Aceleracion(Rodilla)_Y")
columns.append("Aceleracion(Cadera)_X")
columns.append("Aceleracion(Cadera)_Y")
columns.append("Torque(Rodilla)")
columns.append("Torque(Cadera)")


# # Código para recorrer frames del video y realizar cálculos
# Este bloque de código recorre cada frame del video, procesa la imagen utilizando MediaPipe para detectar landmarks de la pose, y guarda los datos en un DataFrame. Luego, calcula el ángulo entre las articulaciones de la cadera, la rodilla y el tobillo. Después, dibuja los landmarks detectados en el video y guarda el video procesado en el archivo de salida. Finalmente, libera los recursos utilizados y guarda los datos de la pose en un archivo CSV.
for i, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)

    # Obtener propiedades del video
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    tiempo_por_frame = 1 / video_fps

    # Inicializar DataFrame y pose para el video actual
    df_completo = pd.DataFrame(columns=columns)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    video_writer = cv2.VideoWriter(output_video_paths[i], cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    frame_number = 0  # Reiniciar el contador de fotogramas para cada video

    # Procesar cada fotograma del video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a RGB (el fotograma)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesar la imagen con MediaPipe y guardar los resultados
        results = pose.process(image)
        # Recolectar y guardar los datos de la pose en el dataframe
        pose_row = {'frame_number': frame_number}

        # Extraer posiciones
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Por cada articulacion, guarda en su posicion de X, Y, Z el resultado
            for landmark in articulaciones:
                pose_row[landmark.name + '_X'] = landmarks[landmark].x * (cadera_a_tobillo / 0.14076531)
                pose_row[landmark.name + '_Y'] = (1 - landmarks[landmark].y) * (cadera_a_rodilla / 0.20725557)
                pose_row[landmark.name + '_Z'] = landmarks[landmark].z
        else:
            for landmark in articulaciones:
                pose_row[landmark.name + '_X'] = None
                pose_row[landmark.name + '_Y'] = None
                pose_row[landmark.name + '_Z'] = None

        pose_row_df = pd.DataFrame(pose_row, index=[pose_row['frame_number']])
        df_completo = pd.concat([df_completo, pose_row_df], ignore_index=True)

        # Agregar los landmarks al gráfico
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=5, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=5, circle_radius=5))

        df_completo.loc[df_completo["frame_number"] == frame_number, "Tiempo"] = tiempo_por_frame * frame_number
        """if frame_number > 0:
            previous_frame = frame_number - 1

            pos_prev_left_hip, pos_prev_left_knee, pos_prev_left_ankle = extraer_posiciones(df_completo, previous_frame, 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE')
            pos_actual_wrist, pos_actual_left_hip, pos_actual_left_knee, pos_actual_left_ankle = extraer_posiciones(df_completo, frame_number, 'LEFT_WRIST', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE')

            # VELOCIDAD ANGULAR
            angulo_anterior = calculate_angle((pos_prev_left_hip[0], pos_prev_left_hip[1]), (pos_prev_left_knee[0], pos_prev_left_knee[1]), (pos_prev_left_ankle[0], pos_prev_left_ankle[1]))
            angulo_actual = calculate_angle((pos_actual_left_hip[0], pos_actual_left_hip[1]), (pos_actual_left_knee[0], pos_actual_left_knee[1]), (pos_actual_left_ankle[0], pos_actual_left_ankle[1]))
            vel_angular = velocidad_angular(angulo_anterior, angulo_actual, tiempo_por_frame)
            df_completo.loc[df_completo["frame_number"] == frame_number, "VelocidadAngular"] = vel_angular

            # VELOCIDAD
            velocidad_cadera = velocidad_instantanea(pos_prev_left_hip, pos_actual_left_hip, tiempo_por_frame)
            velocidad_rodilla = velocidad_instantanea(pos_prev_left_knee, pos_actual_left_knee, tiempo_por_frame)
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_X"] = velocidad_cadera[0]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_Y"] = velocidad_cadera[1]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_X"] = velocidad_rodilla[0]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_Y"] = velocidad_rodilla[1]

            # ACELERACION
            aceleracion_actual_cadera = aceleracion_instantanea(
                df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_X"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Cadera)_X"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_Y"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Cadera)_Y"].iloc[0], tiempo_por_frame)
            
            aceleracion_actual_rodilla = aceleracion_instantanea(
                df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_X"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Rodilla)_X"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_Y"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Rodilla)_Y"].iloc[0], tiempo_por_frame)
            
            df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Rodilla)_X"] = aceleracion_actual_rodilla[0]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Rodilla)_Y"] = aceleracion_actual_rodilla[1]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Cadera)_X"] = aceleracion_actual_cadera[0]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Cadera)_Y"] = aceleracion_actual_cadera[1]"""

    # Escribir el frame procesado en el video de salida
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame_number += 1

    # Liberar recursos y guardar resultados después de procesar cada video
    pose.close()
    video_writer.release()
    cap.release()

    df_completo['LEFT_HIP_Y'] = savgol_filter(df_completo['LEFT_HIP_Y'], window_length = 30, polyorder = 2)
    df_completo['LEFT_KNEE_Y'] = savgol_filter(df_completo['LEFT_KNEE_Y'], window_length = 30, polyorder = 2)
    # Calculo la velocidad a partir de las posiciones suavizadas
    for frame_number in range(1, len(df_completo)):
        previous_frame = frame_number - 1
        
        pos_prev_left_hip = (df_completo.loc[previous_frame, 'LEFT_HIP_X'], df_completo.loc[previous_frame, 'LEFT_HIP_Y'])
        pos_actual_left_hip = (df_completo.loc[frame_number, 'LEFT_HIP_X'], df_completo.loc[frame_number, 'LEFT_HIP_Y'])
        
        pos_prev_left_knee = (df_completo.loc[previous_frame, 'LEFT_KNEE_X'], df_completo.loc[previous_frame, 'LEFT_KNEE_Y'])
        pos_actual_left_knee = (df_completo.loc[frame_number, 'LEFT_KNEE_X'], df_completo.loc[frame_number, 'LEFT_KNEE_Y'])
        
        velocidad_cadera = velocidad_instantanea(pos_prev_left_hip, pos_actual_left_hip, tiempo_por_frame)
        velocidad_rodilla = velocidad_instantanea(pos_prev_left_knee, pos_actual_left_knee, tiempo_por_frame)
        df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_X"] = velocidad_cadera[0]
        df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_Y"] = velocidad_cadera[1]
        df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_X"] = velocidad_rodilla[0]
        df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_Y"] = velocidad_rodilla[1]
    
    df_completo.loc[df_completo["frame_number"] == 0, "Velocidad(Cadera)_X"] = 0
    df_completo.loc[df_completo["frame_number"] == 0, "Velocidad(Cadera)_Y"] = 0
    df_completo.loc[df_completo["frame_number"] == 0, "Velocidad(Rodilla)_X"] = 0
    df_completo.loc[df_completo["frame_number"] == 0, "Velocidad(Rodilla)_Y"] = 0       
    
    #df_completo.interpolate(method='linear',inplace=True)
    df_completo['Velocidad(Cadera)_Y'] = savgol_filter(df_completo['Velocidad(Cadera)_Y'],  window_length = 60, polyorder = 2)
    
    # Calculo la velocidad a partir de las posiciones suavizadas
    for frame_number in range(1, len(df_completo)):
        previous_frame = frame_number - 1
        aceleracion_actual_cadera = aceleracion_instantanea(
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_X"].iloc[0],
            df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Cadera)_X"].iloc[0],
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_Y"].iloc[0],
            df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Cadera)_Y"].iloc[0], tiempo_por_frame)
    
        aceleracion_actual_rodilla = aceleracion_instantanea(
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_X"].iloc[0],
            df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Rodilla)_X"].iloc[0],
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_Y"].iloc[0],
            df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Rodilla)_Y"].iloc[0], tiempo_por_frame)
    
        df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Rodilla)_X"] = aceleracion_actual_rodilla[0]
        df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Rodilla)_Y"] = aceleracion_actual_rodilla[1]
        df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Cadera)_X"] = aceleracion_actual_cadera[0]
        df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Cadera)_Y"] = aceleracion_actual_cadera[1]
    
    df_completo.loc[df_completo["frame_number"] == 0, "Aceleracion(Cadera)_X"] = 0
    df_completo.loc[df_completo["frame_number"] == 0, "Aceleracion(Cadera)_Y"] = 0
    df_completo.loc[df_completo["frame_number"] == 0, "Aceleracion(Rodilla)_X"] = 0
    df_completo.loc[df_completo["frame_number"] == 0, "Aceleracion(Rodilla)_Y"] = 0          
    #df_completo.interpolate(method='linear',inplace=True)
    df_completo['Aceleracion(Cadera)_Y'] = savgol_filter(df_completo['Aceleracion(Cadera)_Y'],  window_length = 30, polyorder = 2)
    
    df_completo.to_csv(output_csv_paths[i], index=False)

    print("Proceso completado. Videos trackeados guardados en:", output_video_paths[i])
    print("Datos de la pose guardados en:", output_csv_paths[i])


# %% [markdown]
# # Gráficos

# %%
# Variables para almacenar trazas de velocidad y aceleración de todos los videos
all_vel_traces = []
all_acc_traces = []
all_pos_traces = []

# Obtener la duración del video más corto
min_duration = min(pd.read_csv(csv_path)['Tiempo'].max() for csv_path in output_csv_paths)

for i, csv_path in enumerate(output_csv_paths):
    df_completo = pd.read_csv(csv_path)
    df_completo['Tiempo'] = df_completo['Tiempo'] * (min_duration / df_completo['Tiempo'].max())
    
    window_size = 50
    left_hip_y_smoothed = df_completo['LEFT_HIP_Y']
    left_knee_y_smoothed = df_completo['LEFT_KNEE_Y']

    #------------POSICIONES DE CADERA Y RODILLA----------------------
    trace1 = go.Scatter(x=df_completo['Tiempo'], y=left_hip_y_smoothed, mode='lines', name=f'Altura de la cadera (Video {i+1})', line=dict(color='blue'))
    trace2 = go.Scatter(x=df_completo['Tiempo'], y=left_knee_y_smoothed, mode='lines', name=f'Posición de la rodilla (Video {i+1})', line=dict(color='red'))

    fig_posiciones = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig_posiciones.add_trace(trace1, row=1, col=1)
    fig_posiciones.add_trace(trace2, row=1, col=1)
    fig_posiciones.update_xaxes(range=[0, min_duration])

    fig_posiciones.update_layout(
        title=f'Evolución de la posición de la cadera y la rodilla con respecto al tiempo (Video {i+1})',
        xaxis=dict(title='Tiempo'),
        yaxis=dict(title='Posición'),  # Invertir eje Y
        legend=dict(x=0.7, y=1.1),
        height=600,
        width=800
    )

    fig_posiciones.show()

    #-------------VELOCIDAD, POSICIÓN Y ACELERACION DE CADERA-------------------
    velocidad_smoothed = df_completo['Velocidad(Cadera)_Y']#.rolling(window=window_size).mean()
    aceleracion_smoothed = df_completo['Aceleracion(Cadera)_Y']#.rolling(window=window_size).mean()
    posicion_cadera_smoothed = df_completo['LEFT_HIP_Y']#.rolling(window=window_size).mean()

    # Crear trazas para velocidad, posición y aceleración de cada video
    vel_trace = go.Scatter(x=df_completo['Tiempo'], y=velocidad_smoothed, mode='lines', name=f'Velocidad de la cadera (Video {i+1})')
    acc_trace = go.Scatter(x=df_completo['Tiempo'], y=aceleracion_smoothed, mode='lines', name=f'Aceleración de la cadera (Video {i+1})')
    pos_trace = go.Scatter(x=df_completo['Tiempo'], y=posicion_cadera_smoothed, mode='lines', name=f'Posición de la cadera (Video {i+1})')

    # Agregar las trazas a las listas de todas las trazas
    all_vel_traces.append(vel_trace)
    all_acc_traces.append(acc_trace)
    all_pos_traces.append(pos_trace)

# Crear figura para la posición de la cadera con todas las trazas superpuestas
fig_pos = go.Figure()
for trace in all_pos_traces:
    fig_pos.add_trace(trace)
fig_pos.update_xaxes(title_text='Tiempo', range=[0, min_duration])
fig_pos.update_yaxes(title_text='Posición de la cadera')
fig_pos.update_layout(title='Posición de la cadera en todos los videos')
fig_pos.show()

# Crear figura para la velocidad de la cadera con todas las trazas superpuestas
fig_velocidad = go.Figure()
for trace in all_vel_traces:
    fig_velocidad.add_trace(trace)
fig_velocidad.update_xaxes(title_text='Tiempo', range=[0, min_duration])
fig_velocidad.update_yaxes(title_text='Velocidad de la cadera')
fig_velocidad.update_layout(title='Velocidad de la cadera en todos los videos')
fig_velocidad.show()

# Crear figura para la aceleración de la cadera con todas las trazas superpuestas
fig_aceleracion = go.Figure()
for trace in all_acc_traces:
    fig_aceleracion.add_trace(trace)
fig_aceleracion.update_xaxes(title_text='Tiempo', range=[0, min_duration])
fig_aceleracion.update_yaxes(title_text='Aceleración de la cadera')
fig_aceleracion.update_layout(title='Aceleración de la cadera en todos los videos')
fig_aceleracion.show()