import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from utils import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Completar con tus rutas
video_path = '/Users/valen/Downloads/Fisica/salto2.MP4'
output_video_path = '/Users/valen/Downloads/Fisica/tracked_video.mp4'
output_csv_path = '/Users/valen/Downloads/Fisica/pose_data.csv'

# Input usuario
peso_persona = 65  # kg
altura_persona = 1.76  # m
longitud_brazo_x = 0.65  # m --> 0.22330 px
longitud_pierna_y = 0.94  # m --> 0.550944 px

# Crear columnas del dataframe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Defino cuales son las articulaciones que me interesa estudiar, en este caso solo necesitamos usar la cadera como punto de referencia
articulaciones = [
    mp_pose.PoseLandmark.LEFT_HIP,
]

columns = ['frame_number']

for landmark in articulaciones:
    columns.append(landmark.name + '_X')
    columns.append(landmark.name + '_Y')
    columns.append(landmark.name + '_Z')

columns.append("Tiempo")
columns.append("Velocidad(Cadera)_X")
columns.append("Velocidad(Cadera)_Y")
columns.append("Energia Potencial(Cadera)")
columns.append("Energia Cinetica(Cadera)")
columns.append("Energia Mecanica(Cadera)")
columns.append("Trabajo")

# Código para recorrer frames del video y realizar cálculos
cap = cv2.VideoCapture(video_path)

# Obtener propiedades del video
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_fps = cap.get(cv2.CAP_PROP_FPS)
tiempo_por_frame = 1 / video_fps

# Inicializar DataFrame y pose para el video actual
df_completo = pd.DataFrame(columns=columns)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

frame_number = 0  # Reiniciar el contador de fotogramas para cada video

altura_cadera_y_inicial = None
altura_cadera_y_actual = None

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
            pos = landmarks[landmark]

            pose_row[landmark.name + '_X'] = pos.x * (longitud_brazo_x / 0.22330)
            pose_row[landmark.name + '_Y'] = (1 - pos.y) * (longitud_pierna_y / 0.55094)

    df_completo = pd.concat([df_completo, pd.DataFrame([pose_row])], ignore_index=True)

    # Agregar los landmarks al gráfico
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=5, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=5, circle_radius=3))

    if frame_number > 0:
        df_completo.loc[df_completo["frame_number"] == frame_number, "Tiempo"] = tiempo_por_frame * frame_number

    # Escribir el frame procesado en el video de salida
    video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    frame_number += 1

# Liberar recursos y guardar resultados después de procesar cada video
pose.close()
video_writer.release()
cap.release()

# Aplicar suavizado de Savitzky-Golay a las posiciones
window_length = 11 
polyorder = 2

df_completo['LEFT_HIP_X'] = savgol_filter(df_completo['LEFT_HIP_X'], window_length, polyorder)
df_completo['LEFT_HIP_Y'] = savgol_filter(df_completo['LEFT_HIP_Y'], window_length, polyorder)

# Calcular las velocidades nuevamente después del suavizado
for frame_number in range(1, len(df_completo)):
    previous_frame = frame_number - 1

    pos_prev_left_hip = (df_completo.loc[previous_frame, 'LEFT_HIP_X'], df_completo.loc[previous_frame, 'LEFT_HIP_Y'])
    pos_actual_left_hip = (df_completo.loc[frame_number, 'LEFT_HIP_X'], df_completo.loc[frame_number, 'LEFT_HIP_Y'])

    # VELOCIDAD
    velocidad_cadera_x, velocidad_cadera_y = velocidad_instantanea(pos_prev_left_hip, pos_actual_left_hip, tiempo_por_frame)
    df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_X"] = velocidad_cadera_x
    df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_Y"] = velocidad_cadera_y

    # Frame 24 es el punto mas bajo de la cadera
    if frame_number > 24:

        altura_cadera_y_inicial = df_completo.loc[24 , 'LEFT_HIP_Y']
        altura_cadera_y_actual = df_completo.loc[frame_number, 'LEFT_HIP_Y']

        altura = altura_cadera_y_actual - altura_cadera_y_inicial
        masa = peso_persona / 9.8

        # ENERGIA POTENCIAL
        energia_potencial_cadera = calcular_energia_potencial(masa, altura, 9.8)
        df_completo.loc[df_completo["frame_number"] == frame_number, "Energia Potencial(Cadera)"] = energia_potencial_cadera

        # ENERGIA CINETICA
        velocidad_total_cadera = np.sqrt((velocidad_cadera_x) ** 2 + (velocidad_cadera_y) ** 2)
        energia_cinetica_cadera = calcular_energia_cinetica(masa, abs(velocidad_total_cadera))
        df_completo.loc[df_completo["frame_number"] == frame_number, "Energia Cinetica(Cadera)"] = energia_cinetica_cadera

        # ENERGIA MECANICA
        energia_mecanica_cadera = energia_potencial_cadera + energia_cinetica_cadera
        df_completo.loc[df_completo["frame_number"] == frame_number, "Energia Mecanica(Cadera)"] = energia_mecanica_cadera

        # TRABAJO
        trabajo_mecanica = df_completo['Energia Mecanica(Cadera)'].diff()
        df_completo.loc[df_completo["frame_number"] == frame_number, "Trabajo"] = trabajo_mecanica   # --> PASARLO a CALORIAS



        #trabajo_total = trabajo_total + abs(df_completo.loc[frame_number, 'Energia Mecanica(Cadera)'])
        #trabajo_total_calorias = trabajo_total / 4.184 # aca ya esta en calorias xq 1 caloria son 4.184 joules
        #
        # Trabajo=energiamecanica.diff() --> sumatoria de todos --> trabajo total --> pasarlo a calorias
trabajo_total = 0
for i in range(26, len(df_completo)):
    trabajo_total = trabajo_total + abs(df_completo.loc[i, 'Trabajo'])
  # print("")

trabajo_total_calorias = trabajo_total / 4.184
print("Trabajo total: ",trabajo_total_calorias)
# Aplicar suavizado de Savitzky-Golay a las velocidades
df_completo['Velocidad(Cadera)_X'] = savgol_filter(df_completo['Velocidad(Cadera)_X'], window_length, polyorder)
df_completo['Velocidad(Cadera)_Y'] = savgol_filter(df_completo['Velocidad(Cadera)_Y'], window_length, polyorder)

df_completo.to_csv(output_csv_path, index=False)

print("Proceso completado. Video trackeado guardado en:", output_video_path)
print("Datos de la pose guardados en:", output_csv_path)


#-----------------GRAFICOS-------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.signal import savgol_filter

# Parámetros de suavizado
window_length = 11
polyorder = 2

# Leer el archivo CSV con los datos
output_csv_path = '/Users/valen/Downloads/Fisica/pose_data.csv'
df_completo = pd.read_csv(output_csv_path)

# Suavizar las energías potencial, cinética y mecánica
energia_potencial_smoothed = savgol_filter(df_completo['Energia Potencial(Cadera)'], window_length, polyorder)
energia_cinetica_smoothed = savgol_filter(df_completo['Energia Cinetica(Cadera)'], window_length, polyorder)
energia_mecanica_smoothed = savgol_filter(df_completo['Energia Mecanica(Cadera)'], window_length, polyorder)
trabajo_mecanica_smoothed = savgol_filter(df_completo['Trabajo'], window_length, polyorder)

# Crear trazas para las energías
trace_energia_potencial = go.Scatter(x=df_completo['Tiempo'], y=energia_potencial_smoothed, mode='lines', name='Energía Potencial de la Cadera', line=dict(color='blue'))
trace_energia_cinetica = go.Scatter(x=df_completo['Tiempo'], y=energia_cinetica_smoothed, mode='lines', name='Energía Cinética de la Cadera', line=dict(color='red'))
trace_energia_mecanica = go.Scatter(x=df_completo['Tiempo'], y=energia_mecanica_smoothed, mode='lines', name='Energía Mecánica de la Cadera', line=dict(color='green'))

trace_trabajo_mecanica = go.Scatter(x=df_completo['Tiempo'], y=trabajo_mecanica_smoothed, mode='lines', name='Trabajo', line=dict(color='purple'))

# Crear la figura con subplots
fig_energias = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
fig_energias.add_trace(trace_energia_potencial, row=1, col=1)
fig_energias.add_trace(trace_energia_cinetica, row=1, col=1)
fig_energias.add_trace(trace_energia_mecanica, row=1, col=1)
fig_energias.add_trace(trace_trabajo_mecanica, row=1, col=1)

# Invertir el eje Y de las energías potencial, cinética y mecánica
#fig_energias.update_yaxes(autorange='reversed', row=1, col=1)

# Actualizar el diseño de la figura
fig_energias.update_layout(
    title='Energía Potencial, Cinética y Mecánica de la Cadera',
    xaxis=dict(title='Tiempo'),
    yaxis=dict(title='Energía (Joules)'), 
    legend=dict(x=0.7, y=1.1),
    height=600,
    width=800
)

# Mostrar la figura de energías invertidas
fig_energias.show()

# Gráfica de las velocidades
trace_velocidad_x = go.Scatter(x=df_completo['Tiempo'], y=df_completo['Velocidad(Cadera)_X'], mode='lines', name='Velocidad en X', line=dict(color='purple'))
trace_velocidad_y = go.Scatter(x=df_completo['Tiempo'], y=df_completo['Velocidad(Cadera)_Y'], mode='lines', name='Velocidad en Y', line=dict(color='orange'))

fig_velocidades = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
fig_velocidades.add_trace(trace_velocidad_x, row=1, col=1)
fig_velocidades.add_trace(trace_velocidad_y, row=1, col=1)

fig_velocidades.update_layout(
    title='Velocidad de la Cadera en X e Y',
    xaxis=dict(title='Tiempo'),
    yaxis=dict(title='Velocidad (m/s)'), 
    legend=dict(x=0.7, y=1.1),
    height=600,
    width=800
)

fig_velocidades.show()