import numpy as np
import math
import cv2

longitud_brazo_x = 0.65  # m --> 0.22330 px
longitud_pierna_y = 0.94  # m --> 0.550944 px

def calculate_angle(a, b, c):
  return np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])

# formato output:([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]).
def extraer_posiciones(df, frame_number, *articulaciones):
    data = []
    # Buscar la fila correspondiente al número de frame
    row = df[df['frame_number'] == frame_number]

    for articulacion in articulaciones:
        x = row[articulacion+ '_X'].iloc[0]
        y = row[articulacion+ '_Y'].iloc[0]

        data.append([x, y])
    return data

# Dado un dataframe y un numero de frame, retorna la velocidad instantánea correspondiente a la fila con el numero de frame pasado por parámetro.
def extraer_velocidad_angular(df, frame_number):
    # Buscar la fila correspondiente al número de frame
    row = df[df['frame_number'] == frame_number]
    return row["VelocidadAngular"].iloc[0]

def velocidad_angular(angulo_inicial, angulo_final, delta_tiempo):
    # Calcular el cambio en el ángulo
    delta_theta = angulo_final - angulo_inicial

    # Calcular la velocidad angular
    # Recordar que omega = theta punto = vel angular = Delta theta / Delta t
    # Donde Delta theta es el cambio en rotación angular y Delta t es el cambio en el tiempo
    # ENTONCES: la velocidad angular se calcula dividiendo la diferencia total del ángulo (delta_theta) por el tiempo transcurrido entre las mediciones (1 / frame_rate).
    angular_velocity = delta_theta / delta_tiempo
    
    return angular_velocity

def velocidad_instantanea(pos_anterior, pos_actual, tiempo):
  dx = pos_actual[0] - pos_anterior[0]
  dy = pos_actual[1] - pos_anterior[1]
  return (dx/tiempo, dy/tiempo)

def aceleracion_instantanea(vel_actual_x, vel_anterior_x, vel_actual_y, vel_anterior_y, tiempo):
  dvx = vel_actual_x - vel_anterior_x
  dvy = vel_actual_y - vel_anterior_y
  return (dvx/tiempo, dvy/tiempo)

def obtener_angulo_barra(pos_left_ankle, pos_left_knee, pos_left_heel, pos_left_foot_index):
  # Se genera el vector que va desde la rodilla al tobillo
  AB = (pos_left_ankle[0] - pos_left_knee[0], pos_left_ankle[1] - pos_left_knee[1])
  # Se genera el vector que va desde la punta del pie al talon
  BD = (pos_left_heel[0] - pos_left_foot_index[0], pos_left_heel[1] - pos_left_foot_index[1])
  # Calculo el producto escalar entre estos dos vectores
  AB_escalar = (AB[0] * BD[0] + AB[1] * BD[1])
  # Calculo la magnitud del primer y segundo vector
  AB_magnitud = (AB[0]**2 + AB[1]**2)**0.5
  BD_magnitud = (BD[0]**2 + BD[1]**2)**0.5
  # Obtengo el angulo entre los 2 vectores
  valor = AB_escalar / (AB_magnitud * BD_magnitud)
  angulo = np.arccos(valor)
  # Ajusto el angulo para que sea el del primer cuadrante, y en base a ese angulo obtengo el de los otros cuadrantes
  if( pos_left_foot_index[1] < pos_left_heel[1]):
    if(angulo > 0):
      if(angulo > np.radians(90)):
        angulo = np.radians(180) - angulo
  else:
    if(angulo > 0):
      if(angulo < np.radians(90)):
        angulo = np.radians(180) - angulo
  return [angulo, - (np.radians(180) - angulo), -angulo, np.radians(180) - angulo]

def graficar_barra(image, pos_left_heel, pos_left_foot_index, pos_left_knee, video_width, video_height):
  normalized_pos_left_foot_index = (pos_left_foot_index[0] * 0.3214562238 / longitud_brazo_x, 1-(pos_left_foot_index[1] * 0.2489993274 / longitud_pierna_y))
  normalized_pos_left_knee = (pos_left_knee[0] * 0.3214562238 / longitud_brazo_x, 1-(pos_left_knee[1] * 0.2489993274 / longitud_pierna_y))
  normalized_pos_left_heel = (pos_left_heel[0] * 0.3214562238 / longitud_brazo_x, 1-(pos_left_heel[1] * 0.2489993274 / longitud_pierna_y))
  distancia_punta_pie_tobillo_x = (0.23 * 0.3214562238) / longitud_brazo_x
  distancia_punta_pie_tobillo_y = (0.0001 * 0.2489993274) / longitud_pierna_y
  vector_puntapie_talon =  (normalized_pos_left_heel[0] - normalized_pos_left_foot_index[0], normalized_pos_left_heel[1] - normalized_pos_left_foot_index[1])
  
  distancia_vector_puntapie_talon = ((vector_puntapie_talon[0] * video_width)**2 + (vector_puntapie_talon[1] * video_height)**2)**0.5
  
  versor_vector_puntapie_talon = ((vector_puntapie_talon[0] * video_width) / distancia_vector_puntapie_talon, (vector_puntapie_talon[1] * video_height) / distancia_vector_puntapie_talon)
  
  vector_puntapie_tobillo = (versor_vector_puntapie_talon[0] * (distancia_punta_pie_tobillo_x * video_width), versor_vector_puntapie_talon[1] * (distancia_punta_pie_tobillo_y * video_height))
  
  #vector_tobillo_rodilla = (normalized_pos_left_knee[0] - vector_puntapie_tobillo[0],normalized_pos_left_knee[1] - vector_puntapie_tobillo[1])
  cv2.arrowedLine(image, (int(normalized_pos_left_foot_index[0] * video_width), int(normalized_pos_left_foot_index[1] * video_height)) , (int((vector_puntapie_talon[0]+ normalized_pos_left_foot_index[0]) * video_width) , int((vector_puntapie_talon[1]+ normalized_pos_left_foot_index[1]) * video_height)) , (0,255,0), 7)
  cv2.arrowedLine(image, (int(vector_puntapie_tobillo[0]+ normalized_pos_left_foot_index[0] * video_width) , int(vector_puntapie_tobillo[1]+ normalized_pos_left_foot_index[1] * video_height)) , (int((normalized_pos_left_knee[0]) * video_width) , int((normalized_pos_left_knee[1]) * video_height)) , (255,0,0), 2)
  cv2.arrowedLine(image, (int(normalized_pos_left_foot_index[0] * video_width) , int(normalized_pos_left_foot_index[1] * video_height)) , (int(vector_puntapie_tobillo[0] + normalized_pos_left_foot_index[0] * video_width) , int(vector_puntapie_tobillo[1]+ normalized_pos_left_foot_index[1] * video_height)) , (255,0,0), 2)
  

def calcular_fuerza_gemelo(df, frame_number, pos_left_knee, pos_left_ankle, pos_left_heel, pos_left_foot_index):
  # Masa del pie, esta masa es el 1.43% del peso total de la persona
  masa_pie = 0.0143 * (70 / 9.8) 
  longitud_pie = ((pos_left_heel[0]-pos_left_foot_index[0])**2 + (pos_left_heel[1]-pos_left_foot_index[1])**2)**0.5
  # Distancia desde el tobillo a donde se aplica la fuerza del gemelo
  distancia_momento_gemelo = ((pos_left_heel[0]-pos_left_ankle[0])**2 + (pos_left_heel[1]-pos_left_ankle[1])**2)**0.5
  # Distancia desde el tobillo a donde se aplica la fuerza del talon
  distancia_momento_normal = ((pos_left_foot_index[0]-pos_left_ankle[0])**2 + (pos_left_foot_index[1]-pos_left_ankle[1])**2)**0.5
  # Obtengo la aceleracion angular del dataframe
  aceleracionAngular = df.loc[df["frame_number"] == frame_number, "AceleracionAngular"].iloc[0]
  # Calculo los angulos a partir de los vectores Tobillo-Rodilla y PuntaPie-Talon
  cuadrantes = obtener_angulo_barra(pos_left_ankle, pos_left_knee, pos_left_heel, pos_left_foot_index)
  # Obtengo angulo entre la rodilla, tobillo, talon y lo paso a radianes para calcular el sen
  angulo_gemelo_talon_cuadrante = cuadrantes[0]
  # Obtengo el angulo que se forma entre la punta del pie, tobillo, y un punto en la direccion del peso
  angulo_gemelo_peso_cuadrante = cuadrantes[2]
  # Obtengo el angulo que se forma entre la punta del pie, tobillo y rodilla
  angulo_gemelo_normal_cuadrante = cuadrantes[1]
  # Distancia desde el centro del pie al tobillo para teorema de Steiner
  distancia_al_centro = 0.08  
  # Momento inercial
  momento_inercial = (1/12 * masa_pie * longitud_pie**2) + (masa_pie * distancia_al_centro**2)
  # Momento del peso
  momento_peso = 70 * distancia_al_centro * math.sin(angulo_gemelo_peso_cuadrante)
  # Momento de la normal
  momento_normal = 70 * distancia_momento_normal * math.sin(angulo_gemelo_normal_cuadrante)
  # Calculo la fuerza que realiza el gemelo
  magnitud_fuerza_gemelo = abs(-((momento_inercial * aceleracionAngular) - momento_normal + momento_peso) / (distancia_momento_gemelo * math.sin(angulo_gemelo_talon_cuadrante)))
  # Al vector fuerza gemelo lo multiplico por la fuerza que realiza este y lo devuelvo
  return magnitud_fuerza_gemelo

def graficar_vector_fuerza(image, magnitud_fuerza_gemelo, pos_left_ankle, pos_left_knee, pos_left_heel, video_width, video_height):
  # Se vuelven a normalizar las posiciones para poder encontrar su coordenada en el video
  normalized_pos_left_ankle = (pos_left_ankle[0] * 0.3214562238 / longitud_brazo_x, 1-(pos_left_ankle[1] * 0.2489993274 / longitud_pierna_y))
  normalized_pos_left_knee = (pos_left_knee[0] * 0.3214562238 / longitud_brazo_x, 1-(pos_left_knee[1] * 0.2489993274 / longitud_pierna_y))
  normalized_pos_left_heel = (pos_left_heel[0] * 0.3214562238 / longitud_brazo_x, 1-(pos_left_heel[1] * 0.2489993274 / longitud_pierna_y))
  # Se genera el vector que va desde el talon a la rodilla
  vector = (normalized_pos_left_knee[0] - normalized_pos_left_ankle[0], normalized_pos_left_knee[1] - normalized_pos_left_ankle[1])
  # Se calcula su distancia
  distancia_vector_pixeles = ((vector[0] * video_width)**2 + (vector[1] * video_height)**2)**0.5
  # Se obtiene su versor para poder despues ser multiplicado por la fuerza y asi cambiar su longitud dinamicamente sobre el video
  versor = ((vector[0] * video_width) / distancia_vector_pixeles, (vector[1] * video_height) /distancia_vector_pixeles)
  cv2.arrowedLine(image, (int(normalized_pos_left_heel[0] * video_width) , int(normalized_pos_left_heel[1] * video_height)) , (int(versor[0] * (magnitud_fuerza_gemelo/2) + normalized_pos_left_heel[0] * video_width) , int(versor[1] * (magnitud_fuerza_gemelo/2) + normalized_pos_left_heel[1] * video_height)) , (0,255,0), 2)
  
#--------------FUNCIONES PARA ENERGIA------------------
def calcular_energia_potencial(masa, altura, g):
    return masa * g * altura

def calcular_energia_cinetica(masa, velocidad):
    return 0.5 * masa * (velocidad ** 2)