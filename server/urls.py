from django.contrib import admin
from django.urls import path
from rest_framework_simplejwt import views as jwt_views
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import keras # o tensorflow.keras
# Definir el modelo y los nombres de las clases
model = keras.models.load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
class Protegida(APIView):
    permission_classes = [IsAuthenticated]

    @csrf_exempt
    def post(self, request):
        # Verificar que el método sea POST y que haya un archivo adjunto
        if request.method == "POST" and request.FILES:
            # Obtener el archivo de video del request
            video = request.FILES["video"] # obtener los datos binarios del video
            # Obtener el valor del nombre del archivo
            nombre = request.POST.get("nombre")
            with open(nombre, "wb") as f: # Escribir el contenido del video en el archivo local 
                f.write(video.read())
            cap = cv2.VideoCapture(nombre) # crear un objeto VideoCapture a partir del video
            # Inicializar una lista vacía para almacenar las detecciones
            detecciones = []
            # Leer el video frame por frame
            while True:
                # Obtener el estado de lectura y el frame actual
                ret, frame = cap.read()
                # Si el estado es verdadero, procesar el frame con el modelo Keras
                if ret:
                    # Cambiar el tamaño de la imagen sin procesar en píxeles (224 de alto, 224 de ancho)
                    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    # Convierte la imagen en una matriz numpy y dale nueva forma a la forma de entrada de los modelos.
                    frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
                    # Normalizar la matriz de imágenes
                    frame = (frame / 127.5) - 1

                    # Predice el modelo
                    prediction = model.predict(frame)
                    index = np.argmax(prediction)
                    class_name = class_names[index]
                    confidence_score = prediction[0][index]
                
                    # Agregar la detección a las listas correspondientes
                    detecciones.append((class_name, confidence_score))
                # Si no, terminar el bucle
                else:
                    break
            # Liberar el objeto de captura de video
            cap.release()
            # Crear un diccionario para contar las ocurrencias de cada dato
            diccionario = {}
            for dato, valor in detecciones:
                # Eliminar el número y el salto de línea del dato
                dato = dato[2:-1]
                if dato not in diccionario:
                    diccionario[dato] = 1
                else:
                    diccionario[dato] += 1
            # Encontrar el dato que más se repite
            maximo = None
            frecuencia = None
            for dato, contador in diccionario.items():
                if frecuencia is None or contador > frecuencia:
                    maximo = dato
                    frecuencia = contador
            # Devolver una respuesta con la detección hecha
            return HttpResponse(maximo)
        # Si no se cumple la condición, devolver una respuesta de error
        else:
            return HttpResponse(None)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/token/', jwt_views.TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', jwt_views.TokenRefreshView.as_view(), name='token_refresh'),
    path('api/detectar-objeto/', Protegida.as_view(), name='detectar-objeto')
]
