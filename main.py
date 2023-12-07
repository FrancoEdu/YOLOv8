from ultralytics import YOLO

# Carregando o modelo pré treinado
model = YOLO("yolov8n.pt")

# source="0": Este parâmetro indica a fonte dos dados
# para previsão. No caso, "0" geralmente se refere à
# entrada de vídeo da webcam. Portanto, o modelo
# estará prevendo objetos em tempo real através
# da webcam.

# show=True: Este parâmetro diz ao modelo para
# mostrar a saída da previsão, ou seja, exibir
# o vídeo com as detecções de objetos sobrepostas.

results = model.predict(source="0", show=True)

# Print do que é exibido em tela
print(results)