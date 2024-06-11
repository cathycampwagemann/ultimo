import torch # Si importo sólo los módulos, me sale error
import os # Si importo sólo los módulos, me sale error
import cv2 # Si importo sólo los módulos, me sale error
import numpy as np # Si importo sólo los módulos, me sale error
import torch.nn as nn
from numpy import array, ndarray, squeeze
from PIL import Image, ImageOps
from cv2 import imread, imwrite, resize, cvtColor, COLOR_BGR2GRAY
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from collections import OrderedDict
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Defino las funciones individuales

def construir_ruta_imagen(nombre_imagen, carpeta_principal_imagenes, subcarpeta=None, categoria=None):
    if subcarpeta and categoria:
        return os.path.join(carpeta_principal_imagenes, subcarpeta, categoria, nombre_imagen)
    elif subcarpeta:
        return os.path.join(carpeta_principal_imagenes, subcarpeta, nombre_imagen)
    else:
        return os.path.join(str(carpeta_principal_imagenes), str(nombre_imagen))

def nuevo_alto_deseado(imagen):
    alto_original, ancho_original, _ = imagen.shape
    if alto_original<670:
      nuevo_alto_deseado = 447
    elif alto_original>1200:
      nuevo_alto_deseado = 1440
    else:
      nuevo_alto_deseado = 800
    return nuevo_alto_deseado

def redimensionar_imagenes(imagen, nuevo_alto_deseado):
    alto_original, ancho_original, _ = imagen.shape
    nuevo_alto_deseado =  nuevo_alto_deseado(imagen)
    factor_redimensionamiento = nuevo_alto_deseado / alto_original
    nuevo_ancho = int(ancho_original * factor_redimensionamiento)
    imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto_deseado))
    return imagen_redimensionada


def convertir_a_modo_L(imagen):
    if isinstance(imagen, np.ndarray):
        imagen_pil = Image.fromarray(imagen)
        imagen_gris = imagen_pil.convert("L")
        return np.array(imagen_gris)

def add_padding(image, target_size=1440):
    ancho, alto = image.size
    alto, ancho = torch.tensor(alto), torch.tensor(ancho)
    padding_vertical = max((target_size - alto) // 2, torch.tensor(0))
    padding_horizontal = max((target_size - ancho) // 2, torch.tensor(0))
    imagen_con_padding = ImageOps.expand(image, (padding_horizontal, padding_vertical, padding_horizontal, padding_vertical), fill=0)
    imagen_con_padding = imagen_con_padding.resize((target_size, target_size))
    return imagen_con_padding

def procesar_imagen_padding(image_paths, output_directory):
    nuevas_rutas = []
    max_size = 1440

    for image_path in image_paths:
        imagen = Image.open(image_path)
        imagen_con_padding = add_padding(imagen, max_size)
        imagen_con_padding.save(image_path)
        nuevas_rutas.append(image_path)
    return nuevas_rutas

def redimensionar_imagenes_con_padding(image_paths, nuevo_alto=720, nuevo_ancho=720):

    nuevas_rutas2 = []

    for image_path in image_paths:
        imagen = cv2.imread(image_path)
        if imagen is not None:
            imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
            cv2.imwrite(image_path, imagen_redimensionada)
            nuevas_rutas2.append(image_path)
    return nuevas_rutas2

def normalizar_imagenes(ruta):
    imagen = load_img(ruta)
    imagen = imagen.convert("L")
    imagen_array = img_to_array(imagen)
    imagen_normalizada = imagen_array / 255.0
    return imagen_normalizada

def procesar_imagen(nombre_imagen, carpeta_principal_imagenes):
    ruta_imagen = construir_ruta_imagen(nombre_imagen, carpeta_principal_imagenes)
    carpeta_imagen_original = os.path.dirname(ruta_imagen)
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo leer la imagen en la ruta: {ruta_imagen}")
        return None 
    imagen_redimensionada = redimensionar_imagenes(imagen, nuevo_alto_deseado)
    imagen_gris = convertir_a_modo_L(imagen_redimensionada)
    ruta_imagen_gris=cv2.imwrite(ruta_imagen,imagen_gris)
    imagen_con_padding = procesar_imagen_padding([ruta_imagen], carpeta_imagen_original)
    imagen_redimensionada2 = redimensionar_imagenes_con_padding(imagen_con_padding, nuevo_alto=720, nuevo_ancho=720)
    imagen_normalizada = normalizar_imagenes(ruta_imagen)
    imagen_normalizada_squeezed = np.squeeze(imagen_normalizada, axis=2)
    imagen_tensor = torch.tensor(imagen_normalizada_squeezed, dtype=torch.float32)
    imagen_tensor = imagen_tensor.unsqueeze(0).unsqueeze(0)
    return imagen_tensor

def predecir_neumonia(modelo, imagen_tensor):
    modelo.eval()
    with torch.no_grad():
        salida = modelo(imagen_tensor)
        _, prediccion = torch.max(salida, 1)
    return prediccion.item()

# Defino el modelo
class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('bn2', nn.BatchNorm2d(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class TransitionLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, num_classes=2):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, in_channels=num_features, growth_rate=growth_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('bn5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool5', nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = torch.flatten(features, 1)
        out = self.classifier(out)
        return out

class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomDenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.densenet(x)
