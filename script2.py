 
from P1 import *

if __name__ == '__main__':

    img = cv.imread('/home/rainor/Escritorio/VA/Entrega/EXAMEN/imageP2.png', 0)

    # highBoost(img, 1, "gaussian", 3)

    # highBoost(img, 2, "gaussian", 3)

    highBoost(img, 3, "gaussian", 3)