 
from P1 import *

if __name__ == '__main__':

    img = cv.imread('/home/rainor/Escritorio/VA/Entrega/EXAMEN/kernel.png', 0)
    kernel = cv.imread('/home/rainor/Escritorio/VA/Entrega/EXAMEN/kernel.png', 0)

    filterImage(img, kernel)