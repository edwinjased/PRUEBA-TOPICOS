'''
Jonathan Fernandez 8-1007-1530
Edwin Gonzalez 2-751-2144 
'''

import tkinter as tk
import numpy as np
from tabulate import tabulate
import scipy.sparse as sp
import scipy.linalg as la

def getHeader():
    #Esta funcion define el encabezado a mostrar en la solucion
    header = []
    for i in range(int(incog.get())):
        header.append(f"X{i+1}")
    for i in range(int(incog.get())):
        header.append(f"ERROR X{i+1}")
    return header

def getIteraciones():
    #Esta funcion obtiene el numero de iteraciones del cuadro de texto
    # Y si el cuadro esta vacio las define como 100
    if(iteraciones_.get() == ""):
        iteraciones = 100
    else:
        iteraciones = int(iteraciones_.get())
    return iteraciones

def getError():
    #Esta funcion obtiene la cantidad de error del cuadro de texto
    # Y si esta vacio lo define como 0.00001
    if(error_.get() == ""):
        error = 0.00001
    else:
        error = float(error_.get())
    return error

def convertirMatriz():
    #Esta funcion convierte el texto ingresado en un array
    n = int(incog.get())
    strOrg = matriz.get('1.0', 'end-1c')
    str2= strOrg.replace('\n' , " ")

    numbers = str2.split()
    num= np.array (numbers)
    num.shape = (n, (n+1))
    mat= num.astype(float)

    #Aqui separa la matriz a y la b
    A= mat[0:n, 0:n]
    b= mat[: ,n]
    b.shape = (n, 1)

    #En esta parte imprime las matices
    imp = "Matriz A"
    imp += "\n"
    imp += tabulate(A, tablefmt="grid", floatfmt=".1f")
    imp += "\n"
    imp += "Matriz B"
    imp += "\n"
    imp += tabulate(b, tablefmt="grid", floatfmt=".1f")
    impresion1.configure(state='normal')
    impresion1.delete('1.0', tk.END)
    impresion1.insert(tk.END, imp)
    impresion1.configure(state='disabled')

    return(A.copy(), b.copy())

def gauss(): #Esta funcion resuelve gauss
    matrizA, matrizB = convertirMatriz()#Esta parte recibe la matriz a y b
    matriz = np.hstack( [matrizA, matrizB]) #Esta parte une la matiz a y b en una sola
    paso = 1
    solution = "\nMétodo de Gauss\n"
    filas = len(matriz)
    columnas = len(matriz[0])
    XlabelVec = []

    for i in range(filas):
        
        # aqui se hace uno el pivote 
        pivot = matriz[i][i]
        for j in range(i, columnas):
            matriz[i][j] = matriz[i][j] / pivot
        solution += f"\nPaso {paso}: "
        solution += "\n"
        paso += 1
        solution += tabulate(matriz, tablefmt="grid", floatfmt=".4f")
        solution += "\n"

        # en esta parte se vuelven cero los numeros de abajo del pivote
        for j in range(i + 1, filas):
            factor = matriz[j][i]
            for k in range(i, columnas):
                matriz[j][k] = matriz[j][k] - (factor * matriz[i][k])
        solution += f"\nPaso {paso}: "
        solution += "\n"
        paso += 1
        solution += tabulate(matriz, tablefmt="grid", floatfmt=".4f")
        solution += "\n"
    x = np.dot(la.inv(matriz[:, :columnas-1]), matriz[:, columnas-1:]) #En esta parte se calcula el vector solucion
    for i in range(1,filas+1):
        XlabelVec.append([f'X{i}'])
    X_resp = np.hstack( [XlabelVec, x])
    solution += "\nSolucion encontrada\n"
    solution += tabulate(X_resp, tablefmt="grid", floatfmt=".4f")
    solution += "\n"
    impresion2.configure(state='normal')
    impresion2.delete('1.0', tk.END)
    impresion2.insert(tk.END, solution)
    impresion2.configure(state='disabled')

def gauss_jordan():
    matrizA, matrizB= convertirMatriz()#Esta parte recibe la matriz a y b
    matriz = np.hstack( [matrizA, matrizB]) #Esta parte une la matiz a y b en una sola
    paso = 1
    solution = "\nMétodo de Gauss-Jordan\n"
    filas = len(matriz)
    columnas = len(matriz[0])

    for i in range(filas):
        
        # Este paso vuelve el pivote 1
        pivot = matriz[i][i]
        for j in range(i, columnas):
            matriz[i][j] /= pivot
        solution += f"\nPaso {paso}: "
        solution += "\n"
        paso += 1
        solution += tabulate(matriz, tablefmt="grid", floatfmt=".4f")

        # Esta parte elimina hace cero abajo y arriba
        for j in range(filas):
            if j != i:
                factor = matriz[j][i]
                for k in range(i, columnas):
                    matriz[j][k] -= factor * matriz[i][k]
        solution += f"\nPaso {paso}: "
        solution += "\n"
        paso += 1
        solution += tabulate(matriz, tablefmt="grid", floatfmt=".4f")
    impresion2.configure(state='normal')
    impresion2.delete('1.0', tk.END)
    impresion2.insert(tk.END, solution)
    impresion2.configure(state='disabled')

def doolittle():
    matrizA, matrizB = convertirMatriz()
    solution = "Método de Doolittle\n"

    A = matrizA.copy()
    B = matrizB.copy()
    B = B.reshape(len(B),1)

    # factorización LU
    P, L, U = la.lu(A,permute_l= False)
    solution += "\nMatriz A\n"
    solution += tabulate(A, tablefmt="grid", floatfmt=".4f")
    solution += "\n"

    solution += "\nMatriz L\n"
    solution += tabulate(L, tablefmt="grid", floatfmt=".4f")
    solution += "\n"

    solution += "\nMatriz U\n"
    solution += tabulate(U, tablefmt="grid", floatfmt=".4f")
    solution += "\n"

    solution += "\n"

    B2= np.dot(P, B)

    y=np.dot(la.inv(L),B2)


    XlabelVec = []
    YlabelVec = []
    for i in range(1,len(B)+1):
        XlabelVec.append([f'X{i}'])
        YlabelVec.append([f'Y{i}'])


    Y_resp = np.hstack( [YlabelVec, y])
    solution += "\nVector Y\n"
    solution += tabulate(Y_resp, tablefmt="grid", floatfmt=".4f")
    solution += "\n"

    x=np.dot(la.inv(U),y)  


    X_resp = np.hstack( [XlabelVec, x])
    solution += "\n"
    solution += "\nSolución Encontrada\n"
    solution += "\nVector X\n"
    solution += tabulate(X_resp, tablefmt="grid", floatfmt=".4f")
    solution += "\n"
    impresion2.configure(state='normal')
    impresion2.delete('1.0', tk.END)
    impresion2.insert(tk.END, solution)
    impresion2.configure(state='disabled')

    return solution

def jacobi():
    matriz, matrizB = convertirMatriz()
    iteraciones = getIteraciones()
    errorg = getError()
    n = len(matriz)
    header = getHeader()
    _ = 1
    solution = "Método de Jacobi\n\n"
    x = np.zeros(n)  # Vector de valores iniciales
    condicional = [True]
    error = np.zeros(n)
    for i in range(n):
        error[i] = 100
    while (any(condicional) and _ <= iteraciones):
        condicional = []
        x_nuevo = np.copy(x)
        solution += f"ITERACION #{_} \n"
        solution += tabulate([np.concatenate((x_nuevo,error))], header, tablefmt="grid" , floatfmt=".4f")
        solution += "\n\n"
        for i in range(n):
            suma = np.dot(matriz[i, :i], x[:i]) + np.dot(matriz[i, i + 1:], x[i + 1:])
            x_nuevo[i] = (matrizB[i] - suma) / matriz[i, i]
            error [i] = abs(x_nuevo[i] - x[i])
        x = x_nuevo
        _ += 1
        for c in range(n):
            if(error[c] > errorg):
                condicional.append(True)
            else:
                condicional.append(False)
    impresion2.configure(state='normal')
    impresion2.delete('1.0', tk.END)
    impresion2.insert(tk.END, solution)
    impresion2.configure(state='disabled')

def gauss_seidel():
    matriz, matrizB = convertirMatriz()
    iteraciones = getIteraciones()
    errorg = getError()
    
    n = len(matriz)
    header = getHeader()
    _ = 1
    x = np.zeros(n)  # Vector de valores iniciales
    solution = "Método de Jacobi\n\n"
    condicional = [True]
    error = np.zeros(n)
    for i in range(n):
        error[i] = 100

    while (any(condicional) and _ <= iteraciones):
        condicional = []
        x_nuevo = np.copy(x)
        solution += f"ITERACION #{_} \n"
        solution += tabulate([np.concatenate((x_nuevo,error))], header, tablefmt="grid" , floatfmt=".4f")
        solution += "\n\n"

        for i in range(n):
            suma = np.dot(matriz[i, :i], x_nuevo[:i]) + np.dot(matriz[i, i + 1:], x[i + 1:])
            x_nuevo[i] = (matrizB[i] - suma) / matriz[i, i]
            error [i] = abs(x_nuevo[i] - x[i])

        x = x_nuevo
        _ += 1
        for c in range(n):
            if(error[c] > errorg):
                condicional.append(True)
            else:
                condicional.append(False)
    impresion2.configure(state='normal')
    impresion2.delete('1.0', tk.END)
    impresion2.insert(tk.END, solution)
    impresion2.configure(state='disabled')


ventana = tk.Tk()
ventana.title("Solucion de sistemas de ecuaciones")

scrollbar = tk.Scrollbar(ventana)
scrollbar.grid(row = 1, column = 2, rowspan = 5,)

boton1 = tk.Button(ventana, text="Gauss", command = gauss, width = 10, height = 1)
boton2 = tk.Button(ventana, text="Gauss-Jordan", command = gauss_jordan, width = 10, height = 1)
boton3 = tk.Button(ventana, text="Dolittle", command = doolittle, width = 10, height = 1)
boton4 = tk.Button(ventana, text="Jacobi", command = jacobi, width = 10, height = 1)
boton5 = tk.Button(ventana, text="Gauss-Seidel", command = gauss_seidel, width = 10, height = 1)

boton1.grid(row = 0, column = 0, padx= 30)
boton2.grid(row = 1, column = 0, padx= 30)
boton3.grid(row = 2, column = 0, padx= 30)
boton4.grid(row = 3, column = 0, padx= 30)
boton5.grid(row = 4, column = 0, padx= 30)

mensaje1 = tk.Label(ventana, text="Introduzca el numero de incognitas")
mensaje1.grid(row = 0, column = 1, padx= 30)

incog = tk.Entry(ventana)
incog.grid(row = 1, column = 1, padx= 30)

mensaje2 = tk.Label(ventana, text="Introduzca el numero de iteraciones")
mensaje2.grid(row = 2, column = 1, padx= 30)

iteraciones_ = tk.Entry(ventana)
iteraciones_.grid(row = 3, column = 1, padx= 30)

mensaje3 = tk.Label(ventana, text="Introduzca el error")
mensaje3.grid(row = 4, column = 1, padx= 30)

error_ = tk.Entry(ventana)
error_.grid(row = 5, column = 1, padx= 30)

mensaje2 = tk.Label(ventana, text = "Introduzca la matriz")
mensaje2.grid(row = 6, column = 1, padx= 30)

matriz = tk.Text(ventana, width = 20, height = 5)
matriz.grid(row = 7, column = 1)

botonv = tk.Button(ventana, text="Verificar Matriz", command = convertirMatriz, width = 11, height = 1)
botonv.grid(row = 8, column = 1, padx= 30)

impresion1 = tk.Text(ventana, wrap=tk.WORD, width=30, height=10, state='disabled')
impresion1.grid(row = 9, column = 1, padx= 30)

scrollbar = tk.Scrollbar(ventana)
scrollbar.grid(row = 1, column = 2, rowspan = 10, padx= 30,)

impresion2 = tk.Text(ventana, wrap=tk.WORD, width=100, height=40, yscrollcommand=scrollbar.set, state='disabled')
impresion2.grid(row = 0, column = 2, rowspan = 10, padx= 30)

"""
3 -0.1 -0.2  7.85
0.1 7 -0.3 -19.3
0.3 -0.2 10 71.4
"""

ventana.mainloop()
