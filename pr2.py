
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing



def main():

    intervalos=[0, 32, 64, 96, 128, 256]
    
    nombres_columnas = ['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras' ,'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending' ,'superfund-right-to-sue', 'crime', 'duty-free-exports' ,'export-administration-act-south-africa' ,'Class']

    nombres_columnas2 = ['MYCT', 'MMIN', 'MMAX', 'CACHE', 'MINCACHE', 'MAXCACHE', 'CLASS']
    pdf = pd.read_csv('cpu.txt', delimiter = ",", comment="%", skiprows=1, names=nombres_columnas2, header=0)


    
    standarizado = preprocessing.scale(pdf["MMAX"])

    
    
    print("HISTOGRAMA DE MEMORIA MÁXIMA")
    plt.hist(pdf["MMAX"], ec="black")
    
    plt.show()
    
    
    
    print("HISTOGRAMA DE MEMORIA MÁXIMA ESTANDARIZADA Y NORMALIZADA")
    plt.hist(standarizado, ec="black", density=True)
    
    plt.show()
    
    plt.scatter(pdf["MMIN"], pdf["MMAX"])
    plt.title("Diagrama de dispersión de MMIN frente a MMAX")
    plt.xlabel("MMIN")
    plt.ylabel("MMAX")
    plt.show()
    

if __name__ == "__main__":
    main()
