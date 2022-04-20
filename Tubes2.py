from asyncio.windows_events import NULL
from tkinter import N
import numpy as np
import pandas as pd
import random
import skfuzzy as fuzz
#import mathplotlib.pyplot as plt

def masukan():

    data = pd.DataFrame({
        'No': [],
        'ID': [],
        'IPK': [],
        'Gaji': []
    })

    for baris in range(1, 101, 1):

        if baris > 26 and baris <= 52:
            labelChar = 'A' + chr(64 + baris - 26)
        elif baris > 52 and baris <= 78:
            labelChar = 'B' + chr(64 + baris - 52)
        elif baris > 78:
            labelChar = 'C' + chr(64 + baris - 78)
        else:
            labelChar = chr(64 + baris)

        df1 = pd.DataFrame({
            'No': [baris],
            'ID': [labelChar],
            'IPK': [round(random.uniform(0.00, 4.00), 2)],
            'Gaji': [round(random.uniform(0, 15), 1)]
        })
        data = pd.concat([data, df1], ignore_index=True, sort=False)   

    data.to_excel('masukan.xlsx', index=False)

def fuzzy():
    x_ipk = np.arange(0.00,4.01,0.10)
    x_gaji = np.arange(0,16,1)
    x_nk = np.arange(0,101,1)

    #Inisiasi diagram cartesius
    ipk_low = fuzz.trapmf(x_ipk,[0,0,2.00,2.75])
    ipk_med = fuzz.trimf(x_ipk,[2.00,2.75,3.25])
    ipk_hig = fuzz.trapmf(x_ipk,[2.75,3.25,4.00,4.00])

    gaji_low = fuzz.trapmf(x_gaji,[0,0,1,3])
    gaji_med = fuzz.trapmf(x_gaji,[1,3,4,6])
    gaji_hig = fuzz.trapmf(x_gaji,[4,6,7,12])
    gaji_vhi = fuzz.trapmf(x_gaji,[7,12,15,15])

    nk_low = fuzz.trapmf(x_nk,[0,0,50,80])
    nk_hig = fuzz.trapmf(x_nk,[50,80,100,100])

    #Fuzzification
    ipk_level_low = fuzz.interp_membership(x_ipk, ipk_low, 4.00)
    ipk_level_med = fuzz.interp_membership(x_ipk, ipk_med, 4.00)
    ipk_level_hig = fuzz.interp_membership(x_ipk, ipk_hig, 4.00)

    gaji_level_low = fuzz.interp_membership(x_gaji, gaji_low, 3)
    gaji_level_med = fuzz.interp_membership(x_gaji, gaji_med, 3)
    gaji_level_hig = fuzz.interp_membership(x_gaji, gaji_hig, 3)
    gaji_level_vhi = fuzz.interp_membership(x_gaji, gaji_vhi, 3)

    #Inisiasi Rule
    rule1 = np.fmin(ipk_level_low, gaji_level_low)
    rule2 = np.fmin(ipk_level_low, gaji_level_med)
    rule3 = np.fmin(ipk_level_low, gaji_level_hig)
    rule4 = np.fmin(ipk_level_low, gaji_level_vhi)

    rule5 = np.fmin(ipk_level_med, gaji_level_low)
    rule6 = np.fmin(ipk_level_med, gaji_level_med)
    rule7 = np.fmin(ipk_level_med, gaji_level_hig)
    rule8 = np.fmin(ipk_level_med, gaji_level_vhi)

    rule9 = np.fmin(ipk_level_hig, gaji_level_low)
    rule10 = np.fmin(ipk_level_hig, gaji_level_med)
    rule11 = np.fmin(ipk_level_hig, gaji_level_hig)
    rule12 = np.fmin(ipk_level_hig, gaji_level_vhi)

    #Conjuction dan Disjunction
    nk_active1 = np.fmin(rule1, nk_low)
    nk_active2 = np.fmin(rule2, nk_low)
    nk_active3 = np.fmin(rule3, nk_low)
    nk_active4 = np.fmin(rule4, nk_low)

    nk_active5 = np.fmin(rule5, nk_hig)
    nk_active6 = np.fmin(rule6, nk_low)
    nk_active7 = np.fmin(rule7, nk_low)
    nk_active8 = np.fmin(rule8, nk_low)

    nk_active9 = np.fmin(rule9, nk_hig)
    nk_active10 = np.fmin(rule10, nk_hig)
    nk_active11 = np.fmin(rule11, nk_hig)
    nk_active12 = np.fmin(rule12, nk_low)
    #nk0 = np.zeros_like(x_nk)

    aggregated = np.fmax(nk_active1,np.fmax(nk_active2,np.fmax(nk_active3,np.fmax(nk_active4,np.fmax(nk_active5,np.fmax(nk_active6,np.fmax(nk_active7,np.fmax(nk_active8,np.fmax(nk_active9,np.fmax(nk_active10,np.fmax(nk_active11, nk_active12)))))))))))

    nk = fuzz.defuzz(x_nk,aggregated, 'centroid')
    print(nk)

if __name__== "__main__":
    masukan()
    fuzzy()