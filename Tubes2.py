import numpy as np
import pandas as pd
import random
import skfuzzy as fuzz

def masukan():

    data = pd.DataFrame({
        'No': [],
        'ID': [],
        'IPK': [],
        'Gaji': [],
        'Hello': []
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
            'Gaji': [round(random.uniform(0, 10), 1)]
        })
        data = pd.concat([data, df1], ignore_index=True, sort=False)   

    data.to_excel('masukan.xlsx', index=False)

def luaran():
    df2 = pd.DataFrame({
        'ID' : [],
        'NK' : []
    })

    data = pd.read_excel('masukan.xlsx', index_col= 'No')
    for baris in range (100):
        labelId = data.iat[baris, 0]
        ipk = data.iat[baris, 1]
        gaji = data.iat[baris, 2]
                
        x_ipk = np.arange(0.00,4.01,0.10)
        x_gaji = np.arange(0,11,1)
        x_nk = np.arange(0,101,1)

        #Inisiasi diagram cartesius
        ipk_low = fuzz.trapmf(x_ipk,[0,0,2.00,2.75])
        ipk_med = fuzz.trimf(x_ipk,[2.00,2.75,3.25])
        ipk_hig = fuzz.trapmf(x_ipk,[2.75,3.25,4.00,4.00])

        gaji_low = fuzz.trapmf(x_gaji,[0,0,1,3])
        gaji_med = fuzz.trapmf(x_gaji,[1,3,4,6])
        gaji_hig = fuzz.trapmf(x_gaji,[4,7,10,10])

        nk_low = fuzz.trapmf(x_nk,[0,0,50,80])
        nk_hig = fuzz.trapmf(x_nk,[50,80,100,100])

        #Fuzzification
        ipk_level_low = fuzz.interp_membership(x_ipk, ipk_low, ipk)
        ipk_level_med = fuzz.interp_membership(x_ipk, ipk_med, ipk)
        ipk_level_hig = fuzz.interp_membership(x_ipk, ipk_hig, ipk)

        gaji_level_low = fuzz.interp_membership(x_gaji, gaji_low, gaji)
        gaji_level_med = fuzz.interp_membership(x_gaji, gaji_med, gaji)
        gaji_level_hig = fuzz.interp_membership(x_gaji, gaji_hig, gaji)

        #Inisiasi Rule
        rule1 = np.fmin(ipk_level_low, gaji_level_low)
        rule2 = np.fmin(ipk_level_low, gaji_level_med)
        rule3 = np.fmin(ipk_level_low, gaji_level_hig)

        rule4 = np.fmin(ipk_level_med, gaji_level_low)
        rule5 = np.fmin(ipk_level_med, gaji_level_med)
        rule6 = np.fmin(ipk_level_med, gaji_level_hig)

        rule7 = np.fmin(ipk_level_hig, gaji_level_low)
        rule8 = np.fmin(ipk_level_hig, gaji_level_med)
        rule9 = np.fmin(ipk_level_hig, gaji_level_hig)

        #Conjuction dan Disjunction
        nk_active1 = np.fmin(rule1, nk_low)
        nk_active2 = np.fmin(rule2, nk_low)
        nk_active3 = np.fmin(rule3, nk_low)
        nk_active4 = np.fmin(rule4, nk_hig)
        nk_active5 = np.fmin(rule5, nk_low)
        nk_active6 = np.fmin(rule6, nk_low)
        nk_active7 = np.fmin(rule7, nk_hig)
        nk_active8 = np.fmin(rule8, nk_hig)
        nk_active9 = np.fmin(rule9, nk_low)

        aggregated = np.fmax(nk_active1,np.fmax(nk_active2,np.fmax(nk_active3,np.fmax(nk_active4,np.fmax(nk_active5,np.fmax(nk_active6,np.fmax(nk_active7,np.fmax(nk_active8,nk_active9))))))))

        nk = fuzz.defuzz(x_nk,aggregated, 'mom')

        df1 = pd.DataFrame({
            'ID' : [labelId],
            'NK' : [nk]
        })

        df2 = pd.concat([df2, df1], ignore_index=True, sort=False)
    df2.sort_values(by=['NK'], inplace=True, ascending = False)

    fz = df2.to_numpy()

    x = [ [0]*2 for j in range(10)]

    for i in range(10):
        for j in range(2):
            x[i][j]= fz[i][j]    

    fuzzy = pd.DataFrame(x, index = [1,2,3,4,5,6,7,8,9,10],columns = ['ID','NK'])
    fuzzy.reset_index(inplace=True)
    fuzzy = fuzzy.rename(columns = {'index':'No'})

    fuzzy.to_excel('luaran.xlsx', index=False)
    

#hasil clone lagi

if __name__== "__main__":
    masukan()
    luaran()
