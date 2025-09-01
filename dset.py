import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-------------------------------------------------------------------------------------

#Load it into a pandas dataframe
columns = ["NAAB","Name","Lineups/Designations","TPI","NM$","CM$","DWP$®","HHP$®","PTAM","CFP","PTAF","PTAF%","PTAP","PTAP%","PL","DPR","LIV","SCS","MAST","Z MAST","PTAT","UDC","FLC","SCE","DCE","SSB","DSB","STA","STR","DFM","RUA","RLS","RTP","FTL","NM$ Rel","FM$","GM$","F SAV","F SAV Dtrs","F SAV Herds","F SAV Rel","Rel","PTAP Dtrs","MT","WT$®","Z MAST Dtrs","Z MAST Rel","Z LAME","Z LAME Dtrs","Z LAME Rel","Z MET","Z MET Dtrs","Z MET Rel","Z RP","Z RP Dtr","Z RP Rel","Z Disp. Abomasum","Z DA Dtr","Z DA Rel","Z KET","Z KET Dtr","Z KET Rel","CW$™","Z CRD","Z CRD Dtrs","Z CRD Rel","Z CS","Z CS Dtrs","Z CS Rel","Z CLIV","Z MF","Z CLIV Dtrs","Z MF Dtrs","Z CLIV Rel","Z MF Rel","MAST Dtrs","MAST Herds","MAST Rel","MET","MET Dtrs","MET Herds","MET Rel","RP","RP Dtrs","RP Herds","Ret. Placenta Rel","DA","Pta DA Num Dtrs","PTA DA Num Herds","Pta DA Rel","KET","KET Dtrs","KET Herds","KET Rel","MF","MF Dtrs","MF Herds","MF Rel","SCE Rel","SCE Obs","DCE Rel","DCE Obs","SCS Rel","PL Rel","LIV Dtrs","LIV Herds","LIV Rel","H LIV","H LIV Dtrs","H LIV Herds","H LIV Rel","DPR Rel","CCR","CCR Rel","HCR","HCR Rel","FI","FI Rel","DSB Obs","DSB Rel","SSB Obs","SSB Rel","PTAT Rel","PTAT Dtrs","BWC","D","SCR","SCR Rel","BOD","RW","GL","GL Dtrs","GL Rel","EFC","EFC Dtrs","EFC Rel","RLR","FTA","FLS","FUA","RUH","RUW","UCL","UDP","FTP","RFI","RFI Dtr","RFI Hrd","RFI Rel","GP$™","aAa","DMS","Beta-Casein","Kappa-Casein","EFI","GFI","Birthdate","Registration Name","Partners","MS","MS Rel","Sire Stack"]
df = pd.read_csv("toros.csv",names = columns)
#si ya hay una fila con nombres, no es necesario usar columns ni names=columns

#-------------------------------------------------------------------------------------

#check the format of each of the columns to verify correct data type
print("*FORMATO DE CADA COLUMNA PARA VERIFICAR EL TIPO DE DATO* \n".center(50))
df.info()
#df.info omite el listado completo, pero se puede forzar con df.info(verbose=True)


#-------------------------------------------------------------------------------------

#decide which varaibles x or columns should stay and which should go based on the
#relevance of the problem.
df_stayed = df.drop(
    [
        'TPI','NM$','CM$','DWP$®','HHP$®','PTAM','CFP','PTAF',
        'PTAF%','PTAP','PTAP%','PL','DPR','LIV','SCS','MAST','Z MAST','UDC','FLC',
        'SCE','DCE','SSB','DSB','NM$ Rel','FM$','GM$','F SAV','F SAV Dtrs','F SAV Herds',
        'F SAV Rel','Rel','PTAP Dtrs','MT','WT$®','Z MAST Dtrs','Z MAST Rel','Z LAME',
        'Z LAME Dtrs','Z LAME Rel','Z MET','Z MET Dtrs','Z MET Rel','Z RP','Z RP Dtr',
        'Z RP Rel','Z Disp. Abomasum','Z DA Dtr','Z DA Rel','Z KET','Z KET Dtr',
        'Z KET Rel','CW$™','Z CRD','Z CRD Dtrs','Z CRD Rel','Z CS','Z CS Dtrs','Z CS Rel',
        'Z CLIV','Z MF','Z CLIV Dtrs','Z MF Dtrs','Z CLIV Rel','Z MF Rel','MAST Dtrs',
        'MAST Herds','MAST Rel','MET','MET Dtrs','MET Herds','MET Rel','RP','RP Dtrs',
        'RP Herds','Ret. Placenta Rel','DA','Pta DA Num Dtrs','PTA DA Num Herds',
        'Pta DA Rel','KET','KET Dtrs','KET Herds','KET Rel','MF','MF Dtrs','MF Herds',
        'MF Rel','SCE Rel','SCE Obs','DCE Rel','DCE Obs','SCS Rel','PL Rel','LIV Dtrs',
        'LIV Herds','LIV Rel','H LIV','H LIV Dtrs','H LIV Herds','H LIV Rel','DPR Rel',
        'CCR','CCR Rel','HCR','HCR Rel','FI','FI Rel','DSB Obs','DSB Rel','SSB Obs',
        'SSB Rel','PTAT Rel','PTAT Dtrs','BWC','D','SCR','SCR Rel','BOD','GL','GL Dtrs',
        'GL Rel','EFC','EFC Dtrs','EFC Rel','FLS','RFI','RFI Dtr','RFI Hrd','RFI Rel',
        'GP$™','aAa','DMS','Beta-Casein','Kappa-Casein','EFI','GFI','Birthdate',
        'Registration Name','Partners','MS','MS Rel','Sire Stack'
    ],
    axis=1
)


print("\n-------------------------------------------------------------")
print("*PORCENTAJE DE VALORES NULOS* \n".center(50))
print(df_stayed.isnull().mean() * 100)

# Eliminar filas con más de 2 NaN
df_col_2NaN = df_stayed[df_stayed.isnull().sum(axis=1) <= 2]
# Reemplazar NaNs restantes con la media de cada columna
df_no_NaN = df_col_2NaN.fillna(df_col_2NaN.mean(numeric_only=True))
# Detectar columnas de tipo string u objeto
string_cols = df_no_NaN.select_dtypes(include=['object', 'string']).columns
# Reemplazar NaN con "ninguno" solo en esas columnas de texto con NaNs
df_no_NaN[string_cols] = df_no_NaN[string_cols].fillna('ninguno')

print("\n-------------------------------------------------------------")
print("*DATA FRAME SIN VALORES NULOS* \n".center(50))
print(df_no_NaN)
print("\n-------------------------------------------------------------")
print("*REVISAR QUE NO HAYA VALORES NUL0S:* \n".center(50))
print(df_no_NaN.isna().sum()) #revisar valores nulos


print("\n-------------------------------------------------------------")
pd.plotting.scatter_matrix(df_no_NaN)
plt.show()

#sns.pairplot(df_no_NaN, hue='Lineups/Designations', height=1.5)
#plt.show()

#-------------------------------------------------------------------------------------
print("\n-------------------------------------------------------------")
#identify which attributes should be x (predictors) and which shold be y (class).
#Then check the general statistics of each column. min max mean null values etc...
df_x = [["PTAT","STA","STR","DFM","RUA","RLS","RTP","FTL","RW","RLR","FTA","FUA","RUH","RUW","UCL","UDP","FTP"]]
df_y = [["Lineups/Designations"]] #"NAAB",

print("\n-------------------------------------------------------------")
print("*MIN Y MAX DE CADA COLUMNA DE DF_X* \n".center(50))
print(df_no_NaN[df_x[0]].describe()) #min, max de cada columna de x (terminal)
print("\n-------------------------------------------------------------")
print("*CANTIDAD DE NaN DE DF_X* \n".center(50))
print(df_no_NaN[df_x[0]].isnull().sum()) #contar valores nulos de x (terminal)
print("\n-------------------------------------------------------------")
print("*CANTIDAD DE NaN DE DF_Y* \n".center(50))
print(df_no_NaN[df_y[0]].isnull().sum()) #contar valores nulos de y (terminal) no hay nadie sin nombre


print("\n-------------------------------------------------------------")
#agregamos una columna con el bias (1) sin agregar ceros hasta que sepa que rayos hacen



