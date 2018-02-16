from astropy.io import fits
import pandas as pd
import glob

sdir = "/data2/cpb405/DR1/"
files = glob.glob(sdir + '*.fits')

d = {'designation':[], 'FILENAME':[]}
for idx, file in enumerate(files):
    with fits.open(file) as hdulist:
        d['designation'].append(hdulist[0].header['DESIG'][7:])
        d['FILENAME'].append(hdulist[0].header['FILENAME'])
df = pd.DataFrame(data=d)
print("FITS files successfully read in...")

catalog_file = "/data2/cpb405/dr1.csv"
catalog = pd.read_csv(catalog_file, sep='|')
catalog.drop_duplicates(subset = 'designation', inplace = True)
print("LAMOST catalog opened...")

df_merged = catalog.merge(df, on='designation', how='inner')

training_class = df_merged["class"].values
STAR = [training_class == "STAR"]
training_class[STAR] = [x[0] for x in df_merged["subclass"].values[STAR]]
df_merged["training_class"] = training_class
print("DataFrames merged...")

df_merged[["FILENAME","training_class"]].to_csv('Training_set.csv')

