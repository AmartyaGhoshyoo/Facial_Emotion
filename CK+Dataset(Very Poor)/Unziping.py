import zipfile
with zipfile.ZipFile("CK+Dataset(Very Poor)/archive.zip") as f:
  f.extractall('CK+Dataset(Very Poor)')