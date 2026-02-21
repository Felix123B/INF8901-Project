import xarray as xr
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def preprocess(data="sst.mnmean.nc"):
    # Load SST
    sst = xr.open_dataset(data)["sst"]
    
    # Deseason + detrend (gridpoint-wise)
    sst_deseason = sst.groupby("time.month") - sst.groupby("time.month").mean("time")
    coef = sst_deseason.polyfit("time", deg=1).polyfit_coefficients
    sst_anom = sst_deseason - xr.polyval(sst_deseason["time"], coef)

    # Tropical Pacific subset + weights
    sst_trop = sst_anom.sel(lon=slice(120, 290), lat=slice(12.5, -20))
    wlat = np.cos(np.deg2rad(sst_trop.lat))

    # Niño 3.4 (area-weighted, 3-mo running mean)
    nino34 = (sst_trop.sel(lon=slice(190, 240), lat=slice(5, -5)).weighted(wlat.sel(lat=slice(5, -5))).mean(("lat", "lon")).rolling(time=3, center=True).mean().dropna("time"))

    # PCA on weighted tropical field
    X = (sst_trop * np.sqrt(wlat)).stack(z=("lat", "lon")).values
    pcs = PCA(n_components=2).fit_transform(X)

    df_pcs = pd.DataFrame(pcs, index=sst_trop.time.to_pandas(), columns=["PC1", "PC2"])
    df_pcs = df_pcs.rolling(3, center=True).mean().dropna()

    data = pd.concat([df_pcs, nino34.to_series().rename("EN_34")], axis=1).dropna()
    
    return data, sst_trop, sst_anom