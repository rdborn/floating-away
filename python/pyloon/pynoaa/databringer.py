import struct
import numpy as np

pres = [1, 10, 100, 1000, 150, 2, 20, 200, 250, 3, 30, 300, 350, 400, 450, 5,
        50, 500, 550, 600, 650, 7, 70, 700, 750, 80, 800, 850, 900, 925, 950, 975]
pres.sort()
DATA_ENTRY_LENGTH = 20
DATA_ENTRY_FORMAT = '>fffff'
P_0 = 1013.25
FT_2_M = 0.3048
M_2_FT = 3.28084

def fetch(min_lat, min_lon, max_lat, max_lon, hrs_ahead):
    data_files = []
    min_latitude = get_lat(min_lat)
    max_latitude = get_lat(max_lat)
    min_longitude = get_lon(min_lon)
    max_longitude = get_lon(max_lon)
    if min_latitude != max_latitude or min_longitude != max_longitude:
        print("WARNING: data request spans multiple files, this functionality is not yet supported. Returning FALSE.")
        return False
    lat = min_latitude
    lon = min_longitude
    alts = hpa2alt(np.array(pres, dtype='float'))
    print(pres)
    print(alts)
    data_files = []
    data = np.array([np.zeros(5)])
    for alt in alts:
        data_files.append([get_file_name(lat, lon, alt, hrs_ahead)])
    for i, data_file in enumerate(data_files):
        __alt__ = alts[i]
        print(data_file[0])
        with open(data_file[0], 'rb') as f:
            raw_data = f.read(DATA_ENTRY_LENGTH)
            while not len(raw_data) < DATA_ENTRY_LENGTH:
                (__lat__, __lon__, vlat, vlon, t) = struct.unpack(DATA_ENTRY_FORMAT, raw_data)
                in_range = (__lat__ >= min_lat) and (__lat__ <= max_lat) and (__lon__ >= min_lon) and (__lon__ <= max_lon)
                if in_range:
                    entry = np.array([__lat__, __lon__, __alt__, vlat, vlon])
                    data = np.append(data, [entry], axis=0)
                raw_data = f.read(DATA_ENTRY_LENGTH)
    return data[1:]

def get_file_name(lat, lon, alt, hrs_ahead):
    latitude = get_lat(lat)
    longitude = get_lon(lon)
    pressure = get_pres(alt)
    hrs = str(hrs_ahead).zfill(3)
    absolute_path = "/home/rdborn/Documents/msl/floating-away/python/pyloon/pynoaa/"
    relative_path = "./"
    data_file = absolute_path + "data/gfs_4_20170701_0000_" + hrs + "/L" + pressure + "/C" + latitude + "_" + longitude + ".gribp"
    return data_file

def get_lat(lat):
    lats = np.linspace(-100, 75, 8)
    return __find__(lat, lats)

def get_lon(lon):
    lons = np.linspace(0, 350, 15)
    return __find__(lon, lons)

def __find__(x, xs):
    x = np.int(x)
    for i, __x__ in enumerate(xs):
        x_cmp1 = np.int(xs[i])
        if (x >= x_cmp1):
            if (i+1) == len(xs):
                return_x = str(np.int(np.floor(__x__)))
                return return_x
            else:
                x_cmp2 = np.int(xs[i+1])
                if (x < x_cmp2):
                    return_x = str(np.int(np.floor(__x__)))
                    return return_x

def get_pres(alt):
    d = np.inf
    hpa = alt2hpa(alt)
    for i, p in enumerate(pres):
        if abs(hpa - p) < d:
            d = abs(hpa - p)
            pressure = str(np.int(np.floor(p)))
    return pressure

def alt2hpa(alt):
    alt_ft = alt * M_2_FT
    C1 = 0.190284
    C2 = 145366.45
    hpa = P_0 * (1. - alt_ft / C2)**(1./C1)
    return hpa

def hpa2alt(hpa):
    C1 = 0.190284
    C2 = 145366.45
    alt = (1. - (hpa / P_0)**C1) * C2
    alt = alt * FT_2_M
    return alt
