import requests
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import overpy

# script for returning elevation from lat, long, based on open elevation data
# which in turn is based on SRTM
def get_elevation(lat, long):
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={lat},{long}')
    r = requests.get(query).json()  # json object, various ways you can extract value
    # one approach is to use pandas json functionality:
    # elevation = pd.io.json.json_normalize(r, 'results')['elevation'].values[0]
    elevation = pd.json_normalize(r, 'results')['elevation'].values[0]
    return elevation

def left_long(y,x):
  left_rad = x*(math.pi/180)-(0.00078534/math.cos(y*math.pi/180))
  left_deg=left_rad*(180/math.pi)
  return left_deg

def right_long(y,x):
  right_rad = x*(math.pi/180)+(0.00078534/math.cos(y*math.pi/180))
  right_deg=right_rad*(180/math.pi)
  return right_deg

def create_box(y,x,km):
  ymin=(y-0.009*km)
  ymax=(y+0.009*km)
  # xmin=(left_long(y,x))
  # xmax=(right_long(y,x))
  ext=km*(np.cos(y*np.pi/180)*6378.1*2*np.pi/360)**(-1)
  xmin=x-ext
  xmax=x+ext
  return [xmin,xmax,ymin,ymax]

coords=[46.99,-121]
[xmin,xmax,ymin,ymax]=create_box(coords[0],coords[1],1)
res=30
steps=(xmax-xmin)/res #steps is used to be neg value bc neg sign
currx=xmin
curry=ymin
ele=0
df=[]
X=[]
Y=[]
queries=[]
while(currx<xmax):
  X.append(currx)
  while(curry<ymax):
    if currx==xmin:
      Y.append(curry)
    queries.append(str(curry)+","+str(currx)+"|")
    curry+=steps
  curry=ymin
  currx+=steps
elevation=[]
maxs=200
for i in tqdm(range(1+int(len(queries)/maxs))):
    # print(len(queries[i*200:min(i*200+200,len(queries))]))
    finalq="".join(queries[i*maxs:min(i*maxs+maxs,len(queries))])[:-1]
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations='+finalq)
    r = requests.get(query).json()
    elevation.extend(pd.json_normalize(r, 'results')['elevation'].values)
try:
    elevation=np.reshape(elevation,(res,-1))
except:
    elevation=np.reshape(elevation,(res+1,-1))


x, y = np.meshgrid(Y,X)
# from scipy.interpolate import RegularGridInterpolator
# interp=RegularGridInterpolator((X, Y), elevation,method='cubic',bounds_error=False)
plt.figure(1,figsize=(10,10))
ax = plt.axes(projection='3d')
# res2=(xmax-xmin)/100
# X=np.arange(xmin,xmax,res2)
# Y=np.arange(ymin,ymax,res2)
# x, y = np.meshgrid(Y,X)
# ele=[]
# for i in tqdm(range(len(X))):
#     ele.append(interp([[i,k] for k in Y]))
ax.plot_surface(x, y, elevation,edgecolor='none',rstride=1, cstride=1,
                cmap='terrain',antialiased=True)
ax.plot([coords[0],coords[0]],[coords[1],coords[1]],[np.min(elevation),np.max(elevation)*1.5],'r-',linewidth=4)

api = overpy.Overpass()
minlen=10
result = api.query("[out:json];way("+str(ymin)+","+str(xmin)+","+str(ymax)+","+str(xmax)+");out;")
waylats=[[float(n.lat) for n in w.get_nodes(resolve_missing=True)] for w in result.ways] # ways x nodes
waylongs=[[float(n.lon) for n in w.get_nodes(resolve_missing=True)] for w in result.ways] # ways x nodes

plt.figure(2)
for way in range(len(waylats)):
  if len(waylongs[way])>minlen:
    plt.plot(waylongs[way],waylats[way],np.zeros(len(waylats[way])))
# ax.set_ylim((ymin,ymax))
# ax.set_xlim((xmin,xmax))
plt.ylim(ymin,ymax)
plt.xlim(xmin,xmax)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    