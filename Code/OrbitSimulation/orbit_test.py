#Import libraries
#math
import numpy as np

#Plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.ion()

#orbit propagator and map
import cartopy.crs as ccrs
from orbit_predictor.sources import EtcTLESource
from orbit_predictor.locations import ARG
from mpl_toolkits.basemap import Basemap

#progressbar
from progressbar import ProgressBar

#Data
import pandas as pd

#image analysis
from PIL import Image
import matplotlib.patches as mpatches

#%%Sub routines
#Function to handle orbit propagation and view
#   Built around orbit_predictor and Basemap
def satellite_loc(predictor,times):
    sat_data = pd.DataFrame(index=dates, columns=["lat", "lon","alt"])
    sat_view = []   
    color = color=np.random.rand(3,)

    pbar = ProgressBar()
    for time in pbar(times):
        #propagate orbit
        lat, lon, alt = predictor.get_position(time).position_llh
        
        #calculate view projection. No resolution (faster)
        m = Basemap(projection='nsper',lon_0=lon,lat_0=lat,
                    satellite_height=alt*1000.,resolution=None)
    
        #append data to frame and list. Assign random color 
        sat_data.loc[time] = (lat, lon,alt)
        sat_view.append([m.boundarylats[1::200],m.boundarylons[1::200]])
        
       
        
    return [sat_data,sat_view,color]

    
#%% Load TLE Files
import os
sat_predictors = []

#iterate through all files
for file in os.listdir("TLE"):
    #Get file to find satellite name
    read = open("TLE/"+file)
    sat_name = read.readlines()[0].rstrip('\n')
    
    #pull source into propagator
    source = EtcTLESource(filename="TLE/"+file)
    sat_predictors.append(source.get_predictor(sat_name))

    
#%%

#propagation dates
point_num = 1100
dates = pd.date_range(start="2018-10-27 00:00", periods=point_num, freq="5S")

sat_data = []

for sat in sat_predictors:
    sat_data.append(satellite_loc(sat,dates))
    


#%% Animate/plot
fig = plt.figure(figsize=(15,25))
ax = plt.axes(projection=ccrs.PlateCarree())

fig2=plt.figure()
ax2 = plt.axes(projection=ccrs.PlateCarree())
ax2.set_global()

def init():
    return

def animate(i):
    #clean slate
    ax.clear()
    ax2.clear()
    
    #set bounds
    ax.stock_img()
    ax2.set_global()
    
    for sat in sat_data:
        sat_pos = sat[0]
        sat_view = sat[1]
        color = sat[2]
        
        #plot satellite point and view radius
        ax.scatter(sat_pos["lon"][i],sat_pos["lat"][i],
                   transform=ccrs.Geodetic(),c=color,s=20)
        
        #find coverage
        bounds = np.vstack((sat_view[i][1],sat_view[i][0])).T
        ax.add_patch(mpatches.Polygon(bounds,facecolor=color,alpha=0.1,
                                       transform=ccrs.Geodetic()))
        ax.plot(sat_view[i][1],sat_view[i][0],
                transform=ccrs.Geodetic(),c=color,linewidth=1)
        
        
        ax2.add_patch(mpatches.Polygon(bounds,facecolor='k',alpha=1,
                                       transform=ccrs.Geodetic()))
    
    #save image
    fig2.savefig('temp.png',bbox_inches="tight",pad_inches=-0.1)
    #analyze image pixel by pixel 
    im = Image.open('temp.png')
    pixels = im.getdata()          # get the pixels as a flattened sequence
    black = (0,0,0,255)
    white_pixels = 0
    for pixel in pixels:
        if pixel != black:
            white_pixels += 1
    coverage = (1.0 - white_pixels/len(pixels))*100.0
    im.close()
        
    ax.set_title('Coverage = %.2f%%'%(coverage))
    
    
anim = animation.FuncAnimation(
        fig, animate, interval=30, frames=point_num-1)
 
plt.draw()
plt.show()

 #Set up formatting for the movie files
mywriter = animation.FFMpegWriter()
anim.save('mymovie.mp4',writer=mywriter)
