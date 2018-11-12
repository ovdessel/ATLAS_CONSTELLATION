# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 21:15:02 2018

@author: ovdessel
"""
#%% Import modules
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

import geopy.distance


import scipy
import scipy.optimize




#%% set parameters

v = 299792458 #speed of light in m/s

transmitter = (0.1,0.2,0) #latitude, longitude, altitude
altitude = 800.

satellite = (1.,1.,altitude*1000.0) #Latitude, longitude, altitude
sat_nodes = 12
sat_length = 20000.

time_err = 1000e-12

#%% sub-routine

#convert long lat to xyc in ecef and vice-versa
import pyproj
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84') 
#x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)

def tdoa_func(point):
    v = 299792458.0 #speed of light
    
    x,y,z = point  #parse point
    functions = [] #build empty return functions
    m_0x,m_0y,m_0z,t0 = measure[0] #parse reference point
        
    #iterate through all data points
    for m in measure:
        if np.all(m == measure[0]):
            continue
        mx,my,mz,mt = m #expand measurment points
        f = (np.sqrt((x-m_0x)**2+(y-m_0y)**2+(z-m_0z)**2) - 
             np.sqrt((x-mx)**2+(y-my)**2+(z-mz)**2))+(np.abs(mt-t0)*v)
        
        functions.append(f)
    
    return functions

def sat_array_build(num_nodes,tot_length,coords):
    #Define satellite array
    sat_array = []
    
    #string of nodes form around a circle. Simplification of Catenoid surface
    #due to drag
    radius = (tot_length*4/np.pi)/2.
    #angle of seperation denoted by # of nodes
    angle_sep = np.pi/(2*(num_nodes-1))

    #convert lat,long,altitude positions into ecef coordinates
    #used as center of circle on xy plane
    x,y,z = pyproj.transform(lla,ecef,coords[0],coords[1],
                             altitude, radians=False)
    #iterate through nodes
    for i in range(0,num_nodes,1):
        #Distribute pseudo-randomly the nodes across pi/2 angles
        #+/- 10 degrees
        angle = angle_sep*i + np.deg2rad(np.random.randint(-250,250)/100.)
        #caculate x/y
        x_n = np.cos(angle)*radius
        y_n = np.sin(angle)*radius
        #append to list
        sat_array.append([x + x_n, y + y_n, z])
    
    #Return an array
    return np.array(sat_array)

# %%===========================================================================
# set Tag location and Satellite
# =============================================================================

#Define transmitter location from latitude,longitude,altitude to ecef
tx_x,tx_y,tx_z = pyproj.transform(lla,ecef,transmitter[0],
                                           transmitter[1],
                                           transmitter[2],
                                  radians=False)
tx_ecef = np.array([tx_x,tx_y,tx_z])

#build satellite array
sat_array = sat_array_build(sat_nodes,sat_length,satellite)


#%% run through 1000 trial points
trial = []
for i in range(0,1000):
    #build satellite array
    #sat_array = sat_array_build(sat_nodes,sat_length,satellite)
    
    #calculate distances and add 1ns noise
    dist = np.linalg.norm(sat_array-tx_ecef,axis=1)
    d_dist = dist-dist[0]
    d_time = d_dist/v + np.random.normal(0,time_err,size=len(dist))
    
    #10 meters of placement noise
    #+np.random.normal(0,1e-9,size=(len(sat_array),3))
    measure = np.hstack((sat_array,
                         np.array([d_time]).T))
    
    #initial guess is placed somewhere +/- 25km of point
    sol,cov = scipy.optimize.leastsq(tdoa_func,(tx_ecef[0]+np.random.randint(-5000,5000),
                                                tx_ecef[1]+np.random.randint(-5000,5000),
                                                tx_ecef[2]+np.random.randint(-5000,5000)),
                                     maxfev=2000,
                                     xtol=1e-15,
                                     ftol=1e-15,
                                     gtol=1e-15
                                     )
    
    trial.append(sol)

#%% Find error
solutions = np.array(trial) #trial data into solutions array

#convert ECEF coordinates into lat/lons
coords = []
coord_error = []
for point in solutions:
    lat,lon,alt = pyproj.transform(ecef,lla,point[0],point[1],point[2])
    
    #lat lon calculate distances
    coord_error.append(geopy.distance.distance(transmitter[0:2],(lat,lon)).km*1000.0)
    coords.append([lat,lon,alt])

coords = np.array(coords)
coord_avg = np.mean(coords,axis=0)

coord_err_avg = np.mean(coord_error)
coord_err_std = np.std(coord_error)


print("Average is: %03fm"%(coord_err_avg))
print("STD Error is: %03fm"%(coord_err_std))


#%%
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if False:
    fig = plt.figure(figsize=plt.figaspect(1)*1.5)
    ax = fig.add_subplot(111, projection='3d')
    
    #ax.scatter(tx_ecef[0],tx_ecef[1],tx_ecef[2],c='r')
    
    ax.scatter(sat_array[:,0],sat_array[:,1],sat_array[:,2],c='b')
    
    set_axes_equal(ax)
    plt.show()

    
    plt.grid()
    plt.show()


#%%
# the histogram of the data

from scipy.stats import halfnorm, kstest

file = open('data_error.txt','w')
for data in coord_error:
    file.write('%f\n'%(data))
file.close()


if True:
    title = ('Altitude %dkm | #Nodes %d\n Baseline %dm | TimeError %.2fps'
                %(altitude,sat_nodes,sat_length,time_err*1e12))
    fig = plt.figure()
    n, bins, patches = plt.hist(coord_error,bins = 50,
                                #range=(-10,10),
                                normed=True,
                                facecolor='g', alpha=0.6)
    
    # add a 'best fit' line
    param = halfnorm.fit(coord_error)
    x = np.linspace(0,3500,100)
    # fitted distribution
    pdf_fitted = halfnorm.pdf(x,loc=param[0],scale=param[1])
    plt.plot(x,pdf_fitted,'r-',alpha=0.8)
    
    #plot test values
    test = kstest(coord_error,'halfnorm')
    plt.text(2500,.0004,s=('Half-normal \nKS-Stat=%.3f \nPvalue = %.3f \nMean = %.3f \nSTD=%.3f'
                             %(test[0],test[1],coord_err_avg,coord_err_std)),
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
    
    plt.axvline(x=coord_err_avg,c='b',alpha=0.5)
    plt.axvline(x=coord_err_avg+coord_err_std,c='k',linestyle='-.',alpha=0.5)
    plt.axvline(x=coord_err_avg-coord_err_std,c='k',linestyle='-.',alpha=0.5)
    
    plt.xlabel('Error (m)')
    plt.title(title)
    #plt.axis([0, 650, 0, 0.007 ])
    plt.grid(True)
    
    plt.show()
    
    plt.savefig('%d_%d_%d_%.2f.png'%(altitude,sat_nodes,sat_length,time_err*1e12))
    
    #print(np.mean(error[:,0]))