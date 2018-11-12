# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:29:02 2018

@author: ovdessel
"""
import numpy as np

output_dir = 'TLE/'


num_sats = 11
num_planes = 6
altitude = 350

inclination = np.degrees(np.arccos(-np.power((6378+altitude)/12352,7/2)))



#ISS (ZARYA)
#1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
#2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537

def sum_digits(num):
    num_str = str(num)
    if '.' in num_str:
        return sum(map(int,num_str.replace('.','')))
    else: return sum(map(int,str(num)))

for raan_adjust in range(0,num_planes):
    for i in range(0,num_sats):
        print(i)
        sat_name = 'sat_%d_%d'%(raan_adjust,i)
        tle_file = open(output_dir+sat_name+'.tle','w')
        
        tle_file.write('%s\n'%(sat_name))
        tle_file.write('1 %05dU 00000A   18001.00000001 -.00002182  00000-0 -36183-4 0  0010\n'%(i))
        
        raan = raan_adjust*(180./num_planes)
        eccentricity = 623
        AoP = 5.
        MeanAnomaly = i*(360/num_sats)
        if raan_adjust%2 == 0:
            MeanAnomaly += (360./num_sats)/2
        
        velocity = np.sqrt(398600.5/(6378.14+(altitude)))
        MeanMotion = (24.0*60.*60.)/(np.pi*2.*(6378.14+altitude)/velocity)
        
        tle_file.write('2 %05d %03d.%04d %03d.%04d %07d %03d.%04d %03d.%04d %08f000010\n'%(i,
                       inclination,
                       inclination%1,
                       raan,
                       raan%1,
                       eccentricity,
                       AoP,
                       AoP%1,
                       MeanAnomaly,
                       MeanAnomaly%1,
                       MeanMotion))
        tle_file.close()
    
    