#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib import rc_context
from yt.units import pc
import numpy as np


ds = yt.load("E:/Gerald/TTbasico/output_00025/info_00025.txt")
print(ds.field_list)
print(ds.derived_field_list)
print (ds.particle_types)
ds.print_stats()
print ("t =", ds.current_time)
print('domain center =', ds.domain_center )
ad = ds.all_data()

print(ad['gas','z'])
print(ad['height'])

#print(ds.field_info['gas', 'velocity_cylindrical_theta'].get_source())
#print(ds.field_info['index', 'cylindrical_z'].get_source())
#print(ds.field_info['gas', 'z'].get_source())
#print(ds.field_info['index', 'cylindrical_radius'].get_source())

mass = ad.include_above('cell_mass', 1e15)
print(ad['cell_mass'])
proj2 = yt.ProjectionPlot(ds, 'x', "density", weight_field="density", data_source=mass)
proj2.save('test')

def thickness_z(field, data):
    thickness = data['gas','z'] / data['cylindrical_radius']
    return thickness

ds.add_field(("gas", "thickness_z"),
             units=None, function=thickness_z,
             sampling_type="cell")

plot1 = yt.ProfilePlot(mass, "cylindrical_radius", "thickness_z", n_bins=1000000, accumulation=True)
plot1.set_unit("cylindrical_radius", "pc")
plot1.set_xlim(10**-1+0.02,10)
plot1.save('test')

plot1 = yt.ProfilePlot(mass, "cylindrical_radius", ["velocity_cylindrical_theta"], n_bins=1000000, accumulation=True)
plot1.set_unit("cylindrical_radius", "pc")
plot1.set_unit("velocity_cylindrical_theta", "km/s")
plot1.set_xlim(10**-2+0.05,10)
plot1.save('velocidad_radial')

plot1 = yt.ProfilePlot(mass, "cylindrical_radius", ["sound_speed"], n_bins=1000000, accumulation=True)
plot1.set_unit("cylindrical_radius", "pc")
plot1.set_unit("sound_speed", "km/s")
plot1.set_xlim(10**-2 + 0.05,10)
plot1.save('velocidad_sonido')





















