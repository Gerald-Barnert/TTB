#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc_context

ds = yt.load("E:/Gerald/TTbasico/output_00013/info_00013.txt")

ds.print_stats()
print(ds.field_list)
print(ds.derived_field_list)
print(ds.field_info["gas", "vorticity_x"].get_source())
print(ds.domain_width)
print (ds.domain_width.in_units("kpc"))
print (ds.domain_width.in_units("au"))
print (ds.domain_width.in_units("mile"))
print (ds.particle_types)
print (ds.particle_types_raw)
print (ds.particle_type_counts)

sp = ds.sphere("max", (5, 'kpc'))
print(sp)
print(list(sp.quantities.keys()))
print(sp.quantities.total_mass())
print ("Redshift =", ds.current_redshift)

p = yt.ProjectionPlot(ds, "z", "density")
#p.zoom(5)
p.save()

p2 = yt.SlicePlot(ds, 'z', "density")
p2.zoom(5)
p2.save()

p3 = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', 'particle_mass')
p3.zoom(500)
p3.save()

sp0 = ds.sphere(ds.domain_center, (5, "kpc"))
bulk_vel = sp0.quantities.bulk_velocity()
sp1 = ds.sphere(ds.domain_center, (5, "kpc"))
sp1.set_field_parameter("bulk_velocity", bulk_vel)
rp0 = yt.create_profile(sp0, 'radius', 'radial_velocity',
                        units = {'radius': 'kpc'},
                        logs = {'radius': False})
rp1 = yt.create_profile(sp1, 'radius', 'radial_velocity',
                        units = {'radius': 'kpc'},
                        logs = {'radius': False})

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(rp0.x.value, rp0["radial_velocity"].in_units("km/s").value,
        rp1.x.value, rp1["radial_velocity"].in_units("km/s").value)

ax.set_xlabel(r"$\mathrm{r\ (kpc)}$")
ax.set_ylabel(r"$\mathrm{v_r\ (km/s)}$")
ax.legend(["Without Correction", "With Correction"])

fig.savefig("%s_profiles.png" % ds)


'''
yt.enable_parallelism()
import collections

# Enable parallelism in the script (assuming it was called with
# `mpirun -np <n_procs>` )
yt.enable_parallelism()

# By using wildcards such as ? and * with the load command, we can load up a
# Time Series containing all of these datasets simultaneously.
ts = yt.load('TTbasico/output_?????/info_?????.txt')

# Calculate and store density extrema for all datasets along with redshift
# in a data dictionary with entries as tuples

# Create an empty dictionary
data = {}

# Iterate through each dataset in the Time Series (using piter allows it
# to happen in parallel automatically across available processors)
for ds in ts.piter():
    ad = ds.all_data()
    extrema = ad.quantities.extrema('density')

    # Fill the dictionary with extrema and redshift information for each dataset
    data[ds.basename] = (extrema, ds.current_redshift)

# Convert dictionary to ordered dictionary to get the right order
od = collections.OrderedDict(sorted(data.items()))

# Print out all the values we calculated.
print("Dataset      Redshift        Density Min      Density Max")
print("---------------------------------------------------------")
for key, val in od.items():
    print("%s       %05.3f          %5.3g g/cm^3   %5.3g g/cm^3" % \
           (key, val[1], val[0][0], val[0][1]))

'''

# Create a slice plot for the dataset.  With no additional arguments,
# the width will be the size of the domain and the center will be the
# center of the simulation box
slc = yt.SlicePlot(ds, 'z', 'density')

# Create a list of a couple of widths and units.
# (N.B. Mpc (megaparsec) != mpc (milliparsec)
widths = [(0.1, 'pc'),
          (1, 'pc'),
          (5, 'pc')]

# Loop through the list of widths and units.
for width, unit in widths:

    # Set the width.
    slc.set_width(width, unit)

    # Write out the image with a unique name.
    slc.save("%s_%010d_%s" % (ds, width, unit))

zoomFactors = [7,8,9,10]

# recreate the original slice
slc = yt.SlicePlot(ds, 'z', 'density')

for zoomFactor in zoomFactors:

    # zoom in
    slc.zoom(zoomFactor)

    # Write out the image with a unique name.
    slc.save("%s_%i" % (ds, zoomFactor))


ts = yt.load('TTbasico/output_000**/info_000**.txt')


plot = yt.SlicePlot(ts[0], 'x', 'density')
fig = plot.plots['density'].figure

# animate must accept an integer frame number. We use the frame number
# to identify which dataset in the time series we want to load
def animate(i):
    ds = ts[i]
    plot._switch_ds(ds)

animation = FuncAnimation(fig, animate, frames=len(ds))
animation.save('animation.gif')

# Override matplotlib's defaults to get a nicer looking font
#with rc_context({'mathtext.fontset': 'stix'}):
#    animation.save('animation.mp4')



ds.add_mesh_sampling_particle_field(('gas', 'temperature'), ptype='all')

print('The temperature at the location of the particles is')
print(ds.r['all', 'cell_gas_temperature'])


#my_galaxy = ds.disk(ds.domain_center, [0.0, 0.0, 1.0], 10*kpc, 3*kpc)
#plot = yt.ProfilePlot(my_galaxy, "radius", 'io_mass')
#plot.save()


plot = yt.SlicePlot(ts[1], 'x', 'density')
fig = plot.plots['density'].figure

# animate must accept an integer frame number. We use the frame number
# to identify which dataset in the time series we want to load
def animate(i):
    ds = ts[i]
    plot._switch_ds(ds)

ani = animation.FuncAnimation(fig, animate, frames=len(ts))
#animation.save('animation')

#Override matplotlib's defaults to get a nicer looking font
#with rc_context({'mathtext.fontset': 'stix'}):
#    animation.save('animation')

writergif = animation.PillowWriter(fps=30)
ani.save('filename.mp4',writer=writergif)


my_galaxy = ds.disk(ds.domain_center, [0.0, 0.0, 1.0], 10*kpc, 3*kpc)
plot = yt.ProfilePlot(my_galaxy, "radius", ["sound_speed"])
plot.set_unit("radius", "pc")
plot.set_unit("sound_speed", "km/s")
plot.save()

my_galaxy = ds.disk(ds.domain_center, [0.0, 0.0, 1.0], 10*kpc, 3*kpc)
plot1 = yt.ProfilePlot(my_galaxy, "radius", ["radial_velocity"])
plot1.set_unit("radius", "pc")
plot1.set_unit("radial_velocity", "km/s")
plot1.save()



ds = yt.load('TTbasico/output_00003/info_00003.txt')

p = yt.ProjectionPlot(ds, "z", ['gaz', 'H_nuclei_density'])
p.set_width(30,'pc')
p.save()