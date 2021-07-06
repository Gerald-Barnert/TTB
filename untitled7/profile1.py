import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from yt.units import pc


ds = yt.load("E:/Gerald/TTbasico2/output_00017/info_00017.txt")


print(ds.field_list)
print(ds.derived_field_list)
print(ds.field_info["index", "height"].get_units())

#field = 'sound_speed'
#field = 'velocity_cylindrical_theta'
#field = 'cylindrical_tangential_velocity'
#field = 'height'
field = 'density'
#field = 'thickness_z'
#field = 'z'
#field = 'density'
ad = ds.all_data()
proj = ds.proj(field, 'z', weight_field='density')
#proj = ad.integrate(field, axis="z")
width = (50, 'pc')
res = [5000, 5000]
frb = proj.to_frb(width, res)
surfer = np.flip(np.array(frb[field]),0)
surfer = np.flip(np.array(frb[field].in_units('g/cm**3')),0)
norm = cm.colors.LogNorm()
ex =  (-width[0]/2, width[0]/2, -width[0]/2,width[0]/2)
plt.imshow(surfer, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$g/cm^{3}$')
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('prueba_projeccion_sinrad_{}'.format(field))

print(surfer)




