import yt
import numpy as np
from yt.units import pc, km, s
import matplotlib.pyplot as plt

time = []
ts = yt.load("E:/Gerald/TTbasico/output_?????/info_?????.txt")
for ds in ts:
    time.append(ds.current_time.in_units('yr'))

ds = yt.load("E:/Gerald/TTbasico/output_00012/info_00012.txt")

field = 'sound_speed'
field = 'density'

#units = 'pc'
units = 'km/s'
units = 'g/cm**2'

proj = ds.proj(field, "z", weight_field=None)
width = (50, 'pc')
res = [100, 100]
frb = proj.to_frb(width, res)
surfer = np.flip(np.array(frb[field].in_units(units)),0)

x_0 = int(res[0]/2)
y_0 = int(res[1]/2)
origen = surfer[x_0][y_0]
corner = surfer[0][0]

def distancia_centro(surfer, i, j):
    x_pixel = np.abs(x_0 - i)
    y_pixel = np.abs(y_0 - j)
    d = np.sqrt(x_pixel**2 + y_pixel**2)
    return [d,i,j]

radio_pixel = []
for i in range(0,res[0]):
    for j in range(0, res[1]):
        r = distancia_centro(surfer, i, j)
        radio_pixel.append(r)

arr = []
for k in range(0, len(radio_pixel)):
    for l in range(0, len(radio_pixel)):
        if radio_pixel[k][0] == radio_pixel[l][0] and k!=l:
            #print(radio_pixel[k], radio_pixel[l])
            r_k = r_l = radio_pixel[k][0]
            i_k = radio_pixel[k][1]
            j_k = radio_pixel[k][2]
            i_l = radio_pixel[l][1]
            j_l = radio_pixel[l][2]
            values = [r_k, i_k, j_k, i_l, j_l]
            #print(values)
            arr.append(values)

for i in range(0, len(arr)):
    for j in range(0, len(arr)):
        if arr[i][0] == arr[j][0] and i!=j:
            indices_mismo_radio = [arr[i][1], arr[i][2], arr[i][3], arr[i][4], arr[j][1], arr[j][2], arr[j][3], arr[j][4]]
            #print(indices_mismo_radio)

suma = []

for indice in range(0,int(res[0]/2)):
    s1 = s2 = s3 = s4 = []
    for j in range(indice, res[0] - indice):
        s1.append((surfer[indice][j]))
        s2.append((surfer[res[0]-indice-1][j]))
    for i in range(indice, res[0] - indice):
        s3.append((surfer[i][indice]))
        s4.append((surfer[i][res[0] - indice-1]))

    sum1 = np.sum(s1)
    sum2 = np.sum(s2)
    sum3 = np.sum(s3)
    sum4 = np.sum(s4)
    sum_tot = sum1 + sum2 + sum3 + sum4
    promedio = sum_tot/(len(s1) + len(s2) + len(s3) + len(s4))
    suma.append(promedio)

suma = np.flip(suma)
r = np.linspace(0,int(width[0]/2),len(suma))

plt.clf()
plt.plot(r, suma, label= r'$t = 1 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
plt.ylabel(r'$c_s [km/s]$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.savefig('plots1\perfil_radial2_{}'.format(field))
#plt.show()

######################################################################################################


ds = yt.load("E:/Gerald/TTbasico/output_00017/info_00017.txt")

field = 'sound_speed'
field = 'density'

#units = 'pc'
units = 'km/s'
units = 'g/cm**2'

proj = ds.proj(field, "z", weight_field=None)
#width = (40, 'pc')
#res = [30, 30]
frb = proj.to_frb(width, res)
surfer = np.flip(np.array(frb[field].in_units(units)),0)

x_0 = int(res[0]/2)
y_0 = int(res[1]/2)
origen = surfer[x_0][y_0]
corner = surfer[0][0]

def distancia_centro(surfer, i, j):
    x_pixel = np.abs(x_0 - i)
    y_pixel = np.abs(y_0 - j)
    d = np.sqrt(x_pixel**2 + y_pixel**2)
    return [d,i,j]

radio_pixel = []
for i in range(0,res[0]):
    for j in range(0, res[1]):
        r = distancia_centro(surfer, i, j)
        radio_pixel.append(r)

arr = []
for k in range(0, len(radio_pixel)):
    for l in range(0, len(radio_pixel)):
        if radio_pixel[k][0] == radio_pixel[l][0] and k!=l:
            #print(radio_pixel[k], radio_pixel[l])
            r_k = r_l = radio_pixel[k][0]
            i_k = radio_pixel[k][1]
            j_k = radio_pixel[k][2]
            i_l = radio_pixel[l][1]
            j_l = radio_pixel[l][2]
            values = [r_k, i_k, j_k, i_l, j_l]
            #print(values)
            arr.append(values)

for i in range(0, len(arr)):
    for j in range(0, len(arr)):
        if arr[i][0] == arr[j][0] and i!=j:
            indices_mismo_radio = [arr[i][1], arr[i][2], arr[i][3], arr[i][4], arr[j][1], arr[j][2], arr[j][3], arr[j][4]]
            #print(indices_mismo_radio)

suma = []

for indice in range(0,int(res[0]/2)):
    s1 = s2 = s3 = s4 = []
    for j in range(indice, res[0] - indice):
        s1.append((surfer[indice][j]))
        s2.append((surfer[res[0]-indice-1][j]))
    for i in range(indice, res[0] - indice):
        s3.append((surfer[i][indice]))
        s4.append((surfer[i][res[0] - indice-1]))

    sum1 = np.sum(s1)
    sum2 = np.sum(s2)
    sum3 = np.sum(s3)
    sum4 = np.sum(s4)
    sum_tot = sum1 + sum2 + sum3 + sum4
    promedio = sum_tot/(len(s1) + len(s2) + len(s3) + len(s4))
    suma.append(promedio)

suma = np.flip(suma)
r = np.linspace(0,int(width[0]/2),len(suma))

plt.plot(r, suma, label=r'$t = 1.5 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
plt.ylabel(r'$c_s [km/s]$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.savefig('plots1\perfil_radial2_{}'.format(field))
#plt.show()




######################################################################################################


ds = yt.load("E:/Gerald/TTbasico/output_00023/info_00023.txt")

field = 'sound_speed'
field = 'density'

#units = 'pc'
units = 'km/s'
units = 'g/cm**2'

proj = ds.proj(field, "z", weight_field=None)
#width = (40, 'pc')
#res = [30, 30]
frb = proj.to_frb(width, res)
surfer = np.flip(np.array(frb[field].in_units(units)),0)

x_0 = int(res[0]/2)
y_0 = int(res[1]/2)
origen = surfer[x_0][y_0]
corner = surfer[0][0]

def distancia_centro(surfer, i, j):
    x_pixel = np.abs(x_0 - i)
    y_pixel = np.abs(y_0 - j)
    d = np.sqrt(x_pixel**2 + y_pixel**2)
    return [d,i,j]

radio_pixel = []
for i in range(0,res[0]):
    for j in range(0, res[1]):
        r = distancia_centro(surfer, i, j)
        radio_pixel.append(r)

arr = []
for k in range(0, len(radio_pixel)):
    for l in range(0, len(radio_pixel)):
        if radio_pixel[k][0] == radio_pixel[l][0] and k!=l:
            #print(radio_pixel[k], radio_pixel[l])
            r_k = r_l = radio_pixel[k][0]
            i_k = radio_pixel[k][1]
            j_k = radio_pixel[k][2]
            i_l = radio_pixel[l][1]
            j_l = radio_pixel[l][2]
            values = [r_k, i_k, j_k, i_l, j_l]
            #print(values)
            arr.append(values)

for i in range(0, len(arr)):
    for j in range(0, len(arr)):
        if arr[i][0] == arr[j][0] and i!=j:
            indices_mismo_radio = [arr[i][1], arr[i][2], arr[i][3], arr[i][4], arr[j][1], arr[j][2], arr[j][3], arr[j][4]]
            #print(indices_mismo_radio)

suma = []

for indice in range(0,int(res[0]/2)):
    s1 = s2 = s3 = s4 = []
    for j in range(indice, res[0] - indice):
        s1.append((surfer[indice][j]))
        s2.append((surfer[res[0]-indice-1][j]))
    for i in range(indice, res[0] - indice):
        s3.append((surfer[i][indice]))
        s4.append((surfer[i][res[0] - indice-1]))

    sum1 = np.sum(s1)
    sum2 = np.sum(s2)
    sum3 = np.sum(s3)
    sum4 = np.sum(s4)
    sum_tot = sum1 + sum2 + sum3 + sum4
    promedio = sum_tot/(len(s1) + len(s2) + len(s3) + len(s4))
    suma.append(promedio)

suma = np.flip(suma)
r = np.linspace(0,int(width[0]/2),len(suma))

plt.plot(r, suma, label=r'$t = 2 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
plt.ylabel(r'$\Sigma [g/cm^{2}]$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('plots_finales\ Sigma')
plt.show()





