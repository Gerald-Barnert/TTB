import yt
import numpy as np
from yt.units import pc, km, s
import matplotlib.pyplot as plt

time = []
ts = yt.load("E:/Gerald/TTbasico/output_?????/info_?????.txt")
for ds in ts:
    time.append(ds.current_time.in_units('yr'))

ds = yt.load("E:/Gerald/TTbasico/output_00012/info_00012.txt")

'''
def thickness(field, data):
    thickness = data['sound_speed'] / data['velocity_cylindrical_theta']
    return thickness

ds.add_field(("gas", "thickness"),
             units=None, function=thickness,
             sampling_type="cell")


my_galaxy = ds.disk(ds.domain_center, [0.0, 0.0, 1.0], 10*pc, 1*pc)
plot1 = yt.ProfilePlot(my_galaxy, "cylindrical_radius", "thickness", n_bins=1000000, accumulation=True)
plot1.set_unit("cylindrical_radius", "pc")
plot1.set_xlim(10**-1+0.02,10)
plot1.save()

my_galaxy = ds.disk(ds.domain_center, [0.0, 0.0, 1.0], 10*pc, 1*pc)
plot1 = yt.ProfilePlot(my_galaxy, "cylindrical_radius", ["sound_speed"], n_bins=1000000, accumulation=True)
plot1.set_unit("cylindrical_radius", "pc")
plot1.set_unit("sound_speed", "km/s")
plot1.set_xlim(10**-2 + 0.05,10)
plot1.save('0')

my_galaxy = ds.disk(ds.domain_center, [0.0, 0.0, 1.0], 10*pc, 1*pc)
plot1 = yt.ProfilePlot(my_galaxy, "cylindrical_radius", "z", n_bins=1000000, accumulation=True)
plot1.set_unit("z", "pc")
plot1.set_unit("cylindrical_radius", "pc")
plot1.set_xlim(10**-2+0.05,10)
plot1.save('1')

my_galaxy = ds.disk(ds.domain_center, [0.0, 0.0, 1.0], 10*pc, 1*pc)
plot1 = yt.ProfilePlot(my_galaxy, "cylindrical_radius", ["velocity_cylindrical_theta"], n_bins=1000000, accumulation=True)
plot1.set_unit("cylindrical_radius", "pc")
plot1.set_unit("velocity_cylindrical_theta", "km/s")
plot1.set_xlim(10**-2+0.05,10)
plot1.save()
'''

print(ds.field_list)
print(ds.derived_field_list)
'''
my_galaxy = ds.disk(ds.domain_center, [0.0, 0.0, 1.0], 10*pc, 1*pc)
profile = yt.create_profile(my_galaxy, 'cylindrical_radius', 'sound_speed', n_bins=1000000, accumulation=True)


profile.set_x_unit('pc')
print(profile.x)
print(profile.field_data)


plt.plot(profile.x.value, profile["sound_speed"].value)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]')
plt.ylabel('speed sound [cm/s]')
plt.show()
'''

field = 'sound_speed'
field2 = 'velocity_cylindrical_theta'
field2 = 'tangential_velocity'
#field = 'height'
#field = 'cylindrical_tangential_velocity'
#field = 'z'
#field = 'density'
#field = 'thickness'
units = 'pc'
#units = 'km/s'

ad = ds.all_data()
proj = ds.proj(field, "z", weight_field='density')
proj2 = ds.proj(field2, "z", weight_field='density')
width = (50, 'pc')
res = [100, 100]
frb = proj.to_frb(width, res)
frb2 = proj2.to_frb(width, res)
surfer = np.flip(np.array(frb[field]),0)
surfer2 = np.flip(np.array(frb2[field2]),0)
surfer = surfer/surfer2

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
        s1.append(surfer[indice][j])
        s2.append(surfer[res[0]-indice-1][j])
    for i in range(indice, res[0] - indice):
        s3.append(surfer[i][indice])
        s4.append(surfer[i][res[0] - indice-1])

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
plt.xlabel('R [pc]')
plt.ylabel('H/R')
#plt.xlim(10**-2, 10)
#plt.savefig('plots1\perfil_radial_{}'.format(field))
#plt.show()



################################################################################################

ds = yt.load("E:/Gerald/TTbasico/output_00017/info_00017.txt")

field = 'sound_speed'
field2 = 'velocity_cylindrical_theta'
field2 = 'tangential_velocity'
units = 'pc'
#units = 'km/s'

ad = ds.all_data()
proj = ds.proj(field, "z", weight_field='density')
proj2 = ds.proj(field2, "z", weight_field='density')
width = (40, 'pc')
res = [30, 30]
frb = proj.to_frb(width, res)
frb2 = proj2.to_frb(width, res)
surfer = np.flip(np.array(frb[field]),0)
surfer2 = np.flip(np.array(frb2[field2]),0)
surfer = surfer/surfer2

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
        s1.append(surfer[indice][j])
        s2.append(surfer[res[0]-indice-1][j])
    for i in range(indice, res[0] - indice):
        s3.append(surfer[i][indice])
        s4.append(surfer[i][res[0] - indice-1])

    sum1 = np.sum(s1)
    sum2 = np.sum(s2)
    sum3 = np.sum(s3)
    sum4 = np.sum(s4)
    sum_tot = sum1 + sum2 + sum3 + sum4
    promedio = sum_tot/(len(s1) + len(s2) + len(s3) + len(s4))
    suma.append(promedio)

suma = np.flip(suma)
r = np.linspace(0,int(width[0]/2),len(suma))

#plt.clf()
plt.plot(r, suma, label= r'$t = 1.5 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
plt.ylabel('H/R', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.xlim(10**-2, 10)
#plt.savefig('plots1\perfil_radial_{}'.format(field))
#plt.show()




################################################################################################

ds = yt.load("E:/Gerald/TTbasico/output_00023/info_00023.txt")

field = 'sound_speed'
field2 = 'velocity_cylindrical_theta'
field2 = 'tangential_velocity'
units = 'pc'
#units = 'km/s'

ad = ds.all_data()
proj = ds.proj(field, "z", weight_field='density')
proj2 = ds.proj(field2, "z", weight_field='density')
width = (40, 'pc')
res = [30, 30]
frb = proj.to_frb(width, res)
frb2 = proj2.to_frb(width, res)
surfer = np.flip(np.array(frb[field]),0)
surfer2 = np.flip(np.array(frb2[field2]),0)
surfer = surfer/surfer2

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
        s1.append(surfer[indice][j])
        s2.append(surfer[res[0]-indice-1][j])
    for i in range(indice, res[0] - indice):
        s3.append(surfer[i][indice])
        s4.append(surfer[i][res[0] - indice-1])

    sum1 = np.sum(s1)
    sum2 = np.sum(s2)
    sum3 = np.sum(s3)
    sum4 = np.sum(s4)
    sum_tot = sum1 + sum2 + sum3 + sum4
    promedio = sum_tot/(len(s1) + len(s2) + len(s3) + len(s4))
    suma.append(promedio)

suma = np.flip(suma)
r = np.linspace(0,int(width[0]/2),len(suma))

#plt.clf()
plt.plot(r, suma, label= r'$t = 2 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
plt.ylabel('H/R', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.xlim(10**-2, 10)
plt.savefig('plots_finales\ H_R')
plt.show()




'''
def distancia_centro_pixel(i, j):
    x_pixel = np.abs(x_0 - i)
    y_pixel = np.abs(y_0 - j)
    d = np.sqrt(x_pixel**2 + y_pixel**2)
    return [d,i,j]

def distancia_centro(i, j):
    x_pixel = np.abs(x_0 - i)
    y_pixel = np.abs(y_0 - j)
    d = np.sqrt(x_pixel**2 + y_pixel**2)
    return d

distancias_pixel = []
distancias = []
for i in range(0,res[0]):
    for j in range(0, res[1]):
        d_pixel = distancia_centro_pixel(i,j)
        d = distancia_centro(i,j)
        distancias.append(d)
        distancias_pixel.append(d_pixel)


lista_misma_dist = []
aux = []
for i in range(0, len(distancias_pixel)):
    misma_dist = []
    for j in range(0, len(distancias_pixel)):
        if distancias_pixel[i][0] == distancias_pixel[j][0] and i!=j:
            #print(distancias_pixel[i], distancias_pixel[j])
            d = distancias_pixel[i][0]
            dist_pix1_i = distancias_pixel[i][1:][0]
            dist_pix1_j = distancias_pixel[i][1:][1]
            dist_pix2_i = distancias_pixel[j][1:][0]
            dist_pix2_j = distancias_pixel[j][1:][1]
            if not d in misma_dist:
                misma_dist.append(d)
                last = aux[len(aux)-1:]
                lista_misma_dist.append(last)
            misma_dist.append(dist_pix1_i)
            misma_dist.append(dist_pix1_j)
            misma_dist.append(dist_pix2_i)
            misma_dist.append(dist_pix2_j)
            #print(misma_dist)
            aux.append(misma_dist)
            print(len(lista_misma_dist))

print('lista_misma_dist=', lista_misma_dist)
lista_misma_dist = np.sort(lista_misma_dist)
print(lista_misma_dist)

lista_misma_dist_final = []
for i in range(1, len(lista_misma_dist)-1):
    if lista_misma_dist[i][0][0] != lista_misma_dist[i+1][0][0]:
        lista_misma_dist_final.append(lista_misma_dist[i])

print(lista_misma_dist_final)
lista_misma_dist = lista_misma_dist_final

print('vaLORES=',surfer[17][11], surfer[3][9], surfer[15][5])
print(718179, 928151, 905184)




final_list = []
for k in range(1,len(lista_misma_dist)):
    list = lista_misma_dist[k][0]
    ss=[]
    pares = []
    for i in range(1,len(list),2):
        par = [list[i], list[i+1]]
        if par not in pares:
            pares.append(par)
            ss.append(surfer[par[0]][par[1]])
        print(par)
        print(pares)
        #ss.append(surfer[list[i]][list[i+1]])
        print(ss, np.mean(ss))
    ss_prom = np.mean(ss)
    print(ss_prom)
    d = list[0]
    final_list.append(d)
    final_list.append(ss_prom)

print('final list=', final_list)

print('AAAAAA= ', surfer[6][6], surfer[6][4], surfer[1][0])

dist = []
prom = []
for i in range(0, len(final_list),2):
    dist.append(final_list[i])
for j in range(1, len(final_list),2):
    prom.append(final_list[j])

#prom = np.flip(prom)
r = np.linspace(0,int(width[0]/2),len(prom))

def pixel_to_width(width, res, pixels):
    output = pixels/res * width
    return output

radio = pixel_to_width(width[0], res[0], np.array(dist))

print('prom=',prom)
print('dist=', radio)

prom = np.abs(np.array(prom)/ np.array(r))

plt.clf()
plt.plot(radio, prom)
plt.xscale('log')
plt.yscale('log')
plt.show()
'''








































































