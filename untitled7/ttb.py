import yt
import numpy as np
import matplotlib.pyplot as plt
from yt.units import pc
from scipy.optimize import curve_fit, newton
from matplotlib import cm

G = 6.672e-8

ts = yt.load("E:/Gerald/TTbasico/output_?????/info_?????.txt")

filenames = ['E:/Gerald/TTbasico/output_00001/sink_00001.csv', 'E:/Gerald/TTbasico/output_00002/sink_00002.csv',
             'E:/Gerald/TTbasico/output_00003/sink_00003.csv', 'E:/Gerald/TTbasico/output_00004/sink_00004.csv',
             'E:/Gerald/TTbasico/output_00005/sink_00005.csv', 'E:/Gerald/TTbasico/output_00006/sink_00006.csv',
             'E:/Gerald/TTbasico/output_00007/sink_00007.csv', 'E:/Gerald/TTbasico/output_00008/sink_00008.csv',
             'E:/Gerald/TTbasico/output_00009/sink_00009.csv', 'E:/Gerald/TTbasico/output_00010/sink_00010.csv',
             'E:/Gerald/TTbasico/output_00011/sink_00011.csv', 'E:/Gerald/TTbasico/output_00012/sink_00012.csv',
             'E:/Gerald/TTbasico/output_00013/sink_00013.csv', 'E:/Gerald/TTbasico/output_00014/sink_00014.csv',
             'E:/Gerald/TTbasico/output_00015/sink_00015.csv', 'E:/Gerald/TTbasico/output_00016/sink_00016.csv',
             'E:/Gerald/TTbasico/output_00017/sink_00017.csv', 'E:/Gerald/TTbasico/output_00018/sink_00018.csv',
             'E:/Gerald/TTbasico/output_00019/sink_00019.csv',
             'E:/Gerald/TTbasico/output_00020/sink_00020.csv', 'E:/Gerald/TTbasico/output_00021/sink_00021.csv',
             'E:/Gerald/TTbasico/output_00022/sink_00022.csv', 'E:/Gerald/TTbasico/output_00023/sink_00023.csv',
             'E:/Gerald/TTbasico/output_00024/sink_00024.csv', 'E:/Gerald/TTbasico/output_00025/sink_00025.csv',
             'E:/Gerald/TTbasico/output_00026/sink_00026.csv', 'E:/Gerald/TTbasico/output_00027/sink_00027.csv',
             'E:/Gerald/TTbasico/output_00028/sink_00028.csv', 'E:/Gerald/TTbasico/output_00029/sink_00029.csv',
             'E:/Gerald/TTbasico/output_00030/sink_00030.csv']

data = []
for f in filenames:
    data.append(np.genfromtxt(f, delimiter=',', skip_header=2, unpack=True))

time = []

for ds in ts:
    time.append(ds.current_time.in_units('yr'))

t_orb = 105e3*np.ones(len(time))
print(time)

x1 = []
x2 = []
y1 = []
y2 = []
z1 = []
z2 = []
m1 = []
m2 = []
acc_rate_1 = []
acc_rate_2 = []


for sink in data:
    x1.append((sink[2][0]))
    x2.append((sink[2][1]))
    y1.append((sink[3][0]))
    y2.append((sink[3][1]))
    z1.append((sink[4][0]))
    z2.append((sink[4][1]))
    m1.append(sink[20][0])
    m2.append(sink[20][1])
    #acc_rate_1.append(sink[12][0])
    #acc_rate_2.append(sink[12][1])

x1 = np.array(x1)
x2 = np.array(x2)
y1 = np.array(y1)
y2 = np.array(y2)
z1 = np.array(z1)
z2 = np.array(z2)
m1 = np.array(m1)
m2 = np.array(m2)
time = np.array(time)

time_acc = time/10**3
plt.clf()
#print(m1,m2, time_acc)
plt.plot(time_acc, m1)
plt.plot(time_acc, m2)
plt.yscale('log')
plt.ylabel(r'$\dot{M} / \dot{M}_{edd}$', size= '15')
plt.xlabel(r'$t[kyr]$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.show()
plt.clf()

def a(x1, x2, y1, y2, z1, z2):

    a_bin = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    #r1 = np.sqrt(x1*x1 + y1*y1)
    #r2 = np.sqrt(x2*x2 + y2*y2)
    #a_bin = r1+r2
    return a_bin

a_binaria = a(x1, x2, y1, y2, z1, z2)
a_0 = a_binaria[0]
t_binaria = time

t_norm = t_binaria / t_orb
a_norm = a_binaria/ a_0

plt.plot(t_norm , a_norm)
plt.loglog()
plt.xlim(0,10**2)
plt.ylim(10**(-1), 10**0+0.1)
plt.ylabel(r'$a / a_0$', size= '15')
plt.xlabel(r'$t/t_{orb}$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.savefig('plots_finales\sep_binaria')
#plt.show()


L_0 = 3.828e26
c = 3e8
eps = 0.1
M_sun = 2e30
def eddington_rate(M_sink):
    edd = 3.2e4*M_sink/M_sun*L_0/(eps*c**2) * 3.15e7
    edd2 = M_sink/(45e6)
    return edd


def acc_rate(m1, m2):
    acc_rate_1 = []
    acc_rate_2 = []
    acc_rate = []
    for i in range(0, len(m1)-1):
        dif1 = m1[i+1] - m1[i]
        dif2 = m2[i+1] - m2[i]
        #print(dif2)
        acc_rate_1.append(dif1)
        acc_rate_2.append(dif2)
    acc_rate.append(acc_rate_1/time[1])
    acc_rate.append(acc_rate_2/time[1])
    return acc_rate

#print(acc_rate(m1,m2))
#print(m1,m2)


acc_rate_1 = np.array(acc_rate_1)
acc_rate_2 = np.array(acc_rate_2)

time_acc = time/10**3
time_acc = time_acc[:len(time)-1]

eddingtonrate1 = eddington_rate(m1)[:len(eddington_rate(m1))-1]
eddingtonrate2 = eddington_rate(m2)[:len(eddington_rate(m2))-1]


plt.clf()
plt.plot(time_acc, acc_rate(m1,m2)[0] / eddingtonrate1)
plt.plot(time_acc, acc_rate(m1,m2)[1] / eddingtonrate2)
plt.ylabel(r'$\dot{M} / \dot{M}_{edd}$', size= '15')
plt.xlabel(r'$t[kyr]$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.yscale('log')
#plt.savefig('plots_finales\ acc_rate')
#plt.show()
plt.clf()




ds = yt.load("E:/Gerald/TTbasico/output_00012/info_00012.txt")
field = 'cell_mass'
proj = ds.proj(field, "z", weight_field='cell_mass')
width = (50, 'pc')
res = [100, 100]
frb = proj.to_frb(width, res)
fn = frb.save_as_dataset(fields=[field])
ds2 = yt.load(fn)

sp = ds2.disk(ds.domain_center,[0,0,1] , 100*pc, 100*pc)
#print(sp.quantities.total_mass())

m_0 = sp.quantities.total_mass()[0]
#print(m_0)


def plummer_mass(a,r):
    output = m_0 * r**3 / (r**2 + a**2)**(3/2)
    return output

#print(plummer_mass(a_binaria[0]/2, a_binaria[0]/2))

def intersec(a,r):
    return plummer_mass(r, a) - m_0/2


r = a_binaria[9]/2
r_12 = newton(intersec, x0= 0.5, args=[r])
#print(r_12)
#print(plummer_mass(a_binaria[0]/2, r_12))
#print(m_0 / plummer_mass(a_binaria[0]/2, r_12))
#print(plummer_mass(m_0, a_binaria[0]/2, a_binaria[0]/2))
a_plummer = r_12
m_plummer = plummer_mass(r, a_plummer)
#print(m_plummer)

m1_g = m1[0]*1e6*M_sun*1e3
#print((m_plummer / (2*m1_g)))

f = 1.00058491

field = 'sound_speed'
proj_ss = ds.proj(field, "z", weight_field='density')
frb_ss = proj_ss.to_frb(width, res)
surfer_ss = np.flip(np.array(frb_ss[field].in_units('cm/s')),0)

field = 'velocity_cylindrical_theta'
field = 'tangential_velocity'
proj_v = ds.proj(field, "z", weight_field='density')
frb_v = proj_v.to_frb(width, res)
surfer_v = np.flip(np.array(frb_v[field].in_units('cm/s')),0)

field = 'cylindrical_radius'
proj_r = ds.proj(field, "z", weight_field='density')
frb_r = proj_r.to_frb(width, res)
surfer_r = np.flip(np.array(frb_r[field].in_units('cm')),0)

field = 'density'
proj_density = ds.proj(field, "z", weight_field=None)
frb_density = proj_density.to_frb(width, res)
surfer_density = np.flip(np.array(frb_density[field].in_units('g/cm**2')),0)

def distancia_centro(i, j):
    x_pixel = np.abs(x_0 - i)
    y_pixel = np.abs(y_0 - j)
    d = np.sqrt(x_pixel**2 + y_pixel**2)
    return d

x_0 = int(res[0]/2)
y_0 = int(res[1]/2)

d_final = []
for i in range(0, res[0]):
    d1 = []
    for j in range(0, res[1]):
        d = distancia_centro(i,j) * 3.086e+18
        d1.append(d)
    d_final.append(d1)

d_final = np.array(d_final)
d_final = d_final * width[0] / res[0]
#surfer_r = d_final

#omega = surfer_v / (np.pi * a_binaria[0]*3.086e+18)
omega = surfer_v / surfer_r
k = np.sqrt( omega**2 * (8*np.pi + 5*f - 9))
Q = surfer_ss * k / (np.pi * G * surfer_density)
#Q = surfer_ss / (np.pi * G * surfer_density)
#print(surfer_ss)
#print(surfer_v)
#print(surfer_r)
#print(surfer_density)
#print(ds.derived_field_list)
#print(surfer_v[50])
#print(surfer_r[50])
#print(omega[50])
#Q = omega

#print(surfer_r)
surfer = Q
#print(surfer)
ex =  (-width[0]/2, width[0]/2, -width[0]/2,width[0]/2)
norm = cm.colors.LogNorm()
plt.imshow(surfer, extent=ex, norm=norm)
plt.xlabel('$y\:[pc]$')
plt.ylabel('$z\:[pc]$')
plt.colorbar()
plt.savefig('prueba_projeccion Q')



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
plt.ylabel(r'$Q$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.savefig('plots1\perfil_radial2_{}'.format(field))
#plt.show()


####################################################################################
ds = yt.load("E:/Gerald/TTbasico/output_00017/info_00017.txt")

field = 'sound_speed'
proj_ss = ds.proj(field, "z", weight_field='density')
frb_ss = proj_ss.to_frb(width, res)
surfer_ss = np.flip(np.array(frb_ss[field].in_units('cm/s')),0)

field = 'velocity_cylindrical_theta'
field = 'tangential_velocity'
proj_v = ds.proj(field, "z", weight_field='density')
frb_v = proj_v.to_frb(width, res)
surfer_v = np.flip(np.array(frb_v[field].in_units('cm/s')),0)

field = 'cylindrical_radius'
proj_r = ds.proj(field, "z", weight_field='density')
frb_r = proj_r.to_frb(width, res)
surfer_r = np.flip(np.array(frb_r[field].in_units('cm')),0)

field = 'density'
proj_density = ds.proj(field, "z", weight_field=None)
frb_density = proj_density.to_frb(width, res)
surfer_density = np.flip(np.array(frb_density[field].in_units('g/cm**2')),0)

omega = surfer_v / surfer_r
k = np.sqrt( omega**2 * (8*np.pi + 5*f - 9))
Q = surfer_ss * k / (np.pi * G * surfer_density)
surfer = Q

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

#plt.clf()
plt.plot(r, suma, label= r'$t = 1.5 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
plt.ylabel(r'$Q$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.savefig('plots1\perfil_radial2_{}'.format(field))
#plt.show()


####################################################################################
ds = yt.load("E:/Gerald/TTbasico/output_00023/info_00023.txt")

field = 'sound_speed'
proj_ss = ds.proj(field, "z", weight_field='density')
frb_ss = proj_ss.to_frb(width, res)
surfer_ss = np.flip(np.array(frb_ss[field].in_units('cm/s')),0)

field = 'velocity_cylindrical_theta'
field = 'tangential_velocity'
proj_v = ds.proj(field, "z", weight_field='density')
frb_v = proj_v.to_frb(width, res)
surfer_v = np.flip(np.array(frb_v[field].in_units('cm/s')),0)

field = 'cylindrical_radius'
proj_r = ds.proj(field, "z", weight_field='density')
frb_r = proj_r.to_frb(width, res)
surfer_r = np.flip(np.array(frb_r[field].in_units('cm')),0)

field = 'density'
proj_density = ds.proj(field, "z", weight_field=None)
frb_density = proj_density.to_frb(width, res)
surfer_density = np.flip(np.array(frb_density[field].in_units('g/cm**2')),0)

omega = surfer_v / surfer_r
k = np.sqrt( omega**2 * (8*np.pi + 5*f - 9))
Q = surfer_ss * k / (np.pi * G * surfer_density)
surfer = Q

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

#plt.clf()
plt.plot(r, suma, label= r'$t = 2 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
plt.ylabel(r'$Q$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('plots_finales\ Q')
plt.show()


'''
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
print(suma)
r = np.linspace(0,int(width[0]/2),len(suma))

#suma = np.abs(suma)

plt.clf()
plt.plot(r, suma)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]')
plt.ylabel('{}'.format(field))
#plt.savefig('plots1\perfil_radial_{}'.format(field))
plt.show()
'''


