import yt
import numpy as np
from yt.units import pc, km, s
import matplotlib.pyplot as plt
from matplotlib import cm

f = 1.00058491
G = 6.67e-8

ts = yt.load("E:/Gerald/TTbasico/output_?????/info_?????.txt")
def filenames(ts, rad):
    filename = []
    if rad==True:
        for i in range(0, len(ts)+1):
            if i < 10:
                filename.append('E:/Gerald/TTbasico/output_0000{}/info_0000{}.txt'.format(i,i))
            else:
                filename.append('E:/Gerald/TTbasico/output_000{}/info_000{}.txt'.format(i,i))
        return filename

    if rad==False:
        for i in range(0, len(ts)+1):
            if i < 10:
                filename.append('E:/Gerald/TTbasico2/output_0000{}/info_0000{}.txt'.format(i,i))
            else:
                filename.append('E:/Gerald/TTbasico2/output_000{}/info_000{}.txt'.format(i,i))
        return filename

def field_surfer(field, ds, w_field, units):
    proj = ds.proj(field, "z", weight_field=w_field)
    frb = proj.to_frb(width, res)
    surfer = np.flip(np.array(frb[field].in_units(units)), 0)
    return surfer

def thickness(sound_speed, tan_vel):
    output = sound_speed / tan_vel
    return output

def toomre(sound_speed, tan_vel, radius, sigma):
    omega = tan_vel / radius
    k = np.sqrt(omega** 2 * (8 * np.pi + 5 * f - 9))
    Q = sound_speed * k / (np.pi * G * sigma)
    return Q

def integrate_field_square(surfer, res):
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
        print(indice)

    suma = np.flip(suma)
    r = np.linspace(0,int(width[0]/2),len(suma))
    return [r, suma]

def distancia_centro(i, j, res):
    x_0 = int(res[0] / 2)
    y_0 = int(res[1] / 2)
    x_pixel = np.abs(x_0 - i)
    y_pixel = np.abs(y_0 - j)
    d = np.sqrt(x_pixel**2 + y_pixel**2)
    d = np.around(d, -1)
    return [d,i,j]

def integrate_field_circle(surfer, res):
    hash_radio = {}
    radio = []
    for i in range(0, res[0]):
        for j in range(0, res[1]):
            r = distancia_centro(i, j, res)
            if r[0] not in radio:
                radio.append(r[0])
            if r[0] not in hash_radio:
                hash_radio[r[0]] = [r[0], [r[1], r[2]]]
            else:
                hash_radio[r[0]].append([r[1], r[2]])
        print(i)

    radio = np.sort(radio)
    radio_field2 = []
    arr_mean = []
    print(len(radio))
    for i in range(0, len(radio)):
        s1 = []
        for j in range(1, len(hash_radio[radio[i]])):
            index = hash_radio[radio[i]][j]
            value_field = surfer[index[0]][index[1]]
            s1.append(value_field)
        mean = np.mean(s1)
        arr_mean.append(mean)
        print(i)

    radio_field2.append(radio)
    radio_field2.append(arr_mean)
    r = radio_field2[0]
    r = r * width[0] / res[0]
    field = radio_field2[1]
    return [r, field]

def mapa_field(surfer, ex, norm, units, rad, name):
    plt.clf()
    plt.imshow(surfer, extent=ex, norm=norm)
    plt.xlabel('$x\:[pc]$')
    plt.ylabel('$y\:[pc]$')
    if units == None:
        plt.colorbar()

    if units != None:
        plt.colorbar().set_label(r'${}$'.format(units))
        plt.show()

    if rad == False:
        plt.savefig('plots_pruebas_sinrad\ mapa_sinrad_{}'.format(name))
        plt.show()

    if rad == True:
        plt.savefig('plots_pruebas_conrad\ mapa_conrad_{}'.format(name))






def plot_perfil_radial(surfers, ylabel, units, rad, name):
    plt.clf()
    t_orb = 1
    for surfer in surfers:
        if t_orb == 1:
            plt.plot(surfer[0], surfer[1], label=r'$t = 1 t_{orb}$')
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('R [pc]', size='15')
            # plt.ylabel(r'$\Sigma [g/cm^{2}]$', size='15')
            plt.ylabel(r'${} {}$'.format(ylabel, units), size='15')
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.gcf().subplots_adjust(left=0.15)

        if t_orb == 1.5:
            plt.plot(surfer[0], surfer[1], label=r'$t = 1.5 t_{orb}$')
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('R [pc]', size='15')
            # plt.ylabel(r'$\Sigma [g/cm^{2}]$', size='15')
            plt.ylabel(r'${} {}$'.format(ylabel, units), size='15')
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.gcf().subplots_adjust(left=0.15)

        if t_orb == 2:
            plt.plot(surfer[0], surfer[1], label=r'$t = 2 t_{orb}$')
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('R [pc]', size='15')
            # plt.ylabel(r'$\Sigma [g/cm^{2}]$', size='15')
            plt.ylabel(r'${} {}$'.format(ylabel, units), size='15')
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.gcf().subplots_adjust(left=0.15)
        t_orb += 0.5

    if rad == False:
        plt.savefig('plots_pruebas_sinrad\ perfil_sinrad_{}'.format(name))
    if rad == True:
        plt.savefig('plots_pruebas_conrad\ perfil_conrad_{}'.format(name))
    plt.show()

rad = False
file_names = filenames(ts, rad)

ds = yt.load(file_names[12])
ds2 = yt.load(file_names[17])
ds3 = yt.load(file_names[23])

width = (50, 'pc')
res = [3800, 3800]
norm = cm.colors.LogNorm()
ex =  (-width[0]/2, width[0]/2, -width[0]/2,width[0]/2)

'''
radius1 = field_surfer('cylindrical_radius', ds, 'density', 'cm')
radius2 = field_surfer('cylindrical_radius', ds2, 'density', 'cm')
radius3 = field_surfer('cylindrical_radius', ds3, 'density', 'cm')
sound_speed1 = field_surfer('sound_speed', ds, 'density', 'cm/s')
sound_speed2 = field_surfer('sound_speed', ds2, 'density', 'cm/s')
sound_speed3 = field_surfer('sound_speed', ds3, 'density', 'cm/s')
tan_vel1 = field_surfer('tangential_velocity', ds, 'density', 'cm/s')
tan_vel2 = field_surfer('tangential_velocity', ds2, 'density', 'cm/s')
tan_vel3 = field_surfer('tangential_velocity', ds3, 'density', 'cm/s')
sigma1 = field_surfer('density', ds, None, 'g/cm**2')
sigma2 = field_surfer('density', ds2, None, 'g/cm**2')
sigma3 = field_surfer('density', ds3, None, 'g/cm**2')
H1 = thickness(sound_speed1, tan_vel1)
H2 = thickness(sound_speed2, tan_vel2)
H3 = thickness(sound_speed3, tan_vel3)
Q1 = toomre(sound_speed1, tan_vel1, radius1, sigma1)
Q2 = toomre(sound_speed2, tan_vel2, radius2, sigma2)
Q3 = toomre(sound_speed3, tan_vel3, radius3, sigma3)

surfer = H1
surfer2 = H2
surfer3 = H3

circle_int1 = integrate_field_circle(surfer, res)
circle_int2 = integrate_field_circle(surfer2, res)
circle_int3 = integrate_field_circle(surfer3, res)

r = circle_int1[0]
suma = circle_int1[1]

r2 = circle_int2[0]
suma2 = circle_int2[1]

r3 = circle_int3[0]
suma3 = circle_int3[1]

plt.clf()
plt.plot(r, suma, label= r'$t = 1 t_{orb}$')
plt.plot(r2, suma2, label= r'$t = 1.5 t_{orb}$')
plt.plot(r3, suma3, label= r'$t = 2 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
#plt.ylabel(r'$\Sigma [g/cm^{2}]$', size='15')
plt.ylabel('H/R', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('plots_pruebas_sinrad\ H_R')
plt.show()

toomre1 = integrate_field_circle(Q1, res)
toomre2 = integrate_field_circle(Q2, res)
toomre3 = integrate_field_circle(Q3, res)

r = toomre1[0]
suma = toomre1[1]

r2 = toomre2[0]
suma2 = toomre2[1]

r3 = toomre3[0]
suma3 = toomre3[1]

plt.clf()
plt.plot(r, suma, label= r'$t = 1 t_{orb}$')
plt.plot(r2, suma2, label= r'$t = 1.5 t_{orb}$')
plt.plot(r3, suma3, label= r'$t = 2 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
#plt.ylabel(r'$\Sigma [g/cm^{2}]$', size='15')
plt.ylabel('Q', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('plots_pruebas_sinrad\ Q')
plt.show()

density1 = integrate_field_circle(sigma1, res)
density2 = integrate_field_circle(sigma2, res)
density3 = integrate_field_circle(sigma3, res)

r = density1[0]
suma = density1[1]

r2 = density2[0]
suma2 = density2[1]

r3 = density3[0]
suma3 = density3[1]

plt.clf()
plt.plot(r, suma, label= r'$t = 1 t_{orb}$')
plt.plot(r2, suma2, label= r'$t = 1.5 t_{orb}$')
plt.plot(r3, suma3, label= r'$t = 2 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
plt.ylabel(r'$\Sigma [g/cm^{2}]$', size='15')
#plt.ylabel('Q', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('plots_pruebas_sinrad\ Sigma')
plt.show()

ss1 = integrate_field_circle(sound_speed1, res)
ss2 = integrate_field_circle(sound_speed2, res)
ss3 = integrate_field_circle(sound_speed3, res)

r = ss1[0]
suma = ss1[1]

r2 = ss2[0]
suma2 = ss2[1]

r3 = ss3[0]
suma3 = ss3[1]

plt.clf()
plt.plot(r, suma, label= r'$t = 1 t_{orb}$')
plt.plot(r2, suma2, label= r'$t = 1.5 t_{orb}$')
plt.plot(r3, suma3, label= r'$t = 2 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
#plt.ylabel(r'$\Sigma [g/cm^{2}]$', size='15')
plt.ylabel(r'$c_s [cm/s]$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('plots_pruebas_sinrad\ c_s')
plt.show()

vel1 = integrate_field_circle(tan_vel1, res)
vel2 = integrate_field_circle(tan_vel2, res)
vel3 = integrate_field_circle(tan_vel3, res)

r = vel1[0]
suma = vel1[1]

r2 = vel2[0]
suma2 = vel2[1]

r3 = vel3[0]
suma3 = vel3[1]

plt.clf()
plt.plot(r, suma, label= r'$t = 1 t_{orb}$')
plt.plot(r2, suma2, label= r'$t = 1.5 t_{orb}$')
plt.plot(r3, suma3, label= r'$t = 2 t_{orb}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R [pc]', size='15')
#plt.ylabel(r'$\Sigma [g/cm^{2}]$', size='15')
plt.ylabel(r'$v [cm/s]$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('plots_pruebas_sinrad\ v')
plt.show()

radius1 = field_surfer('cylindrical_radius', ds, 'density', 'cm')
radius2 = field_surfer('cylindrical_radius', ds2, 'density', 'cm')
radius3 = field_surfer('cylindrical_radius', ds3, 'density', 'cm')
sound_speed1 = field_surfer('sound_speed', ds, 'density', 'cm/s')
sound_speed2 = field_surfer('sound_speed', ds2, 'density', 'cm/s')
sound_speed3 = field_surfer('sound_speed', ds3, 'density', 'cm/s')
tan_vel1 = field_surfer('tangential_velocity', ds, 'density', 'cm/s')
tan_vel2 = field_surfer('tangential_velocity', ds2, 'density', 'cm/s')
tan_vel3 = field_surfer('tangential_velocity', ds3, 'density', 'cm/s')
sigma1 = field_surfer('density', ds, None, 'g/cm**2')
sigma2 = field_surfer('density', ds2, None, 'g/cm**2')
sigma3 = field_surfer('density', ds3, None, 'g/cm**2')
H1 = thickness(sound_speed1, tan_vel1)
H2 = thickness(sound_speed2, tan_vel2)
H3 = thickness(sound_speed3, tan_vel3)
Q1 = toomre(sound_speed1, tan_vel1, radius1, sigma1)
Q2 = toomre(sound_speed2, tan_vel2, radius2, sigma2)
Q3 = toomre(sound_speed3, tan_vel3, radius3, sigma3)

ss1 = integrate_field_circle(sound_speed1, res)
ss2 = integrate_field_circle(sound_speed2, res)
ss3 = integrate_field_circle(sound_speed3, res)
toomre1 = integrate_field_circle(Q1, res)
toomre2 = integrate_field_circle(Q2, res)
toomre3 = integrate_field_circle(Q3, res)
height1 = integrate_field_circle(H1, res)
height2 = integrate_field_circle(H2, res)
height3 = integrate_field_circle(H3, res)
vel1 = integrate_field_circle(tan_vel1, res)
vel2 = integrate_field_circle(tan_vel2, res)
vel3 = integrate_field_circle(tan_vel3, res)
density1 = integrate_field_circle(sigma1, res)
density2 = integrate_field_circle(sigma2, res)
density3 = integrate_field_circle(sigma3, res)

ss = [ss1, ss2, ss3]
toomrex = [toomre1, toomre2, toomre3]
H = [height1, height2, height3]
v = [vel1, vel2, vel3]
sig = [density1, density2, density3]

plot_perfil_radial(ss, 'c_s', '[cm/s]', rad=rad, name='c_s')
plot_perfil_radial(toomrex, 'Q', '', rad=rad, name='Q')
plot_perfil_radial(H, 'H/R', '', rad=rad, name='H_R')
plot_perfil_radial(v, 'v', '[cm/s]', rad=rad, name='v')
plot_perfil_radial(sig, '\Sigma', '[g/cm^{2}]', rad=rad, name='Sigma')
'''

#mapa_field(sound_speed1, ex, norm, 'g/cm^{2}', rad=False, name='density_17')
#plot_perfil_radial(dens, '\Sigma', '[g/cm^{2}]', rad=True, name='density_17')


tan_vel1 = field_surfer('tangential_velocity', ds, 'density', 'cm/s')
radius1 = field_surfer('cylindrical_radius', ds, 'density', 'cm')
sigma1 = field_surfer('density', ds, None, 'g/cm**2')
sound_speed1 = field_surfer('sound_speed', ds, 'density', 'cm/s')
Q1 = toomre(sound_speed1, tan_vel1, radius1, sigma1)

plt.clf()
plt.imshow(Q1, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$Q$')
#plt.colorbar()
plt.text(width[0]/4.8, 8.7* width[0]/20, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
#plt.text(width[0]/4.8, 8.7* width[0]/20, 'Con radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_Q_1')
#plt.savefig('plots_pruebas_conrad\ mapa_Q_1')
plt.show()

'''
plt.clf()
plt.imshow(Q2, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
#plt.colorbar().set_label(r'$g/cm^{2}$')
plt.colorbar().set_label('Q')
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_Q_2')
#plt.savefig('plots_pruebas_conrad\ mapa_Q_2')

plt.clf()
plt.imshow(Q3, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
#plt.colorbar().set_label(r'$g/cm^{2}$')
plt.colorbar().set_label('Q')
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_Q_3')
#plt.savefig('plots_pruebas_conrad\ mapa_Q_3')


plt.clf()
plt.imshow(sigma1, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$\Sigma [g/cm^{2}]$')
#plt.colorbar()
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_sigma_1')
#plt.savefig('plots_pruebas_conrad\ mapa_sigma_1')

plt.clf()
plt.imshow(sigma2, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$\Sigma [g/cm^{2}]$')
#plt.colorbar()
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_sigma_2')
#plt.savefig('plots_pruebas_conrad\ mapa_sigma_2')

plt.clf()
plt.imshow(sigma3, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$\Sigma [g/cm^{2}]$')
#plt.colorbar()
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_sigma_3')
#plt.savefig('plots_pruebas_conrad\ mapa_sigma_3')


plt.clf()
plt.imshow(tan_vel1, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$ v [cm/s]$')
#plt.colorbar()
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_v_1')
#plt.savefig('plots_pruebas_conrad\ mapa_v_1')

plt.clf()
plt.imshow(tan_vel2, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$ v [cm/s]$')
#plt.colorbar()
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_v_2')
#plt.savefig('plots_pruebas_conrad\ mapa_v_2')

plt.clf()
plt.imshow(tan_vel3, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$ v [cm/s]$')
#plt.colorbar()
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_v_3')
#plt.savefig('plots_pruebas_conrad\ mapa_v_3')


plt.clf()
plt.imshow(sound_speed1, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$c_s [cm/s]$')
#plt.colorbar()
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_ss_1')
#plt.savefig('plots_pruebas_conrad\ mapa_ss_1')

plt.clf()
plt.imshow(sound_speed2, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$c_s [cm/s]$')
#plt.colorbar()
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_ss_2')
#plt.savefig('plots_pruebas_conrad\ mapa_ss_2')

plt.clf()
plt.imshow(sound_speed3, extent=ex, norm=norm)
plt.xlabel('$x\:[pc]$')
plt.ylabel('$y\:[pc]$')
plt.colorbar().set_label(r'$c_s [cm/s]$')
#plt.colorbar()
plt.text(11, 22, 'Sin radiación', bbox={'facecolor': 'white', 'pad': 5})
plt.savefig('plots_pruebas_sinrad\ mapa_ss_3')
#plt.savefig('plots_pruebas_conrad\ mapa_ss_3')
'''

