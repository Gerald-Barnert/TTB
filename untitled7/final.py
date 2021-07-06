import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


ts_conrad = yt.load("E:/Gerald/TTbasico/output_?????/info_?????.txt")
ts_sinrad = yt.load("E:/Gerald/TTbasico2/output_?????/info_?????.txt")

def filenames(ts, rad):
    filename = []
    if rad==True:
        for i in range(1, len(ts)+1):
            if i < 10:
                filename.append('E:/Gerald/TTbasico/output_0000{}/info_0000{}.txt'.format(i,i))
            else:
                filename.append('E:/Gerald/TTbasico/output_000{}/info_000{}.txt'.format(i,i))
        return filename

    if rad==False:
        for i in range(1, len(ts)+1):
            if i < 10:
                filename.append('E:/Gerald/TTbasico2/output_0000{}/info_0000{}.txt'.format(i,i))
            else:
                filename.append('E:/Gerald/TTbasico2/output_000{}/info_000{}.txt'.format(i,i))
        return filename

def filenames_sink(ts, rad):
    filename = []
    if rad==True:
        for i in range(1, len(ts)+1):
            if i < 10:
                filename.append('E:/Gerald/TTbasico/output_0000{}/sink_0000{}.csv'.format(i,i))
            else:
                filename.append('E:/Gerald/TTbasico/output_000{}/sink_000{}.csv'.format(i,i))
        return filename

    if rad==False:
        for i in range(1, len(ts)+1):
            if i < 10:
                filename.append('E:/Gerald/TTbasico2/output_0000{}/sink_0000{}.csv'.format(i,i))
            else:
                filename.append('E:/Gerald/TTbasico2/output_000{}/sink_000{}.csv'.format(i,i))
        return filename

time_conrad = []
time_sinrad = []

for ds in ts_conrad:
    time_conrad.append(ds.current_time.in_units('yr'))

for ds in ts_sinrad:
    time_sinrad.append(ds.current_time.in_units('yr'))


data_conrad = []
data_sinrad = []

filename_conrad = filenames_sink(ts_conrad, rad=True)
filename_sinrad = filenames_sink(ts_sinrad, rad=False)

for f in filename_conrad:
    data_conrad.append(np.genfromtxt(f, delimiter=',', skip_header=2, unpack=True))

for f in filename_sinrad:
    data_sinrad.append(np.genfromtxt(f, delimiter=',', skip_header=2, unpack=True))

t_orb = 105e3*np.ones(len(time_conrad))

def a(x1, x2, y1, y2, z1, z2):
    a_bin = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    return a_bin

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

L_0 = 3.828e26
c = 3e8
eps = 0.1
M_sun = 2e30
def eddington_rate(M_sink):
    edd = 3.2e4*M_sink/M_sun*L_0/(eps*c**2) * 3.15e7
    edd2 = M_sink/(45e6)
    return edd

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

for sink in data_conrad:
    x1.append((sink[2][0]))
    x2.append((sink[2][1]))
    y1.append((sink[3][0]))
    y2.append((sink[3][1]))
    z1.append((sink[4][0]))
    z2.append((sink[4][1]))
    m1.append(sink[20][0])
    m2.append(sink[20][1])

x1 = np.array(x1)
x2 = np.array(x2)
y1 = np.array(y1)
y2 = np.array(y2)
z1 = np.array(z1)
z2 = np.array(z2)
m1 = np.array(m1)
m2 = np.array(m2)
time = np.array(time_conrad)

a_binaria = a(x1, x2, y1, y2, z1, z2)
a_0 = a_binaria[0]
print(a_binaria)
t_binaria = time

t_norm = t_binaria / t_orb
a_norm = a_binaria/ a_0

#print(m1 / m2[0])
#print(m2 / m2[0])

m1_norm = m1 / m2[0]
m2_norm = m2 / m2[0]

#print(m1_norm)
#print(m2_norm)
#print(t_norm)

plt.clf()
plt.plot(t_norm[5:], m1_norm[5:] )
plt.plot(t_norm[5:], m2_norm[5:] )
plt.ylabel(r'$M_{BH} / M_0$', size= '15')
plt.xlabel(r'$t/t_{orb}$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.yticks(np.arange(0.9994, 1.0012, step=0.0006))
#plt.yticks([1.0001, np.max(m1_norm)])
#plt.ticklabel_format(axis="y", style='sci', scilimits=(-4,-4))
#plt.yscale('log')
plt.legend(title= 'Con radiación')
#plt.savefig('plots_finales\ acc_rate_conrad')
#plt.show()
plt.clf()

#plt.plot(t_norm[5:] , a_norm[5:], label='Con radiación')

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

for sink in data_sinrad:
    x1.append((sink[2][0]))
    x2.append((sink[2][1]))
    y1.append((sink[3][0]))
    y2.append((sink[3][1]))
    z1.append((sink[4][0]))
    z2.append((sink[4][1]))
    m1.append(sink[20][0])
    m2.append(sink[20][1])

x1 = np.array(x1)
x2 = np.array(x2)
y1 = np.array(y1)
y2 = np.array(y2)
z1 = np.array(z1)
z2 = np.array(z2)
m1 = np.array(m1)
m2 = np.array(m2)
time = np.array(time_sinrad)

a_binaria = a(x1, x2, y1, y2, z1, z2)
a_0 = a_binaria[0]
print(a_binaria)
t_binaria = time

t_norm = t_binaria / t_orb
a_norm = a_binaria/ a_0

plt.plot(t_norm[5:] , a_norm[5:], label='Sin radiación')
plt.legend()
plt.loglog()
plt.xlim(0,10)
plt.ylim(10**(-1), 10**0+0.1)
plt.ylabel(r'$a / a_0$', size= '15')
plt.xlabel(r'$t/t_{orb}$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.savefig('plots_finales\sep_binaria_final')
plt.show()
plt.clf()


acc_rate_1 = np.array(acc_rate_1)
acc_rate_2 = np.array(acc_rate_2)

time_acc = time/10**3
time_acc = time_acc[:len(time)-1]

eddingtonrate1 = eddington_rate(m1)[:len(eddington_rate(m1))-1]
eddingtonrate2 = eddington_rate(m2)[:len(eddington_rate(m2))-1]

plt.clf()
#plt.plot(time_acc[5:], acc_rate(m1,m2)[0][5:] / eddingtonrate1[5:])
#plt.plot(time_acc[5:], acc_rate(m1,m2)[1][5:] / eddingtonrate2[5:])
plt.plot(t_norm[5:], m1[5:] / m1[0])
plt.plot(t_norm[5:], m2[5:]/ m2[0])
plt.ylabel(r'$M_{BH} / M_0$', size= '15')
plt.xlabel(r'$t/t_{orb}$', size='15')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
#plt.yscale('log')
plt.legend(title= 'Sin radiación')
#plt.savefig('plots_finales\ acc_rate_sinrad')
#plt.show()
plt.clf()




