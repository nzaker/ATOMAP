import numpy as np
import atomap.api as am
import atomap.testing_tools as tt

#####
s = am.dummy_data.get_dumbbell_signal()
s.plot()
s._plot.signal_plot.figure.savefig('maketestdata/dumbbell.png')

#####
t1 = tt.MakeTestData(20, 20)
t1.add_atom(10, 10)
s1 = t1.signal
s1.plot()
s1._plot.signal_plot.figure.savefig('maketestdata/t1.png')

#####
t2 = tt.MakeTestData(200, 200)
x, y = np.mgrid[0:200:10j, 0:200:10j]
x, y = x.flatten(), y.flatten()
t2.add_atom_list(x, y)
s2 = t2.signal
s2.plot()
s2._plot.signal_plot.figure.savefig('maketestdata/t2.png')

#####
t3 = tt.MakeTestData(200, 200)
x, y = np.mgrid[0:200:20, 0:200:20]
x, y = x.flatten(), y.flatten()
t3.add_atom_list(x, y, sigma_x=2, sigma_y=1.5, amplitude=20, rotation=0.4)

x, y = np.mgrid[10:200:20, 10:200:20]
x, y = x.flatten(), y.flatten()
t3.add_atom_list(x, y, sigma_x=2, sigma_y=2, amplitude=40)
t3.add_image_noise(sigma=0.1)
s3 = t3.signal
s3.plot()
s3._plot.signal_plot.figure.savefig('maketestdata/t3.png')
