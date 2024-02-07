import numpy as np
import enterprise.constants as const
from scipy.optimize import newton

class Ephemeris:

    def __init__(self):

        self.planets = {}
        #
        # orbital elements taken from : https://ssd.jpl.nasa.gov/planets/approx_pos.html
        # 
        # mass = planet mass [kg]
        # T = orbital period [days]
        # inc : inclination of orbit [deg, deg/century]
        # Om : right ascension of ascending node [deg, deg/century]
        # omega : argument of periapsis [deg, deg/century]
        # a : semi major axis [AU. AU/century]
        # l0 : mean longitude at epoch [deg, deg/century]
        #
        self.planets['mercury'] = {'mass':3.301*1e23, 'T':87.9691 , 'inc':[7.00497902, -0.00594749], 'Om':[48.33076593, -0.12534081], 'omega':[77.45779628, 0.16047689], 'a':[0.38709843, 0.00000037], 'e':[0.20563661, 0.00001906], 'l0':[252.25032350, 149472.67411175]}
        self.planets['venus'] = {'mass':4.867*1e24, 'T':224.7, 'inc':[3.39467605, -0.00078890], 'Om':[76.67984255, -0.27769418], 'omega':[131.60246718, 0.00268329], 'a':[0.72332102, 0.00000390], 'e':[0.00676399, -0.00004107], 'l0':[181.97909950, 58517.81538729]}
        self.planets['earth'] = {'mass':5.972*1e24, 'T':365.25636, 'inc':[-0.00001531, -0.01294668], 'Om':[0., 0.], 'omega':[102.93768193, 0.32327364], 'a':[1.00000018, 0.00000562], 'e':[0.01673163, -0.00004392], 'l0':[100.46457166, 35999.37244981]}
        self.planets['mars'] = {'mass':6.417*1e23, 'T':687.0, 'inc':[1.84969142, -0.00813131], 'Om':[49.55953891, -0.29257343], 'omega':[-23.94362959, 0.44441088], 'a':[1.52371243, 0.00001847], 'e':[0.09336511, 0.00007882], 'l0':[-4.55343205, 19140.30268499]}
        self.planets['jupiter'] = {'mass':1.899*1e27, 'T':4331, 'inc':[1.30439695, -0.00183714], 'Om':[100.47390909, 0.20469106], 'omega':[14.72847983, 0.21252668], 'a':[5.20248019, -0.00011607], 'e':[0.04853590, -0.00013253], 'l0':[34.39644051, 3034.74612775]}
        self.planets['saturn'] = {'mass':5.685*1e26, 'T':10747, 'inc':[2.48599187, 0.00193609], 'Om':[113.66242448, -0.28867794], 'omega':[92.59887831, -0.41897216], 'a':[9.54149883,-0.00125060 ], 'e':[0.05550825, -0.00050991], 'l0':[49.95424423, 1222.49362201]}
        self.planets['uranus'] = {'mass':8.683*1e25, 'T':30589, 'inc':[0.77263783, -0.00242939], 'Om':[74.01692503, 0.04240589], 'omega':[170.95427630, 0.40805281], 'a':[19.18797948, -0.00196176], 'e':[0.04685740, -0.00004397], 'l0':[313.23810451, 428.48202785]}
        self.planets['neptune'] = {'mass':1.024*1e26, 'T':59800, 'inc':[1.77004347, 0.00035372], 'Om':[131.78422574, -0.00508664], 'omega':[44.96476227, -0.32241464], 'a':[30.06952752, 0.00026291], 'e':[0.00895439, 0.00005105], 'l0':[-55.12002969, 218.45945325]}

        self.planet_names = [*self.planets]
        self.mass_ss = const.Msun + np.sum([self.planets[planet]['mass'] for planet in [*self.planets]])
        
    def do_rotation_op_to_eq(self, vec, Om, omega, inc):

        ec = 23.43 * np.pi/180
        inc *= np.pi/180
        Om *= np.pi/180
        omega *= np.pi/180
        rot = np.array([[np.cos(Om)*np.cos(omega) - np.sin(Om)*np.cos(inc)*np.sin(omega), -np.cos(Om)*np.sin(omega) - np.sin(Om)*np.cos(inc)*np.cos(omega), 0.],
                        [np.sin(Om)*np.cos(omega) + np.cos(Om)*np.cos(inc)*np.sin(omega), -np.sin(Om)*np.sin(omega) + np.cos(Om)*np.cos(inc)*np.cos(omega), 0.],
                        [np.sin(inc)*np.sin(omega), np.sin(inc)*np.cos(omega), 0.]])
        rot_ec = np.array([[1., 0., 0],
                            [0., np.cos(ec), -np.sin(ec)],
                            [0., np.sin(ec), np.cos(ec)]])

        return np.dot(rot_ec, np.dot(rot, vec))
    
    def mean_anomaly(self, times, T, l0):

        M = 2*np.pi / T * times / const.day + l0 * np.pi / 180

        return M
    
    def solve_kepler_equation(self, M, e):

        E = np.zeros(len(M))
        E[0] = newton(lambda x : M[0] - (x - e*np.sin(x)), M[0])
        for i in range(len(M)-1):
            E[i+1] = newton(lambda x : M[i+1] - (x - e*np.sin(x)), E[i])

        return E

    def compute_orbit(self, times, T, Om, omega, inc, a, e, l0, mass=None):

        # if a is None:
        #     a = (const.GMsun * (T*const.day)**2 / (4*np.pi**2))**(1/3) / const.c
        # else:
        #     a *= const.AU / const.c

        # redefine orbital elements at epoch t0
        t0 = (times[0] / 24 / 3600 - 2451545) / 36525
        Om = Om[0] + Om[1] * t0
        omega = omega[0] + omega[1] * t0
        inc = inc[0] + inc[1] * t0
        a = (a[0] + a[1] * t0) * const.AU / const.c
        e = e[0] + e[1] * t0
        l0 = l0[0] + l0[1] * t0

        # mean anomaly
        M = self.mean_anomaly(times, T, l0 - omega)
        M = np.mod(M, 2*np.pi)

        # solve kepler equation for eccentric anomaly
        E = self.solve_kepler_equation(M, e)

        # orbit trajectory in orbital plane
        x = a * np.cos(E - e)
        y = a * np.sqrt(1 - e**2) * np.sin(E)
        z = np.zeros(len(times))

        # rotation from orbital plane to equatorial plane
        pos = np.vstack((x, y, z)).T
        pos_eq = np.zeros(np.shape(pos))
        for i, v0 in enumerate(pos):
            pos_eq[i] = self.do_rotation_op_to_eq(v0, Om, omega - Om, inc)

        return pos_eq
    
    def get_orbit_planet(self, times, planet):

        return self.compute_orbit(times, **self.planets[planet])
    
    def get_planet_ssb(self, times):

        planetssb = np.empty((len(times), len(self.planet_names), 6))
        for i, planet in enumerate(['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']):
            planetssb[:, i, :3] = self.get_orbit_planet(times, planet)
        return planetssb

    def get_sunssb(self, times):

        sunssb = np.zeros((len(times), 3))
        for planet in [*self.planets]:
            sunssb -= self.planets[planet]['mass'] / const.Msun * self.get_orbit_planet(times, planet)

        return sunssb
    
    def add_planet(self, name, mass, T, inc, Om, omega, a, e, l0):

        self.planets[name] = {'mass':mass, 'T':T , 'inc':inc, 'Om':Om, 'omega':omega, 'a':a, 'e':e, 'l0':l0}
        self.mass_ss = const.Msun + np.sum([self.planets[planet]['mass'] for planet in [*self.planets]])
        self.planet_names = [*self.planets]

    def roemer_delay(self, toas, psr_pos, planet, d_mass=0., d_Om=0., d_omega=0., d_inc=0., d_a=0., d_e=0., d_l0=0.):

        # get parameters
        t0 = (toas[0] / 24 / 3600 - 2451545) / 36525
        mass = self.planets[planet]['mass']
        T = self.planets[planet]['T']
        Om = self.planets[planet]['Om'][0] + self.planets[planet]['Om'][1] * t0
        omega = self.planets[planet]['omega'][0] + self.planets[planet]['omega'][1] * t0
        inc = self.planets[planet]['inc'][0] + self.planets[planet]['inc'][1] * t0
        a = self.planets[planet]['a'][0] + self.planets[planet]['a'][1] * t0
        e = self.planets[planet]['e'][0] + self.planets[planet]['e'][1] * t0
        l0 = self.planets[planet]['l0'][0] + self.planets[planet]['l0'][1] * t0

        # compute deviation from SSB position estimate
        d_ssb = (mass + d_mass) * self.compute_orbit(toas, T, Om+d_Om, omega+d_omega, inc+d_inc, a+d_a, e+d_e, l0+d_l0) - mass * self.get_orbit_planet(toas, planet)
        d_ssb /= self.mass_ss

        # get Roemer delay
        dt_roemer = -np.dot(d_ssb, psr_pos)

        return dt_roemer