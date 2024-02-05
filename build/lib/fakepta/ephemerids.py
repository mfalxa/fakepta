import numpy as np
import enterprise.constants as const

class Ephemerids:

    def __init__(self):

        self.planets = {}
        # mass = planet mass [kg]
        # T = orbital period [days]
        # inc : inclination of orbit [deg]
        # Om : right ascension of ascending node [deg]
        # omega : argument of periapsis [deg]
        # a : semi major axis [AU]
        # l0 : mean longitude at epoch [deg]
        self.planets['mercury'] = {'mass':3.301*1e23, 'T':87.9691 , 'inc':7.00487, 'Om':48.33167, 'omega':77.45645, 'a':0.38709893, 'e':0.20563069, 'l0':252.25}
        self.planets['venus'] = {'mass':4.867*1e24, 'T':224.7, 'inc':3.39471, 'Om':76.68069, 'omega':131.53298, 'a':0.72333199, 'e':0.00677323, 'l0':181.98}
        self.planets['earth'] = {'mass':5.972*1e24, 'T':365.25636, 'inc':0.00005, 'Om':-11.26064, 'omega':102.94719, 'a':1.00000011, 'e':0.01671022, 'l0':100.47}
        self.planets['mars'] = {'mass':6.417*1e23, 'T':687.0, 'inc':1.85061, 'Om':49.57854, 'omega':336.04084, 'a':1.52366231, 'e':0.09341233, 'l0':355.43}
        self.planets['jupiter'] = {'mass':1.899*1e27, 'T':4331, 'inc':1.30530, 'Om':100.55615, 'omega':14.75385, 'a':5.20336301, 'e':0.04839266, 'l0':34.33}
        self.planets['saturn'] = {'mass':5.685*1e26, 'T':10747, 'inc':2.48446, 'Om':113.71504, 'omega':92.43194, 'a':9.53707032, 'e':0.05415060, 'l0':50.08}
        self.planets['uranus'] = {'mass':8.683*1e25, 'T':30589, 'inc':0.76986, 'Om':74.22988, 'omega':170.96424, 'a':19.19126393, 'e':0.04716771, 'l0':314.20}
        self.planets['neptune'] = {'mass':1.024*1e26, 'T':59800, 'inc':1.76917, 'Om':131.72169, 'omega':44.97135, 'a':30.06896348, 'e':0.00858587, 'l0':304.11}

        self.planet_names = [*self.planets]
        self.mass_ss = const.Msun + np.sum([self.planets[planet]['mass'] for planet in [*self.planets]])
        
    def do_rotation_pf_to_eq(self, vec, Om, omega, inc):

        inc += 23.4
        inc *= np.pi/180
        Om *= np.pi/180
        omega *= np.pi/180
        rot = np.array([[np.cos(Om)*np.cos(omega) - np.sin(Om)*np.cos(inc)*np.sin(omega), -np.cos(Om)*np.sin(omega) - np.sin(Om)*np.cos(inc)*np.cos(omega), np.sin(Om)*np.sin(inc)],
                        [np.sin(Om)*np.cos(omega) + np.cos(Om)*np.cos(inc)*np.sin(omega), -np.sin(Om)*np.sin(omega) + np.cos(Om)*np.cos(inc)*np.cos(omega), -np.cos(Om)*np.sin(inc)],
                        [np.sin(inc)*np.sin(omega), np.sin(inc)*np.cos(omega), np.cos(inc)]])
        
        return np.dot(rot, vec)

    def compute_orbit(self, times, T, Om, omega, inc, a, e, l0, mass=None):

        if a is None:
            a = (const.GMsun * (T*const.day)**2 / (4*np.pi**2))**(1/3) / const.c
        else:
            a *= const.AU / const.c

        # mean anomaly
        M = 2*np.pi / T * times / const.day + l0 * np.pi / 180

        # first order eccentric perturbation
        r = a * (1 - e * np.cos(M))

        x = r * np.cos(M)
        y = r * np.sin(M)
        z = np.zeros(len(times))

        # rotation to equatorial plane
        pos = np.vstack((x, y, z)).T
        pos_eq = np.zeros(np.shape(pos))
        for i, v0 in enumerate(pos):
            pos_eq[i] = self.do_rotation_pf_to_eq(v0, Om, omega, inc)

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

        mass = self.planets[planet]['mass']
        T = self.planets[planet]['T']
        Om = self.planets[planet]['Om']
        omega = self.planets[planet]['omega']
        inc = self.planets[planet]['inc']
        a = self.planets[planet]['a']
        e = self.planets[planet]['e']
        l0 = self.planets[planet]['l0']

        d_ssb = (mass + d_mass) * self.compute_orbit(toas, T, Om+d_Om, omega+d_omega, inc+d_inc, a+d_a, e+d_e, l0+d_l0) - mass * self.get_orbit_planet(toas, planet)
        d_ssb /= self.mass_ss

        dt_roemer = np.dot(d_ssb, psr_pos)

        return dt_roemer