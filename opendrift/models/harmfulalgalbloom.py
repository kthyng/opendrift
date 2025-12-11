# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2025, Kristen Thyng, TetraTech


import numpy as np
import logging; logger = logging.getLogger(__name__)
from opendrift.models.oceandrift import Lagrangian3DArray, OceanDrift
from opendrift.config import CONFIG_LEVEL_ESSENTIAL, CONFIG_LEVEL_BASIC, CONFIG_LEVEL_ADVANCED


class HarmfulAlgalBloomElement(Lagrangian3DArray):
    """
    Extending Lagrangian3DArray with specific properties for modeling harmful algal blooms.
    """

    variables = Lagrangian3DArray.add_variables([
        # ('diameter', {'dtype': np.float32,
        #               'units': 'm',
        #               'default': 0.0014}),  # for NEA Cod
        # ('neutral_buoyancy_salinity', {'dtype': np.float32,
        #                                'units': 'PSU',
        #                                'default': 31.25}),  # for NEA Cod
        # ('stage_fraction', {'dtype': np.float32,  # to track percentage of development time completed
        #                     'units': '',
        #                     'default': 0.}),
        # ('hatched', {'dtype': np.uint8,  # 0 for eggs, 1 for larvae
        #              'units': '',
        #              'default': 0}),
        # ('length', {'dtype': np.float32,
        #             'units': 'mm',
        #             'default': 0}),
        # ('weight', {'dtype': np.float32,
        #             'units': 'mg',
        #             'default': 0.08}),
        # ('survival', {'dtype': np.float32,  # Not yet used
        #               'units': '',
        #               'default': 1.})
        ('dead',  {'dtype': np.float32,
                     'units': '',
                     'default': 0.}),
        # This is for calculating relative biomass changes and when a particle dies
        ('biomass',  {'dtype': np.float32,
                     'units': '',
                     'default': 1.}),])


class HarmfulAlgalBloom(OceanDrift):
    """Buoyant particle trajectory model based on the OpenDrift framework.

        Developed at MET Norway

        Generic module for particles that are subject to vertical turbulent
        mixing with the possibility for positive or negative buoyancy

        Particles could be e.g. oil droplets, plankton, or sediments

    """

    ElementType = HarmfulAlgalBloomElement

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_height': {'fallback': 0},
        'sea_surface_wave_significant_height': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 100},
        'ocean_vertical_diffusivity': {'fallback': 0.01, 'profiles': True},
        'ocean_mixed_layer_thickness': {'fallback': 50},
        'sea_water_temperature': {'fallback': 10, 'profiles': True},
        'sea_water_salinity': {'fallback': 34, 'profiles': True},
        'sea_surface_wave_stokes_drift_x_velocity': {'fallback': 0},
        'sea_surface_wave_stokes_drift_y_velocity': {'fallback': 0},
        # 'x_sea_water_velocity': {'fallback': 0},
        # 'y_sea_water_velocity': {'fallback': 0},
        # 'sea_surface_height': {'fallback': 0},
        # 'sea_surface_wave_significant_height': {'fallback': 0},
        # 'x_wind': {'fallback': 0},
        # 'y_wind': {'fallback': 0},
        # 'land_binary_mask': {'fallback': None},
        # 'sea_floor_depth_below_sea_level': {'fallback': 100},
        # 'ocean_vertical_diffusivity': {'fallback': 0.01, 'profiles': True},
        # 'ocean_mixed_layer_thickness': {'fallback': 50},
        # 'sea_water_temperature': {'fallback': 10, 'profiles': True},
        # 'sea_water_salinity': {'fallback': 34, 'profiles': True},
        # 'sea_surface_wave_stokes_drift_x_velocity': {'fallback': 0},
        # 'sea_surface_wave_stokes_drift_y_velocity': {'fallback': 0},
    }


    def __init__(self, *args, **kwargs):

        # Calling general constructor of parent class
        super(HarmfulAlgalBloom, self).__init__(*args, **kwargs)

        # IBM configuration options
        self._add_config({
            # 'IBM:fraction_of_timestep_swimming':
            #     {'type': 'float', 'default': 0.15,
            #      'min': 0.0, 'max': 1.0, 'units': 'fraction',
            #      'description': 'Fraction of timestep swimming',
            #      'level': CONFIG_LEVEL_ADVANCED},
            'hab:temperature_pref_min': {'type':'float', 'default':10.,
                                'min': -5., 'max': 40., 'units': 'degrees',
                                'description': 'Minimum temperature for preferred temperature range; cells have regular growth.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:temperature_pref_max': {'type':'float', 'default':20.,
                                'min': -5., 'max': 40., 'units': 'degrees',
                                'description': 'Maximum temperature for preferred temperature range; cells have regular growth.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:temperature_death_min': {'type':'float', 'default':0.,
                                'min': -5., 'max': 40., 'units': 'degrees',
                                'description': 'Minimum temperature for living. Below this temperature, cells have high mortality rate. Between this and temperature_viable_min, cells have no growth.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:temperature_death_max': {'type':'float', 'default':40.,
                                'min': -5., 'max': 40., 'units': 'degrees',
                                'description': 'Maximum temperature for living. Above this temperature, cells have high mortality rate. Between temperature_viable_max and this parameter, cells have no growth.',
                                'level': CONFIG_LEVEL_BASIC},            

            'hab:salinity_pref_min': {'type':'float', 'default':30.,
                                'min': 0., 'max': 50., 'units': 'psu',
                                'description': 'Minimum salinity for preferred salinity range; cells have regular growth.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:salinity_pref_max': {'type':'float', 'default':40.,
                                'min': 0., 'max': 50., 'units': 'psu',
                                'description': 'Maximum salinity for preferred salinity range; cells have regular growth.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:salinity_death_min': {'type':'float', 'default':20.,
                                'min': 0., 'max': 50., 'units': 'psu',
                                'description': 'Minimum salinity for living. Below this salinity, cells have high mortality rate. Between this and salinity_viable_min, cells have no growth.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:salinity_death_max': {'type':'float', 'default':50.,
                                'min': 0., 'max': 50., 'units': 'psu',
                                'description': 'Maximum salinity for living. Above this salinity, cells have high mortality rate. Between salinity_viable_max and this parameter, cells have no growth.',
                                'level': CONFIG_LEVEL_BASIC},

            'hab:mortality_rate_high': {'type':'float', 'default':1.,
                                'min': 0., 'max': 10., 'units': 'days^-1',
                                'description': 'Rate of mortality applied below outside viable conditions defined by hab:temperature_death_min, hab:temperature_death_max, hab:salinity_death_min, and hab:salinity_death_max.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:mortality_rate_medium': {'type':'float', 'default':0.5,
                                'min': 0., 'max': 10., 'units': 'days^-1',
                                'description': 'Rate of mortality applied in edge conditions defined by hab:temperature_death_min, hab:temperature_death_max, hab:salinity_death_min, hab:salinity_death_max, hab:temperature_pref_min, hab:temperature_pref_max, hab:salinity_pref_min, and hab:salinity_pref_max.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:mortality_rate_low': {'type':'float', 'default':0.1,
                                'min': 0., 'max': 10., 'units': 'days^-1',
                                'description': 'Rate of mortality applied in preferred conditions defined by hab:temperature_pref_min, hab:temperature_pref_max, hab:salinity_pref_min, and hab:salinity_pref_max.',
                                'level': CONFIG_LEVEL_BASIC},

            'hab:growth_rate_high': {'type':'float', 'default':0.7,
                                'min': 0., 'max': 10., 'units': 'days^-1',
                                'description': 'Rate of growth applied in preferred conditions defined by hab:temperature_pref_min, hab:temperature_pref_max, hab:salinity_pref_min, and hab:salinity_pref_max.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:growth_rate_medium': {'type':'float', 'default':0.2,
                                'min': 0., 'max': 10., 'units': 'days^-1',
                                'description': 'Rate of growth applied in edge conditions defined by hab:temperature_death_min, hab:temperature_death_max, hab:salinity_death_min, hab:salinity_death_max, hab:temperature_pref_min, hab:temperature_pref_max, hab:salinity_pref_min, and hab:salinity_pref_max.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:growth_rate_low': {'type':'float', 'default':0.,
                                'min': 0., 'max': 10., 'units': 'days^-1',
                                'description': 'Rate of growth applied below viable conditions defined by hab:temperature_death_min, hab:temperature_death_max, hab:salinity_death_min, and hab:salinity_death_max.',
                                'level': CONFIG_LEVEL_BASIC},
            
            'hab:biomass_dead_threshold': {'type':'float', 'default':1e-3,
                                'min': 0., 'max': 1., 'units': 'fraction',
                                'description': 'Threshold of biomass below which a particle is considered dead and deactivated.',
                                'level': CONFIG_LEVEL_ADVANCED},

            'hab:vertical_behavior': {'type': 'enum',
                                'enum': ['none', 'band', 'diel_band'],
                                'default': 'none',
                                'description': 'Vertical behavior: no active movement, fixed band, or diel band migration.',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:swim_speed': {'type': 'float',
                                'default': 0.001,  # 1 mm/s
                                'units': 'm/s',
                                'min': 0.0,
                                'max': 100,
                                'description': 'Maximum active vertical swimming speed (m/s).',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:band_center_depth': {'type': 'float',
                                'default': -10.0,
                                'units': 'm',
                                'min': -10000, 'max': 0.0,
                                'description': 'Target center of preferred depth band (m, negative down).',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:band_half_width': {'type': 'float',
                                'default': 5.0,
                                'units': 'm',
                                'min': 0.0, 'max': 5000,
                                'description': 'Half-width of preferred depth band (m).',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:diel_day_depth': {'type': 'float',
                                'default': -20.0,
                                'units': 'm',
                                'min': -10000, 'max': 0.0,
                                'description': 'Target depth during daytime (m, negative).',
                                'level': CONFIG_LEVEL_BASIC},
            'hab:diel_night_depth': {'type': 'float',
                                'default': -5.0,
                                'units': 'm',
                                'min': -10000, 'max': 0.0,
                                'description': 'Target depth during nighttime (m, negative).',
                                'level': CONFIG_LEVEL_BASIC},

        }
            )

        # Are these on or off when uncommented?
        # self._set_config_default('drift:vertical_mixing', True)
        # self._set_config_default('drift:vertical_mixing_at_surface', True)
        # self._set_config_default('drift:vertical_advection_at_surface', True)

    # ---------------------------------------------------------------------
    # Time helpers: UTC -> local solar hour
    # ---------------------------------------------------------------------

    def _current_utc_hour(self):
        """Current model time in UTC hours [0, 24)."""
        t = self.time  # datetime in UTC
        return t.hour + t.minute / 60.0 + t.second / 3600.0


    def _local_solar_hour(self, lon_deg):
        """
        Convert UTC model time to local solar time for given longitude(s).
        lon_deg: degrees East (scalar or array).
        Returns hours in [0, 24).
        """
        utc_hour = self._current_utc_hour()
        # 15° longitude = 1 hour; positive lon east of Greenwich.
        local = utc_hour + lon_deg / 15.0
        # Wrap into [0, 24)
        return np.mod(local, 24.0)

    # ---------------------------------------------------------------------
    # Fast sunrise/sunset approximation based on simulation date
    # ---------------------------------------------------------------------

    def _approx_sunrise_sunset(self, lat_deg):
        """
        Fast approximation of sunrise/sunset for the current simulation date.

        lat_deg: latitude in degrees (scalar or array).
        Uses day-of-year from self.time (UTC), which corresponds to the
        simulation date.
        Returns (sunrise, sunset) in local solar hours [0, 24).
        """
        # Ensure arrays for vectorized operations
        lat = np.asarray(lat_deg, dtype=float)

        # Day of year from the simulation clock (UTC)
        N = self.time.timetuple().tm_yday

        # Declination (degrees)
        decl_deg = 23.44 * np.cos(2.0 * np.pi * (N + 10.0) / 365.0)
        decl_rad = np.radians(decl_deg)
        lat_rad  = np.radians(lat)

        # Hour angle at sunrise/sunset
        cosH = -np.tan(lat_rad) * np.tan(decl_rad)
        cosH = np.clip(cosH, -1.0, 1.0)   # handle polar extremes
        H = np.arccos(cosH)               # radians

        # Convert to hours: 15° per hour, H in radians
        H_deg = np.degrees(H)
        daylight_hours = 2.0 * H_deg / 15.0

        sunrise = 12.0 - daylight_hours / 2.0
        sunset  = 12.0 + daylight_hours / 2.0

        return sunrise, sunset   # same shape as lat

    # ---------------------------------------------------------------------
    # Diel behavior using date + UTC + longitude
    # ---------------------------------------------------------------------

    def _target_depth_diel(self, z):
        """
        Target depth for diel_band behavior.

        - Uses self.time (UTC) for the simulation date.
        - Converts UTC to local solar time via longitude.
        - Computes sunrise/sunset from latitude and date.
        """
        lat = np.asarray(self.elements.lat, dtype=float)
        lon = np.asarray(self.elements.lon, dtype=float)

        # Local solar hour for each particle
        local_hour = self._local_solar_hour(lon)  # same shape as lon

        # Sunrise/sunset (local solar time) for each particle based on lat + date
        sunrise, sunset = self._approx_sunrise_sunset(lat)

        # Daytime mask
        is_day = (local_hour >= sunrise) & (local_hour < sunset)

        day_z = self.get_config('hab:diel_day_depth')
        night_z = self.get_config('hab:diel_night_depth')

        # Choose depth per particle
        target = np.where(is_day, day_z, night_z).astype(z.dtype)

        return target

    # ---------------------------------------------------------------------
    # Band behavior and dispatcher as before
    # ---------------------------------------------------------------------

    def _target_depth_band(self, z):
        z0 = self.get_config('hab:band_center_depth')
        half_w = self.get_config('hab:band_half_width')

        band_min = z0 - half_w
        band_max = z0 + half_w

        target = np.where(
            (z >= band_min) & (z <= band_max),
            z0,
            np.where(z < band_min, band_min, band_max),
        )
        return target


    def _apply_vertical_behavior(self):
        behavior = self.get_config('hab:vertical_behavior')
        if behavior == 'none':
            return

        z = self.elements.z.copy()
        dt = self.time_step.total_seconds()
        swim_speed = self.get_config('hab:swim_speed')

        if swim_speed <= 0.0 or dt <= 0.0:
            return

        if behavior == 'band':
            target_z = self._target_depth_band(z)
        elif behavior == 'diel_band':
            target_z = self._target_depth_diel(z)
        else:
            return

        dz = target_z - z
        max_step = swim_speed * dt
        dz_step = np.clip(dz, -max_step, max_step)

        self.elements.z += dz_step

        # # Clamp to seafloor and surface if desired
        # if hasattr(self.elements, 'sea_floor_depth_below_sea_level'):
        #     bottom = -self.elements.sea_floor_depth_below_sea_level
        #     self.elements.z = np.maximum(self.elements.z, bottom)
        # self.elements.z = np.minimum(self.elements.z, 0.0)

    def classify_zone(self, value, death_min_key, death_max_key, pref_min_key, pref_max_key):
        """Classify environmental suitability zone for a scalar field (T or S).
        
        value: array of scalar values (e.g., temperature or salinity)

        Returns: array of strings: 'lethal', 'tolerable', 'preferred'
        """
        death_min = self.get_config(death_min_key)
        death_max = self.get_config(death_max_key)
        pref_min = self.get_config(pref_min_key)
        pref_max = self.get_config(pref_max_key)

        zones = np.empty_like(value, dtype='<U10')

        # Lethal
        inds = (value < death_min) | (value > death_max)
        zones[inds] = 'lethal'

        # Tolerable (between death and preferred on either side)
        inds = ((value < pref_min) & (value > death_min)) | \
            ((value > pref_max) & (value < death_max))
        zones[inds] = 'tolerable'

        # Preferred
        inds = (value >= pref_min) & (value <= pref_max)
        zones[inds] = 'preferred'

        return zones
        
    def calculate_mortality_rates(self, temperature, salinity):
        temperature_zones = self.classify_zone(
            temperature,
            'hab:temperature_death_min',
            'hab:temperature_death_max',
            'hab:temperature_pref_min',
            'hab:temperature_pref_max'
        )
        salinity_zones = self.classify_zone(
            salinity,
            'hab:salinity_death_min',
            'hab:salinity_death_max',
            'hab:salinity_pref_min',
            'hab:salinity_pref_max'
        )

        mortality_rates = np.zeros_like(temperature, dtype=np.float32)

        lethal = (temperature_zones == 'lethal') | (salinity_zones == 'lethal')
        mortality_rates[lethal] = self.get_config('hab:mortality_rate_high')

        tolerable = ((temperature_zones == 'tolerable') & (salinity_zones != 'lethal')) | \
                    ((salinity_zones == 'tolerable') & (temperature_zones != 'lethal'))
        mortality_rates[tolerable] = self.get_config('hab:mortality_rate_medium')

        preferred = (temperature_zones == 'preferred') & (salinity_zones == 'preferred')
        mortality_rates[preferred] = self.get_config('hab:mortality_rate_low')

        return mortality_rates
    
    def calculate_growth_rates(self, temperature, salinity):
        temperature_zones = self.classify_zone(
            temperature,
            'hab:temperature_death_min',
            'hab:temperature_death_max',
            'hab:temperature_pref_min',
            'hab:temperature_pref_max'
        )
        salinity_zones = self.classify_zone(
            salinity,
            'hab:salinity_death_min',
            'hab:salinity_death_max',
            'hab:salinity_pref_min',
            'hab:salinity_pref_max'
        )

        growth_rates = np.zeros_like(temperature, dtype=np.float32)

        lethal = (temperature_zones == 'lethal') | (salinity_zones == 'lethal')
        growth_rates[lethal] = self.get_config('hab:growth_rate_low')  # usually 0

        tolerable = ((temperature_zones == 'tolerable') & (salinity_zones != 'lethal')) | \
                    ((salinity_zones == 'tolerable') & (temperature_zones != 'lethal'))
        growth_rates[tolerable] = self.get_config('hab:growth_rate_medium')

        preferred = (temperature_zones == 'preferred') & (salinity_zones == 'preferred')
        growth_rates[preferred] = self.get_config('hab:growth_rate_high')

        return growth_rates
        
    def calculate_change_in_biomass(self):
        """Calculate change in biomass based on temperature and salinity.
        
        dB/dt = (growth_rate(T) - mortality_rate(T))*B
    
        both growth_rate and mortality_rate depend on temperature only here.
        B is the biomass.
        """
        
        temp = self.environment.sea_water_temperature
        sal = self.environment.sea_water_salinity
        
        # Growth
        growth_rates = self.calculate_growth_rates(temp, sal)
        
        # Mortality
        mortality_rates = self.calculate_mortality_rates(temp, sal)

        net_rates = growth_rates - mortality_rates

        # Update biomass
        self.elements.biomass *= np.exp(net_rates * (self.time_step.total_seconds() / 86400))  # Convert rates from per day to per timestep  

    def kill_elements(self):
        """Deactivate elements with biomass below threshold."""
        low_biomass = np.where(self.elements.biomass < self.get_config('hab:biomass_dead_threshold'))[0]
        if len(low_biomass) > 0:
            logger.debug('Deactivating %s elements due to low biomass' % len(low_biomass))
            self.deactivate_elements(low_biomass, reason="dead")      

    def update(self):
                
        self.calculate_change_in_biomass()
        
        self.kill_elements()

        # self.update_fish_larvae()
        self.advect_ocean_current()

        # # Advect particles due to surface wind drag,
        # # according to element property wind_drift_factor
        # self.advect_wind()
    
        # self.update_terminal_velocity()
        self.vertical_mixing()
        # self.larvae_vertical_migration()

        self._apply_vertical_behavior()
