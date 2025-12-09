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
        }
            )

        # Are these on or off when uncommented?
        # self._set_config_default('drift:vertical_mixing', True)
        # self._set_config_default('drift:vertical_mixing_at_surface', True)
        # self._set_config_default('drift:vertical_advection_at_surface', True)

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
