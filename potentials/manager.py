import torch
from icecream import ic
import potentials.potentials as potentials

class PotentialManager:
    '''
        Class to define a set of potentials from the given config object and to apply all of the specified potentials
        during each cycle of the inference loop.

        Author: NRB 
    '''

    def __init__(self, potentials_config, ppi_config, diffuser_config):
        
        self.guide_scale = potentials_config.guide_scale
        self.guide_decay = potentials_config.guide_decay

        if potentials_config.guiding_potentials is None: setting_list = []
        else: setting_list = [self.parse_potential_string(potstr) for potstr in potentials_config.guiding_potentials]

        # PPI potentials require knowledge about the binderlen which may be detected at runtime
        # This is a mechanism to still allow this info to be used in potentials - NRB 
        if not ppi_config.binderlen is None:
            binderlen_update   = { 'binderlen': ppi_config.binderlen }
            hotspot_res_update = { 'hotspot_res': ppi_config.hotspot_res }

            for setting in setting_list:
                if setting['type'] in potentials.require_binderlen:
                    setting.update(binderlen_update)

                if setting['type'] in potentials.require_hotspot_res:
                    setting.update(hotspot_res_update)

        self.potentials_to_apply = self.initialize_all_potentials(setting_list)
        self.T = diffuser_config.T
        
    def is_empty(self):
        '''
            Check whether this instance of PotentialManager actually contains any potentials
        '''

        return len(self.potentials_to_apply) == 0

    def parse_potential_string(self, potstr):
        '''
            Parse a single entry in the list of potentials to be run to a dictionary of settings for that potential.

            An example of how this parsing is done:
            'setting1:val1,setting2:val2,setting3:val3' -> {setting1:val1,setting2:val2,setting3:val3}
        '''

        setting_dict = {entry.split(':')[0]:entry.split(':')[1] for entry in potstr.split(',')}

        for key in setting_dict:
            if not key == 'type': setting_dict[key] = float(setting_dict[key])

        return setting_dict

    def initialize_all_potentials(self, setting_list):
        '''
            Given a list of potential dictionaries where each dictionary defines the configurations for a single potential,
            initialize all potentials and add to the list of potentials to be applies
        '''

        to_apply = []

        for potential_dict in setting_list:
            assert(potential_dict['type'] in potentials.implemented_potentials), f'potential with name: {potential_dict["type"]} is not one of the implemented potentials: {potentials.implemented_potentials}'

            to_apply.append( potentials.implemented_potentials[potential_dict['type']](**{k: potential_dict[k] for k in potential_dict.keys() - {'type'}}) )

        return to_apply

    def compute_all_potentials(self, seq, xyz):
        '''
            This is the money call. Take the current sequence and structure information and get the sum of all of the potentials that are being used
        '''

        potential_list = [potential.compute(seq,xyz) for potential in self.potentials_to_apply]
        potential_stack = torch.stack(potential_list, dim=0)

        return torch.sum(potential_stack, dim=0)

    def get_guide_scale(self, t):
        '''
        Given a timestep and a decay type, get the appropriate scale factor to use for applying guiding potentials
        
        Inputs:
        
            t (int, required):          The current timestep
        
        Output:
        
            scale (int):                The scale factor to use for applying guiding potentials
        
        '''
        
        implemented_decay_types = {
                'constant': lambda t: self.guide_scale,
                # Linear interpolation with y2: 0, y1: guide_scale, x2: 0, x1: T, x: t
                'linear'  : lambda t: t/self.T * self.guide_scale,
                'quadratic' : lambda t: t**2/self.T**2 * self.guide_scale,
                'cubic' : lambda t: t**3/self.T**3
        }
        
        if self.guide_decay not in implemented_decay_types:
            sys.exit(f'decay_type must be one of {implemented_decay_types.keys()}. Received decay_type={self.guide_decay}. Exiting.')
        
        return implemented_decay_types[self.guide_decay](t)


        
