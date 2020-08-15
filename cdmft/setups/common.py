class CycleSetupCommon:
    """
    realization needs self.h_int, self.gloc, self.g0, self.se, self.mu, self.global_moves,
    self.quantum_numbers
    """
    def initialize_cycle(self):
        return {'h_int': self.h_int,'g_local': self.gloc, 'weiss_field': self.g0,
                'self_energy': self.se, 'mu': self.mu, 'global_moves': self.global_moves,
                'quantum_numbers': self.quantum_numbers}

    def set_data(self, storage, load_mu = True):
        """
        loads the data of g_imp_iw, g_weiss_iw, se_imp_iw, mu from storage into the corresponding
        objects
        The data is copied, storage returns objects that are all BlockGf's and can not init
        a selfconsistency cycle
        """
        g = storage.load('g_imp_iw')
        self.gloc << g
        try: # TODO backward compatibility
            self.g0 << storage.load('g_weiss_iw')
        except KeyError:
            pass
        self.se << storage.load('se_imp_iw')
        if load_mu:
            self.mu = storage.load('mu')
