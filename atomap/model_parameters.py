class SublatticeParameterBase:
    def __init__(self):
        self.color = 'red'
        self.name = "Base Sublattice"
        self.sublattice_order = None

    def __repr__(self):
        return '<%s, %s>' % (
            self.__class__.__name__,
            self.name
            )


class GenericSublattice(SublatticeParameterBase):
    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.color = 'red'
        self.tag = 'S0'
        self.image_type = 0
        self.name = "Sublattice 0"
        self.sublattice_order = 0
        self.refinement_config = {
                'config': [
                    ['image_data', 1, 'center_of_mass'],
                    ['image_data', 1, 'gaussian'],
                    ],
                'neighbor_distance': 0.35}


class PerovskiteOxide110SublatticeACation(SublatticeParameterBase):
    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.name = "A-cation"
        self.tag = "A"
        self.color = 'blue'
        self.image_type = 0
        self.zone_axis_list = [
                {'number': 0, 'name': '110'},
                {'number': 1, 'name': '100'},
                {'number': 2, 'name': '11-2'},
                {'number': 3, 'name': '112'},
                {'number': 5, 'name': '111'},
                {'number': 6, 'name': '11-1'},
                ]
        self.sublattice_order = 0
        self.refinement_config = {
                'config': [
                    ['image_data', 2, 'gaussian'],
                    ],
                'neighbor_distance': 0.35}


class PerovskiteOxide110SublatticeBCation(SublatticeParameterBase):
    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.name = "B-cation"
        self.tag = "B"
        self.color = 'green'
        self.image_type = 0
        self.zone_axis_list = [
                {'number': 0, 'name': '110'},
                {'number': 1, 'name': '100'},
                {'number': 2, 'name': '11-2'},
                {'number': 3, 'name': '112'},
                {'number': 5, 'name': '111'},
                {'number': 6, 'name': '11-1'}, ]
        self.sublattice_order = 1
        self.sublattice_position_sublattice = "A-cation"
        self.sublattice_position_zoneaxis = "100"
        self.refinement_config = {
                'config': [
                    ['image_data', 1, 'center_of_mass'],
                    ['image_data', 1, 'gaussian'],
                    ],
                'neighbor_distance': 0.25}
        self.atom_subtract_config = [
                {
                    'sublattice': 'A-cation',
                    'neighbor_distance': 0.35,
                    },
                ]


class PerovskiteOxide110SublatticeOxygen(SublatticeParameterBase):
    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.name = "Oxygen"
        self.tag = "O"
        self.color = 'red'
        self.image_type = 1
        self.zone_axis_list = [
                {'number': 0, 'name': '110'},
                {'number': 1, 'name': '100'},
                {'number': 2, 'name': '11-2'},
                {'number': 3, 'name': '112'},
                {'number': 5, 'name': '111'},
                {'number': 6, 'name': '11-1'}, ]
        self.sublattice_order = 2
        self.sublattice_position_sublattice = "B-cation"
        self.sublattice_position_zoneaxis = "110"
        self.refinement_config = {
                'config': [
                    ['image_data', 1, 'center_of_mass'],
                    ['image_data', 1, 'gaussian'],
                    ],
                'neighbor_distance': 0.25}
        self.atom_subtract_config = [
                {
                    'sublattice': 'A-cation',
                    'neighbor_distance': 0.35,
                    },
                {
                    'sublattice': 'B-cation',
                    'neighbor_distance': 0.30,
                    },
                ]


class ModelParametersBase:
    def __init__(self):
        self.peak_separation = None
        self.name = None
        self.sublattice_list = []

    def __repr__(self):
        return '<%s, %s>' % (
            self.__class__.__name__,
            self.name,
            )

    def get_sublattice_from_order(self, order_number):
        for sublattice in self.sublattice_list:
            if order_number == sublattice.sublattice_order:
                return(sublattice)
        return(False)

    @property
    def number_of_sublattices(self):
        return(len(self.sublattice_list))


class GenericStructure(ModelParametersBase):
    def __init__(self):
        ModelParametersBase.__init__(self)
        self.peak_separation = None
        self.name = 'A structure'

        self.sublattice_list = [
            GenericSublattice(),
        ]


class PerovskiteOxide110(ModelParametersBase):
    def __init__(self):
        ModelParametersBase.__init__(self)
        self.name = "Peroskite 110"
        self.peak_separation = 0.127

        self.sublattice_list = [
            PerovskiteOxide110SublatticeACation(),
            PerovskiteOxide110SublatticeBCation(),
            PerovskiteOxide110SublatticeOxygen(),
            ]


class SrTiO3_110(PerovskiteOxide110):
    def __init__(self):
        PerovskiteOxide110.__init__(self)
        self.sublattice_names = "Sr", "Ti", "O"
        Ti_sublattice_position = {
                "sublattice": "Sr",
                "zoneaxis": "100"}
        O_sublattice_position = {
                "sublattice": "Ti",
                "zoneaxis": "110"}
        self.sublattice_position = [
                Ti_sublattice_position,
                O_sublattice_position]
