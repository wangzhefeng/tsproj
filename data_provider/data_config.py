# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 1.0.121121
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


config = {
    # TODO level 1-1
    "energy_storage": {
        "energy_storage_1": None,
        "energy_storage_2": None,
        "energy_storage_3": None,
        "energy_storage_4": None,
    },
    # TODO level 1-2
    "diesel_generators": {
        "diesel_generators_1": None,
        "diesel_generators_2": None,
        "diesel_generators_3": None,
        "diesel_generators_4": None,
        "diesel_generators_5": None,
        "diesel_generators_6": None,
        "diesel_generators_7": None,
    },
    # level 1-3
    "city": {
        # TODO
        "power": {
            "line_A": None,
            "line_B": None,
        },
        #! level 1 final target
        "power_pred": {
            "line_A": None,
            "line_B": None,
        },
        # level 2
        "transformer": {
            # level 3-1
            "ups_crac": {
                # TODO
                "power": {
                    "line_A1": [],
                    "line_B1": [],
                    "line_A2": [],
                    "line_B2": [],
                },
                #! level 3 target
                "power_pred": {
                    "line_A1": None,
                    "line_B1": None,
                    "line_A2": None,
                    "line_B2": None,
                },
                # level 4
                "rooms": {
                    # TODO
                    "power": {
                        "line_A1": [],
                        "line_B1": [],
                        "line_A2": [],
                        "line_B2": [],
                    },
                    #! level 4 target
                    "power_pred": {
                        "line_A1": None,
                        "line_B1": None,
                        "line_A2": None,
                        "line_B2": None,
                    },
                    # level 5
                    "room_201": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        #! level 5 target
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        # level 6
                        "cabinet_row_A": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            #! level 6 target
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_A01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_B": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_B01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_C": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_C01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_D": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_D01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_E": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_E01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_F": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_F01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_G": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_G01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_H": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_H01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_I": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_I01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_J": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_J01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                    },
                    "room_202": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "cabinet_row_A": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_A01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_B": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_B01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_C": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_C01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C06": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_D": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_D01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D05": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                    },
                    "room_203": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "cabinet_row_A": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_A01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_B": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_B01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_C": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_C01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_D": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_D01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_E": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_E01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_F": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_F01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_G": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_G01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_H": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_H01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_I": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_I01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_J": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_J01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                    },
                    "room_204": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "cabinet_row_A": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_A01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_A04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_B": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_B01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_B04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_C": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_C01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_C04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_D": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_D01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_D04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_E": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_E01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_E04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_F": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_F01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_F04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_G": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_G01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_G04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_H": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_H01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_H04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_I": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_I01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_I04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                        "cabinet_row_J": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "env_vars": {
                                "temp": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                                "humi": {
                                    "hot_asile": [],
                                    "cold_asile": [],
                                },
                            },
                            "crac_J01": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J02": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J03": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                            "crac_J04": {
                                "power": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                                "power_pred": {
                                    "line_A": None,
                                    "line_B": None,
                                },
                            },
                        },
                    },
                },
            },
            "transformer_201_202": {
                "power": {
                    "room_201": "(201PD)2AN1a01.xlsx",
                    "room_202": "(202PD)2AN1b01.xlsx",
                },
                "power_pred": {
                    "room_201": None,
                    "room_202": None,
                },
                "ups": {
                    "power": {
                        
                    },
                    "power_pred": {
                        
                    },
                    
                },
            },
            "transformer_203_204": {
                "power": {
                    "room_203": "(204PD)2AN2a01.xlsx",
                    "room_204": "(203PD)2AN2b01.xlsx",
                },
                "power_pred": {
                    "room_203": None,
                    "room_204": None,
                },
            },
            # level 3-2
            "ups": {
                # TODO
                "power": {
                    "line_A1": [
                        "2GU1a1(201PD)UPS本体.xlsx",
                        "2GU1a2(201PD)UPS本体.xlsx",
                        "2GU1a3(201PD)UPS本体.xlsx",
                        "2GU1a4(201PD)UPS本体.xlsx",
                    ],
                    "line_B1": [
                        "2GU1b1(PD202)UPS本体.xlsx",
                        "2GU1b2(PD202)UPS本体.xlsx",
                        "2GU1b3(PD202)UPS本体.xlsx",
                        "2GU1b4(PD202)UPS本体.xlsx",
                    ],
                    "line_A2": [
                        "2GU2a1(204PD)UPS本体.xlsx",
                        "2GU2a2(204PD)UPS本体.xlsx",
                        "2GU2a3(204PD)UPS本体.xlsx",
                        "2GU2a4(204PD)UPS本体.xlsx",
                    ],
                    "line_B2": [
                        "2GU2b1(PD203)UPS本体.xlsx",
                        "2GU2b2(PD203)UPS本体.xlsx",
                        "2GU2b3(PD203)UPS本体.xlsx",
                        "2GU2b4(PD203)UPS本体.xlsx",
                    ],
                },
                #! level 3-2 target
                "power_pred": {
                    "line_A": None,
                    "line_B": None,
                },
                "power_dist_room": {
                    "room_201_202": {
                        "ups_1": {},
                        "ups_2": {},
                        "ups_3": {},
                        "ups_4": {},
                        "ups_5": {},
                        "ups_6": {},
                        "ups_7": {},
                        "ups_8": {},
                    },
                    "room_203_204": {
                        "ups_1": {},
                        "ups_2": {},
                        "ups_3": {},
                        "ups_4": {},
                        "ups_5": {},
                        "ups_6": {},
                        "ups_7": {},
                        "ups_8": {},
                    },
                },
                # level 4
                "room_201": {
                    "power": {
                        "line_A": None,
                        "line_B": None,
                    },
                    #! level 4 target
                    "power_pred": {
                        "line_A": None,
                        "line_B": None,
                    },
                    # level 5
                    "cabinet_row_A": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        #! level 5 target
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        # level 6
                        "cabinet_A01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            #! level 6 target
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_B": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_B01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_B03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },   
                        },
                        "cabinet_B04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_C": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_C01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_D": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_D01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_D04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_E": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_E01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_F": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_F01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_G": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_G01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_G03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_H": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_H01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_I": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_I01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_J": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_J01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_J03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_J04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                },
                "room_202": {
                    "power": {
                        "line_A": None,
                        "line_B": None,
                    },
                    "power_pred": {
                        "line_A": None,
                        "line_B": None,
                    },
                    "cabinet_row_A": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_A01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },  
                        },
                        "cabinet_A03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_B": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_B01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },  
                        },
                        "cabinet_B03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_B04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_C": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_C01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_D": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_D01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },  
                        },
                        "cabinet_D03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                },
                "room_203": {
                    "power": {
                        "line_A": None,
                        "line_B": None,
                    },
                    "power_pred": {
                        "line_A": None,
                        "line_B": None,
                    },
                    "cabinet_row_A": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_A01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_A03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_B": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_B01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_C": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_C01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_D": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_D01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_E": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_E01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_F": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_F01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_G": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_G01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_H": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_H01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_I": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_I01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_J": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_J01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                },
                "room_204": {
                    "power": {
                        "line_A": None,
                        "line_B": None,
                    },
                    "power_pred": {
                        "line_A": None,
                        "line_B": None,
                    },
                    "cabinet_row_A": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_A01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_A03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_A10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_B": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_B01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_B11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_C": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_C01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_C03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_C11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_D": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_D01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_D11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_E": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_E01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_E10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_F": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_F01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_F03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_F11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_G": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_G01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_G11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_H": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_H01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            }, 
                        },
                        "cabinet_H03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_H11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_I": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_I01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_I10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                    "cabinet_row_J": {
                        "power": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "power_pred": {
                            "line_A": None,
                            "line_B": None,
                        },
                        "env_vars": {
                            "temp": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                            "humi": {
                                "hot_asile": [],
                                "cold_asile": [],
                            },
                        },
                        "cabinet_J01": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J02": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J03": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J04": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J05": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J06": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J07": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J08": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J09": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J10": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                        "cabinet_J11": {
                            "power": {
                                "line_A": None,
                                "line_B": None,
                            },
                            "power_pred": {
                                "line_A": None,
                                "line_B": None,
                            },
                        },
                    },
                },
            },
        },
    },
}


geo1 = "transformer|low_voltage_incoming_line|UPS_device|UPS_output|cabinet_row|cabinet"
geo2 = "transformer|low_voltage_incoming_line|battery_room_CRAC"
geo3 = "transformer|low_voltage_incoming_line|power_distribution_room_CRAC"
geo4 = "transformer|low_voltage_incoming_line|IT_room_CRAC"
geo5 = "transformer|low_voltage_incoming_line|UPS_device|UPS_output|power_distribution_room_CRAC"
geo6 = "transformer|low_voltage_incoming_line|UPS_device|UPS_output|IT_room_CRAC"


config2 = {
    "module_1": {
        "line_A": {
            "transformer": "2TA1",
            "power_distribution_box": "201",
            "low_voltage_incoming_line": {
                "data_name": ["2AN1a1-1"],
                "data_file": [],
            },
            "UPS": {
                "UPS_group_1": {
                    "UPS_device": {
                        "data_name": ["2GU1a1", "2GU1a2"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU1a1-1", "2ANU1a1-2"],
                        "data_file": [],
                    },
                    "cabinet_row": {
                        "room_201": {
                            "cabinet_row_F": {
                                "data_name": {
                                    "cabinet_row": [],
                                    "cabinet": [],
                                },
                                "data_file": [],
                            },
                            "cabinet_row_G": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_H": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_I": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_J": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                        "room_202": {
                            "cabinet_row_C": {
                                "data_name": [],
                                "data_file": [],
                            }, 
                            "cabinet_row_D": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                    },
                },
                "UPS_group_2": {
                    "UPS_device": {
                        "data_name": ["2GU1a3", "2GU1a4"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU1a4-1", "2ANU1a4-2"],
                        "data_file": [],
                    },
                    "cabinet_row": {
                        "room_201": {
                            "cabinet_row_A": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_B": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_C": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_D": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_E": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                        "room_202": {
                            "cabinet_row_A": {
                                "data_name": [],
                                "data_file": [],
                            }, 
                            "cabinet_row_B": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                    },
                },
            },
            "battery_room_CRAC": {
                "data_name": ["2AP11"],
                "data_file": [],
            },
            # TODO
            "power_distribution_room_CRAC": {
                "data_name": ["2AP1b4", "2AP1b5"],
                "data_file": [],
            },
            # # TODO
            # "IT_room_CRAC": {
            #     "data_name": ["2AP1b1","2AP1b2","2AP1b3"],
            #     "data_file": [],
            # },
            # # TODO
            # "operator_room": {
            #     "data_name": ["2AP1b6"],
            #     "data_file": [],
            # },
        },
        "line_B": {
            "transformer": "2TB1",
            "power_distribution_box": "202",
            "low_voltage_incoming_line": {
                "data_name": ["2AN1b1-1"],
                "data_file": [],
            },
            "UPS": {
                "UPS_group_1": {
                    "UPS_device": {
                        "data_name": ["2GU1b1", "2GU1b2"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU1b1-1", "2ANU1b1-2"],
                        "data_file": [],
                    },
                    "cabinet_row": {
                        "room_201": {
                            "cabinet_row_F": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_G": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_H": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_I": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_J": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                        "room_202": {
                            "cabinet_row_C": {
                                "data_name": [],
                                "data_file": [],
                            }, 
                            "cabinet_row_D": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                    },
                },
                "UPS_group_2": {
                    "UPS_device": {
                        "data_name": ["2GU1b3", "2GU1b4"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU1b4-1", "2ANU1b4-2"],
                        "data_file": [],
                    },
                    "cabinet_row": {
                        "room_201": {
                            "cabinet_row_A": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_B": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_C": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_D": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_E": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                        "room_202": {
                            "cabinet_row_A": {
                                "data_name": [],
                                "data_file": [],
                            }, 
                            "cabinet_row_B": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                    },
                },
                "UPS_group_3": {
                    "UPS_device": {
                        "data_name": ["2GU1b5"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU1b7-1"],
                        "data_file": [],
                    },
                    # TODO
                    "power_distribution_room_CRAC": {
                        "data_name": ["2AP1a4", "2AP1a5"],
                        "data_file": [],
                    },
                    # # TODO
                    # "IT_room_CRAC": {
                    #     "data_name": ["2AP1a1", "2AP1a2", "2AP1a3"],
                    #     "data_file": [],
                    # },
                    # # TODO
                    # "operator_room": {
                    #     "data_name": ["2AP1a6"],
                    #     "data_file": [],
                    # },
                },
            },
            "battery_room_CRAC": {
                "data_name": ["2AP12"],
                "data_file": [],
            },
        },
    },
    "module_2": {
        "line_B": {
            "transformer": "2TB2",
            "power_distribution_box": "203",
            "low_voltage_incoming_line": {
                "data_name": ["2AN2b1-1"],
                "data_file": [],
            },
            "UPS": {
                "UPS_group_1": {
                    "UPS_device": {
                        "data_name": ["2GU2b1", "2GU2b2"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU2b1-1", "2ANU2b1-2"],
                        "data_file": [],
                    },
                    "cabinet_row": {
                        "room_203": {
                            "cabinet_row_A": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_B": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_C": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_D": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_E": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_F": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_G": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_H": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_I": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_J": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                    },
                },
                "UPS_group_2": {
                    "UPS_device": {
                        "data_name": ["2GU2b3", "2GU2b4"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU2b4-1", "2ANU2b4-2"],
                        "data_file": [],
                    },
                    "cabinet_row": {
                        "room_204": {
                            "cabinet_row_A": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_B": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_C": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_D": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_E": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_F": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_G": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_H": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_I": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_J": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                    },
                },
                "UPS_group_3": {
                    "UPS_device": {
                        "data_name": ["2GU2b5"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU2b7-1"],
                        "data_file": [],
                    },
                    # TODO
                    "power_distribution_room_CRAC": {
                        "data_name": ["2AP2a5", "2AP2a6"],
                        "data_file": [],
                    },
                    # # TODO
                    # "IT_room_CRAC": {
                    #     "data_name": ["2AP2a1", "2AP2a2", "2AP2a3", "2AP2a4"],
                    #     "data_file": [],
                    # },
                    # # TODO
                    # "operator_room": {
                    #     "data_name": ["2AP2a7"],
                    #     "data_file": [],
                    # },
                }
            },
            "battery_room_CRAC": {
                "data_name": ["2AP21"],
                "data_file": [],
            },
        },
        "line_A": {
            "transformer": "2TA2",
            "power_distribution_box": "204",
            "low_voltage_incoming_line": {
                "data_name": ["2AN2a1-1"],
                "data_file": [],
            },
            "UPS": {
                "UPS_group_1": {
                    "UPS_device": {
                        "data_name": ["2GU2a1", "2GU2a2"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU2a1-1", "2ANU1a2-2"],
                        "data_file": [],
                    },
                    "cabinet_row": {
                        "room_203": {
                            "cabinet_row_A": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_B": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_C": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_D": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_E": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_F": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_G": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_H": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_I": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_J": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                    },
                },
                "UPS_group_2": {
                    "UPS_device": {
                        "data_name": ["2GU2a3", "2GU2a4"],
                        "data_file": [],
                    },
                    "UPS_output": {
                        "data_name": ["2ANU2a4-1", "2ANU2a4-2"],
                        "data_file": [],
                    },
                    "cabinet_row": {
                        "room_203": {
                            "cabinet_row_A": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_B": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_C": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_D": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_E": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_F": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_G": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_H": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_I": {
                                "data_name": [],
                                "data_file": [],
                            },
                            "cabinet_row_J": {
                                "data_name": [],
                                "data_file": [],
                            },
                        },
                    },
                },
            },
            "battery_room_CRAC": {
                "data_name": ["2AP22"],
                "data_file": [],
            },
            # TODO
            "power_distribution_room_CRAC": {
                "data_name": ["2AP2b5", "2AP2b6"],
                "data_file": [],
            },
            # # TODO
            # "IT_room_CRAC": {
            #     "data_name": ["2AP2b1", "2AP2b2", "2AP2b3", "2AP2b4"],
            #     "data_file": [],
            # },
            # # TODO
            # "operator_room": {
            #     "data_name": ["2AP2b7"],
            #     "data_file": [],
            # },
        },
    },
}




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
