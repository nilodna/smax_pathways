import cdsapi
import numpy as np

c = cdsapi.Client()

years = [1998]
months = [str(m).zfill(2) for m in np.arange(1, 13)]

for y in years:
    for m in months:

        dic = { 'format': 'zip',
                'variable': 'mass_concentration_of_chlorophyll_a',
                'projection': 'regular_latitude_longitude_grid',
                'year': str(y),
                'month': str(m).zfill(2),
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
        ],
        'version': '5_0_1',
    }


    c.retrieve('satellite-ocean-colour', dic, f"chla_{y}{m}.zip")
