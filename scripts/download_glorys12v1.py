import copernicusmarine
import pandas as pd

# two lists with months of intereset
times1 = pd.date_range('20220101', '20220601', freq='MS') # first day of the month
times2 = pd.date_range('20220101', '20220601', freq='M') # last day of the month


for t1,t2 in zip(times1, times2):
    begin, final = t1, t2
    
    copernicusmarine.subset(
      dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
      variables=["so", "thetao", "uo", "vo", "zos"],
      minimum_longitude=-52,
      maximum_longitude=-30,
      minimum_latitude=-35,
      maximum_latitude=-16,
      start_datetime=t1.strftime('%Y-%m-%d'),
      end_datetime=t2.strftime('%Y-%m-%d'),
      minimum_depth=0.49402499198913574,
      maximum_depth=109.72930145263672,
      credential_file='.copernicus_credentials',
      output_filename=f'../data/glorys/smax_{t1.strftime('%Y%m')}'
    )