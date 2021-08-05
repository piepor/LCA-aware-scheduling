import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os

plot_path = './plots/results'
technologiesDF = pd.read_pickle('data/technologiesDFFilled.pkl')
match_tech = {'Biomass': 'biomass', 'Fossil Hard coal': 'hard coal', 'Fossil Coal-derived gas': 'coal gases', 
             'Fossil Gas': 'natural gas', 'Fossil Oil': 'HFO', 'Geothermal': 'geothermal', 'Hydro Pumped Storage': 'hydro',
             'Hydro': 'hydro', 'Solar': 'photovoltaic', 'Wind Onshore': 'wind'}
impacts = ['climate change total', 'respiratory effects, inorganics']
match_tech = {'Biomass': 'Biomass', 'Fossil Hard coal': 'Fossil Hard coal', 'Fossil Coal-derived gas': 'Fossil Coal-derived gas', 
             'Fossil Gas': 'Fossil Gas', 'Fossil Oil': 'Fossil Oil', 'Geothermal': 'Geothermal', 'Hydro Pumped Storage': 'Hydro Pumped Storage',
             'Hydro': 'Hydro', 'Solar': 'Solar', 'Wind Onshore': 'Wind Onshore'}
impact_df = pd.read_csv('data/use-phase-unitary-impacts.csv')

for i, technology in enumerate(technologiesDF.keys()):
    coeff = impact_df[impact_df['Impact category'] == impacts[0]][match_tech[technology]]
    ##breakpoint()
    if i == 0:
        total_impact = technologiesDF[technology] * coeff.values[0]
    else:
        total_impact += technologiesDF[technology] * coeff[0]

total_impact = total_impact.rename('Climate Change')
total_production = technologiesDF.sum(axis=1)
total_impact_prod = total_impact/total_production

#fig = make_subplots(rows=6, cols=4)
fig = go.Figure()
for i in range(0, 24):
    #if i != 10 and i !=11:
    x_names = 24*['{}-{}'.format(i, i+1)]
    total_impact_hour = total_impact_prod.loc[total_impact.index.to_series().dt.hour == i]
    y_plot = total_impact_hour.values[:]
    #breakpoint()
    #fig.add_trace(go.Box(y=y_plot, boxmean=True, name='{}-{}'.format(i, i+1)))
    if i < 10:
        name_trace = '0{} '.format(i)
    else:
        name_trace = '{}:00'.format(i)
    fig.add_trace(go.Box(y=y_plot, boxmean=True, name=name_trace,
                         showlegend=False, marker_color='blue'))
                  #row=int(np.floor(i/4)+1), col=int(i%4+1))
    #print("{} {}".format(int(np.floor(i/4)+1), int(i%4+1)))

fig.update_layout(font=dict(size=32))
fig.update_yaxes(title_text='[kg_CO2eq / kWh]')
fig.show()
fig.write_html(os.path.join(plot_path, 'hour-distribution.html'))
