import plotly.graph_objects as go
import pandas as pd
import pickle
from plotly.subplots import make_subplots
import os

file_paths = ['./results/dict-results-direct-impact.pkl', 
              './results/dict-results-composed-impact.pkl']
plot_path = './plots/results'

model_types = ['linear regr', 'nn 32', 'nn 64', 'rnn 32', 'rnn 64']
#look_up_table = {'LINEAR 32': 'linear regr',
#                 'NN 32': 'nn 32',
#                 'NN 64': 'nn 64',
#                 'RNN 32': 'rnn 32',
#                 'RNN 64': 'rnn 64'}
technologies = ['Biomass', 'Fossil Hard coal', 'Fossil Coal-derived gas', 'Fossil Gas', 'Fossil Oil', 
                'Geothermal', 'Hydro Pumped Storage', 'Hydro', 'Solar', 'Wind Onshore']
prediction_types = ['composed', 'direct']

dict_ord = {}
dict_ord_tech = {}
for file_path in file_paths:
    prediction_type = file_path.split('-')[2]
    print('Processing {}'.format(prediction_type))
    with open(file_path, 'rb') as file:
        dict_results = pickle.load(file)
    for impact in dict_results.keys():
        print('---------- {}'.format(impact))
        for model_type in dict_results[impact].keys():
            print('---------------- {}'.format(model_type))
            for unit in dict_results[impact][model_type].keys():
                print('------------------- {}'.format(unit))
                if prediction_type == 'composed':
                    for element in dict_results[impact][model_type][unit].keys():
                        for metric in dict_results[impact][model_type][unit][element].keys(): 
                            dict_ord_tech['{}-{}-{}-{}-{}'.format(metric, impact, element, model_type, unit)] = \
                                dict_results[impact][model_type][unit][element][metric]
                for metric in dict_results[impact][model_type][unit][prediction_type].keys():
                    dict_ord['{}-{}-{}-{}-{}'.format(impact, metric, prediction_type, model_type, unit)] = \
                        dict_results[impact][model_type][unit][prediction_type][metric]

#marker_colors = {'Biomass': 'darkred', 'Fossil Hard coal': 'darkblue', 
#                 'Fossil Coal-derived gas': 'purple', 'Fossil Gas': 'darkgoldenrod',
#                 'Fossil Oil': 'darksalmon', 'Geothermal': 'darkgreen', 'Hydro Pumped Storage': 'navy',
#                 'Hydro': 'orange', 'Solar': 'dimgray', 'Wind Onshore': 'deepskyblue',
#                 'Climate change': 'red', 'Particulate Matter': 'blue'}
marker_colors = {'Biomass': 'darkred', 'Fossil Hard coal': 'darkblue', 
                 'Fossil Coal-derived gas': 'purple', 'Fossil Gas': 'darkgoldenrod',
                 'Fossil Oil': 'darksalmon', 'Geothermal': 'darkgreen', 'Hydro Pumped Storage': 'navy',
                 'Hydro': 'orange', 'Solar': 'dimgray', 'Wind Onshore': 'deepskyblue',
                 'climate change total': 'red', 'respiratory effects, inorganics': 'blue'}

#impact_df = pd.read_csv('data/climate-change-impacts.csv')
impact_df = pd.read_csv('data/use-phase-unitary-impacts.csv')
#match_tech = {'Biomass': 'biomass', 'Fossil Hard coal': 'hard coal', 'Fossil Coal-derived gas': 'coal gases', 
#             'Fossil Gas': 'natural gas', 'Fossil Oil': 'HFO', 'Geothermal': 'geothermal', 'Hydro Pumped Storage': 'hydro',
#             'Hydro': 'hydro', 'Solar': 'photovoltaic', 'Wind Onshore': 'wind'}
match_tech = {'Biomass': 'Biomass', 'Fossil Hard coal': 'Fossil Hard coal', 'Fossil Coal-derived gas': 'Fossil Coal-derived gas', 
             'Fossil Gas': 'Fossil Gas', 'Fossil Oil': 'Fossil Oil', 'Geothermal': 'Geothermal', 'Hydro Pumped Storage': 'Hydro Pumped Storage',
             'Hydro': 'Hydro', 'Solar': 'Solar', 'Wind Onshore': 'Wind Onshore'}

impacts = ['climate change total', 'respiratory effects, inorganics']
for model_type in ['LINEAR', 'NN', 'RNN']:
    for metric in ['rmse',  'nrmse', 'smape']:
        subplot_titles = ('Technologies contributions', 'Total impacts')
        for j, unit in enumerate(['32', '64']):
            fig = make_subplots(rows=1, cols=2,
                                specs=[[{"secondary_y":True}, {"secondary_y":True}]])
            if model_type == 'LINEAR' and unit == '64':
                continue
            #for i, impact in enumerate(['Climate change', 'Particulate Matter']):
            for i, impact in enumerate(impacts):
                impact_coeffs = impact_df[impact_df['Impact category'] == impact]
                for k, technology in enumerate(technologies):
                    impact_name = match_tech[technology]
                    impact_coeff = impact_coeffs[impact_name].values
                    tech_sel = dict_ord_tech['{}-{}-{}-{}-{}'.format(
                        metric, impact, technology, model_type, unit)]
                    perc = dict_ord_tech['{}-{}-{}-{}-{}'.format(
                        'percentage', impact, technology, model_type, unit)]
                    if k == 0:
                        total_y = tech_sel*perc*impact_coeff
                    else:
                        total_y += tech_sel*perc*impact_coeff
                for technology in technologies:
                    impact_name = match_tech[technology]
                    impact_coeff = impact_coeffs[impact_name].values
                    tech_sel = dict_ord_tech['{}-{}-{}-{}-{}'.format(
                        metric, impact, technology, model_type, unit)]
                    perc = dict_ord_tech['{}-{}-{}-{}-{}'.format(
                        'percentage', impact, technology, model_type, unit)]
                    fig.add_trace(go.Box(
                        x=[technology]*len(list(tech_sel)), y=100*tech_sel*perc*impact_coeff/total_y, showlegend=False, 
                        marker_color=marker_colors[technology], boxmean=True), row=1, col=i+1)
                imp_sel = dict_ord_tech['{}-{}-{}-{}-{}'.format(
                    metric, impact, 'composed', model_type, unit)]
                #if impact == 'Particulate Matter' and metric == 'rmse':
                if impact == 'respiratory effects, inorganics' and metric == 'rmse':
                    secondary_y = True
                else:
                    secondary_y = False

            title = '{} - {} - {}'.format(metric.upper(), model_type, str(unit))
            fig.update_yaxes(range=[0, 100])
            fig.update_layout(font=dict(size=32))
            fig.update_yaxes(title_text='% of daily {}'.format(metric.upper()), row=1, col=1)
            fig.update_yaxes(title_text='% of daily {}'.format(metric.upper()), row=1, col=2)
            fig.write_html(os.path.join(plot_path, '{}-composition-boxplot.html'.format(title.replace(' ', ''))))
            fig.show(renderer='chromium')

marker_colors = {'direct': 'red', 'composed': 'blue'}
#subplot_titles = ('Climate Change', 'Particulate Matter')
subplot_titles = tuple(impacts)
for metric in ['rmse', 'nrmse', 'smape']:
    fig = make_subplots(rows=1, cols=2)
    #for i, impact in enumerate(['Climate change', 'Particulate Matter']):
    for i, impact in enumerate(impacts):
        for prediction_type in ['direct', 'composed']:
            y_input = []
            x_names = []
            if i == 0:
                legend = True
            else:
                legend = False
            for model_type in ['LINEAR', 'NN', 'RNN']:
                for j, unit in enumerate(['32', '64']):
                    if model_type == 'LINEAR' and unit == '64':
                        continue
                    metric_impact = dict_ord['{}-{}-{}-{}-{}'.format(
                        impact, metric, prediction_type, model_type, unit)]
                    y_input.extend(list(metric_impact))
                    if model_type == 'LINEAR':
                        name = '{}'.format(model_type)
                    else:
                        name = '{} {} units'.format(model_type, unit)
                    x_names.extend([name.lower()]*len(list(metric_impact)))
            if prediction_type == 'composed':
                name_plot = 'ETMF'
            else:
                name_plot = 'DF'
            fig.add_trace(go.Box(
                x=x_names, y=y_input, name=name_plot, marker_color=marker_colors[prediction_type],
                showlegend=legend, boxmean=True, offsetgroup=prediction_type), row=1, col=i+1)
    if metric == 'nrmse':
        fig.update_yaxes(range=[0, 0.4])
    elif metric == 'smape':
        fig.update_yaxes(range=[0, 30])
    fig.update_layout(boxmode='group', font=dict(size=32))
    if metric == 'smape':
        fig.update_yaxes(title_text='daily SMAPE [%]', row=1, col=1)
        fig.update_yaxes(title_text='daily SMAPE [%]', row=1, col=2)
    elif metric == 'nrmse':
        fig.update_yaxes(title_text='daily NRMSE [-]', row=1, col=1)
        fig.update_yaxes(title_text='daily NRMSE [-]', row=1, col=2)
    fig.write_html(os.path.join(plot_path, '{}-boxplot.html'.format(metric.upper())))
    fig.show(renderer='chromium')
