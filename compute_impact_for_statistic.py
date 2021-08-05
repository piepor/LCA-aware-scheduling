import plotly.graph_objects as go
import tensorflow as tf
import pickle
import os
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

tfk = tf.keras 

def rmse_func(forecast, target):
    #return np.sqrt(np.mean(np.power(forecast - target, 2)))
    return np.sqrt(np.mean(np.power(forecast - target, 2), axis=1))

def smape_func(forecast, target):
    #return np.mean(np.mean((np.abs(forecast - target) / (np.abs(forecast) + np.abs(target))) * 100, axis=1), axis=0)
    return (np.abs(forecast - target) / (np.abs(forecast) + np.abs(target))) * 100

def series_to_supervised(df, lags_in, lags_forecast):
    # name of the series considered
    var_name = df.head().name
    cols, names = [], []
    # appending the series shifted by one on the right in order to have
    # columns from var(t-lags_in), ..., var(t-1)
    for i in range(lags_in, 0, -1):
        cols.append(df.shift(i))
        # column name
        names.append("".join([var_name, "(t-", str(i), ")"]))
    # create new pandas dataframe
    df_x = pd.concat(cols, axis=1)
    df_x.columns = names
    # dropping row with not a number items (it means they are out of the interval)
    df_x = df_x.dropna()
    # taking only the rows with hour == 0 because we are forecsting only one time
    # at midnight the next day
    df_x = df_x[df_x.index.hour == 0]
    cols, names = [], []
    # appending the series shifted by minus one on the right in order to 
    # have columns from var(t), ..., var(t+lags_forecast)
    for i in range(0, lags_forecast):
        cols.append(df.shift(-i))
        if i == 0:
            names.append("".join([var_name, "(t)"]))
        else:
            names.append("".join([var_name, "(t+", str(i), ")"]))
    # same operations as before
    df_y = pd.concat(cols, axis=1)
    df_y.columns = names
    df_y = df_y.dropna()
    df_y = df_y[df_y.index.hour == 0][int(lags_in/24):]
    df_x = df_x[df_x.index.isin(df_y.index)]
    return df_x, df_y

def compute_direct_impact(df_tech, df_impacts, impacts):
    # assign to every column in dftech a parto of the string
    # in dfImpacts columns
#    match_tech = {'Biomass': 'biomass', 'Fossil Hard coal': 'hard coal', 'Fossil Coal-derived gas': 'coal gases', 
#                 'Fossil Gas': 'natural gas', 'Fossil Oil': 'HFO', 'Geothermal': 'geothermal', 'Hydro Pumped Storage': 'hydro',
#                 'Hydro': 'hydro', 'Solar': 'photovoltaic', 'Wind Onshore': 'wind'}
    match_tech = {'Biomass': 'Biomass', 'Fossil Hard coal': 'Fossil Hard coal', 'Fossil Coal-derived gas': 'Fossil Coal-derived gas', 
                 'Fossil Gas': 'Fossil Gas', 'Fossil Oil': 'Fossil Oil', 'Geothermal': 'Geothermal', 'Hydro Pumped Storage': 'Hydro Pumped Storage',
                 'Hydro': 'Hydro', 'Solar': 'Solar', 'Wind Onshore': 'Wind Onshore'}
    df_tech_impact = pd.DataFrame()
    for impact in impacts:
        direct_impact = 0
        for technology in df_tech.columns:
            impactName = match_tech[technology]
            direct_impact += df_tech[technology] * df_impacts[df_impacts['Impact category'] == impact][impactName].values
        df_tech_impact[impact] = direct_impact
    return df_tech_impact

#impacts = ['Climate change', 'Particulate Matter']
impacts = ['climate change total', 'respiratory effects, inorganics']
#match_tech = {'Biomass': 'biomass', 'Fossil Hard coal': 'hard coal', 'Fossil Coal-derived gas': 'coal gases', 
#             'Fossil Gas': 'natural gas', 'Fossil Oil': 'HFO', 'Geothermal': 'geothermal', 'Hydro Pumped Storage': 'hydro',
#             'Hydro': 'hydro', 'Solar': 'photovoltaic', 'Wind Onshore': 'wind'}
match_tech = {'Biomass': 'Biomass', 'Fossil Hard coal': 'Fossil Hard coal', 'Fossil Coal-derived gas': 'Fossil Coal-derived gas', 
             'Fossil Gas': 'Fossil Gas', 'Fossil Oil': 'Fossil Oil', 'Geothermal': 'Geothermal', 'Hydro Pumped Storage': 'Hydro Pumped Storage',
             'Hydro': 'Hydro', 'Solar': 'Solar', 'Wind Onshore': 'Wind Onshore'}

path_df = './data/technologiesDFFilled.pkl'
technologies_df = pd.read_pickle('data/technologiesDFFilled.pkl')
indeces = pd.read_pickle('./data/indeces.pkl')

total_sum = technologies_df.sum(axis=1).to_frame(name='total')
scaler = StandardScaler()
scaled_values = scaler.fit_transform(technologies_df.values)
technologies_df = pd.DataFrame(scaled_values, index=technologies_df.index, columns=technologies_df.columns)
impact_df = pd.read_csv('data/use-phase-unitary-impacts.csv')

#compute production
#for i, technology in enumerate(technologies_df.keys()):
#    # select technology
#    df = technologies_df[technology]
#    x, y = series_to_supervised(df, lags_in=7*24, lags_forecast=24)
#    x = x.loc[indeces.index]
#    y = y.loc[x.index]
#    x_test = x.iloc[int(perc_train*len(x))+int(perc_vali*len(x)):]
#    y_test = y.loc[x_test.index]
#    if i == 0:
#        total_production = np.zeros_like(y_test.values)
#    total_production += (y_test.values * scaler.scale_[i] + scaler.mean_[i]) 

perc_train = 0.8
perc_vali = 0.1
perc_test = 0.1
n_units = [32, 64]
model_types = ['LINEAR', 'NN', 'RNN']
dict_results = {}

total_x, total_y = series_to_supervised(total_sum['total'], lags_in=7*24, lags_forecast=24)
total_x = total_x.loc[indeces.index]
total_y = total_y.loc[total_x.index]
total_x_test = total_x.iloc[int(perc_train*len(total_x))+int(perc_vali*len(total_x)):]
total_y_test = total_y.loc[total_x_test.index]
for impact in impacts:
    impact_coeffs = impact_df[impact_df['Impact category'] == impact]
    total_impact_df_act = pd.DataFrame()
    total_impact_act = 0
    total_impact_df_pred = pd.DataFrame()
    dict_results[impact] = {}
    for k, model_type in enumerate(model_types):
        dict_results[impact][model_type] = {}
        for j, n_unit in enumerate(n_units):
            message = "".join(['Processing ', impact, ' - ', model_type, ' - ', str(n_unit)])
            print(message)
            if model_type == 'LINEAR':
                LINEAR = True
                NN = False
                RNN = False
            elif model_type == 'NN':
                LINEAR = False
                NN = True
                RNN = False
            elif model_type == 'RNN':
                LINEAR = False
                NN = False
                RNN = True

            if LINEAR and j > 0:
                continue
            if LINEAR:
                results_path = os.path.join('results', 'linear-regression')
                models_path = os.path.join('models', 'linear-regression')
            elif NN:
                results_path = os.path.join('results', 'nn', "".join([str(n_unit), 'neurons']))
                models_path = os.path.join('models', 'nn', "".join([str(n_unit), 'neurons']))
            elif RNN:
                results_path = os.path.join('results', 'rnn', "".join([str(n_unit), 'neurons']))
                models_path = os.path.join('models', 'rnn', "".join([str(n_unit), 'neurons']))

            file_path = os.path.join(results_path, "".join([impact.replace(" ", "-"),'-composed-metrics.log']))
            try:
                os.remove(file_path)
            except Exception:
                pass
            with open(file_path, 'a') as file:
                file.write("".join([impact, " composed model"]))
            dict_results[impact][model_type][str(n_unit)] = {}
            for i, technology in enumerate(technologies_df.keys()):
                # select technology
                df = technologies_df[technology]
                impact_name = match_tech[technology]
                impact_coeff = impact_coeffs[impact_name]
                x, y = series_to_supervised(df, lags_in=7*24, lags_forecast=24)
                x = x.loc[indeces.index]
                y = y.loc[x.index]
                x_test = x.iloc[int(perc_train*len(x))+int(perc_vali*len(x)):]
                y_test = y.loc[x_test.index]
                if i == 0:
                    total_impact_pred = np.zeros_like(y_test.values)

                if LINEAR:
                    model_path = os.path.join(models_path, "".join([technology.replace(" ", "-"), "-model.pkl"]))
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                else:
                    model_path = os.path.join(models_path, technology.replace(" ", "-"))
                    model = tfk.models.load_model(model_path)
                        
                if RNN:
                    x_rnn = x_test.values[:, :, np.newaxis]
                    forecast = model.predict(x_rnn)
                else:
                    forecast = model.predict(x_test.values)
                total_impact_pred += (forecast * scaler.scale_[i] + scaler.mean_[i]) * impact_coeff.values
                # compute the total actual impact only in the raw case
                if i == 0 and j == 0 and k == 0:
                    total_impact_act = (y_test.values * scaler.scale_[i] + scaler.mean_[i]) * impact_coeff.values
                elif j == 0 and k == 0:
                    total_impact_act += (y_test.values * scaler.scale_[i] + scaler.mean_[i]) * impact_coeff.values
                unscaled_forecast = (forecast * scaler.scale_[i] + scaler.mean_[i]) * impact_coeff.values
                unscaled_actual = (y_test.values * scaler.scale_[i] + scaler.mean_[i]) * impact_coeff.values
                # compute usage percentage
                perc = np.mean(unscaled_actual / total_y_test.values, axis=1)
                partial_rmse = rmse_func(unscaled_forecast, unscaled_actual)
                partial_nrmse = partial_rmse / (unscaled_actual.max() - unscaled_actual.min())
                partial_smape = smape_func(unscaled_forecast, unscaled_actual)
                partial_smape = np.mean(partial_smape, axis=1)
                dict_results[impact][model_type][str(n_unit)][technology] = {'rmse': partial_rmse,
                                                                             'nrmse': partial_nrmse, 
                                                                             'smape': partial_smape,
                                                                             'percentage': perc}

            # total_impact_pred and total_impact_act are numpy arrays
            #breakpoint()
            total_impact_pred = total_impact_pred / total_y_test.values
            if j == 0 and k == 0:
                total_impact_act = total_impact_act / total_y_test.values
            total_rmse = rmse_func(total_impact_pred, total_impact_act)
            total_nrmse = total_rmse / (total_impact_act.max() - total_impact_act.min())
            total_smape = np.mean(smape_func(total_impact_pred, total_impact_act), axis=1)
            dict_results[impact][model_type][str(n_unit)]['composed'] = {'rmse': total_rmse,
                                                                         'nrmse': total_nrmse,
                                                                         'smape': total_smape}
            with open('./results/dict-results-composed-impact.pkl', 'wb') as fp:
                pickle.dump(dict_results, fp, protocol=pickle.HIGHEST_PROTOCOL)

            with open(file_path, 'a') as file:
                #file.write("".join(["\n---- FOLD ", str(fold), '----']))
                file.write("".join(["\nRMSE: ", str(total_rmse)]))
                file.write("".join(["\nNRMSE: ", str(total_nrmse)]))
                file.write("".join(["\nSMAPE: ", str(total_smape)]))

technologies_df = pd.read_pickle('data/technologiesDFFilled.pkl')
df_total_impact = compute_direct_impact(technologies_df, impact_df, impacts)
# scale dataframe
scaler_dir = StandardScaler()
scaled_values = scaler_dir.fit_transform(df_total_impact.values)
df_total_impact = pd.DataFrame(scaled_values, index=df_total_impact.index, columns=df_total_impact.columns)
dict_results = {}
for i, impact in enumerate(impacts):
    impact_coeffs = impact_df[impact_df['Impact category'] == impact]
    total_direct_impact = df_total_impact[impact]
    x, y = series_to_supervised(total_direct_impact, lags_in=7*24, lags_forecast=24)
    x = x.loc[indeces.index]
    y = y.loc[x.index]
    x_total = x.iloc[int(perc_train*len(x))+int(perc_vali*len(x)):]
    actual = y.loc[x_total.index]
    dict_results[impact] = {}
    for k, model_type in enumerate(model_types):
        dict_results[impact][model_type] = {}
        for j, n_unit in enumerate(n_units):
            message = "".join(['Processing ', impact, ' - ', model_type, ' - ', str(n_unit)])
            print(message)
            if model_type == 'LINEAR':
                LINEAR = True
                NN = False
                RNN = False
            elif model_type == 'NN':
                LINEAR = False
                NN = True
                RNN = False
            elif model_type == 'RNN':
                LINEAR = False
                NN = False
                RNN = True

            if LINEAR and j > 0:
                continue
            if LINEAR:
                results_path = os.path.join('results', 'linear-regression')
                models_path = os.path.join('models', 'linear-regression')
            elif NN:
                results_path = os.path.join('results', 'nn', "".join([str(n_unit), 'neurons']))
                models_path = os.path.join('models', 'nn', "".join([str(n_unit), 'neurons']))
            elif RNN:
                results_path = os.path.join('results', 'rnn', "".join([str(n_unit), 'neurons']))
                models_path = os.path.join('models', 'rnn', "".join([str(n_unit), 'neurons']))

            dict_results[impact][model_type][str(n_unit)] = {}
            if LINEAR:
                model_path = os.path.join(models_path, "".join([impact.replace(" ", "-"), "-model.pkl"]))
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
            else:
                model_path = os.path.join(models_path, impact.replace(" ", "-"))
                model = tfk.models.load_model(model_path)

            if RNN:
                x_rnn = x_total.values[:, :, np.newaxis]
                total_forecast = model.predict(x_rnn)
            else:
                total_forecast = model.predict(x_total.values)

            # total_impact_pred and total_impact_act are numpy arrays
            unscaled_forecast = total_forecast * scaler_dir.scale_[i] + scaler_dir.mean_[i] 
            unscaled_actual = actual.values * scaler_dir.scale_[i] + scaler_dir.mean_[i]
            # new unity measure
            unscaled_forecast = unscaled_forecast / total_y_test.values
            unscaled_actual = unscaled_actual / total_y_test.values
            rmse = rmse_func(unscaled_forecast, unscaled_actual)
            nrmse = rmse / (unscaled_actual.max() - unscaled_actual.min())
            #smape = smape_func((total_forecast * scaler_dir.scale_[i] + scaler_dir.mean_[i]), 
            #        (actual.values * scaler_dir.scale_[i] + scaler_dir.mean_[i]))
            smape = smape_func(unscaled_forecast, unscaled_actual)
            smape = np.mean(smape, axis=1)
            dict_results[impact][model_type][str(n_unit)]['direct'] = {'rmse': rmse,
                                                                       'nrmse': nrmse,
                                                                       'smape': smape}
            with open('./results/dict-results-direct-impact.pkl', 'wb') as fp:
                pickle.dump(dict_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
