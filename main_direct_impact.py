import tensorflow as tf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import adfuller
import os

SAVE_INDECES = False
computeAutocorrelation = False
checkMissingDays = False
creatingDF = False
pathDF = './data/technologiesDFFilled.pkl'
plotData = False
fillingDF = False
trainingModel = True
lags = 2000
plotAutoCorrelations = False
pathAnalysis = './time-series-analysis/analysis'
TSA = False
LINEAR = True
NN = False
RNN = False
resultsPath = None
modelsPath = None
#nUnit = 64


def plotACF(series, lags, pathSave):
    results, confInt = acf(series.values, nlags=lags, alpha=.05, fft=True)
    xArray = np.arange(0, results.shape[0])
    # concatenate [UP, DOWN] depend on xArray + xArray[::-1]
    fig = go.Figure(go.Scatter(x=xArray, y=results, mode='lines+markers', line=dict(color='darkslateblue')))
    xArray = np.concatenate([xArray, xArray[::-1]])
    up = confInt[:, 1] - results
    down = confInt[:, 0] - results
    fig.add_trace(go.Scatter(x=xArray, y=np.concatenate([up, down[::-1]]),
                             fill='toself', fillcolor='darkturquoise', line_color='darkturquoise', 
                             opacity=0.5, showlegend=False))
    fig.update_layout(title="".join([series.name, " autocorrelation"])) 
    fig.write_html("".join([pathSave, '/', series.name.replace(" ", "-"), '.html']))

def plotPACF(series, lags, pathSave):
    results, confInt = pacf(series.values, nlags=lags, alpha=.05)
    xArray = np.arange(0, results.shape[0])
    # concatenate [UP, DOWN] depend on xArray + xArray[::-1]
    fig = go.Figure(go.Scatter(x=xArray, y=results, mode='lines+markers', line=dict(color='darkslateblue')))
    xArray = np.concatenate([xArray, xArray[::-1]])
    up = confInt[:, 1] - results
    down = confInt[:, 0] - results
    fig.add_trace(go.Scatter(x=xArray, y=np.concatenate([up, down[::-1]]),
                             fill='toself', fillcolor='darkturquoise', line_color='darkturquoise', 
                             opacity=0.5, showlegend=False))
    fig.update_layout(title="".join([series.name, " partial autocorrelation"])) 
    fig.write_html("".join([pathSave, '/', series.name.replace(" ", "-"), '.html']))

def fillDF(df):
    # extract deltatime
    deltaTime = df.index.to_series().diff()
    # find deltas > 1 day
    endMissingPeriod = deltaTime[deltaTime.dt.days > 1]
    dfToAdd = pd.DataFrame()
    for i, missingDays in enumerate(endMissingPeriod):
        end = endMissingPeriod.index[i]
        # select same day in the previous period
        selectDay = df[(df.index < end) & (df.index.dayofweek == end.dayofweek)].sort_index(ascending=False)
        selectDay = selectDay.index[0]
        selectDateTime = datetime(int(selectDay.year), int(selectDay.month), int(selectDay.day),
                                  hour=0, minute=0, second=0)
        # select period
        selectDF = df[(df.index < selectDateTime) 
                      & (df.index > selectDateTime - missingDays)]
        # create new index 
        newDates = pd.date_range(start=end-missingDays+timedelta(hours=1), 
                                 end=end-timedelta(hours=1), freq='H')
        selectDF.index = newDates
        df = pd.concat([df, selectDF]).sort_index()
    return df

def createDF(data, keys, emptyDF):
    for key in keys:
        totalData = np.hstack((data[key]['y_TRAIN'].flatten(), data[key]['y_VALI'].flatten()))
        totalData = np.hstack((totalData, data[key]['y_TEST'].flatten()))
        emptyDF[key] = totalData
    return technologiesDF

def seriesToSupervised(df, lagsIn, lagsForecast):
    # name of the series considered
    varName = df.head().name
    cols, names = [], []
    # appending the series shifted by one on the right in order to have
    # columns from var(t-lagsIn), ..., var(t-1)
    for i in range(lagsIn, 0, -1):
        cols.append(df.shift(i))
        # column name
        names.append("".join([varName, "(t-", str(i), ")"]))
    # create new pandas dataframe
    dfX = pd.concat(cols, axis=1)
    dfX.columns = names
    # dropping row with not a number items (it means they are out of the interval)
    dfX = dfX.dropna()
    # taking only the rows with hour == 0 because we are forecsting only one time
    # at midnight the next day
    dfX = dfX[dfX.index.hour == 0]
    cols, names = [], []
    # appending the series shifted by minus one on the right in order to 
    # have columns from var(t), ..., var(t+lagsForecast)
    for i in range(0, lagsForecast):
        cols.append(df.shift(-i))
        if i == 0:
            names.append("".join([varName, "(t)"]))
        else:
            names.append("".join([varName, "(t+", str(i), ")"]))
    # same operations as before
    dfY = pd.concat(cols, axis=1)
    dfY.columns = names
    dfY = dfY.dropna()
    dfY = dfY[dfY.index.hour == 0][int(lagsIn/24):]
    dfX = dfX[dfX.index.isin(dfY.index)]
    return dfX, dfY

def rmseFunc(forecast, target):
    return np.sqrt(np.mean(np.power(forecast - target, 2)))

def smapeFunc(forecast, target):
    return np.mean(np.mean((np.abs(forecast - target) / (np.abs(forecast) + np.abs(target))) * 100, axis=1), axis=0)

def analyzeTimeSeries(series, pathAnalysis):
    print("".join(["--> Analise ", series.name, " time series"]))
    # analysis seasonality
    res = seasonal_decompose(series, model='add')
    fig = make_subplots(rows=4, shared_xaxes=True, 
                        subplot_titles=("".join([series.name, ' time series']), 'Trend', 'Seasonality', 'Residuals'))
    fig.add_trace(go.Scatter(x=series.index, y=series.values, showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=series.index, y=res.trend, showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=series.index, y=res.seasonal, showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=series.index, y=res.resid, showlegend=False), 4, 1)
    fig.write_html("".join([pathAnalysis, '/', series.name.replace(" ", "-"), '-seasonal-analysis.html']))
    # fft analysis seasonal components
    yf = fft(res.seasonal.values)
    N = res.seasonal.shape[0]
    y = 2.0/N * np.abs(yf[0:N//2])
    T=1/24 # samples are taken every hour, 1/24 of day
    xf = fftfreq(N, T)[:N//2]
    fig = go.Figure(go.Scatter(x=xf, y=y))
    fig.update_layout(title="".join([series.name.replace(" ", "-"), " Seasonal Component FFT"]),
                      xaxis_title='1 / day')
    fig.write_html("".join([pathAnalysis, '/fft/', series.name.replace(" ", "-"), '-seasonal-FFT.html']))
    # fft analysis trend components
    trendDropNa = res.trend.dropna()
    yf = fft(trendDropNa.values)
    N = trendDropNa.shape[0]
    y = 2.0/N * np.abs(yf[0:N//2])
    T=1/24 # samples are taken every hour, 1/24 of day
    xf = fftfreq(N, T)[:N//2]
    fig = go.Figure(go.Scatter(x=xf, y=y))
    fig.update_layout(title="".join([series.name.replace(" ", "-"), " Trend Component FFT"]),
                      xaxis_title='1 / day')
    fig.write_html("".join([pathAnalysis, '/fft/', series.name.replace(" ", "-"), '-trend-FFT.html']))
    # stationarity test
    # actual series
    testRes = adfuller(series, autolag='AIC')
    with open("".join([pathAnalysis, '/', series.name.replace(" ", "-"), "-stationary.test"]), 'a') as file:
        file.write("\n\n-------- Dickey-Fuller Test --------")
        file.write("\n Null Hypotesis: Non stationarity exist in data.")
        file.write("\n Test statistic must be less than T values at 1%, 5%, 10% to reject null hyotesis")
        file.write("\n P-value must be less than 0.05 (for 95% confidence interval)")
        file.write("".join(["\n\n\n--- Actual ", series.name, " time series ---"]))
        file.write("".join(["\nTest Statistic: ", str(testRes[0])]))
        file.write("".join(["\nP-value: ", str(testRes[1])]))
        file.write("".join(["\nNumber of lags used : ", str(testRes[2])]))
        file.write("".join(["\nNumber of observations used : ", str(testRes[3])]))
        file.write("".join(["\nT value at 1%: ", str(testRes[4]['1%'])]))
        file.write("".join(["\nT value at 5%: ", str(testRes[4]['5%'])]))
        file.write("".join(["\nT value at 10%: ", str(testRes[4]['10%'])]))
    # differentiated series
    series = series.diff().dropna()
    testRes = adfuller(series, autolag='AIC')
    with open("".join([pathAnalysis, '/', series.name.replace(" ", "-"), "-stationary.test"]), 'a') as file:
        file.write("".join(["\n\n\n--- Differentiated ", series.name, " time series ---"]))
        file.write("".join(["\nTest Statistic: ", str(testRes[0])]))
        file.write("".join(["\nP-value: ", str(testRes[1])]))
        file.write("".join(["\nNumber of lags used : ", str(testRes[2])]))
        file.write("".join(["\nNumber of observations used : ", str(testRes[3])]))
        file.write("".join(["\nT value at 1%: ", str(testRes[4]['1%'])]))
        file.write("".join(["\nT value at 5%: ", str(testRes[4]['5%'])]))
        file.write("".join(["\nT value at 10%: ", str(testRes[4]['10%'])]))

def computeDirectImpact(dfTech, dfImpacts, impacts):
    # assign to every column in dftech a parto of the string
    # in dfImpacts columns
#    matchTech = {'Biomass': 'biomass', 'Fossil Hard coal': 'hard coal', 'Fossil Coal-derived gas': 'coal gases', 
#                 'Fossil Gas': 'natural gas', 'Fossil Oil': 'HFO', 'Geothermal': 'geothermal', 'Hydro Pumped Storage': 'hydro',
#                 'Hydro': 'hydro', 'Solar': 'photovoltaic', 'Wind Onshore': 'wind'}
    # new impacts
    matchTech = {'Biomass': 'Biomass', 'Fossil Hard coal': 'Fossil Hard coal', 'Fossil Coal-derived gas': 'Fossil Coal-derived gas', 
                 'Fossil Gas': 'Fossil Gas', 'Fossil Oil': 'Fossil Oil', 'Geothermal': 'Geothermal', 'Hydro Pumped Storage': 'Hydro Pumped Storage',
                 'Hydro': 'Hydro', 'Solar': 'Solar', 'Wind Onshore': 'Wind Onshore'}
    dfTechImpact = pd.DataFrame()
    for impact in impacts:
        directImpact = 0
        for technology in dfTech.columns:
            impactName = matchTech[technology]
            directImpact += dfTech[technology] * dfImpacts[dfImpacts['Impact category'] == impact][impactName].values
        dfTechImpact[impact] = directImpact
    return dfTechImpact
    
if plotAutoCorrelations:
    technologiesDF = pd.read_pickle('./data/technologiesDF.pkl')
    scaler = StandardScaler()
    scaledValues = scaler.fit_transform(technologiesDF.values)
    technologiesDF = pd.DataFrame(scaledValues, index=technologiesDF.index, columns=technologiesDF.columns)
    pathACF = './plot/autocorrelations/original'
    pathPACF = './plot/partial-autocorrealtions/original'
    for technology in technologiesDF.keys():
        print("".join(['Computing correlations for ', technology, ' original']))
        plotACF(technologiesDF[technology], lags, pathACF)
        plotPACF(technologiesDF[technology], lags, pathPACF)
    technologiesDF = pd.read_pickle('./data/technologiesDFFilled.pkl')
    scaler = StandardScaler()
    scaledValues = scaler.fit_transform(technologiesDF.values)
    technologiesDF = pd.DataFrame(scaledValues, index=technologiesDF.index, columns=technologiesDF.columns)
    pathACF = './plot/autocorrelations/filled'
    pathPACF = './plot/partial-autocorrealtions/filled'
    for technology in technologiesDF.keys():
        print("".join(['Computing correlations for ', technology, ' filled']))
        plotACF(technologiesDF[technology], lags, pathACF)
        plotPACF(technologiesDF[technology], lags, pathPACF)

if creatingDF:
    # Create pandas df
    data = pd.read_pickle("./data/data_ITA_v5.pkl")
    totalIdx = []
    totalIdx.extend(data['y_TRAIN_index'])
    totalIdx.extend(data['y_VALI_index'])
    totalIdx.extend(data['y_TEST_index'])
    # list of difference between days
    timeDelta = list(map(lambda x, y: x - y, totalIdx[1:], totalIdx[:-1]))
    numberMissingDays = [0 if delta.days == 1 else 1 for delta in timeDelta]
    if checkMissingDays:
        # check if all day are present
        numberMissingDays = sum(numberMissingDays)
        print("There are " + str(numberMissingDays) + " missing days")
    technologies = ['Biomass', 'Fossil Hard coal', 'Fossil Coal-derived gas', 'Fossil Gas', 'Fossil Oil',
                    'Geothermal', 'Hydro Pumped Storage', 'Hydro', 'Solar', 'Wind Onshore']
    # Create list of datetime including hours
    dates =  []
    for day in totalIdx:
        for hour in range(24):
            dates.append(datetime(day.year, day.month, day.day, hour, 0))
    technologiesDF = pd.DataFrame(index=dates)
    technologiesDF = createDF(data, technologies, technologiesDF)
    technologiesDF.to_pickle("./data/technologiesDF.pkl")
elif fillingDF:
    technologiesDF = pd.read_pickle(pathDF)
    technologiesDF = fillDF(technologiesDF)
    technologiesDF.to_pickle("./data/technologiesDFFilled.pkl")
else:
    technologiesDF = pd.read_pickle(pathDF)

if plotData:
    for technology in technologiesDF.keys():
        fig = go.Figure(data=go.Scatter(x=technologiesDF.index, y=technologiesDF[technology]))
        fig.update_layout(title=technology + ' time series')
        fig.show()
        fig.write_html("./plot/time-series/" + technology.replace(" ", "-"))

if TSA:
    os.system("rm ./time-series-analysis/analysis/*.test")
    technologiesDF = pd.read_pickle('./data/technologiesDFFilled.pkl')
    for technology in technologiesDF.keys():
        analyzeTimeSeries(technologiesDF[technology], pathAnalysis)

if trainingModel:
# general for loop
    nUnits = [32, 64]
    modelTypes = ['LINEAR', 'NN', 'RNN']
    #impacts = ['Climate change', 'Particulate Matter']
    impacts = ['climate change total', 'respiratory effects, inorganics']

    for modelType in modelTypes:
        for j, nUnit in enumerate(nUnits):
            if modelType == 'LINEAR':
                LINEAR = True
                NN = False
                RNN = False
            elif modelType == 'NN':
                LINEAR = False
                NN = True
                RNN = False
            elif modelType == 'RNN':
                LINEAR = False
                NN = False
                RNN = True

            if LINEAR and j > 0:
                continue

            if LINEAR:
                resultsPath = './results/linear-regression/'
                modelsPath = './models/linear-regression/'
            elif NN:
                resultsPath = "".join(['./results/nn/', str(nUnit), 'neurons']) 
                modelsPath = "".join(['./models/nn/', str(nUnit), 'neurons']) 
            elif RNN:
                resultsPath = "".join(['./results/rnn/', str(nUnit), 'neurons']) 
                modelsPath = "".join(['./models/rnn/', str(nUnit), 'neurons']) 
            try:
                os.stat(resultsPath)
            except:
                os.mkdir(resultsPath)
            try:
                os.stat(modelsPath)
            except:
                os.mkdir(modelsPath)
            technologiesDF = pd.read_pickle('data/technologiesDFFilled.pkl')
            #impactDF = pd.read_csv('data/climate-change-impacts.csv')
            impactDF = pd.read_csv('data/use-phase-unitary-impacts.csv')
            impactDF = computeDirectImpact(technologiesDF, impactDF, impacts)
            secDay = 24*60*60
            secWeek = 7*secDay
                # create sinusoidal signals for week and hour
            dateSeconds = technologiesDF.index.map(datetime.timestamp)
#    technologiesDF['Day sin'] = np.sin(dateSeconds * (2 * np.pi / secDay))
#    technologiesDF['Day cos'] = np.cos(dateSeconds * (2 * np.pi / secDay))
#    technologiesDF['Week sin'] = np.sin(dateSeconds * (2 * np.pi / secWeek))
#    technologiesDF['Week cos'] = np.cos(dateSeconds * (2 * np.pi / secWeek))
#technologiesDF['Weekday'] = technologiesDF.index.dayofweek
#dummies = pd.get_dummies(technologiesDF['Weekday'])
#technologiesDF = pd.concat([technologiesDF, dummies], axis=1)
#technologiesDF = technologiesDF.drop('Weekday')
# Prepare data for linear regression
# From autocorrelations info take an history of 7 days
# using k-fold cross validation
            #kfSplit = TimeSeriesSplit(n_splits=6, test_size = 7*4)
            percTrain = 0.8
            percVali = 0.1
            percTest = 0.1
# scale dataframe
            scaler = StandardScaler()
            scaledValues = scaler.fit_transform(impactDF.values)
            impactDF = pd.DataFrame(scaledValues, index=impactDF.index, columns=impactDF.columns)
            for i, impact in enumerate(impactDF.keys()):
                # select technology
                df = impactDF[impact]
                x, y = seriesToSupervised(df, lagsIn=7*24, lagsForecast=24)
                #fold = 0
                #filePath = "".join([resultsPath, impact.replace(" ", "-"),'-metrics.log'])
                #with open(filePath, 'a') as file:
                #    file.write("".join([impact, " model"]))
                r2Tot = 0
                rmseTot = 0
                smapeTot = 0
                r2TotRescaled = 0
                rmseTotRescaled = 0
                smapeTotRescaled = 0
                MAX_EPOCHS = 200
                patience = 30
                filePath = os.path.join(resultsPath, "".join([impact.replace(" ", "-"),'-metrics.log']))
                try:
                    os.remove(filePath)
                except Exception:
                    pass
                with open(filePath, 'a') as file:
                    file.write("".join([impact, " model"]))
                
                #for periodTrain, periodVali in kfSplit.split(x):
                    #fold += 1
                    #xTrain = x.iloc[periodTrain]
                    #yTrain = y.iloc[periodTrain]
                    #xVali = x.iloc[periodVali]
                    #yVali = y.iloc[periodVali]
                if SAVE_INDECES and i == 0:
                    x = x.sample(frac=1)
                    y = y.loc[x.index]
                    x.to_pickle('./data/indeces.pkl')
                else:
                    indeces = pd.read_pickle('./data/indeces.pkl')
                    x = x.loc[indeces.index]
                    y = y.loc[x.index]

                xTrain = x.iloc[0:int(percTrain*len(x))]
                xVali = x.iloc[int(percTrain*len(x)):int(percTrain*len(x))+int(percVali*len(x))]
                xTest = x.iloc[int(percTrain*len(x))+int(percVali*len(x)):]
                yTrain = y.loc[xTrain.index]
                yVali = y.loc[xVali.index]
                yTest = y.loc[xTest.index]
                if LINEAR:
                    # fit linear regression
                    model = LinearRegression().fit(xTrain.values, yTrain.values)
                if NN:
                    model = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=nUnit, activation='relu'),
                    tf.keras.layers.Dense(units=24)
                    ])
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                      patience=patience,
                                                                      mode='min',
                                                                      restore_best_weights=True)

                    model.compile(loss=tf.losses.MeanSquaredError(),
                                  optimizer=tf.optimizers.Adam(),
                                  metrics=[tf.metrics.MeanAbsoluteError()])

                    model.fit(xTrain.values, yTrain.values, batch_size=32, epochs=MAX_EPOCHS, 
                              validation_data=(xVali.values, yVali.values), callbacks=[early_stopping])
                if RNN:
#                    model = tf.keras.Sequential([
#                    tf.keras.layers.LSTM(units=nUnit, activation='relu', return_sequences=False),
#                    tf.keras.layers.Dense(units=24)
#                    ])
                    input_series = tf.keras.Input(shape=(7*24,1))
                    pred, stateh, statec = tf.keras.layers.LSTM(units=nUnit, activation='tanh', 
                                                       return_sequences=False, return_state=True)(input_series)

                    concat = tf.concat([pred, stateh], axis=1)
                    pred_out = tf.keras.layers.Dense(units=24)(concat)
                    model = tf.keras.Model(inputs=input_series, outputs=pred_out)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                      patience=patience,
                                                                      mode='min')
                    model.compile(loss=tf.losses.MeanSquaredError(),
                                  optimizer=tf.optimizers.Adam(),
                                  metrics=[tf.metrics.MeanAbsoluteError()])

                    xTrainRNN = xTrain.values[:, :, np.newaxis]
                    yTrainRNN = yTrain.values[:, :, np.newaxis]
                    xValiRNN = xVali.values[:, :, np.newaxis]
                    yValiRNN = yVali.values[:, :, np.newaxis]
                    xTestRNN = xTest.values[:, :, np.newaxis]
                    yTestRNN = yVali.values[:, :, np.newaxis]
                    model.fit(xTrainRNN, yTrainRNN, batch_size=32, epochs=MAX_EPOCHS, 
                              validation_data=(xValiRNN, yValiRNN), callbacks=[early_stopping])
                if RNN:
                    forecast = model.predict(xTestRNN)
                else:
                    forecast = model.predict(xTest.values)
                r2 = r2_score(forecast, yTest.values)
                rmse = rmseFunc(forecast, yTest.values)
                smape = smapeFunc(forecast, yTest.values)
                r2Tot += r2
                #r2Tot = r2Tot/2
                rmseTot += rmse
                #rmseTot = rmseTot/2
                smapeTot += smape
                #smapeTot = smapeTot/2
                with open(filePath, 'a') as file:
                    #file.write("".join(["\n---- FOLD ", str(fold), '----']))
                    file.write("".join(["\nR2: ", str(r2)]))
                    file.write("".join(["\nRMSE: ", str(rmse)]))
                    file.write("".join(["\nSMAPE: ", str(smape)]))
                r2 = r2_score(forecast*scaler.scale_[i] + scaler.mean_[i], 
                              yTest.values*scaler.scale_[i] + scaler.mean_[i])
                rmse = rmseFunc(forecast*scaler.scale_[i] + scaler.mean_[i], 
                                yTest.values*scaler.scale_[i] + scaler.mean_[i])
                smape = smapeFunc(forecast*scaler.scale_[i] + scaler.mean_[i], 
                                  yTest.values*scaler.scale_[i] + scaler.mean_[i])
                r2TotRescaled += r2
                #r2TotRescaled = r2TotRescaled/2
                rmseTotRescaled += rmse
                #rmseTotRescaled = rmseTotRescaled/2
                smapeTotRescaled += smape
                #smapeTotRescaled = smapeTotRescaled/2
                with open(filePath, 'a') as file:
                    file.write("".join(["\nR2 RESCALED: ", str(r2)]))
                    file.write("".join(["\nRMSE RESCALED: ", str(rmse)]))
                    file.write("".join(["\nSMAPE RESCALED: ", str(smape)]))

#        examples = np.random.randint(0, len(xVali), 3)
#        fig = make_subplots(3 , 1)
#        for i, example in enumerate(examples):
#            xPlot = xVali.iloc[example]
#            yPlot = yVali.iloc[example]
#            if RNN:
#                xRNN = xPlot.values[np.newaxis, :, np.newaxis]
#                forecast = model.predict(xRNN)
#            else:
#                forecast = model.predict(xPlot.values[np.newaxis, :])
#            #print(forecast)
#            fig.add_trace(go.Scatter(y=yPlot, mode='lines+markers'), i+1, 1)
#            fig.add_trace(go.Scatter(y=forecast[0, :], mode='markers'), i+1, 1)
#        fig.show()
                    
                with open(filePath, 'a') as file:
                    file.write("".join(['\n---- General metrics ----']))
                    file.write("".join(["\nR2: ", str(r2Tot)]))
                    file.write("".join(["\nRMSE: ", str(rmseTot)]))
                    file.write("".join(["\nSMAPE: ", str(smapeTot)]))
                    file.write("".join(["\nR2 RESCALED: ", str(r2TotRescaled)]))
                    file.write("".join(["\nRMSE RESCALED: ", str(rmseTotRescaled)]))
                    file.write("".join(["\nSMAPE RESCALED: ", str(smapeTotRescaled)]))
                if LINEAR:
                    fileModel = os.path.join(modelsPath, "".join([impact.replace(" ", "-"), "-model.pkl"]))
                    pickle.dump(model, open(fileModel, 'wb'))
                else:
                    model.save(os.path.join(modelsPath, impact.replace(" ", "-")))
