import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from html_settings import html
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm

st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
st.markdown(html, unsafe_allow_html=True)

st.write('''
# Opady deszczu na terenie Szwajcarii
Poniższy notebook zawiera analizę danych przestrzennych zebranych w ramach SIC97 (szczegółowe informacje dostępne pod linkiem) **[LINK](https://wiki.52north.org/AI_GEOSTATS/AI_GEOSTATSData)**.

Powyższe dane zostały zebrane 8 maja 1986 roku i dotyczą opadów deszczu (z dokładnością do 1/10 mm) w wybranych 467 
lokalizacjach na obszarze Szwajcarii. W ramach SIC97 uczestnicy otrzymali pomiary ze 100 losowo wybranych 
lokalizacji, a ich zadaniem było oszacowanie opadów w pozostałych 367 miejscach. Oprócz danych dotyczących samych 
opadów, dostępny jest także plik DEM (z ang. Digital Elevation Model, zobacz też NMT - Numeryczny Model Terenu), 
który zawiera informacje o wysokości topograficznej powierzchni terenu z dokładnością do 1 km x 1 km.''')

df_training_set = pd.read_csv(
    'sic_obs.dat',
    index_col=0,
    names=["id", "x-coordinate", "y-coordinate", "rainfall"]
)

df_full_set = pd.read_csv(
    'sic_full.dat',
    index_col=0,
    names=["id", "x-coordinate", "y-coordinate", "rainfall"],
    skiprows=6
)

mask = (~df_full_set.index.isin(df_training_set.index))
df_testing_set = df_full_set[mask]
st.write(df_testing_set)

st.write('''Powyżej wczytany plik DEM opisuje wysokość topograficzną powierzchni terenu, dokładne informacje zawarte są w pliku SIC978_Readme.pdf.

A Digital Elevation Model (DEM): grid data were made available in which the values are the elevation (in meters)
Surfer Format is available surfdem.grd.
Normal ascii Arc/INFO format file is also available (demstd.grd) where we have
ncols 376 {number of columns}
nrows 253{ number of rows}
xllcorner -185556.375 {X lower left corner}
yllcorner -127261.5234 {Y lower left corner}
cellsize 1009.975 {cell size in meters}
NODATA_value -9999
Warto te dane zwizualizować i nanieść na nie granice Szwajcarii.
''')

xllcorner = -185556.375
xtrcorner = 194194.225
yllcorner = -127261.5234375
ytrcorner = 128262.152

cellsize = 1009.975

df = pd.read_csv(
    'demstd.grd',
    header=None,
    skiprows=6,
    sep='\s'
)

fig = sns.heatmap(df).get_figure()

height_above_sea_level = []

for _, row in df_training_set.iterrows():
    x_result = round((row['x-coordinate'] - xllcorner) / cellsize)
    y_result = 253 - round((row['y-coordinate'] - yllcorner) / cellsize)

    height_above_sea_level.append(df.loc[y_result, x_result])

df_training_set['elevation'] = height_above_sea_level

height_above_sea_level = []

for _, row in df_testing_set.iterrows():
    x_result = round((row['x-coordinate'] - xllcorner) / cellsize)
    y_result = 253 - round((row['y-coordinate'] - yllcorner) / cellsize)

    height_above_sea_level.append(df.loc[y_result, x_result])

df_testing_set['elevation'] = height_above_sea_level

df__training_features = df_training_set[['x-coordinate', 'y-coordinate', 'elevation']]
df_training_target = df_training_set['rainfall']

df_testing_features = df_testing_set[['x-coordinate', 'y-coordinate', 'elevation']]
df_testing_target = df_testing_set['rainfall']

scaler = StandardScaler()
df__training_features = scaler.fit_transform(df__training_features)

df_testing_features = scaler.transform(df_testing_features)

n_neighbors = 4
neighbors = KNeighborsRegressor(n_neighbors=5)
neighbors.fit(df__training_features, df_training_target)

y_test_pred = neighbors.predict(df_testing_features)
y_test_true = df_testing_target
score_knn = r2_score(y_test_true, y_test_pred)

pred_df = pd.DataFrame(y_test_pred, columns=['Values'])
pred_df = pred_df.astype(int)

regression_model = DecisionTreeRegressor(random_state=0, max_depth=3)
regression_model.fit(df__training_features, df_training_target)

y_test_pred_dt = regression_model.predict(df_testing_features)
y_train_pred = regression_model.predict(df__training_features)
score_dt = r2_score(df_testing_target, y_test_pred_dt)

regression_model_svm = svm.SVR(kernel="rbf", C=100, gamma=1.5, epsilon=0.1)
regression_model_svm.fit(df__training_features, df_training_target)

y_test_pred_svm = regression_model_svm.predict(df_testing_features)
score_svm = r2_score(df_testing_target, y_test_pred_svm)


def make_rainfall_plots():
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(18, 9))

    sns.scatterplot(
        x=df_testing_set['x-coordinate'],
        y=df_testing_set['y-coordinate'],
        s=df_testing_set['rainfall'] / 2,
        hue=df_testing_set['rainfall'],
        palette=sns.color_palette('ch:start=.2,rot=.3', as_cmap=True),
        zorder=3,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('True rainfall')

    sns.scatterplot(
        x=df_testing_set['x-coordinate'],
        y=df_testing_set['y-coordinate'],
        s=y_test_pred_svm / 2,
        hue=y_test_pred_svm,
        palette=sns.color_palette('ch:start=.2,rot=-.3', as_cmap=True),
        zorder=3,
        ax=axes[0, 1]
    ).set(title='SVM r2 score: ' + str(score_svm))

    sns.scatterplot(
        x=df_testing_set['x-coordinate'],
        y=df_testing_set['y-coordinate'],
        s=y_test_pred_dt / 2,
        hue=y_test_pred_dt,
        palette=sns.color_palette('ch:start=-.6,rot=-.3', as_cmap=True),
        zorder=3,
        ax=axes[1, 0]
    ).set(title='Decision Tree r2 score: ' + str(score_dt))

    sns.scatterplot(
        x=df_testing_set['x-coordinate'],
        y=df_testing_set['y-coordinate'],
        s=y_test_pred / 2,
        hue=y_test_pred,
        palette=sns.color_palette('ch:start=.7,rot=-.3', as_cmap=True),
        zorder=3,
        ax=axes[1, 1]
    ).set(title='KNN r2 score: ' + str(score_knn))
    plt.suptitle('Rainfall in Switzerland')

    return fig


st.pyplot(make_rainfall_plots())

# from matplotlib.legend import Legend
# from matplotlib import colors, cm
# import copy
#
# url = 'https://drive.google.com/file/d/1ISUJH33fJZZt-ug9b4WhdKxGSe7DNFSy/view?usp=sharing'
# url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
# df = pd.read_csv(
#     url,
#     header=None,
#     skiprows=6,
#     sep='\s+'
# )
#
# # wykorzystanie funkcji plt.imread() do wczytania pliku z granicami Szwajcarii
# url = 'https://drive.google.com/file/d/1NhmPj8DNifvkIZy5VA6ag9HNPFfB_bWc/view?usp=sharing'
# url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
# img = plt.imread(url)
#
# df_testing_set["x_coord"] = round((df_testing_set["x-coordinate"] - xllcorner) / cellsize)
# df_testing_set["y_coord"] = 253 - round((df_testing_set["y-coordinate"] - yllcorner) / cellsize)

option = st.selectbox(
    'Which model would you like to choose',
    ('KNN', 'Decision Tree', 'SVM'))

if option == 'KNN':

    def predict_user_point_SVM(my_points, x_input):
        # colors_land1 = plt.cm.terrain(np.linspace(0.35, 0.42, 200))
        # colors_land2 = plt.cm.terrain(np.linspace(0.45, 0.85, 2000))
        #
        # _colors = np.vstack((colors_land1, colors_land2))
        # cut_terrain_map = colors.LinearSegmentedColormap.from_list('cut_terrain', _colors)
        #
        # img_transparent_back = copy.copy(img)
        # img_transparent_back[img_transparent_back < 0.005] = np.nan
        #
        # with sns.plotting_context("talk"):
        #     fig, ax = plt.subplots(figsize=(15, 9))
        #     sns.heatmap(
        #         data=df,
        #         vmin=0,
        #         vmax=5000,
        #         cmap=cut_terrain_map,
        #         zorder=1
        #     )
        #
        #     ax.imshow(
        #         img_transparent_back,
        #         extent=[-30, 410, 265, -10],
        #         zorder=2,
        #         alpha=0.25,  # transparentność
        #
        #     )

        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(18, 9))

        sns.scatterplot(
            x=df_testing_set['x-coordinate'],
            y=df_testing_set['y-coordinate'],
            s=y_test_pred / 2,
            hue=y_test_pred,
            palette=sns.color_palette('ch:start=.7,rot=-.3', as_cmap=True),
        ).set(title='KNN r2 score: ' + str(score_knn))

        _, labels_init = axis.get_legend_handles_labels()
        n_labels = len(labels_init)

        legend1 = axis.legend()
        axis.add_artist(legend1)

        for my_point in my_points.values():
            my_point_scaled = scaler.transform(my_point)
            my_point_pred = neighbors.predict(my_point_scaled)
            my_point_pred = np.round(my_point_pred)
            sns.scatterplot(
                x=[my_point[0][0]],
                y=[my_point[0][1]],
                s=my_point_pred[0] / 2,
                label=my_point_pred[0]
            )

            # arrowprops = dict(
            #     arrowstyle="->",
            #     connectionstyle="angle, angleA = 0, angleB = 90,\
            #     rad = 10")
            #
            # plt.annotate(
            #     my_point_pred[0],
            #     xy=(my_point[0][0], my_point[0][1] - 10),
            #     xytext=(my_point[0][0], my_point[0][1] - 25000),
            #     arrowprops=arrowprops
            # )

        handles, labels = axis.get_legend_handles_labels()
        handles, labels = handles[n_labels:], labels[n_labels:]

        if handles:
            legend2 = axis.legend(
                handles=handles,
                labels=[float(label) for label in labels],
                loc='upper left'
            )
            axis.add_artist(legend2)

        return fig

if option == 'Decision Tree':

    def predict_user_point_SVM(my_points, x_input):
        # colors_land1 = plt.cm.terrain(np.linspace(0.35, 0.42, 200))
        # colors_land2 = plt.cm.terrain(np.linspace(0.45, 0.85, 2000))
        #
        # _colors = np.vstack((colors_land1, colors_land2))
        # cut_terrain_map = colors.LinearSegmentedColormap.from_list('cut_terrain', _colors)
        #
        # img_transparent_back = copy.copy(img)
        # img_transparent_back[img_transparent_back < 0.005] = np.nan
        #
        # with sns.plotting_context("talk"):
        #     fig, ax = plt.subplots(figsize=(15, 9))
        #     sns.heatmap(
        #         data=df,
        #         vmin=0,
        #         vmax=5000,
        #         cmap=cut_terrain_map,
        #         zorder=1
        #     )
        #
        #     ax.imshow(
        #         img_transparent_back,
        #         extent=[-30, 410, 265, -10],
        #         zorder=2,
        #         alpha=0.25,  # transparentność
        #
        #     )

        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(18, 9))

        sns.scatterplot(
            x=df_testing_set['x-coordinate'],
            y=df_testing_set['y-coordinate'],
            s=y_test_pred_dt / 2,
            hue=y_test_pred_dt,
            palette=sns.color_palette('ch:start=.6,rot=-.3', as_cmap=True),
        ).set(title='Decision Tree r2 score: ' + str(score_dt))

        _, labels_init = axis.get_legend_handles_labels()
        n_labels = len(labels_init)

        legend1 = axis.legend()
        axis.add_artist(legend1)

        for my_point in my_points.values():
            my_point_scaled = scaler.transform(my_point)
            my_point_pred = regression_model.predict(my_point_scaled)
            my_point_pred = np.round(my_point_pred)
            sns.scatterplot(
                x=[my_point[0][0]],
                y=[my_point[0][1]],
                s=my_point_pred[0] / 2,
                label=my_point_pred[0]
            )

            # arrowprops = dict(
            #     arrowstyle="->",
            #     connectionstyle="angle, angleA = 0, angleB = 90,\
            #     rad = 10")
            #
            # plt.annotate(
            #     my_point_pred[0],
            #     xy=(my_point[0][0], my_point[0][1] - 10),
            #     xytext=(my_point[0][0], my_point[0][1] - 25000),
            #     arrowprops=arrowprops
            # )

        handles, labels = axis.get_legend_handles_labels()
        handles, labels = handles[n_labels:], labels[n_labels:]

        if handles:
            legend2 = axis.legend(
                handles=handles,
                labels=[float(label) for label in labels],
                loc='upper left'
            )
            axis.add_artist(legend2)

        return fig

if option == 'SVM':

    def predict_user_point_SVM(my_points, x_input):
        # colors_land1 = plt.cm.terrain(np.linspace(0.35, 0.42, 200))
        # colors_land2 = plt.cm.terrain(np.linspace(0.45, 0.85, 2000))
        #
        # _colors = np.vstack((colors_land1, colors_land2))
        # cut_terrain_map = colors.LinearSegmentedColormap.from_list('cut_terrain', _colors)
        #
        # img_transparent_back = copy.copy(img)
        # img_transparent_back[img_transparent_back < 0.005] = np.nan
        #
        # with sns.plotting_context("talk"):
        #     fig, ax = plt.subplots(figsize=(15, 9))
        #     sns.heatmap(
        #         data=df,
        #         vmin=0,
        #         vmax=5000,
        #         cmap=cut_terrain_map,
        #         zorder=1
        #     )
        #
        #     ax.imshow(
        #         img_transparent_back,
        #         extent=[-30, 410, 265, -10],
        #         zorder=2,
        #         alpha=0.25,  # transparentność
        #
        #     )

        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(18, 9))

        sns.scatterplot(
            x=df_testing_set['x-coordinate'],
            y=df_testing_set['y-coordinate'],
            s=y_test_pred_svm / 2,
            hue=y_test_pred_svm,
            palette=sns.color_palette('ch:start=.2,rot=-.3', as_cmap=True),
        ).set(title='SVM r2 score: ' + str(score_svm))

        _, labels_init = axis.get_legend_handles_labels()
        n_labels = len(labels_init)

        legend1 = axis.legend()
        axis.add_artist(legend1)

        for my_point in my_points.values():
            my_point_scaled = scaler.transform(my_point)
            my_point_pred = regression_model_svm.predict(my_point_scaled)
            my_point_pred = np.round(my_point_pred)
            sns.scatterplot(
                x=[my_point[0][0]],
                y=[my_point[0][1]],
                s=my_point_pred[0] / 2,
                label=my_point_pred[0]
            )

            # arrowprops = dict(
            #     arrowstyle="->",
            #     connectionstyle="angle, angleA = 0, angleB = 90,\
            #     rad = 10")
            #
            # plt.annotate(
            #     my_point_pred[0],
            #     xy=(my_point[0][0], my_point[0][1] - 10),
            #     xytext=(my_point[0][0], my_point[0][1] - 25000),
            #     arrowprops=arrowprops
            # )

        handles, labels = axis.get_legend_handles_labels()
        handles, labels = handles[n_labels:], labels[n_labels:]

        if handles:
            legend2 = axis.legend(
                handles=handles,
                labels=[float(label) for label in labels],
                loc='upper left'
            )
            axis.add_artist(legend2)

        return fig


x_input = st.slider('X-coordinate: ', -175000, 200000, 12500, step=1000)
y_input = st.slider('Y-coordinate: ', -110000, 110000, 0, step=1000)
elevation_input = st.slider('Elevation: ', 0, 1000, 500, step=10)

if st.button('Apply'):
    if f'{x_input}, {y_input}, {elevation_input}' not in st.session_state:
        st.session_state[f'{x_input}, {y_input}, {elevation_input}'] = ((x_input, y_input, elevation_input),)
    st.pyplot(predict_user_point_SVM(st.session_state, x_input))
    # st.table(st.session_state)

else:
    if st.session_state:
        st.pyplot(predict_user_point_SVM(st.session_state, x_input))
    else:
        st.pyplot(predict_user_point_SVM(dict(), x_input))
    # st.table(st.session_state)
