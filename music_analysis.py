import pyarrow.parquet as pa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px

st.title(':orange[💮Classifying the Audio Genres]🌞')
st.write(":green[**|GUVI|**]")
#) reading parquet file
table = pa.read_table(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\final_project\0000 (1).parquet") 
#table

#) to count columns and rows
#table.shape

#) converting table to pandas
df = table.to_pandas() 
#df.head()

#) to get info of dataframe
#df.info

#) to get describe of dataframe
#df.describe

#) to get column name
#df.columns

#) to check the columns with data
#df[['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name']]

#) unique in artist column
#df['artists'].unique()

#) unique in album column
#df['album_name'].unique()

#) unique in track column
#df['track_name'].unique()

#) to check the columns with data
#df[['popularity','duration_ms','explicit','danceability','energy']]

#) to check the columns with data
#df[['key', 'loudness', 'mode', 'speechiness', 'acousticness','instrumentalness']]

#) to check the columns with data
#df[['liveness', 'valence', 'tempo', 'time_signature','track_genre']]

#) to get unique in track_genre column 
#df['track_genre'].unique()

#df.columns

st.subheader(':green[(a) Data analysis🌞]')
selectBox=st.selectbox(":blue[**data analysis:**] ", ['dataframe',
                                                            'scatterplot1',
                                                            'scatterplot2',
                                                            'scatterplot3',
                                                            'scatterplot4',
                                                            ])
#)showing original dataframe
if selectBox == 'dataframe':
    st.markdown("\n#### :red[1.1 original dataframe]\n")
    data = df.head(5)
    st.dataframe(data.style.applymap(lambda x: 'color:purple'))

#) scatterplot 1 - duration vs popularity
elif selectBox == 'scatterplot1':
    st.markdown("\n#### :green[2.1 scatterplot 1 - duration vs popularity]\n")
    fig = px.scatter(
    df,
    x="duration_ms",
    y="popularity",
    color='explicit',
    log_x=True,
    size_max=60,
    )
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
    #) observation 1:
    st.markdown("\n#### :green[2.2 observation 1]\n")
    st.write("• people mostly like only less time duration music compared to the lengthy music")
    st.write("• lyrics not required for popularity")
    st.write("• people like both melody and beat songs")

#) scatterplot 2 - danceability vs popularity
elif selectBox == 'scatterplot2':
    st.markdown("\n#### :violet[3.1 scatterplot 2 - danceability vs popularity]\n")
    fig = px.scatter(
    df,
    x="danceability",
    y="popularity",
    color='explicit',
    log_x=True,
    size_max=60,
    )
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
    #) observation 2:
    st.markdown("\n#### :violet[3.2 observation 2]\n")
    st.write("• danceability influences popularity. people like more dancability songs")

#) scatterplot 3 - energy vs popularity
elif selectBox == 'scatterplot3':
    st.markdown("\n#### :violet[4.1 scatterplot 3 - energy vs popularity]\n")
    fig = px.scatter(
    df,
    x="energy",
    y="popularity",
    color='mode',
    log_x=True,
    size_max=60,
    )
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
    #)observation 3:
    st.markdown("\n#### :violet[4.2 observation 3]\n")
    st.write("• people like (above average) medium energy songs. high noisy songs are not so much popular")
    st.write("• very low tone songs low popularity score")

#) scatterplot 4 - loudness vs popularity
elif selectBox == 'scatterplot4':
    st.markdown("\n#### :violet[5.1 scatterplot 4 - loudness vs popularity]\n")
    fig = px.scatter(
    df,
    x="loudness",
    y="popularity",
    color='mode',
    log_x=True,
    size_max=60,
    )
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
    #) observation 4
    st.markdown("\n#### :violet[5.2 observation 4]\n")
    st.write("• low loud songs are not popular. high loudness songs are famous")

#) to check the null values in pandas
#df.isna().sum()

#) filling the null values
df = df.ffill()
df = df.bfill()

#) to check the null values in pandas
#df.isna().sum()

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df.drop(['Unnamed: 0','track_id','artists','album_name','track_name'],axis=1,inplace=True)
df = pd.get_dummies(df,['track_genre'])
X = df.drop(['popularity'],axis=1)
y = df['popularity']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

st.subheader(":violet[(b) ML model]")
#) linear regression
if (st.button(':blue[linear regression]')):
    st.markdown("\n#### :violet[linear regression]\n")
    st.error(f"********{type(model).__name__}***********")
    st.info("********Train data*********")
    st.write(mean_squared_error(y_train,train_pred))
    st.info("********Test data*********")
    st.write(mean_squared_error(y_test,test_pred))
    
