import streamlit as st
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import xgboost
import joblib 
import sys
import logging
import os
from typing import Union
from google.cloud import storage
from io import StringIO
import altair as alt
import openpyxl
import validar_preprocesar_predecir_organizarrtados
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

# import clases_valid_cartera as val
# import clases_prepro as pre
# import clases_rdoMod as rdo
from sklearn.preprocessing import OneHotEncoder


#-------------------------------------------------------IAP GCP
app = Flask(__name__)

#----------------------------------------------------B
# Configure this environment variable via app.yaml
#CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']

@app.route('/')
def index() -> str:
    return """
<form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="datos" name="datos">
    <input type="submit">
</form>
"""
@app.route('/upload', methods=['POST'])
def upload(csvdata, bucketname, blobname):
    client = storage.Client()
    bucket = client.get_bucket(bucketname)
    blob = bucket.blob(blobname)
    blob.upload_from_string(csvdata)
    gcslocation = 'gs://{}/{}'.format(bucketname, blobname)
    logging.info('Uploaded {} ...'.format(gcslocation))
    return gcslocation

@app.errorhandler(500)
def server_error(e: Union[Exception, int]) -> str:
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500
#-----------------------------------------------------

CERTS = None
AUDIENCE = None

def certs():
    """Returns a dictionary of current Google public key certificates for
    validating Google-signed JWTs. Since these change rarely, the result
    is cached on first request for faster subsequent responses.
    """
    import requests

    global CERTS
    if CERTS is None:
        response = requests.get(
            'https://www.gstatic.com/iap/verify/public_key'
        )
        CERTS = response.json()
    return CERTS

def get_metadata(item_name):
    """Returns a string with the project metadata value for the item_name.
    See https://cloud.google.com/compute/docs/storing-retrieving-metadata for
    possible item_name values.
    """
    import requests

    endpoint = 'http://metadata.google.internal'
    path = '/computeMetadata/v1/project/'
    path += item_name
    response = requests.get(
        '{}{}'.format(endpoint, path),
        headers={'Metadata-Flavor': 'Google'}
    )
    metadata = response.text
    return metadata

def audience():
    """Returns the audience value (the JWT 'aud' property) for the current
    running instance. Since this involves a metadata lookup, the result is
    cached when first requested for faster future responses.
    """
    global AUDIENCE
    if AUDIENCE is None:
        project_number = get_metadata('numeric-project-id')
        project_id = get_metadata('project-id')
        AUDIENCE = '/projects/{}/apps/{}'.format(
            project_number, project_id
        )
    return AUDIENCE

def validate_assertion(assertion):
    """Checks that the JWT assertion is valid (properly signed, for the
    correct audience) and if so, returns strings for the requesting user's
    email and a persistent user ID. If not valid, returns None for each field.
    """
    from jose import jwt
    try:
        info = jwt.decode(
            assertion,
            certs(),
            algorithms=['ES256'],
            audience=audience()
            )
        return info['email'], info['sub']
    except Exception as e:
        print('Failed to validate assertion: {}'.format(e), file=sys.stderr)
        return None, None

def download_excel(df_v,nombre='LogErrores'):
    df_v.to_excel(nombre+'.xlsx', index=False)
    with open(nombre+'.xlsx', 'rb') as file:
        contents = file.read()
    st.download_button(label='Descargar '+nombre, data=contents, file_name=nombre+'.xlsx')

@app.route('/', methods=['GET'])
def say_hello():
    from flask import request
    assertion = request.headers.get('X-Goog-IAP-JWT-Assertion')
    email, id = validate_assertion(assertion)
    page = "<h1>Hello {}</h1>".format(email)
    return page

#------------------------------------------------------

def generar_graficos(df_t, configuraciones,mayus=True):
                        for config in configuraciones:
                            df_group = df_t.groupby(by=config['groupby'], as_index=True)['NIT9'].count()
                            df_group = pd.DataFrame(df_group)
                            # st.write(df_group)
                            
                            if mayus ==True:
                                df_group = df_group.reindex(config['order'])   # Reordenar el DataFrame según el orden deseado
                            else:
                                df_group.sort_values(by='NIT9', ascending = False, inplace=True)       #Ordenar de mayor a menor

                            df_group.reset_index(inplace=True, drop=False) # Extrae indice a columna
                            df_group.dropna(inplace=True)    
                            df_group.reset_index(inplace=True, drop=True)
                            # st.write(df_group)

                            df_group.rename({'NIT9': 'Cantidad', config['groupby']: config['y_axis']}, axis=1, inplace=True)
                            df_group['Cantidad'] = pd.to_numeric(df_group['Cantidad'])

                            if mayus ==True:
                                keys = config['order']
                                values = config['order_f']

                                diccionario = dict(zip(keys, values))

                                df_group[config['y_axis']] = df_group[config['y_axis']].replace(diccionario)
  
                            df_group[config['y_axis']] = pd.Categorical(df_group[config['y_axis']], ordered=True)

                            df_group['Porcentaje'] = df_group['Cantidad'] / df_group['Cantidad'].sum() * 100
                            df_group['Porcentaje'] = df_group['Porcentaje'].round(2)
                            df_group['Porcentaje'] = df_group['Porcentaje'].apply(lambda x: ' {:.2f}%'.format(x))
                            
                            # st.write(df_group)
                            

                            if mayus==True:
                                bar = alt.Chart(df_group).mark_bar().encode(
                                    x=alt.X('Cantidad', axis=alt.Axis(title=config['x_axis_title'])),
                                    y=alt.Y(config['y_axis'] + ":N", sort=list(df_group[config['y_axis']]),axis=alt.Axis(ticks=False)),
                                    tooltip=[config['y_axis']+":N",'Cantidad:Q', 'Porcentaje:O'],
                                    text=alt.Text('Porcentaje:N')
                                ).configure_mark(color='#311557').configure_view(fill="none").configure_axis(grid=False)
                            else:
                                bar = alt.Chart(df_group).mark_bar().encode(
                                    x=alt.X('Cantidad', axis=alt.Axis(title=config['x_axis_title'])),
                                    # y=config['y_axis'] + ":N",
                                    y=alt.Y(config['y_axis'] + ":N", sort=list(df_group[config['y_axis']]),axis=alt.Axis(ticks=False)),
                                    tooltip=[config['y_axis']+":N",'Cantidad:Q', 'Porcentaje:O'],
                                    text=alt.Text('Porcentaje:N')
                                ).configure_mark(color='#311557').configure_view(fill="none").configure_axis(grid=False)


                            config['col'].altair_chart(bar, use_container_width=True, theme="streamlit")
                            st.write("")


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_excel()#.encode('utf-8')

def main():
    
    #model=''
    # Se carga el modelo
   # if model=='':
    #    with open(MODEL_PATH, 'rb') as file:
     #       model = joblib.load(file)
    
    # Configura titulo e icon de pagina        
    st.set_page_config(page_title="Modelo Analítico Enel X", page_icon="img/Icono.ico", layout="wide")
    
    # st.markdown("""
    # <style>
    #     #MainMenu, header, footer {visibility: hidden;}
    # </style>
    # """,unsafe_allow_html=True)
    
    # Agregar estilo personalizado desde el archivo CSS
    # Leer el contenido del archivo CSS
    css = open('styles.css', 'r').read()

    # Agregar estilo personalizado
    st.markdown(
        f'<style>{css}</style>',
        unsafe_allow_html=True )




    # Variable que controla la visibilidad de la imagen
    b=False
    vista0,vista1,vista2,vista3 = st.tabs(['Inicio',"Resultado múltiples clientes", "Reporte descriptivo", "Resultado modelo unitario"])

    if not b:
        with vista0:
            # Verifica el valor de b
            # Muestra la imagen
            # image = 'img/Entorno_enel.jpg'
            image = 'img/Ciudad_Enel.jpg'
            st.image(image)




    # st.image("img/Entorno_enel.jpg" ,use_column_width=True)#
    # st.empty()

    # Menú y logo
    st.sidebar.image("img/logo3.png")
    st.sidebar.write("")

    #Estilo botón
    st.markdown("""
            <style>
            div.stButton > button:hover {
                background-color:#f0f2f6;
                color:#461e7d
            }
            </style>""", unsafe_allow_html=True)

    #a, b, c = False, False, False

    
     
    with st.sidebar.expander("MODELO MÚLTIPLES CLIENTES ", expanded = False):

        datos = st.file_uploader("Subir archivos: ", type = ["xlsx"])
        # b=False
        if datos is not None:
           
            dataframe = pd.read_excel(datos,index_col=0)
            # st.write(dataframe)

            #subir archivo al bucket en gcloud
            #urlarchivo = upload(bytes_data,CLOUD_STORAGE_BUCKET,'datos')
            try:
                dataframe['FECHACONSTITUCION']=dataframe['FECHACONSTITUCION'].astype('datetime64[ns]')
                dataframe['NUMERODEEMPLEADOS']=dataframe['NUMERODEEMPLEADOS'].astype('int64')
            except: pass
            # Validación archivo
            ob = validar_preprocesar_predecir_organizarrtados.Modelos_2(dataframe)
            df_v, text, final_flag = ob.Validar_todo()
            
            if final_flag == False:
                st.write(df_v)
                download_excel(df_v)
            else:     
                st.write(text)
                b = st.button("Ejecutar Modelo",type="primary")
                


    with st.sidebar.expander("MODELO UNITARIO ", expanded = False):
        # Lectura de datos
        nit = st.number_input("Digite el número del Nit",min_value=1000000,max_value=99999999999)
        actEcon = st.text_input("Actividad económica",value='Administración Empresarial')
        tamEmp = st.selectbox("Tamaño de la empresa",['Gran Empresa','Mediana Empresa','Pequeña Empresa'])
        flegal = st.selectbox("Forma Legal",['SAS', 'LTDA', 'SA', 'ESAL', 'SUCURSALEXTRANJERA', 'SCA','UNDEFINED', 'SCS', 'PERSONANATURAL'])
        numEmpl = st.number_input("Número de empleados",min_value=1,step=1)
        activos = st.number_input("Activos Totales")
        ingresosOp = st.number_input("Total Ingresos Operativos")
        TotPatr = st.number_input("Total Patrimonio")
        ganDespImpto = st.number_input("Ganancias después de Impuestos")
        fecha_constitucion = st.date_input("Fecha de constitución", min_value=date(1000, 1, 1), max_value=date.today())
        consprom = st.number_input("Consumo promedio kWh",min_value=0)

        
        # button
        c = st.button("Ejecutar Modelo 1",type="primary")
        dataframe_u = pd.DataFrame({'NIT9':nit,
                                    'ACTIVIDADPRINCIPAL(EMIS)':actEcon,
                                    'TAMANOEMPRESA':tamEmp,
                                    'FORMALEGAL':flegal,
                                    'NUMERODEEMPLEADOS':numEmpl,
                                    'ACTIVOSTOTALES':activos,
                                    'TOTALINGRESOOPERATIVO':ingresosOp,
                                    'TOTALDEPATRIMONIO':TotPatr,                                     
                                    'GANANCIASDESPUESDEIMPUESTOS':ganDespImpto,
                                    'FECHACONSTITUCION':fecha_constitucion,
                                    'CONSPROM':consprom}, index=[1])
        
        try:
            dataframe_u['FECHACONSTITUCION']=dataframe_u['FECHACONSTITUCION'].astype('datetime64[ns]')
            dataframe_u['NUMERODEEMPLEADOS']=dataframe_u['NUMERODEEMPLEADOS'].astype('int64')
        except:
            pass
        if c == True:
            # Si b es True, ocultar la primera vista

            # Crear un espacio vacío
            # placeholder = st.empty()
            # # Actualizar el espacio vacío con la segunda vista
            # placeholder.tabs(["Resultado Múltiples Clientes", "Reporte Descriptivo", "Resultado Modelo Unitario"])#, index=1


            # placeholder = st.empty()
            # Actualiza el espacio vacío para eliminar la imagen
            # placeholder.empty()
            
            ob_u = validar_preprocesar_predecir_organizarrtados.Modelos_2(dataframe_u)
            # df_v_u, text_u, final_flag_u = ob_u.Validar_todo()

            # if final_flag_u == False:
            #     st.write(df_v_u)
            #     download_excel(df_v_u)
            # else:     
            #     st.write(text_u)
            #     pass
        else:
            pass


        #with st.sidebar.expander("CUSTOMER JOURNEY ", expanded = False):
        #    a = st.button("Visualizar",type="primary")        

    if b == True:
         # Crear un espacio vacío
        # placeholder = st.empty()
        # # Actualizar el espacio vacío con la segunda vista
        # placeholder.tabs(["Resultado Múltiples Clientes", "Reporte Descriptivo", "Resultado Modelo Unitario"])#, index=1

        # st.markdown("<h1 style='text-align: center;'>Modelo Predictivo</h1>", unsafe_allow_html=True)
        # st.write("")
        # vista1,vista2,vista3 = st.tabs(["Resultado Múltiples Clientes", "Reporte Descriptivo", "Resultado Modelo Unitario"])

        with vista1:    # Modelo Multiples Clientes
            try:                  
                #st.write("")
                #st.write("")
                st.subheader("Top 3 de recomendaciones")
                col111,col113 = st.columns(spec=[1,1.75]) #col112, 
                #st.write(dataframe.head())

                Xi,Xf = ob.predict_proba()
                # st.write(Xf)

                # configuraciones = [
                #     {
                #         'groupby': 'Producto_1',
                #         'count_col': 'NIT9',
                #         'x_axis_title': 'Registros',
                #         'y_axis': 'Primer recomendación',
                #         'col': col112,
                #         # 'order': ['SINCATALOGAR','MENORA5000', 'ENTRE5000Y10000', 'ENTRE10000Y55000',  'MAYORA55000'],
                #         # 'order_f':['Sin catalogar','Menor a 5000',  'Entre 5000 y 10000','Entre 10000 y 55000',   'Mayor a 55000']
                #     }
                #     ]


                hm_df = pd.DataFrame({'index':['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA', 'FIBRA_OPTICA',
                                                'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']})
                
                # productos = ['Producto_1','Producto_2','Producto_3','Producto_4',
                #         'Producto_5','Producto_6','Producto_7','Producto_8'] 
                productos = ['Producto_1','Producto_2','Producto_3'] # Solo 3 primeras
                
                for i in productos:
                    hm_df=pd.merge(hm_df, pd.DataFrame(Xf[i].value_counts(dropna=False)).reset_index(drop=False),how='outer', on='index')


                # Suma # primeras predicciones
                df_tmp = pd.DataFrame(hm_df['index'].copy())
                df_tmp.rename({'index':'Productos'},axis=1, inplace=True)
                df_tmp['Top 3'] = hm_df.sum(axis=1)
                
                df_tmp['Porcentaje'] = df_tmp['Top 3'] / df_tmp['Top 3'].sum() * 100
                df_tmp['Porcentaje'] = df_tmp['Porcentaje'].round(2)
                # df_tmp['Porcentaje'] = df_tmp['Porcentaje'].apply(lambda x: ' {:.2f}%'.format(x))

                #
                keys = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA', 
                        'FIBRA_OPTICA','REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']
                values = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS DE CARGA', 
                          'FIBRA OPTICA', 'REDES ELECTRICAS', 'ILUMINACION', 'CUENTAS NUEVAS']
                diccionario = dict(zip(keys, values))

                df_tmp['Productos'] = df_tmp['Productos'].replace(diccionario) # Corrijo nombre de los productos

                # Obtener la paleta de colores 'Purples'
                colors = plt.cm.Purples(range(256))
                # Seleccionar los tres tonos deseados
                C = [colors[80], colors[170], colors[255]]

                # fig, axes = plt.subplots(2, 4, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [1, 1,1,]})
                fig = plt.figure()
                gs = GridSpec(2, 4)

                # ################################ Gráfico de barras
                # # Definir el orden deseado de las barras
                # orden = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS DE CARGA', 
                #         'FIBRA OPTICA', 'REDES ELECTRICAS', 'ILUMINACION', 'CUENTAS NUEVAS']
                # orden = list(reversed(orden))
                # # Ordenar el DataFrame según el orden deseado
                # df_tmp = df_tmp.set_index('Productos').loc[orden].reset_index()
                # # Crear los gráficos para la primera fila

                # ax2 = fig.add_subplot(gs[1, 3])#axes[1,1]
                # bar = ax2.barh(df_tmp['Productos'], df_tmp['Top 3'], color="#311557")  # Cambiar el color a purple
                # ax2.set_xlabel('Top 3')
                # ax2.set_yticklabels('')#df_tmp['Productos'], fontsize=10
                # ax2.set_xticklabels('')
                # ax2.set_xticks([])
                # ax2.margins(y=0.01)

                # ax2.tick_params(axis='y', left=False)  # Quitar los ticks del eje y
                # ax2.spines['top'].set_visible(False)  # Quitar la línea superior del marco
                # ax2.spines['right'].set_visible(False)  # Quitar la línea derecha del marco
                # ax2.spines['bottom'].set_visible(False)  # Quitar la línea inferior del marco
                # ax2.spines['left'].set_visible(False)  # Quitar la línea izquierda del marco

                # # Agregar los porcentajes a las barras
                # for bar, porcentaje in zip(bar, df_tmp['Porcentaje']):
                #     width = bar.get_width()
                #     ax2.text(width+4, bar.get_y() + bar.get_height() / 2, f'{porcentaje:.2f}%', ha='left', va='center', fontsize=10)

                # ################################## Mapa de calor
                # ax1 = fig.add_subplot(gs[1, 0:3])#axes[1,0]
                # heatmap = sns.heatmap(hm_df.loc[:, productos],
                #                     annot=True,
                #                     # yticklabels=['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS DE CARGA', 'FIBRA ÓPTICA',
                #                     #             'REDES ELÉCTRICAS', 'ILUMINACIÓN', 'CUENTAS NUEVAS'],
                #                     yticklabels=['Instalaciones', 'Mantenimientos', 'Estudios', 'Aumentos de carga', 'Fibras ópticas',
                #                                 'Redes eléctricas', 'Iluminación', 'Cuentas nuevas'],


                #                     xticklabels=['Recomendación 1', 'Recomendación 2', 'Recomendación 3'],
                #                     cbar=False,
                #                     fmt='.0f',
                #                     cmap="Purples",
                #                     annot_kws={"fontsize": 10},
                #                     ax=ax1)
                
                # # etiq = ax1.get_xticklabels()

                # ax1.set_xticklabels(ax1.get_xticklabels(),rotation=0,fontsize=8)
                # ax1.set_yticklabels(ax1.get_yticklabels(),rotation=0,fontsize=8)
                # ax1.set_xlabel('')  # Quitar etiqueta del eje x
                # ax1.set_ylabel('')  # Quitar etiqueta del eje y
                # ax1.set_title('')  # Quitar el título
                # # ax1.set_yticks([])

                # ################################ DONA
                # bar_width = 0.95
                # bar_spacing = 1
                # etiquetas = ['Alta','Media','Baja']
                # X = [0,  0,  0]
                # Y = [0.3,0,-0.3]

                # pro_df = pd.DataFrame({'index':['Baja','Media','Alta']})
                # for i in ['Probabilidad_1',	'Probabilidad_2',	'Probabilidad_3']:
                #     pro_df=pd.merge(pro_df, pd.DataFrame(Xf[i].value_counts(dropna=False)).reset_index(drop=False),how='outer', on='index')
                # pro_df.drop('index', axis=1, inplace=True)
                # pro_df = pro_df.fillna(0)

                # pro_df['p1'] = pro_df['Probabilidad_1']/pro_df['Probabilidad_1'].sum() *100 #,'Probabilidad_2','Probabilidad_3'
                # pro_df['p2'] = pro_df['Probabilidad_2']/pro_df['Probabilidad_2'].sum() *100 #,'Probabilidad_2','Probabilidad_3'
                # pro_df['p3'] = pro_df['Probabilidad_3']/pro_df['Probabilidad_3'].sum() *100 #,'Probabilidad_2','Probabilidad_3'

                # # pro_df = pro_df[::-1]

                # categorias = ['Probabilidad 1', 'Probabilidad 2', 'Probabilidad 3']
                # valores1 = pro_df.iloc[:,0].tolist()
                # valores2 = pro_df.iloc[:,1].tolist()
                # valores3 = pro_df.iloc[:,2].tolist()

                # # Calcula los porcentajes
                # total1 = sum(valores1)
                # porcentajes1 = pro_df['p1'][::-1].tolist()

                # total2 = sum(valores2)
                # porcentajes2 = pro_df['p2'][::-1].tolist() 

                # total3 = sum(valores3)
                # porcentajes3 = pro_df['p3'][::-1].tolist() 

                # ########
                # ax3 = fig.add_subplot(gs[0, 0])
                # # Grafico de dona
                # patches, text = ax3.pie(valores1, labels=['','',''], startangle=90, colors = C,wedgeprops={'edgecolor': 'white'})

                # # Centro del círculo para convertirlo en un gráfico de dona
                # circulo_centro = plt.Circle((0, 0), 0.70, fc='white')
                # fig.gca().add_artist(circulo_centro)

                # # Agregar los porcentajes y los nombres dentro del espacio vacío de la dona
                # fs = 7
                # for i, p in enumerate(patches):

                #     ax3.text(X[i], Y[i], f'{etiquetas[i]+" "}{porcentajes1[i]:.1f}%', ha='center', va='center',
                #             fontsize=fs)

                # ########
                # ax4 = fig.add_subplot(gs[0, 1])
                # # Grafico de dona
                # patches, text = ax4.pie(valores2, labels=['','',''], startangle=90, colors = C,wedgeprops={'edgecolor': 'white'})

                # # Centro del círculo para convertirlo en un gráfico de dona
                # circulo_centro = plt.Circle((0, 0), 0.70, fc='white')
                # fig.gca().add_artist(circulo_centro)

                # # Agregar los porcentajes y los nombres dentro del espacio vacío de la dona

                # for i, p in enumerate(patches):

                #     ax4.text(X[i], Y[i], f'{etiquetas[i]+" "}{porcentajes2[i]:.1f}%', ha='center', va='center',
                #             fontsize=fs)
                # #########
                # ax5 = fig.add_subplot(gs[0, 2])
                # # Grafico de dona
                # patches, text = ax5.pie(valores3, labels=['','',''], startangle=90, colors = C,wedgeprops={'edgecolor': 'white'})

                # # Centro del círculo para convertirlo en un gráfico de dona
                # circulo_centro = plt.Circle((0, 0), 0.70, fc='white')
                # fig.gca().add_artist(circulo_centro)

                # # Agregar los porcentajes y los nombres dentro del espacio vacío de la dona

                # for i, p in enumerate(patches):

                #     ax5.text(X[i], Y[i], f'{etiquetas[i]+" "}{porcentajes3[i]:.1f}%', ha='center', va='center',
                #             fontsize=fs)

                # # ax[0, 4].axis('off')
                # # Ajustar el espaciado entre los subplots
                # # plt.subplots_adjust(wspace=0.5)
                # # plt.subplots_adjust(hspace=0.5)

                # # Mostrar el gráfico
           
                # # # Ajustar el espaciado entre los subplots
                # plt.subplots_adjust(wspace=0.05)
                # fig.subplots_adjust(hspace=0)
                # # plt.subplots_adjust(hspace=0.3)
                def dona(prod, x, y, title, tamano_donut=1, tamano_centro=0.75):
                    valores3 = df_prob_prod.loc[:, prod].tolist()
                    porcentajes3 = df_prob_prod.loc[:, 'P_' + prod].tolist()

                    ax3 = fig.add_subplot(gs[x, y])

                    patches, text = ax3.pie(valores3, labels=['', '', ''], startangle=90, colors=C, wedgeprops={'edgecolor': 'white'})

                    # Centro del círculo para convertirlo en un gráfico de dona
                    circulo_centro = plt.Circle((0, 0), tamano_centro, fc='white')
                    fig.gca().add_artist(circulo_centro)

                    # Agregar los porcentajes y los nombres dentro del espacio vacío de la dona
                    for i, p in enumerate(patches):
                        ax3.text(X[i], Y[i], f'{str(int(valores3[i]))+" "+etiquetas[i] +" "}({porcentajes3[i]:.1f}%)',
                                ha='center', va='center', fontsize=fs)
                    ax3.set_title(title, fontsize=fs)


                merged_df = pd.DataFrame(index=['Alta','Media','Baja'])
                df_prob_prod = pd.DataFrame()

                productos = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA', 
                            'FIBRA_OPTICA', 'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']

                for prod in productos:

                    df_tmp1 = pd.DataFrame(Xf[Xf['Producto_1']==prod]['Probabilidad_1'].value_counts())
                    df_tmp2 = pd.DataFrame(Xf[Xf['Producto_2']==prod]['Probabilidad_2'].value_counts())
                    df_tmp3 = pd.DataFrame(Xf[Xf['Producto_3']==prod]['Probabilidad_3'].value_counts())
                    
                    merged_df = pd.DataFrame(index=['Alta','Media','Baja'])

                    merged_df = merged_df.merge(df_tmp1, left_index=True, right_index=True, how = 'outer')
                    merged_df = merged_df.merge(df_tmp2, left_index=True, right_index=True, how = 'outer')
                    merged_df = merged_df.merge(df_tmp3, left_index=True, right_index=True, how = 'outer')


                    merged_df = merged_df.fillna(0)
                    merged_df['Total'] =merged_df.sum(axis=1) 
                    df_prob_prod[prod] = merged_df['Total']

                df_prob_prod = df_prob_prod.reindex(['Alta','Media','Baja'])

                for prod in productos:
                     df_prob_prod['P_'+prod] = np.round(df_prob_prod[prod]/df_prob_prod[prod].sum() *100, 2) 


                colors = plt.cm.Purples(range(256))
                # Seleccionar los tres tonos deseados
                C = [ colors[255], colors[170],colors[80]]

                fig = plt.figure(figsize=(10,6))
                gs = GridSpec(2, 4)

                # Crear los gráficos para la primera fila

                bar_width = 0.95
                bar_spacing = 1

                etiquetas = ['Alta','Media','Baja']
                X = [0,  0,  0]
                Y = [0.3,0,-0.3]
                fs = 7

                ######## INSTALACIONES
                dona('INSTALACIONES',0 ,0, 'Instalaciones')

                ######## Mantenimiento
                dona('MANTENIMIENTO',0 , 1, 'Mantenimiento')

                ######## Estudios
                dona('ESTUDIOS',0 , 2, 'Estudios')

                #AUMENTOS_CARGA
                dona('AUMENTOS_CARGA',0 , 3, 'Aumentos de carga')

                #FIBRA OPTICA
                dona('FIBRA_OPTICA',1 , 0, 'Fibras ópticas')

                #REDESELECTRICAS
                dona('REDESELECTRICAS',1 , 1, 'Redes eléctricas')

                #ILUMINACION
                dona('ILUMINACION',1 , 2, 'Iluminación')

                #CUENTASNUEVAS
                dona('CUENTASNUEVAS',1 , 3, 'Cuentas nuevas')



                # # Mostrar el gráfico
                # col113.pyplot(fig)
                st.write(fig)

                st.write("")


                # st.write(Xf)
                download_excel(Xf, 'Resultado')
                

                # ####################################################### TORTA 
                # # col111
                # df_torta = Xf.loc[:,['Producto_1','Producto_2','Producto_3','Producto_4','Producto_5','Producto_6','Producto_7','Producto_8','Probabilidad_1','Probabilidad_2','Probabilidad_3']]

                # filtered_colors = ['#C5AAFF','#5A3E66','#C5AAFF','#5A3E66','#C5AAFF','#5A3E66','#C5AAFF','#5A3E66']
                # fig = px.sunburst(df_torta, path=['Producto_1','Producto_2','Producto_3','Producto_4','Producto_5','Producto_6','Producto_7','Producto_8'],
                #       color='Producto_1',
                #       color_discrete_sequence=filtered_colors,
                #       maxdepth=2)


                # fig.update_layout(
                # width=300,  # ajusta el valor según tus necesidades
                # height=300)  # ajusta el valor según tus necesidades
                # fig.update_layout(
                # margin=dict(l=0, r=0, t=0, b=0) )
                # # Mostrar el gráfico en Streamlit 
                # col111.write("")
                # col111.plotly_chart(fig)




            except UnboundLocalError:
                st.warning('No ha cargado un archivo para procesar!. En el menú de la izquierda cargar archivo en la sección Modelo Múltiples Variables')
        
        with vista2:    # Descriptiva
            try:
                tab1, tab2, tab3,tab4 = st.tabs(["Consumo", "Ventas", "Económico","Demográficas"])
                # source = dataframe.copy()

                #source.columns = dataframe.columns.str.lower()
                #source = source.applymap(lambda x: x.lower() if type(x)==str else x)   
                #st.write(source)
            
                df_t = ob.transform_load()#_graf

                with tab1:
                    st.write("")
                    st.write("")
                    st.subheader("CONSUMO")
                    col7,col8,col71 = st.columns(spec=[1,3,1]) 
                    #col8.image("img/cons.png",use_column_width="always")
                    # print(source.columns)

                    # df_group = df_t.groupby(by='RANGOCONSUMO', as_index=False)['NIT9'].count()
                    # df_group.rename({'NIT9':'Cantidad',
                    #                  'RANGOCONSUMO':'Rango de Consumo'}, axis=1, inplace=True)
                    # df_group['Cantidad'] = pd.to_numeric(df_group['Cantidad'])
                    # df_group['Porcentaje'] = df_group['Cantidad'] / df_group['Cantidad'].sum() * 100
                    # df_group['Porcentaje'] = df_group['Porcentaje'].round(2)
                    # # df_group['Porcentaje'] = df_group['Porcentaje'].apply(lambda x: str(x) + '%')
                    # df_group['Porcentaje'] = df_group['Porcentaje'].apply(lambda x: '{:.2f}%'.format(x))

                    # # Gráfico de barras con etiquetas de valor y porcentaje
                    # bar = alt.Chart(df_group).mark_bar().encode(
                    #     x=alt.X('Cantidad', axis=alt.Axis(title='Registros')),
                    #     y="Rango de Consumo:N",
                    #     tooltip=['Cantidad:Q', 'Porcentaje:O'],
                    #     text=alt.Text('Porcentaje:N')#alt.Text('count:Q', format='.0f')
                    # ).configure_mark(color='#311557').configure_view(fill="none").configure_axis(grid=False)

                    # col8.altair_chart(bar, use_container_width=True, theme="streamlit")
                    configuraciones = [
                        {
                            'groupby': 'RANGOCONSUMO',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Clientes',
                            'y_axis': 'Rango de Consumo',
                            'col': col8,
                            'order': ['SINCATALOGAR','MENORA5000', 'ENTRE5000Y10000', 'ENTRE10000Y55000',  'MAYORA55000'],
                            'order_f':['Sin catalogar','Menor a 5000',  'Entre 5000 y 10000','Entre 10000 y 55000',   'Mayor a 55000']
                        }
                        ]

                    generar_graficos(df_t, configuraciones)

                with tab2:
                    st.write("")
                    st.write("")
                    st.subheader("VENTAS")
                    st.write("")

                    col1,col2 = st.columns(2)#, gap="medium"   
                    # Configuraciones de los gráficos
                    configuraciones = [
                        {
                            'groupby': 'RANGODECOMPRA($)',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Clientes',
                            'y_axis': 'Rango de compra',
                            # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                            'col': col1,
                            'order': ['SINCATALOGAR', 'NOCOMPRADOR', 'PEQUENOCOMPRADOR', 'MEDIANOCOMPRADOR', 'GRANCOMPRADOR', 'COMPRADORMEGAPROYECTOS'],  # Orden deseado de las categorías
                            'order_f':['Sin catalogar','No comprador','Pequeno comprador','Mediano comprador','Gran comprador', 'Comprador megaproyectos']
                        },
                        {
                            'groupby': 'RANGORECURRENCIACOMPRA',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Clientes',
                            'y_axis': 'Recurrencia de compra',
                            # 'chart_title': 'Gráfico 2 RangoRecurrenciaCompra',
                            'col': col2,
                            'order': ['SINCATALOGAR', 'NOCOMPRADOR', 'UNICACOMPRA', 'BAJARECURRENCIA', 'RECURRENCIAMEDIA', 'GRANRECURRENCIA'],  # Orden deseado de las categorías
                            'order_f':['Sin catalogar', 'No comprador','Unica compra','Baja recurrencia','Recurrencia media','Gran recurrencia'],
                        
                        },
                        {
                            'groupby': 'TIPOCLIENTE#OPORTUNIDADES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Clientes',
                            'y_axis': 'Tipo de cliente por numero de oportunidades',
                            # 'chart_title': 'Gráfico 3 TIPOCLIENTE#OPORTUNIDADES',
                            'col': col1,
                            'order':['SINCATALOGAR', 'NICOMPRA-NICOTIZA', 'SOLOCOTIZAN', 'COTIZANMASDELOQUECOMPRAN', 
                                    'COMPRANYCOTIZAN', 'COMPRANMASDELOQUECOTIZAN', 'SIEMPRECOMPRAN'],  # Orden deseado de las categorías
                            'order_f':['Sin catalogar','Ni compra - ni cotiza','Solo cotizan','Cotizan mas de lo que compran',
                                       'Compran y cotizan','Compran  mas de lo que cotizan', 'Siempre compran'],
                        },
                        {
                            'groupby': 'TIPOCLIENTE$OPORTUNIDADES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Clientes',
                            'y_axis': 'Tipo de cliente por valor de oportunidades',
                            # 'chart_title': 'Gráfico 4 TIPOCLIENTE$OPORTUNIDADES',
                            'col': col2,
                            'order':['SINCATALOGAR', 'NICOMPRA-NICOTIZA', 'SOLOCOTIZAN', 'COTIZANMASDELOQUECOMPRAN', 
                                    'COMPRANYCOTIZAN', 'COMPRANMASDELOQUECOTIZAN', 'SIEMPRECOMPRAN'],
                            'order_f':['Sin catalogar','Ni compra - ni cotiza','Solo cotizan','Cotizan mas de lo que compran',
                                       'Compran y cotizan','Compran  mas de lo que cotizan', 'Siempre compran']   # Orden deseado de las categorías
                        }
                        ]

                    generar_graficos(df_t, configuraciones)

                with tab3:
                    st.write("")
                    st.write("")
                    st.subheader("ECONOMICAS")
                    col4,col5 = st.columns(2,gap="medium")  
                    
                    # bar= alt.Chart(source).mark_bar().encode(x='count()',y="TamanoEmpresa:N").configure_mark(color='#311557')
                    # col4.altair_chart(bar,use_container_width=True,theme="streamlit")   
                    # #col4.image("img/econ1.png")
                    # bar= alt.Chart(source).mark_bar().encode(x='count()',y="CategorizacionSectores:N").configure_mark(color='#311557')
                    # col5.altair_chart(bar,use_container_width=True,theme="streamlit")
                    # #col4.image("img/econ2.png")
                    # bar= alt.Chart(source).mark_bar().encode(x='count()',y="EstatusOperacional:N").configure_mark(color='#311557')
                    # col4.altair_chart(bar,use_container_width=True,theme="streamlit")
                    # st.write("")

                    # Configuraciones de los gráficos
                    configuraciones = [
                        {
                            'groupby': 'TAMANOEMPRESA',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Clientes',
                            'y_axis': 'Tamaño de la empresa',
                            # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                            'col': col4,
                            'order': ['SINCATALOGAR', 'PEQUENAEMPRESA', 'MEDIANAEMPRESA', 'GRANEMPRESA'],  # Orden deseado de las categorías
                            'order_f':['Sin catalogar', 'Pequena empresa', 'Mediana empresa','Gran empresa']   # Orden deseado de las categorías
                            
                        },
                        {
                            'groupby': 'CATEGORIZACIONSECTORES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Clientes',
                            'y_axis': 'Categoria del sector',
                            # 'chart_title': 'Gráfico 2 RangoRecurrenciaCompra',
                            'col': col5,
                            'order': ['SINCATALOGAR', 'OTROSSECTORES', 'SECTORALTOVALOR'] ,
                            'order_f':['Sin catalogar', 'Otros sectores', 'Sector alto valor']   # Orden deseado de las categorías
                            
                              # Orden deseado de las categorías
                        },
                        {
                            'groupby': 'ESTATUSOPERACIONAL',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Clientes',
                            'y_axis': 'Estatus operacional',
                            # 'chart_title': 'Gráfico 3 TIPOCLIENTE#OPORTUNIDADES',
                            'col': col4,
                            'order': ['NOSECONOCEELESTATUS', 'BAJOINVESTIGACIONLEGAL', 'OPERACIONAL'] , # Orden deseado de las categorías
                            'order_f': ['No se conoce el estatus', 'Bajo investigacion legal',  'Operacional']   # Orden deseado de las categorías
                        }]
                    generar_graficos(df_t, configuraciones)

                with tab4:
                    st.write("")
                    st.write("")
                    st.subheader("DEMOGRÁFICAS")
                    col9,col0,col91 = st.columns(spec=[1,3,1])
                    # #col9.image("img/dem.png",use_column_width="always")
                    # bar= alt.Chart(source).mark_bar().encode(x='count()',y="CategoriaDepartamento:N").configure_mark(color='#311557')
                    # col0.altair_chart(bar,use_container_width=True,theme="streamlit")
                    # st.write("")

# Configuraciones de los gráficos
                    configuraciones = [
                        {
                            'groupby': 'CATEGORIADEPARTAMENTO',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Clientes',
                            'y_axis': 'Categoria de departamento',
                            # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                            'col': col0,
                            'order': ['NOSECONOCEELDEPARTAMENTO', 'OTROSDEPARTAMENTOS', 'COSTA', 'CUNDINAMARCA', 'BOGOTADC'],  # Orden deseado de las categorías
                            'order_f': ['No se conoce el departamento',  'Otros departamentos',  'Costa',  'Cundinamarca',  'Bogota DC']   # Orden deseado de las categorías
                        
                        }]
                    generar_graficos(df_t, configuraciones)

            except UnboundLocalError:
                st.warning('No ha cargado un archivo para procesar!. En el menú de la izquierda cargar archivo en la sección Modelo Múltiples Variables')
                 
    if c== True:
        # st.empty()
        # st.markdown("<h1 style='text-align: center;'>Modelo Predictivo</h1>", unsafe_allow_html=True)
        # st.write("")
        # vista1,vista2,vista3 = st.tabs(["Resultado Múltiples Clientes", "Reporte Descriptivo", "Resultado Modelo Unitario"])

        with vista3: # Modelo Unitario
            try:       
                st.write("")
                #st.write(dataframe.head())
 
                Xi,Xf = ob_u.predict_proba()
                st.write(Xf)
                download_excel(Xf, 'Resultado')
    #             )
            except UnboundLocalError:
                st.warning('No ha cargado un archivo para procesar!. En el menú de la izquierda cargar archivo en la sección Modelo Múltiples Variables')

               
    #         with vista3:
    #             if c == True:
    #                 #x_in = pd.read_excel('./Data_cla_1.xlsx',index_col=0)
    #                 x_1 = x_in.iloc[:1,:]
    #                 #x_1.reset_index()
    #                 st.write(x_1)
    #                 ytest_e = model_prediction(x_1,model[0])
    #                 ytest_d = pd.DataFrame(np.int_(ytest_e))
    #                 st.write(ytest_d)
    #                 #st.write("")
    #                 #bar= alt.Chart(ytest_d).mark_bar().encode(x='count()',y="0:N").configure_mark(color='#311557')
    #                 #st.altair_chart(bar,use_container_width=True,theme="streamlit")
    #                 #st.image("img/Modelo1.png",use_column_width="always")
    #                 #st.write("")
    #                 #x_in =[np.float_(N.title()),
    #                 #            np.float_(P.title()),
    #                 #            np.float_(K.title()),
    #                 #            np.float_(Temp.title()),
    #                 #            np.float_(Hum.title()),
    #                 #            np.float_(pH.title()),
    #                 #            np.float_(rain.title())]
    #                 #predictS = model_prediction(x_in, model)
    #                 #st.success('EL CULTIVO RECOMENDADO ES: {}'.format(predictS[0]).upper())
    #                 #st.write("")
    #                 #if datos != '':
    #                 #    dataframe = pd.read_csv(datos)
    #                 #    st.write(dataframe.head())
                        
    #                 #    def convert_df(df):
    #                         # IMPORTANT: Cache the conversion to prevent computation on every rerun
    #                 #        return df.to_csv().encode('utf-8')

    #                 #    csv = convert_df(dataframe)
                        
    #                 #    st.download_button(
    #                 #        label="Descargar archivo",
    #                 #        data=csv,
    #                 #        file_name='large_df.csv',
    #                 #        mime='text/csv',
    #                 #    )
    #                 #st.download_button("Descargar archivo",data=datos)
    #             else:
    #                 st.warning('No ha diligenciado las variables para ejecutar el modelo unitario!')                
    # else:
    #     st.image("img/Img_presentacion2.jpg",use_column_width="always")

if __name__ == '__main__':
    main()