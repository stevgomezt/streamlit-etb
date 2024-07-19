import logging  # Para registrar mensajes de registro
import sys  # Para acceder a argumentos y rutas del sistema
from typing import Union  # Para proporcionar anotaciones de tipos

import numpy as np  # Para cálculos numéricos
import pandas as pd  # Para manipulación y análisis de datos
import altair as alt  # Para visualización interactiva de datos
import matplotlib.pyplot as plt  # Para visualización estática de datos
import plotly.express as px  # Para visualización interactiva y expresiva de datos
import plotly.graph_objects as go  # Para crear gráficos interactivos avanzados
import plotly.colors as colors_plotly  # Para trabajar con colores en Plotly

# Para construir aplicaciones web con Flask
from flask import Flask, request, jsonify, render_template, url_for
# Para interactuar con servicios de almacenamiento en la nube de Google
from google.cloud import storage

import streamlit as st  # Para crear aplicaciones web interactivas y paneles de control

# Módulo personalizado para validar, preprocesar y predecir datos
import validar_preprocesar_predecir_organizarrtados

from datetime import date  # Para trabajar con fechas

import base64  # Para realizar codificación y decodificación en base64
import json  # Para trabajar con datos en formato JSON
import pickle  # Para serializar y deserializar objetos en Python
import uuid  # Para generar identificadores únicos
import re  # Para trabajar con expresiones regulares

# Para Botones sin recargar

# -----------------------------------------------------IAP GCP

app = Flask(__name__)

# -----------------------------------------------------

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

# -----------------------------------------------------


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
    from json import jwt
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


def download_excel(df_v, nombre='LogErrores', col=st):
    df_v.to_excel(nombre+'.xlsx', index=False)
    filename = nombre+'.xlsx'
    with open(nombre+'.xlsx', 'rb') as file:
        contents = file.read()
    if col == st:
        # st.download_button(label='Descargar '+nombre, data=contents, file_name=nombre+'.xlsx')
        download_button_str = download_button(contents, filename, nombre)
        st.markdown(download_button_str, unsafe_allow_html=True)

    else:
        # col.download_button(label='Descargar '+nombre, data=contents, file_name=nombre+'.xlsx')
        download_button_str = download_button(contents, filename, nombre)
        col.markdown(download_button_str, unsafe_allow_html=True)


def download_button(object_to_download, download_filename, button_text, pickle_it=False):

    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                -webkit-box-align: center;
                align-items: center;
                -webkit-box-pack: center;
                justify-content: center;
                padding: 0.25rem 2.75rem;
                margin: 0px;
                line-height: 1.6;
                user-select: none;
                background-color: white;
                color: #0091ff;
                height: 35px;
                width: auto;
                border: solid 1px #0091ff;
                font-size: 15px;
                border-radius: 0px;
                font-family: "Roboto", sans-serif;
                border-radius: 5px;
                margin-left: 16px;
                width: 200px;
                text-decoration: none !important;
            }}
            
            #{button_id}:focus {{
                box-shadow: none;
                outline: none;
                text-decoration: none !important;
            }}
            
            #{button_id}:focus:not(:active) {{
                border-color: #0091ff;
                background-color: white;
                color: #0091ff;
                text-decoration: none !important;
            }}
            
            #{button_id}:hover {{
                border-color: #0091ff;
                background-color: #0091ff;
                color: white;
                text-decoration: none !important;
            }}
            
            #{button_id}:active {{
                border-color: #0091ff;
                -webkit-box-shadow: 0px 0px 5px 0px rgba(0, 0, 0, 0.75);
                -moz-box-shadow: 0px 0px 5px 0px rgba(0, 0, 0, 0.75);
                box-shadow: 0px 0px 5px 0px rgba(0, 0, 0, 0.75);
                background-color: white;
                color: #0091ff;
                text-decoration: none !important;
            }}
        </style> """

    dl_link = custom_css + \
        f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


def download_excel_torta(df_v, nombre='LogErrores', col=st):
    df_v.to_excel(nombre+'.xlsx', index=False)

    filename = nombre+'.xlsx'
    with open(filename, 'rb') as file:
        contents = file.read()

    if col == st:
        download_button_str = download_button(contents, filename, nombre)
        st.markdown(download_button_str, unsafe_allow_html=True)

    else:
        download_button_str = download_button(contents, filename, nombre)
        col.markdown(download_button_str, unsafe_allow_html=True)


# ,cols = [col11, col12, col13, col14,col15]}
def botones_descarga(Xf=None, variable='RangoConsumo', col=None):

    for categoria in Xf[variable].unique():

        download_excel_torta(
            Xf[Xf[variable] == categoria], nombre=categoria, col=col)


def download_txt(nombre, logs):

    # Especifica la ruta y el nombre de archivo del archivo de texto
    archivo_txt = "archivo.txt"

    # Abre el archivo en modo escritura
    with open(archivo_txt, "w") as archivo:
        # Escribe cada valor de la lista en una línea separada
        for valor in logs:
            archivo.write(valor + "\n \n")
    # Leer el contenido del archivo
    with open(archivo_txt, 'rb') as file:
        contents = file.read()

    # Descargar el archivo

    st.download_button(nombre + '.txt', data=contents, file_name="archivo.txt")


@app.route('/', methods=['GET'])
def say_hello():
    from flask import request
    assertion = request.headers.get('X-Goog-IAP-JWT-Assertion')
    email, id = validate_assertion(assertion)
    page = "<h1>Hello {}</h1>".format(email)
    return page

# ------------------------------------------------------


def agregar_k(valor):
    return str(valor) + 'K'


def generar_graficos(df_t, configuraciones, mayus=True, color=1, auto_orden=False, total=False):
    for config in configuraciones:
        df_group = df_t.groupby(by=config['groupby'], as_index=True)[
            'NIT9'].count()
        df_group = pd.DataFrame(df_group)
        # st.write(df_group)

        if mayus == True:
            if not auto_orden:
                # Reordenar el DataFrame según el orden deseado
                df_group = df_group.reindex(config['order'])
            else:
                df_group.sort_values(by='NIT9', ascending=False, inplace=True)
        else:
            # Ordenar de mayor a menor
            df_group.sort_values(by='NIT9', ascending=False, inplace=True)

        # Extrae indice a columna
        df_group.reset_index(inplace=True, drop=False)
        df_group.dropna(inplace=True)
        df_group.reset_index(inplace=True, drop=True)
        # st.write(df_group)

        df_group.rename(
            {'NIT9': 'Cantidad_n', config['groupby']: config['y_axis']}, axis=1, inplace=True)
        df_group['Cantidad_n'] = pd.to_numeric(df_group['Cantidad_n'])
        df_group['Cantidad_n'] = df_group['Cantidad_n']*100

        if mayus == True:

            keys = config['order']
            values = config['order_f']

            diccionario = dict(zip(keys, values))

            df_group[config['y_axis']] = df_group[config['y_axis']
                                                  ].replace(diccionario)

        df_group[config['y_axis']] = pd.Categorical(
            df_group[config['y_axis']], ordered=True)

        df_group['Porcentaje'] = df_group['Cantidad_n'] / \
            df_group['Cantidad_n'].sum() * 100
        df_group['Porcentaje'] = df_group['Porcentaje'].round(2)
        df_group['Porcentaje'] = df_group['Porcentaje'].apply(
            lambda x: ' {:.2f}%'.format(x))

        # df_group['Cantidad'] = df_group['Cantidad_n'].apply(agregar_k)

        # st.write(df_group)
        if color == 0:
            color_b = "#757575"
        if color == 1:
            color_b = '#0076BA'
        elif color == 2:
            color_b = '#162055'
        elif color == 3:
            color_b = '#0378A6'
        elif color == 4:
            color_b = '#DDF2FD'
        elif color == 5:
            color_b = '#79B4D9'
        elif color == 6:
            color_b = '#0076BA'
        elif color == 7:
            color_b = '#162055'

        if mayus == True:
            if total == False:

                # nuevos_valores_xticks = [5,10,15]#'5 K', '10 K', '15 K', '20 K'
                bar = alt.Chart(df_group).mark_bar().encode(
                    x=alt.X('Cantidad_n', axis=alt.Axis(
                        ticks=True, title=config['x_axis_title'],
                        # values=nuevos_valores_xticks
                    ),
                        # scale=alt.Scale(type='ordinal')
                    ),
                    y=alt.Y(config['y_axis'] + ":N", sort=list(
                        df_group[config['y_axis']]), axis=alt.Axis(ticks=False, title='')),
                    tooltip=[config['y_axis']+":N",
                             'Cantidad_n:Q', 'Porcentaje:O'
                             ],
                    text=alt.Text('Porcentaje:N')
                ).configure_mark(color=color_b).configure_view(fill="none").configure_axis(grid=False).configure_axisY(labelFont='Roboto').configure_axisX(ticks=True, labels=True)

                config['col'].altair_chart(
                    bar, use_container_width=True, theme="streamlit")
                st.write("")
            else:

                bar = alt.Chart(df_group).mark_bar().encode(
                    x=alt.X('Cantidad', axis=alt.Axis(
                        ticks=False, title=config['x_axis_title'])),
                    y=alt.Y(config['y_axis'] + ":N", sort=list(
                        df_group[config['y_axis']]), axis=alt.Axis(ticks=False, title='')),
                    tooltip=[config['y_axis']+":N",
                             'Cantidad:Q', 'Porcentaje:O'],
                    text=alt.Text('Porcentaje:N')
                ).configure_mark(color=color_b).configure_view(fill="none").configure_axis(grid=False).configure_axisX(ticks=False, labels=False).configure_axisY(labelFont='Roboto')

                config['col'].altair_chart(
                    bar, use_container_width=True, theme="streamlit")
                st.write("")

        else:
            bar = alt.Chart(df_group).mark_bar().encode(
                x=alt.X('Cantidad', axis=alt.Axis(
                    ticks=False, title=config['x_axis_title'])),
                # y=config['y_axis'] + ":N",
                y=alt.Y(config['y_axis'] + ":N", sort=list(df_group[config['y_axis']]),
                        axis=alt.Axis(ticks=False, title=None)),  # , title=None
                tooltip=[config['y_axis']+":N", 'Cantidad:Q', 'Porcentaje:O'],
                text=alt.Text('Porcentaje:N')
            ).configure_mark(color=color_b).configure_view(fill="none").configure_axis(grid=False).configure_axisY(labelFont='Roboto')

            config['col'].altair_chart(
                bar, use_container_width=True, theme="streamlit")
            st.write("")

# ------------------------------------------------------


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_excel()  # .encode('utf-8')


def dona_plotly(df_prob_prod, producto='INSTALACIONES', col=None, titulo=None, tamano_pantalla=(400, 400)):

    valores = df_prob_prod.loc[:, producto].astype(int)*100
    etiquetas = ['Alta', 'Media', 'Baja']

    # Colores personalizados
    colores = ['#0076BA', '#162055', '#79B4D9']

    total = sum(valores)
    conteos = [str(valor) for valor in valores]
    porcentajes = [f'{(valor/total)*100:.1f}%' for valor in valores]

    fig = go.Figure(data=[
        go.Pie(
            labels=etiquetas,
            values=valores,
            hole=0.55,
            textinfo='none',  # 'label+text+percent',
            # text=conteos,
            hovertemplate='%{label}',  # '%{label}<br>%{text} (%{percent})',
            marker=dict(colors=colores)
        )
    ])
    if titulo:
        fig.update_layout(title={
            'text': titulo,
            'y': 0.95,
            'yanchor': 'top',
            'font': {'size': 15, 'family': "Roboto, sans-serif", 'color': '#002f6c'}
        }
        )

    fig.update_layout(width=tamano_pantalla[0], height=tamano_pantalla[1])

    # Ocultar el legend
    fig.update_layout(
        showlegend=False
    )
    fig.update_traces(
        text=conteos,
        textinfo='label+text+percent',  # Activa el texto personalizado
        textposition='outside'  # Mueve el texto fuera de la dona
    )

# -----------------------------------------------------

    # Reducir el tamaño de las etiquetas
    fig.update_traces(
        textfont=dict(
            size=13  # Tamaño de la fuente de las etiquetas
        )
    )

# -----------------------------------------------------

    col.plotly_chart(fig, use_container_width=True)
    # Encabezado inicial
    # header = st.empty()


def espacio(col, n):
    if n > 0:
        for i in range(n):
            col.write('')


def scatter_plot(df, col=None):
    # Definir los colores base
    color_azul = '#162055'
    color_amarillo = '#0076BA'

    # Crear la paleta de color
    colores = [color_azul, color_amarillo]

    # Crear la escala de color continua
    colorscale = colors_plotly.make_colorscale(colores)
    # Crear el gráfico scatter utilizando plotly express
    fig = px.scatter(df, x='DEPARTAMENTO', y='ACTIVIDADES',
                     # 'OPORTUNIDADESCOTIZADAS(#)',
                     color='OPORTUNIDADESVENDIDAS', size='OPORTUNIDADESCOTIZADAS($)',
                     # 'Plasma'#px.colors.sequential.Cividis#'Plotly3'#'matter_r'#'purples_r'
                     color_continuous_scale=colorscale
                     )

    # Personalizar el diseño del gráfico

    fig.update_layout(coloraxis_colorbar=dict(len=1, ypad=0))

    fig.update_layout(xaxis_title='Departamento', yaxis_title='Actividad económica',
                      coloraxis_colorbar=dict(title='Ventas'), width=875, height=500)

    fig.update_layout(coloraxis_colorbar=dict(
        tickmode='array',  # Usar modo de ticks de arreglo
        tickvals=list(range(0, 27, 2)),  # Valores de los ticks personalizados
        ticktext=list(range(0, 27, 2))  # Etiquetas de los ticks personalizados
    ))

    fig.update_traces(
        hovertemplate='<b>Departamento</b>: %{x}<br>'
        '<b>Actividad económica</b>: %{y}<br>'
        '<b>Oportunidades vendidas</b>: %{marker.color}<br>'
        '<b>Oportunidades cotizadas</b>: %{marker.size:,}<extra></extra>'
    )

    col.plotly_chart(fig, use_container_width=True,)

    col.write('')


def main():

    # Configura titulo e icon de pagina
    st.set_page_config(page_title="2024 © ETB S.A. ESP. Todos los derechos reservados. Música Autorizada Por Acinpro.",
                       page_icon="img/Icono.png", layout="wide")

    # Leer el contenido del archivo CSS
    css = open('styles.css', 'r').read()

    # Agregar estilo personalizado
    st.markdown(
        f'<style>{css}</style>',
        unsafe_allow_html=True)

    # Variable que controla la visibilidad de la imagen
    b = False
    vista2, vista1 = st.tabs(
        ["Reporte descriptivo", "Resultado modelo"])  # 'Inicio', vista0,

    # Bloque de estilo CSS para centrar la imagen LOGO
    st.markdown("""
    <style>
        .sidebar .sidebar-content {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # Imagen centrada en la barra lateral con tamaño de 150px
    st.sidebar.image("img/logo.png", width=180, output_format="PNG",
                     caption='2024 © ETB S.A. ESP. Todos los derechos reservados. Música Autorizada Por Acinpro.')

    st.markdown(
        """
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        </head>

        <div style="display: flex; justify-content: center;">
            <a href="https://co.linkedin.com/company/mundial-seguros-s-a-" style="color: #0057b8; margin: 0 30px;">
                <i class="fab fa-linkedin" style="font-size: 30px;"></i>
            </a>
            <a href="https://www.instagram.com/segurosmundial/?hl=es" style="color: #0057b8; margin: 0 30px;">
                <i class="fab fa-instagram" style="font-size: 30px;"></i>
            </a>
            <a href="https://www.facebook.com/segurosmundial/?locale=es_LA" style="color: #0057b8; margin: 0 30px;">
                <i class="fab fa-facebook" style="font-size: 30px;"></i>
            </a>
            <a href="https://twitter.com/SegurosMundial" style="color: #0057b8; margin: 0 30px;">
                <i class="fab fa-twitter" style="font-size: 30px;"></i>
            </a>
        </div>
    """,
        unsafe_allow_html=True
    )

    with st.sidebar.expander("MODELO MÚLTIPLES CLIENTES", expanded=False):

        try:
            datos = st.file_uploader("Subir archivos: ", type=["xlsx"])
            # b=False
            if datos is not None:

                dataframe = pd.read_excel(datos)
                dataframe.index = range(1, len(dataframe)+1)

                try:
                    dataframe['FECHACONSTITUCION'] = dataframe['FECHACONSTITUCION'].astype(
                        'datetime64[ns]')
                except:
                    pass

                # try:

                # Validación archivo
                original_len = len(dataframe.copy())
                ob = validar_preprocesar_predecir_organizarrtados.Modelos_2(
                    dataframe)
                df_v, text, final_flag = ob.Validar_todo()

# -----------------------------------------------------

                if final_flag == False:

                    logs, logs_riesgo, indices_posibles = ob.Logs()

                    if '1' not in logs_riesgo:

                        tx_registros_aptos = str('Registros aptos para recomendar: ') + str(len(
                            indices_posibles)/10)+'K ('+str(round(100*(len(indices_posibles))/original_len, 2))+'%)'
                        st.success(tx_registros_aptos, icon="✅")
                        b = st.button("Ejecutar Modelo", type="primary")

                    download_txt(logs=logs, nombre='Log_errores')

                    for i, j in zip(range(len(logs)), logs_riesgo):

                        if i == 0:       # Si es el primer log agrega '¡Ups! Parece que hay un problema.'

                            if j == '1':
                                st.write(
                                    '<div align="center"><h2>¡Ups! Parece que hay un problema.</h2></div>', unsafe_allow_html=True)

                            if (len(logs[i]) > 150) & (j == '1'):
                                st.warning(logs[i][:172]+'...', icon="⚠️")

                            elif (len(logs[i]) > 150) & (j == 0):
                                st.info(logs[i][:172]+'...', icon="ℹ️")

                            elif (len(logs[i]) <= 150) & (j == '1'):
                                st.warning(logs[i][:], icon="⚠️")

                            elif (len(logs[i]) <= 150) & (j == 0):
                                st.info(logs[i][:], icon="ℹ️")
                        else:
                            if (len(logs[i]) > 150) & (j == '1'):
                                st.warning(logs[i][:172]+'...', icon="⚠️")

                            elif ((len(logs[i]) > 150) & (j == 0)):
                                st.info(logs[i][:172]+'...', icon="ℹ️")

                            elif ((len(logs[i]) <= 150) & (j == '1')):
                                st.warning(logs[i], icon="⚠️")

                            elif ((len(logs[i]) <= 150) & (j == 0)):
                                st.info(logs[i], icon="ℹ️")

                    st.write('')

                else:
                    st.success(text+' (100%)', icon="✅")
                    b = st.button("Ejecutar Modelo", type="primary")

        except UnboundLocalError:
            st.warning('Error. Problemas con caracteristicas del archivo.')

    if b == True:
        # -----------------------------------------------------13/06/23

        with vista1:    # Modelo Multiples Clientes
            try:
                Xi, Xf = ob.predict_proba()

                # Modifico nombres de categorias
                keys = ['SINCATALOGAR', 'MENORA5000',
                        'ENTRE5000Y10000', 'ENTRE10000Y55000',  'MAYORA55000']
                values = ['Sin catalogar', 'Menor a 5000 kW⋅h',
                          'Entre 5000 y 10000 kW⋅h', 'Entre 10000 y 55000 kW⋅h']

                dic_rango_consumo = dict(zip(keys, values))

                Xf['RANGOCONSUMO'] = Xf['RANGOCONSUMO'].replace(
                    dic_rango_consumo)
                # st.write(Xi)

                hm_df = pd.DataFrame({'index': ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA', 'FIBRA_OPTICA',
                                                'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']})

                productos = ['Producto_1', 'Producto_2',
                             'Producto_3']  # Solo 3 primeras
                # print(hm_df,Xf)
                for i in productos:
                    hm_df = pd.merge(hm_df, pd.DataFrame(Xf[i].value_counts(
                        dropna=False)).reset_index(drop=False), how='outer', on='index')

                # Suma # primeras predicciones
                df_tmp = pd.DataFrame(hm_df['index'].copy())
                df_tmp.rename({'index': 'Productos'}, axis=1, inplace=True)

                # print(hm_df.columns)                                                                        ##------------

                df_tmp['Top 3'] = hm_df[['Producto_1',
                                         'Producto_2', 'Producto_3']].sum(axis=1)

                df_tmp['Porcentaje'] = df_tmp['Top 3'] / \
                    df_tmp['Top 3'].sum() * 100
                df_tmp['Porcentaje'] = df_tmp['Porcentaje'].round(2)
                # df_tmp['Porcentaje'] = df_tmp['Porcentaje'].apply(lambda x: ' {:.2f}%'.format(x))

                #
                keys = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA',
                        'FIBRA_OPTICA', 'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']
                values = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS DE CARGA',
                          'FIBRA OPTICA', 'REDES ELECTRICAS', 'ILUMINACION', 'CUENTAS NUEVAS']
                diccionario = dict(zip(keys, values))

                df_tmp['Productos'] = df_tmp['Productos'].replace(
                    diccionario)  # Corrijo nombre de los productos

                # Obtener la paleta de colores 'Purples'
                colors = plt.cm.Purples(range(256))
                # Seleccionar los tres tonos deseados
                C = [colors[80], colors[170], colors[255]]

                merged_df = pd.DataFrame(index=['Alta', 'Media', 'Baja'])
                df_prob_prod = pd.DataFrame()

                productos = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA',
                             'FIBRA_OPTICA', 'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']

                for prod in productos:

                    df_tmp1 = pd.DataFrame(
                        Xf[Xf['Producto_1'] == prod]['Probabilidad_1'].value_counts())
                    df_tmp2 = pd.DataFrame(
                        Xf[Xf['Producto_2'] == prod]['Probabilidad_2'].value_counts())
                    df_tmp3 = pd.DataFrame(
                        Xf[Xf['Producto_3'] == prod]['Probabilidad_3'].value_counts())

                    merged_df = pd.DataFrame(index=['Alta', 'Media', 'Baja'])

                    merged_df = merged_df.merge(
                        df_tmp1, left_index=True, right_index=True, how='outer')
                    merged_df = merged_df.merge(
                        df_tmp2, left_index=True, right_index=True, how='outer')
                    merged_df = merged_df.merge(
                        df_tmp3, left_index=True, right_index=True, how='outer')

                    merged_df = merged_df.fillna(0)
                    merged_df['Total'] = merged_df.sum(axis=1)
                    df_prob_prod[prod] = merged_df['Total']

                df_prob_prod = df_prob_prod.reindex(['Alta', 'Media', 'Baja'])

                for prod in productos:
                    df_prob_prod['P_'+prod] = np.round(
                        df_prob_prod[prod]/df_prob_prod[prod].sum() * 100, 2)

                # st.write(df_prob_prod)

                container0 = st.container()
                container0.markdown(
                    """
                    <style>
                    .custom-container {
                        background-color: #9e9ac8;
                        padding: 2.5px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                container0.markdown(
                    f'<div class="custom-container"></div>', unsafe_allow_html=True)

                # Crear el primer contenedor
                container1 = st.container()
                # Aplicar CSS personalizado al contenedor
                container1.markdown(
                    """
                    <style>
                    .custom-container {
                        background-color: #f2f0f7;
                        padding: 2.5px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Crear el segundo contenedor
                container2 = st.container()
                # Dividir el primer contenedor en dos columnas
                col1_container1, col2_container1, col3_container1 = container1.columns(spec=[
                                                                                       2.5, 2.3, 1])
                # Dividir el segundo contenedor en tres columnas
                col1_container2, col2_container2 = container2.columns(
                    2)  # , col3_container2, col4_container2

# -----------------------------------------------------

                # # Título con tamaño y color configurables
                tamaño1 = 30  # Tamaño1 del título
                tamaño2 = 60  # Tamaño2 del título
                color1 = '#757575'  # Color del título en formato hexadecimal
                color2 = '#9e9ac8'

                texto1 = 'Total clientes analizados'  # +'\n'+str(len(Xf))
                texto2 = str('  '+str(len(Xf)/10)+' K')  # +' Clientes')

                # col1_container1.text_align("center")
                container1.markdown(
                    f'<div class="custom-container"></div>', unsafe_allow_html=True)

                col1_container1.markdown(
                    f'<h1 style="text-align: center; font-size: {tamaño1}px; family: "Roboto, sans-serif"; color: {color1}; ">{texto1}</h1>', unsafe_allow_html=True)

                col1_container1.markdown(
                    f'<h1 style="text-align: center; font-size: {tamaño2}px; family: "Roboto, sans-serif"; color: {color2} ">{texto2}</h1>', unsafe_allow_html=True)

                configuraciones = [
                    {
                        'groupby': 'Producto_1',
                        'count_col': 'NIT9',
                        'x_axis_title': None,
                        'y_axis': 'Producto 1',
                        # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                        'col': col2_container1,
                        # Orden deseado de las categorías
                        'order': ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA', 'FIBRA_OPTICA', 'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS'],
                        # Orden deseado de las categorías']ACIONES','MANTENIMIENTO','ESTUDIOS','AUMENTOS_CARGA','FIBRA_OPTICA','REDESELECTRICAS','ILUMINACION','CUENTASNUEVAS0
                        'order_f': ['Disfruta Tranquilo', 'Vida', 'Desempleo', 'Responsabilidad Civil Familiar', 'AP Segurísimo', 'Cyber Esencial', 'SOAT', 'Arriendos']
                    }]

                # espacio(0,0,1,9)
                espacio(col2_container1, 1)
                generar_graficos(
                    Xf, auto_orden=True, configuraciones=configuraciones, color=0, total=False)
                espacio(col3_container1, 9)
                # col3_container1)

                Xf = Xf.loc[:, ['NIT9', 'Producto_1',	'Probabilidad_1',	'Valor_probabilidad1',	'Producto_2',
                                'Probabilidad_2',	'Valor_probabilidad2',	'Producto_3',	'Probabilidad_3', 'Valor_probabilidad3',
                                'Producto_4',	'Probabilidad_4',	'Valor_probabilidad4',	'Producto_5',	'Probabilidad_5',
                                'Valor_probabilidad5',	'Producto_6',	'Probabilidad_6',	'Valor_probabilidad6',	'Producto_7',
                                'Probabilidad_7',	'Valor_probabilidad7',	'Producto_8',	'Probabilidad_8',	'Valor_probabilidad8']]

                dic1 = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA', 'FIBRA_OPTICA', 'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']
                dic2 = ['Disfruta Tranquilo', 'Vida', 'Desempleo', 'Responsabilidad Civil Familiar', 'AP Segurísimo', 'Cyber Esencial', 'SOAT', 'Arriendos']
                Xf = Xf.replace(dict(zip(dic1, dic2)))
                download_excel(Xf, 'Resultado', col=col2_container1)

                # INSTALACIONES
                dona_plotly(df_prob_prod=df_prob_prod, producto='INSTALACIONES',
                            titulo='Disfruta Tranquilo', col=col1_container2)

                # ######## Estudios
                # dona('ESTUDIOS',0 , 2, 'c')
                dona_plotly(df_prob_prod=df_prob_prod, producto='ESTUDIOS',
                            titulo='Vida', col=col1_container2)
                
                # ######## Mantenimiento
                # dona('MANTENIMIENTO',0 , 1, 'Mantenimiento')
                dona_plotly(df_prob_prod=df_prob_prod, producto='MANTENIMIENTO',
                            titulo='Desempleo', col=col2_container2)

                # #FIBRA OPTICA
                # dona('FIBRA_OPTICA',1 , 0, 'Fibras ópticas')
                dona_plotly(df_prob_prod=df_prob_prod, producto='FIBRA_OPTICA',
                            titulo='Responsabilidad Civil Familiar', col=col1_container2)

                # #AUMENTOS_CARGA
                # dona('AUMENTOS_CARGA',0 , 3, 'Aumentos de carga')
                dona_plotly(df_prob_prod=df_prob_prod, producto='AUMENTOS_CARGA',
                            titulo='AP Segurísimo', col=col2_container2)

                # #CUENTASNUEVAS
                # dona('CUENTASNUEVAS',1 , 3, 'Cuentas nuevas')
                dona_plotly(df_prob_prod=df_prob_prod, producto='CUENTASNUEVAS',
                            titulo='Cyber Esencial', col=col2_container2)

                # #ILUMINACION
                # dona('ILUMINACION',1 , 2, 'Iluminación')
                dona_plotly(df_prob_prod=df_prob_prod, producto='ILUMINACION',
                            titulo='SOAT', col=col1_container2)
                
                # #REDESELECTRICAS
                # dona('REDESELECTRICAS',1 , 1, 'Redes eléctricas')
                dona_plotly(df_prob_prod=df_prob_prod, producto='REDESELECTRICAS',
                            titulo='Arriendos', col=col2_container2)



            except UnboundLocalError:
                st.warning(
                    'Error. En el menú de la izquierda cargar archivo en la sección Modelo múltiples clientes')

        with vista2:    # Descriptiva
            try:
                tab4, tab2 = st.tabs(
                    ["Demográfico", "Ventas"])

                df_t, _ = ob.transform_load()  # _graf
                df_t = df_t.copy()
# -----------------------------------------------------

                with tab2:
                    st.write("")
                    st.write("")
                    # st.subheader("VENTAS")
                    # st.write("")
                    col1, col2, col3 = st.columns(spec=[1, 5, 1])

                    col2.subheader("Clientes con producto")

                    # Configuraciones de los gráficos
                    configuraciones = [
                        {
                            'groupby': 'RANGODECOMPRA($)',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Rango de compra',
                            # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                            'col': col2,
                            # Orden deseado de las categorías
                            'order': ['SINCATALOGAR', 'NOCOMPRADOR', 'PEQUENOCOMPRADOR', 'MEDIANOCOMPRADOR', 'GRANCOMPRADOR', 'COMPRADORMEGAPROYECTOS'],
                            'order_f': ['Vida', 'Disfruta Tranquilo', 'Desempleo', 'Cyber Esencial', 'SOAT', 'No comprador']
                        }]
                    generar_graficos(df_t, configuraciones, color=1)

                    col2.subheader("Oferta último semestre")
                    configuraciones = [
                        {
                            'groupby': 'RANGORECURRENCIACOMPRA',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Recurrencia de compra',
                            # 'chart_title': 'Gráfico 2 RangoRecurrenciaCompra',
                            'col': col2,
                            # Orden deseado de las categorías
                            'order': ['SINCATALOGAR', 'NOCOMPRADOR', 'UNICACOMPRA', 'BAJARECURRENCIA', 'RECURRENCIAMEDIA', 'GRANRECURRENCIA'],
                            'order_f': ['Sin catalogar', 'Vida', 'Disfruta Tranquilo', 'Cyber Esencial', 'AP Segurísimo', 'SOAT']
                            # 'order_f':['Sin catalogar', 'No comprador','Unica compra','Baja recurrencia','Recurrencia media','Gran recurrencia']
                        }]
                    generar_graficos(df_t, configuraciones, color=2)

                    col2.subheader("Frecuencia de contacto")
                    configuraciones = [
                        {
                            'groupby': 'TIPOCLIENTE#OPORTUNIDADES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Tipo de cliente por numero de oportunidades',
                            'col': col2,
                            'order': ['SINCATALOGAR', 'NICOMPRA-NICOTIZA', 'SOLOCOTIZAN', 'COTIZANMASDELOQUECOMPRAN','COMPRANYCOTIZAN', 'COMPRANMASDELOQUECOTIZAN', 'SIEMPRECOMPRAN'],  # Orden deseado de las categorías
                            'order_f': ['Sin catalogar', 'Entre 31 y 60 días', 'Entre 61 y 90 días', 'Entre 91 y 120 días',
                                        'Entre 121 y 150 días', 'Entre a 151 y 180 días', 'Mayores a 180 días'],
                        }]
                    generar_graficos(df_t, configuraciones, color=3)

                    col2.subheader("Valor de prima")
                    configuraciones = [
                        {
                            'groupby': 'TIPOCLIENTE$OPORTUNIDADES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Tipo de cliente por valor de oportunidades',
                            # 'chart_title': 'Gráfico 4 TIPOCLIENTE$OPORTUNIDADES',
                            'col': col2,
                            'order': ['SINCATALOGAR', 'NICOMPRA-NICOTIZA', 'SOLOCOTIZAN', 'COTIZANMASDELOQUECOMPRAN',
                                      'COMPRANYCOTIZAN', 'COMPRANMASDELOQUECOTIZAN', 'SIEMPRECOMPRAN'],
                            'order_f': ['Sin catalogar', 'Menos a 40 mil', 'Entre 40 mil y 60 mil', 'Entre 60 mil y 80 mil',
                                        'Entre 80 mil y 100 mil', 'Entre 100 mil y 120 mil', 'Mayor a 120 mil']   # Orden deseado de las categorías
                        }]
                    generar_graficos(df_t, configuraciones, color=4)

                with tab4:
                    st.write("")
                    st.write("")

                    col31, col32, col33 = st.columns(spec=[1, 5, 1])

                    df_c = ob.Agrupar_actividades('ACTIVIDADPRINCIPAL(EMIS)')

                    # --------------------------------------- 16/06/23

                    configuraciones = [
                        {
                            'groupby': 'ACTIVIDADES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Sector económico',
                            'col': col32,
                            'order':  ['SERVICIOS', 'AGROPECUARIO', 'INDUSTRIAL', 'TRANSPORTE',  'COMERCIO', 'FINANCIERO', 'CONSTRUCCION', 'ENERGETICO', 'COMUNICACIONES'],
                            'order_f': ['Generación Baby Boomers', 'Millennials',  'Generación Z', 'Transporte', 'Generación X', 'Generación Baby boomers', 'Construcción', 'Energético', 'Comunicaciones']
                        }
                    ]

                    col32.subheader("Generación digital")
                    # ob.generar_graficos_pie(configuraciones)
                    col32.plotly_chart(ob.generar_graficos_pie(
                        configuraciones, paleta=1, width=500, height=300), use_container_width=True)

                    # -------------------------------------------

                    # Configuraciones de los gráficos barras

                    configuraciones = [
                        {
                            'groupby': 'TAMANOEMPRESA',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Tamaño de la empresa',
                            # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                            'col': col32,
                            # Orden deseado de las categorías
                            'order': ['SINCATALOGAR', 'PEQUENAEMPRESA', 'MEDIANAEMPRESA', 'GRANEMPRESA'],
                            # Orden deseado de las categorías
                            'order_f': ['Sin catalogar', 'Profesional', 'Tecnólogo', 'Bachiller']
                        },
                        {
                            'groupby': 'CATEGORIZACIONSECTORES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Categoria del sector',
                            # 'chart_title': 'Gráfico 2 RangoRecurrenciaCompra',
                            'col': col32,
                            'order': ['SINCATALOGAR', 'OTROSSECTORES', 'SECTORALTOVALOR'],
                            # Orden deseado de las categorías
                            'order_f': ['Sin catalogar', 'Empleado', 'Independiente']
                        },
                        {
                            'groupby': 'ESTATUSOPERACIONAL',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Estatus operacional',
                            # 'chart_title': 'Gráfico 3 TIPOCLIENTE#OPORTUNIDADES',
                            'col': col32,
                            # Orden deseado de las categorías
                            'order': ['NOSECONOCEELESTATUS', 'BAJOINVESTIGACIONLEGAL', 'OPERACIONAL'],
                            # Orden deseado de las categorías
                            'order_f': ['No se conoce el estatus', 'Bajo investigacion legal',  'Operacional']
                        }]

                    col32.subheader("Nivel educativo")
                    generar_graficos(df_c, configuraciones[0:1], color=3)

                    col32.subheader("Situación laboral")
                    generar_graficos(df_c, configuraciones[1:2], color=4)
# Demograficas
                    st.write("")
                    st.write("")

                    col000, col0, col002, col003 = st.columns(
                        spec=[0.35, 5, 1, 0.25])

                    # Configuraciones de los gráficos
                    configuraciones = [
                        {
                            'groupby': 'CATEGORIADEPARTAMENTO',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Categoria de departamento',
                            # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                            'col': col0,
                            # Orden deseado de las categorías
                            'order': ['NOSECONOCEELDEPARTAMENTO', 'OTROSDEPARTAMENTOS', 'COSTA', 'CUNDINAMARCA', 'BOGOTADC'],
                            # Orden deseado de las categorías
                            'order_f': ['No se conoce el departamento',  'Otros departamentos',  'Costa',  'Cundinamarca',  'Bogotá DC']

                        }]

            except UnboundLocalError:
                st.warning(
                    'No ha cargado un archivo para procesar!. En el menú de la izquierda cargar archivo en la sección Modelo Múltiples Variables')


if __name__ == '__main__':
    main()
