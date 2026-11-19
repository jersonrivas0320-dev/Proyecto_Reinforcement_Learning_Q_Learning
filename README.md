# Optimización del abastecimiento de inventarios mediante Q-Learning

## 1. Descripción del proyecto

Este proyecto desarrolla un modelo de aprendizaje por refuerzo basado en el algoritmo Q-Learning para optimizar las decisiones de abastecimiento de inventarios en una tienda de calzado.

El entorno simula tres categorías de productos:

- Deportivo.
- Formal.
- Sandalias.

El agente aprende a seleccionar cantidades de abastecimiento según los niveles de inventario disponibles, considerando ingresos por ventas, costos de almacenamiento, agotamiento de productos y costos de abastecimiento.

## 2. Objetivo

Evaluar la capacidad del algoritmo Q-Learning para mejorar las decisiones de abastecimiento y comparar su desempeño con una estrategia tradicional de abastecimiento fijo.

## 3. Tecnologías utilizadas

- Python.
- NumPy.
- Pandas.
- Matplotlib.
- Google Colab.
- Jupyter Notebook.
- Algoritmo Q-Learning.

## 4. Archivos del repositorio

- `Proyecto_Reinforcement_Learning_Q_Learning.ipynb`: notebook con el desarrollo del proyecto, código, entrenamiento y resultados.
- `Proyecto_Reinforcement_Learning_Q_Learning.py`: script de Python con el código exportado del notebook.

## 5. Parámetros principales

- 3 categorías de productos.
- 27 estados posibles.
- 27 acciones posibles.
- 10.000 episodios de entrenamiento.
- 30 días simulados por episodio.
- 300.000 días de entrenamiento.

## 6. Instalación y ejecución

Se recomienda utilizar Google Colab para ejecutar el proyecto:

1. Ingresar a https://colab.research.google.com/
2. Seleccionar la opción para cargar un notebook.
3. Subir el archivo `Proyecto_Reinforcement_Learning_Q_Learning.ipynb`.
4. Ejecutar las celdas en orden.
5. Revisar los resultados, tablas y gráficas generadas.

Las principales bibliotecas utilizadas pueden instalarse con:

```bash
pip install numpy pandas matplotlib
```

El proyecto utiliza datos de demanda generados mediante simulación, por lo que no requiere archivos externos de datos.

## 7. Resultados principales

En las 100 evaluaciones controladas de 1.000 días cada una se obtuvieron los siguientes resultados:

- Recompensa promedio de Q-Learning: 170.772,76.
- Recompensa promedio de la estrategia tradicional: -14.342,30.
- Diferencia promedio: 185.115,06.

El análisis estadístico mediante una prueba t para muestras relacionadas mostró una diferencia estadísticamente significativa entre ambas estrategias (p menor que 0,001).

## 8. Autor

Yerson Ronaldo Rivas Ruiz

Especialización en Analítica y Ciencia de Datos

Universidad de Cundinamarca

2026
