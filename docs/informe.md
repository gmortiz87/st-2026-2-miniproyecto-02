# Informe del miniproyecto 2

Llene cada sección reemplazando los marcadores `{{...}}`, sin quitar las
secciones. Las cifras salen de lo que imprime `python miniproyecto.py` y las
figuras son las que ese guion deja en `figuras/`. Las fórmulas van en LaTeX
(`$...$` o `$$...$$`).

## 1. La serie y el modelo a superar

La serie es la **lluvia mensual (mm)** de la estación **CDEA El Topacio**
(código 2629020601), en la cuenca del río Pance (Cali), suministrada por la
**CVC** para uso académico. Va de **enero de 2015 a febrero de 2026: 134
meses**. Se trae la decisión del miniproyecto 1: **no hay huecos** (ningún mes
quedó por debajo del 80 % de días con dato), y los **7 atípicos** del residuo
de una STL robusta (2022-02, 2022-05, 2022-06, 2023-03, 2023-10, 2024-04 y
2025-12) **se conservan**: son lluvia real o meses registrados con totales
diarios, un posible artefacto de medición que no se puede corregir sin otra
fuente. `preparar` lo comprueba con `assert` y los imprime.

![La serie](../figuras/serie.png)

El modelo a superar es el mejor del miniproyecto 1, un
**SARIMAX$(0,0,0)(0,1,1)_{12}$ con el número de días del mes** como exógena,
ajustado sobre la lluvia dividida por 100:

$$y_t - y_{t-12} = \beta\,(d_t - d_{t-12}) + \varepsilon_t + \Theta\,\varepsilon_{t-12},
\qquad \Theta = -0.860,\quad \beta = -100.9\ \text{mm por día},$$

con $d_t$ los días del mes $t$ (ajuste con toda la serie). Tiene **3
parámetros** ($\beta$, $\Theta$ y la varianza). Pronostica cada mes con un
patrón casi fijo, cercano al promedio de ese mes en los años anteriores. El
efecto de los días del mes sale de unos pocos febreros bisiestos, con signo
sin sentido físico y valor p 0.095: probablemente es casualidad, como se
concluyó en el miniproyecto 1.

| Modelo | RMSE | MAE | MAPE (%) |
|---|---|---|---|
| SARIMAX$(0,0,0)(0,1,1)_{12}$ + días del mes | **113.79** | **91.75** | 54.31 |
| Estacional ingenuo | 154.93 | 112.39 | **43.90** |

Las métricas **coinciden exactamente** con las del miniproyecto 1: mismo
*backtest* (13 orígenes, de 2024-02 a 2025-02, horizonte 1 a 12), misma
preparación y misma clase de statsmodels (`ARIMA`).

**El MAPE de 54 %.** En la sustentación del miniproyecto 1 se preguntó cómo
mejorarlo. Buena parte viene de la métrica: el MAPE divide por la lluvia real,
y en un mes seco un error pequeño en milímetros es enorme en porcentaje. Los
meses con menos de 100 mm son el 28 % de los pronósticos y aportan el 59 % del
MAPE; solo dos meses, **enero de 2025 (53.8 mm) y junio de 2024 (31.4 mm),
aportan el 50 %**. Por eso este trabajo decide por **RMSE**, y en la sección 5
se vuelve sobre la pregunta con todos los modelos.

## 2. La tabla supervisada

Cada mes $t$ es una fila con 14 columnas, todas calculadas con meses
anteriores a $t$:

| Columna | Fórmula | Qué aporta |
|---|---|---|
| `rezago_1` … `rezago_12` | $z_{t-k}$, $k = 1, \dots, 12$ | la lluvia de los doce meses anteriores; `rezago_12` es el mismo mes del año pasado |
| `mes` | el número del mes de $t$ (1 a 12) | la temporada: se conoce de antemano por el calendario |
| `media_3` | $\frac{1}{3}(z_{t-1} + z_{t-2} + z_{t-3})$ | si el trimestre reciente viene húmedo o seco |

**Qué aprenden los modelos: el nivel**, es decir, la lluvia del mes tal cual
($z_t = y_t$), no su diferencia con el mismo mes del año anterior. Se
decidió con los cuatro modelos en sus valores por defecto, en el mismo
*backtest*:

| Modelo | RMSE con el nivel | RMSE con la diferencia |
|---|---|---|
| Random Forest | **135.0** | 161.4 |
| XGBoost | **134.1** | 164.3 |
| KNN | **145.2** | 169.0 |
| SVR | 163.3 | **152.0** |

El nivel gana en tres de cuatro. La diferencia anual de la lluvia es casi
ruido: su única memoria clara es la del rezago 12 (la que captura el SARIMA),
y esa ya está en la tabla como columna. Con el nivel, además, la tabla tiene
**98 filas** antes del *backtest* (2016-01 a 2024-02) en lugar de 86, porque
no se pierde un año en la diferencia. Solo el SVR mejora con la diferencia, y
es porque así casi repite el año anterior: su RMSE (152.0) queda pegado al del
estacional ingenuo (154.9).

**La prueba de que no hay fuga.** `comprobar_sin_fuga` toma cada uno de los
134 meses, le suma 10 000 mm a su lluvia y vuelve a armar la tabla: la fila de
ese mes, que es lo que el modelo ve para pronosticarlo, tiene que quedar
idéntica. Si alguna columna cambia, lanza `AssertionError`. El guion imprime:

```
sin fuga: en los 134 meses, cambiar y(t) no cambia la fila de t (14 columnas: rezago_1, rezago_12, mes, media_3, ...)
```

La prueba sí detecta la fuga cuando existe: si la media móvil se calcula sin
el `shift(1)`, falla en la fila de 2015-03 y señala la columna `media_3`.

**El pronóstico recursivo.** Para pronosticar 12 meses desde un origen,
`recursivo` agrega el mes siguiente sin dato, arma su fila con la tabla,
pronostica, escribe el pronóstico como si fuera un dato y pasa al mes
siguiente. Desde el segundo mes, los rezagos más recientes son pronósticos,
no observaciones, así que los errores se acumulan con el horizonte.

**El escalador** vive dentro de cada modelo (KNN y SVR son `make_pipeline`
con `StandardScaler`; las redes escalan en `Red.fit`). Como `con_tabla` crea y
ajusta un modelo nuevo en cada origen con las filas de ese entrenamiento, la
media y la desviación salen solo de los meses anteriores al origen.

## 3. Aprendizaje automático

**Cómo se eligieron los hiperparámetros.** Con `GridSearchCV` y
`TimeSeriesSplit` sobre la tabla de `serie.iloc[:-24]`, es decir, las **98
filas de 2016-01 a 2024-02**. Los 24 meses del *backtest* no participan en
la elección: si decidieran, el *backtest* dejaría de ser una prueba y pasaría
a ser parte del ajuste. La validación son **4 cortes de 12 meses** al final de
esa tabla: el primero entrena con las 50 filas más antiguas y valida el año
siguiente, y cada corte agrega un año. Nunca se valida con meses anteriores a
los de entrenamiento. Se elige la combinación de menor RMSE promedio.

Es el error **a un mes**, con rezagos observados. El *backtest* después mide
el pronóstico recursivo a 12 meses, que es más difícil. Por eso los RMSE de
validación (105 a 112 mm) son menores que los del *backtest* (131 a 150 mm).

| Modelo | Rejilla | Elegido | RMSE de validación |
|---|---|---|---|
| KNN | `n_neighbors` ∈ {5, 10, 15, 20} | 20 vecinos | 112.0 |
| SVR | `C` ∈ {1, 10, 100}, `epsilon` ∈ {1, 10, 30} mm | C = 10, ε = 10 mm | 107.7 |
| Random Forest | `max_depth` ∈ {2, 3, 6}, `min_samples_leaf` ∈ {1, 3, 6} | profundidad 2, 3 por hoja | 108.6 |
| XGBoost | `max_depth` ∈ {2, 3, 5}, `learning_rate` ∈ {0.01, 0.03, 0.1} | profundidad 5, tasa 0.01 | **105.7** |

**Las rejillas.** En el SVR, el objetivo es la lluvia en mm sin escalar, así
que $\varepsilon$ es una tolerancia en milímetros: el valor por defecto
(0.1 mm) no tiene sentido y se buscó entre 1 y 30 mm. Una primera rejilla
(KNN de 3 a 15 vecinos, SVR con C de 10 a 1000, Random Forest con
profundidad 3, 6 o sin límite, XGBoost con tasa de 0.03 a 0.3) eligió en el
**borde suave** de tres modelos. Se corrió una vez hacia ese borde,
decidiendo solo con la validación, y la validación **siguió eligiendo lo más
suave**: más vecinos, árboles más bajos y aprendizaje más lento. Es una
señal de la serie, no de la rejilla: un modelo suave pronostica cerca del
promedio de cada mes, y la validación lo prefiere porque, fuera de la
estacionalidad, la tabla trae poca información que no sea ruido. Por eso no
se siguió corriendo la rejilla, que habría terminado en "pronosticar el
promedio".

![Importancia de las variables](../figuras/importancia.png)

**Qué variables pesan.** En los dos modelos, la primera es **`rezago_12`**,
la lluvia del mismo mes del año anterior: 0.41 de la importancia en Random
Forest y 0.20 en XGBoost. Le siguen:

- en Random Forest, `rezago_3` (0.18) y luego `mes` y `rezago_9`, con 0.06
  cada uno;
- en XGBoost, que reparte más, `media_3` (0.11), `mes` (0.10) y `rezago_3`
  (0.09).

Coincide con la PACF del miniproyecto 1. Sobre $y_t - y_{t-12}$, la PACF
salía de la banda en los rezagos 7, 12, 20 y 24, y la barra aislada en 12 es
la que llevó al término estacional del SARIMA. Aquí `rezago_12` es,
de lejos, lo que más usan los árboles. En cambio, **`rezago_7` es la variable
menos usada por Random Forest** (0.009) y de las menos usadas por XGBoost
(0.04). Eso respalda lo que se concluyó entonces: la barra en 7 probablemente
es azar. Los rezagos 20 y 24 no están en la tabla, que llega hasta 12. Que
`mes` y `media_3` pesen dice que los árboles también usan la temporada y el
trimestre reciente, algo que el SARIMA no ve.

**Resultados en el *backtest*** (13 orígenes, horizonte 1 a 12):

| Modelo | RMSE | MAE | MAPE (%) | Parámetros |
|---|---|---|---|---|
| SARIMAX$(0,0,0)(0,1,1)_{12}$ + días del mes | **113.79** | **91.75** | 54.31 | 3 |
| KNN, 20 vecinos | 131.82 | 115.11 | 76.41 | 1540 |
| XGBoost | 133.45 | 106.61 | 67.66 | 7946 |
| Random Forest | 138.01 | 115.58 | 70.30 | 700 |
| SVR | 150.05 | 131.56 | 78.68 | 1531 |
| Estacional ingenuo | 154.93 | 112.39 | **43.90** | 0 |

**Entre los cuatro gana KNN por RMSE (131.8), casi empatado con XGBoost
(133.5), que tiene el menor MAE (106.6).** Los cuatro le ganan al estacional
ingenuo en RMSE, pero **ninguno al SARIMAX**: el mejor queda 18 mm por encima,
con cientos o miles de "parámetros" frente a 3. En los modelos de tabla, los
parámetros son lo que `contar_parametros` cuenta: datos guardados en KNN y
SVR, nodos en los árboles.

Hay un detalle que vale la pena decir. Con la primera rejilla, que en
validación era un poco peor, el *backtest* daba un poco mejor (KNN 131.5,
XGBoost 131.5, Random Forest 135.4). Se dejó la segunda, porque **la elección
la hace la validación y no el *backtest***, y una diferencia de 2 a 3 mm entre
rejillas es menor que lo que separa a los modelos.

**El MAPE.** Los cuatro tienen un MAPE peor que el SARIMAX (67 a 79 % frente
a 54 %). Al ser suaves, pronostican cerca del promedio en los meses secos, y
ahí el error relativo se dispara. Es la misma lección de la sección 1: el
MAPE castiga los meses de poca lluvia, y un modelo que se acerca al promedio
la paga cara.

## 4. Aprendizaje profundo

{{Las tres redes: arquitectura, número de parámetros y cómo recibe cada una
la entrada (columnas sueltas o secuencia). Cuál es la tercera red y por qué
la eligió el grupo.}}

![Curvas de pérdida](../figuras/perdida.png)

{{Lectura de las curvas: dónde empieza el sobreajuste, cuántas épocas se
eligieron para cada red y qué tamaño de MLP quedó.}}

{{La mejor red con cinco semillas: el RMSE de cada una, la media y la
desviación. Si la diferencia con el clásico es mayor o menor que esa
desviación.}}

## 5. La tabla y la elección

{{Copie aquí `resultados/metricas.csv` como tabla de Markdown, ordenada por
RMSE.}}

| Modelo | RMSE | MAE | MAPE (%) | Parámetros | Segundos |
|---|---|---|---|---|---|
| {{...}} | {{...}} | {{...}} | {{...}} | {{...}} | {{...}} |

![Pronósticos desde el último origen](../figuras/pronosticos.png)

{{Si alguna red no supera al clásico, a qué lo atribuye el grupo y qué dato
de la serie lo muestra: el número de meses frente al número de parámetros,
el mes en que falla, la curva de pérdida o la variación entre semillas.}}

## 6. Conclusión

{{El modelo que el grupo pondría en uso y lo que se paga por él frente al
clásico: cuántos datos necesita, cuánto tarda en el *backtest* y qué tanto
se puede explicar su pronóstico.}}
