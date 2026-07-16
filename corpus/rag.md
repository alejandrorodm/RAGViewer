# Sistemas RAG: Generación Aumentada por Recuperación

## Qué es un RAG

La Generación Aumentada por Recuperación (RAG, por sus siglas en inglés) es una arquitectura que combina un modelo de lenguaje con un sistema de búsqueda de información. En lugar de depender únicamente de lo que el modelo memorizó durante su entrenamiento, un RAG recupera fragmentos relevantes de una base de conocimiento externa y los inserta en el contexto del modelo antes de generar la respuesta. Esto reduce las alucinaciones y permite responder sobre datos privados o recientes.

El flujo típico tiene dos fases. En la fase de ingesta, los documentos se trocean en fragmentos, cada fragmento se convierte en un vector numérico mediante un modelo de embeddings, y esos vectores se almacenan en una base de datos vectorial. En la fase de consulta, la pregunta del usuario se convierte en un vector con el mismo modelo, se buscan los fragmentos más cercanos en el espacio vectorial, y los mejores candidatos se pasan al modelo de lenguaje como contexto.

## Chunking: el arte de trocear documentos

El chunking consiste en dividir los documentos en fragmentos manejables. Es una de las decisiones con más impacto en la calidad de un RAG. Si los fragmentos son demasiado pequeños, pierden contexto y se vuelven ambiguos. Si son demasiado grandes, mezclan varios temas y el vector resultante queda diluido, además de desperdiciar la ventana de contexto del modelo.

La estrategia más simple es el troceo de tamaño fijo: cortar cada N caracteres. Funciona, pero puede partir frases por la mitad. Para mitigarlo se usa el solapamiento (overlap): cada fragmento repite los últimos caracteres del anterior, de modo que la información que cae en una frontera aparece completa en al menos uno de los dos fragmentos.

Estrategias más sofisticadas respetan la estructura del texto: cortar por frases, por párrafos o por encabezados de sección. El chunking semántico va más allá y agrupa frases consecutivas mientras sus embeddings sean similares, cortando cuando detecta un cambio de tema.

## Embeddings y espacio vectorial

Un modelo de embeddings transforma texto en un vector de números reales, típicamente de entre 256 y 3072 dimensiones. La propiedad clave es que textos con significados parecidos producen vectores cercanos en ese espacio. La palabra «perro» y la palabra «cachorro» quedan cerca; «perro» y «hipoteca» quedan lejos.

Esta propiedad convierte la búsqueda semántica en un problema geométrico: encontrar los fragmentos relevantes para una pregunta equivale a encontrar los vectores más cercanos al vector de la pregunta. A diferencia de la búsqueda por palabras clave, la búsqueda vectorial encuentra fragmentos que hablan del mismo concepto aunque no compartan ni una sola palabra con la consulta.

## Métricas de similitud

Para medir cuán cerca están dos vectores existen varias métricas. La similitud del coseno mide el ángulo entre los dos vectores e ignora sus longitudes: vale 1 si apuntan en la misma dirección, 0 si son perpendiculares y -1 si son opuestos. Es la métrica más usada en recuperación semántica porque la dirección de un embedding codifica el significado, mientras que su longitud suele ser ruido.

La distancia euclídea mide la longitud del segmento recto que une las puntas de los dos vectores. A diferencia del coseno, sí es sensible a la magnitud. Con vectores normalizados a longitud 1, el ranking por distancia euclídea coincide exactamente con el ranking por coseno.

El producto escalar (dot product) combina ángulo y magnitud: premia vectores que apuntan en la misma dirección y además son largos. Algunos modelos de embeddings se entrenan específicamente para usarse con producto escalar.

## Reranking: una segunda opinión

La búsqueda vectorial es rápida pero imprecisa: comprime todo el significado de un fragmento en un solo vector. El reranking añade una segunda etapa más costosa y más precisa. Un modelo cross-encoder recibe la pregunta y un fragmento candidato juntos, y produce una puntuación de relevancia leyendo ambos textos a la vez, palabra por palabra.

El patrón habitual es recuperar entre 20 y 100 candidatos con búsqueda vectorial barata, y luego reordenarlos con el cross-encoder, quedándose con los 3 a 5 mejores. Este embudo de dos etapas mejora notablemente la precisión: fragmentos que parecían relevantes por proximidad vectorial caen al fondo, y fragmentos que la búsqueda vectorial subestimó suben a las primeras posiciones.
