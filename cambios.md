# Reunión 26/9/2025

## RAG
Mejore el RAG, por las pruebas parece haber funcionado

1. Probe hacer mas preprocesamiento del pdf porque el loader no detectaba bien los títulos y secciones entonces perdía info del contexto. Funcionó relativamente bien pero me di cuenta que era muy especifico a este documento y tal vez no generalizaba bien a otros archivos


2. Investigando más técnicas lo que encontre fue [Parent Document Retriever](https://python.langchain.com/docs/how_to/parent_document_retriever/). En el RAG tradicional toma un documento lo separa en splits y se hace el embedding de cada split. Acá a cada split (Parent Doc) lo vuelve a dividir (Child docs) y sobre esos se hace el embedding. En el retrieval, primero recupera los hijos y despues los padres de esos para devolver documentos más grandes con más contexto.
Basicamente los embeddings son más especificos sin hacer que el LLM pierda contexto
 
![alt text](images/image.png)

![alt text](images/image3.png)

Hay otras alternativas pero por el momento me parecía que era agregar mas complejidad y como esto funcionaba no las probe: 
 - [Rerankers](https://www.pinecone.io/learn/series/rag/rerankers/): *Maximize retrieval recall by retrieving plenty of documents and then maximize LLM recall by minimizing the number of documents that make it to the LLM. To do that, we reorder retrieved documents and keep just the most relevant for our LLM — to do that, we use reranking.*
 - [Hybrid Search](https://python.langchain.com/docs/how_to/hybrid/): *The standard search in LangChain is done by vector similarity. However, a number of vector store implementations also support more advanced search combining vector similarity search and other search techniques (full-text, BM25, and so on). This is generally referred to as "Hybrid" search.*
  
 >[!note] Volviendo al preprocesamiento
Creo que va a valer la pena hacerlo igualmente, por ahora la carga es simple y sola saca texto de los archivos, hay formas de parsear tablas, imagenes, gráficos, etc. Una opción interesante es [Docling](https://python.langchain.com/docs/integrations/document_loaders/docling/)

### Preguntas de prueba y respuestas que me dio ChatGPT, para comparar con el doc de prueba

- ¿Cuáles son las cinco fases del modelo de proceso propuesto?
> Project Definition
> 
> Domain Survey
> 
> Data and Knowledge Exploitation
> 
> Data Understanding
> 
> Data Preparation


- ¿Qué tareas se realizan en la fase de Project Definition?

> Definir el propósito del sistema inteligente.
>
>Especificar el tipo de necesidad (clasificación, segmentación, predicción, etc.).
>
>Identificar características iniciales: stakeholders, repositorios de datos disponibles, expertos de dominio.
>
>Establecer una taxonomía inicial que guíe la explotación.

- ¿Qué objetivo tiene la fase de Domain Survey?

>Determinar el alcance y limitaciones del proyecto.
>
>Realizar un estudio de factibilidad para decidir si es posible desarrollar un sistema inteligente que responda a la necesidad.


- ¿Qué beneficios busca la metodología respecto a la calidad de los datos?

>Mitigar incertidumbre.
>
>Mejorar calidad de datos.
>
>Hacer más eficiente la ejecución de proyectos de análisis.


- ¿Qué sección menciona explícitamente la “Data and Knowledge Exploitation phase”?

> La sección Process Model (Fase 3 en la lista de cinco fases).

## Carga de documentos

Ya funciona, sigue el mismo procedimiento explicado recien. 

Por el momento solo permite cargar PDF para simplificar el proceso. Como dije en la nota estaría bueno agregar preprocesamiento para poder incluir diferentes fuentes y no solo lo textual. 

El documento original de prueba se sigue cargando por defecto, para hacerlo mas rápido

Me gustaría agregar una manera de ver los documentos cargados y poder eliminarlos 

## En progreso: Persistir chat y documentos cargados, distintos chats.

Empece a hacer que los docs subidos, el historial de chats, etc. se persista y poder crear distintos chats, con id y thread unicos, para mantener conversaciones separadas.