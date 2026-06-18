# Changelog

**Languages / زبان‌ها:** [English](CHANGELOG.md) · [فارسی](CHANGELOG.fa.md) · [العربية](CHANGELOG.ar.md) · [Español](CHANGELOG.es.md) · [简体中文](CHANGELOG.zh.md)

## Extracción de DOCX / XLSX para documentos reenviados (2026-06-17)

Extracción de texto de documentos Office reenviados, realizada localmente y sin clave de API.

### Comportamiento
- Reenviar un archivo `.docx` o `.xlsx` (detectado por la extensión o el tipo MIME) ahora extrae su texto y lo almacena en una bandeja buscable. A diferencia de OCR/PDF, que requiere Gemini, esta ruta es **totalmente local** (no se necesita clave de API ni red).

### Diseño
- Un módulo puro `office.py` con `detect_office_kind`, `extract_docx_text`, `extract_xlsx_text` y el despachador `extract_office_text` — usando solo la biblioteca estándar (`zipfile` + `xml.etree`). La coincidencia de etiquetas se realiza sobre el local-name para que se admitan ambas variantes de espacio de nombres (transitional/strict).
- DOCX: texto de párrafos (incluido dentro de tablas) uniendo los runs. XLSX: resolución de sharedStrings + cadenas inline + números, con separadores de tabulación/nueva línea y separación de hojas.
- `NotebookBot._media_route` devuelve la nueva ruta `"office"`, y `_process_forwarded_media` la ejecuta sin necesidad de un servicio ni de `enabled`.

### Pruebas
- `tests/test_office.py`: detección de tipo, extracción de DOCX (unión de runs/párrafos), XLSX (shared/inline/número, varias hojas, sharedStrings ausentes) y rechazo de un formato desconocido.
- `tests/test_inbox_media.py`: enrutamiento de documentos Office y orquestación completa sin ningún servicio.

## 0.3.0 — Conjunto de funciones de Notebook (2026-06-17)

Resumen de la versión: además del núcleo del MVP (ingesta, transcripción, búsqueda/pregunta, bandeja de reenviados, motor de reglas, trabajos de importación, MCP), se añadió lo siguiente:

- Infraestructura: CI (ruff + pytest) en cada push/PR.
- Organización: agrupación por temas (`/topics` con etiquetado por LLM), `/timeline`, colecciones (`/collection`, además de `/summarize`/`/export --collection`), gestión de etiquetas (`/tag rename|delete`).
- Contenido: procesamiento de medios reenviados (transcripción + OCR/PDF + extracción local de DOCX/XLSX), reglas de IA (`/rule add-ai`) y etiquetado automático opcional (`/airules`), reenvío automático a un canal de archivo (`/setarchive`).
- Salida/revisión: `/digest`, `/export` (Markdown), `/stats`, `/recent` y los endpoints web `/api/{stats,recent,timeline}` con un panel de Biblioteca en el dashboard.
- Nuevas herramientas MCP: `list_topics`, `timeline`, `archive_stats`, `list_recent`.

Sin nuevas dependencias; conjunto de pruebas completo. Los detalles de cada elemento están en las entradas siguientes.

## Etiquetado automático opcional con IA en los reenvíos (2026-06-12)

Completando las reglas de IA: ejecutarlas automáticamente en los nuevos reenvíos, de forma opcional.

### Comportamiento
- `/airules on|off` (desactivado de forma predeterminada) controla si las reglas de IA se ejecutan automáticamente en cada nuevo reenvío. Cuando está activado, se realiza una llamada al LLM por elemento; **las importaciones masivas de canales nunca se clasifican automáticamente** (coste controlado).
- Requiere una clave de Gemini; si no hay ninguna clave presente al activarlo, se te notifica.

### Diseño
- `IngestionPipeline` ganó un parámetro opcional `ai_classifier`; `_apply_rules` también aplica las reglas de IA cuando está presente (absorbiendo los errores). Solo la ruta de la bandeja de reenviados lo conecta (y solo cuando el usuario ha optado por activarlo y tiene una clave).
- Una columna `ai_autotag` en `bot_users` con una migración idempotente; `Repository.set_ai_autotag` y el helper `_ai_classifier_for_user`.

### Pruebas
- `tests/test_ai_autotag.py`: aplicar reglas de IA solo con un clasificador, absorber los errores del clasificador, persistir la configuración, el gating en `_ai_classifier_for_user` y el manejador de `/airules`.

## Consultar colecciones (2026-06-12)

Completando las colecciones: ahora un cuaderno se puede resumir o exportar.

### Comportamiento
- `/summarize --collection <name>` y `/export --collection <name>` resumen o exportan a Markdown todos los elementos que tengan cualquiera de las etiquetas de la colección (de `items_for_tags`). Si la colección no existe, se muestra un mensaje apropiado.

### Componentes
- El helper puro `NotebookBot._extract_collection(args)` (análisis del flag `--collection <name>`) y `_collection_items` (resolución de una colección a elementos + una etiqueta de alcance).

### Pruebas
- `tests/test_collections.py`: `_extract_collection`, `/summarize --collection` (unión de etiquetas y una colección inexistente) y `/export --collection` (contenido correcto del documento).

## Colecciones / cuadernos (2026-06-12)

Agrupar varias etiquetas bajo un único "cuaderno" (colección).

### Comportamiento
- `/collection new <name>` (nombre de una sola palabra), `/collection add <name> <tag>`, `/collection list`, `/collection remove <name>` y `/collection show <name>`, que muestra los elementos que tienen cualquiera de las etiquetas de la colección (distintos, los más nuevos primero). Todo limitado al propietario.

### Componentes
- Las tablas `collections` y `collection_tags` (con un índice único en `(owner_id, name)`).
- Métodos de Repository: `create_collection`, `add_collection_tag`, `list_collections`, `collection_tags`, `remove_collection` e `items_for_tags(owner_id, tags, limit)` (unión distinta).

### Pruebas
- `tests/test_collections.py`: CRUD y adición de etiquetas, aislamiento por usuario, `items_for_tags` (unión/distinto/alcance) y las rutas completas del manejador (new/add/list/show/remove + errores).

## Panel de Biblioteca del dashboard (2026-06-12)

- El panel web ganó una tarjeta "Library" que, al hacer clic en un botón, llama a `/api/stats` y `/api/recent` y muestra un resumen del archivo (recuento de elementos/fuentes/etiquetas y tipos de medios) y los elementos más recientes.
- Una prueba de humo (smoke test) en `tests/test_web_api.py` que comprueba la presencia del panel y las referencias a los endpoints en `INDEX_HTML`.

## API web: stats / recent / timeline (2026-06-12)

Paridad del panel web con las nuevas capacidades (una capa de API JSON).

### Comportamiento
- Tres endpoints de solo lectura `GET /api/stats`, `GET /api/recent?limit=N` y `GET /api/timeline?granularity=month|day` que devuelven el archivo del dashboard (propietario fijo `0`). Como el resto de la API, están protegidos con `WEB_API_TOKEN` (o loopback cuando no se ha establecido ningún token) y usan los mismos métodos de repositorio y las funciones puras `recent_rows`/`build_timeline`/`archive_stats`.
- El helper `_query_int` para una lectura segura y acotada de los parámetros de consulta numéricos.

### Fuera de alcance (seguimiento)
- Mostrar estos datos en la interfaz HTML del dashboard (por ahora, solo API JSON).

### Pruebas
- `tests/test_web_api.py`: salida de `/api/stats`, `/api/recent` (límite máximo y ordenación) y `/api/timeline`, y el requisito de autenticación cuando no es local.

## Explorar elementos recientes (2026-06-12)

Exploración rápida de los elementos más recientes — un complemento de `/timeline` y `/digest`.

### Comportamiento
- `/recent [n]` (10 de forma predeterminada, máximo 50) muestra los elementos más recientes con fuente, fecha, fragmento y enlace. La herramienta MCP `list_recent` devuelve la misma lista.

### Componentes
- El módulo puro `recent.py` con `recent_rows(items, *, limit, snippet_chars)` (normalización de espacios en blanco y truncamiento del fragmento); alimentado por `timeline_items` (los más nuevos primero).

### Pruebas
- `tests/test_recent.py`: normalización/límite de `recent_rows`, fuente desconocida, el manejador (ordenación de nuevo→antiguo y un archivo vacío) y la herramienta MCP.

## Gestión de etiquetas (2026-06-12)

Gestión manual de etiquetas (renombrar / fusionar / eliminar).

### Comportamiento
- `/tag rename <old> -> <new>` renombra una etiqueta; si `<new>` ya existe, las dos etiquetas se fusionan (sin un error de clave duplicada). `/tag delete <tag>` elimina la etiqueta de todos los elementos. Ambos limitados al propietario.

### Componentes
- `Repository.rename_tag` (INSERT OR IGNORE y luego DELETE para una fusión segura) y `Repository.delete_tag`.

### Pruebas
- `tests/test_tag_management.py`: renombrar, fusionar con una etiqueta existente, eliminar, aislamiento por usuario y las rutas del manejador (rename/delete/usage/missing).

## Resumen periódico reciente (digest) (2026-06-12)

"Dime qué me perdí" — un resumen con IA del contenido reciente.

### Comportamiento
- `/digest [days]` (predeterminado 7, rango de 1 a 90): resume el contenido registrado en los últimos N días usando el mismo motor `summarize`. Sin una clave de Gemini, recurre a un resumen simple (recuento de elementos + fuentes); sin contenido reciente, muestra un mensaje apropiado.

### Componentes
- `Repository.recent_items(owner_id, since_date, limit)` — elementos con `message_date >= since` (los más nuevos primero).

### Pruebas
- `tests/test_digest.py`: filtrado por fecha/propietario en `recent_items` y las rutas del manejador (sin contenido, recurso sin clave y uso de summarize con una clave).

## Estadísticas del archivo (2026-06-12)

Una visión general del archivo mediante `/stats` y la herramienta MCP `archive_stats`.

### Comportamiento
- `/stats` muestra el recuento de elementos, fuentes y etiquetas, los recuentos por tipo de medio y el rango temporal (primera/última fecha). La herramienta MCP `archive_stats` devuelve la misma salida.

### Componentes
- `Repository.archive_stats(owner_id)` con consultas de agregación (limitadas al propietario).
- El módulo puro `stats.py` con `format_stats(stats)`.

### Pruebas
- `tests/test_stats.py`: formateo (vacío/poblado), agregación y alcance por usuario en `archive_stats` y la herramienta MCP.

## Exportación a Markdown (2026-06-12)

Exportar el archivo a un archivo Markdown descargable.

### Comportamiento
- `/export [--source <url>] [--tag <tag>]` convierte todo el archivo, una sola fuente o una sola etiqueta en un documento Markdown (con título, fuente, enlace y texto de cada elemento) y lo envía al usuario como un archivo.

### Componentes
- El módulo puro `export.py` con `build_markdown_export(scope_label, items)`.
- `TelegramBotApi.send_document` para subir el archivo.
- El manejador `_handle_export`, que escribe el documento en un archivo temporal, lo envía y limpia.

### Pruebas
- `tests/test_export.py`: estructura de Markdown y campos ausentes, y la orquestación del manejador (envío del documento con el contenido correcto y el mensaje de archivo vacío).

## Etiquetas de temas con LLM (2026-06-12)

Nombrar los grupos de `/topics` con un LLM (si hay una clave de Gemini presente).

### Comportamiento
- `/topics` y la herramienta MCP `list_topics` ahora construyen la etiqueta de cada grupo con una llamada al LLM (Gemini) basada en textos de muestra del grupo; sin una clave, o ante un error/respuesta vacía, recurren a una etiqueta basada en los términos significativos más frecuentes (`top_terms`). La salida del bot está escapada en HTML.

### Diseño
- En `clustering.py`: las funciones puras `build_label_prompt` y `parse_topic_label`, y `label_cluster(texts, *, generate)` con una llamada al LLM inyectada; `build_topics` ganó un parámetro opcional `namer` que construye una etiqueta por grupo y recurre a la alternativa en caso de error/vacío.
- En `bot.py` y `mcp_server.py`, el namer solo se construye cuando hay una clave de Gemini presente.

### Pruebas
- `tests/test_clustering.py`: construcción/análisis del prompt de la etiqueta, `label_cluster` con un generate inyectado y `build_topics` con un namer (etiquetado y alternativa en caso de error/vacío).

## Procesamiento de medios reenviados (2026-06-12)

Completando la bandeja de reenviados: los medios reenviados se descargan y se convierten en texto buscable.

### Comportamiento
- Los archivos de audio/vídeo/voz/video_note se transcriben automáticamente (el mismo `TranscriptionService`), y las fotos y los documentos PDF/imagen se convierten en texto mediante OCR (Gemini multimodal). El texto extraído se almacena en la bandeja, se etiqueta, se incrusta (embedding) y se hace buscable con `/search`/`/ask` (y también se reenvía automáticamente si una etiqueta coincide).
- Si no hay clave de Gemini, o el tipo de medio no es compatible, se notifica al usuario y solo se almacena la referencia/leyenda.

### Componentes
- `TelegramBotApi.get_file` + `download_file` (y `file_base_url`) para descargar el archivo desde la Bot API.
- `provider_http.gemini_extract_document` (OCR/extracción de texto multimodal) y el ligero `ExtractionService`, a la par de `TranscriptionService`.
- En `bot.py`: los helpers puros `_forward_file_ref` (selección de archivo, mayor tamaño de foto) y `_media_route` (ruta transcribe/extract), y el núcleo de orquestación `_process_forwarded_media` con inyección de servicio y descarga para pruebas totalmente sin conexión.

### Fuera de alcance (seguimiento)
- Extracción de DOCX/Excel y procesamiento de medios en la ruta de importación completa del canal.

### Pruebas
- `tests/test_inbox_media.py`: selección y enrutamiento de archivos, orquestación para transcribe/extract, rechazo cuando falta un servicio/ruta/descarga, absorción de errores del servicio y `file_base_url`.

## Reglas basadas en IA (2026-06-12)

Reglas de etiqueta basadas en LLM, junto a las reglas de palabra clave existentes.

### Comportamiento
- `/rule add-ai <criterion> -> <tag>` define una regla con un criterio en lenguaje natural; `/rule list` muestra el tipo de cada regla con un icono (📝 palabra clave / 🤖 ia).
- Las reglas de IA se evalúan solo durante `/rule apply` (una llamada al LLM por elemento, cubriendo todas las reglas de IA). Sin una clave de Gemini se ignoran, y esto se informa en la salida. Las reglas de palabra clave se aplican en cada ingesta como antes.
- `match_tags` ahora omite las reglas de IA en las rutas automáticas.

### Diseño
- El módulo `rules.py` con las funciones puras `build_classify_prompt` y `parse_classified_tags`, y `classify_ai_tags(text, ai_rules, *, generate)`, que inyecta la llamada al LLM para que siga siendo totalmente comprobable sin conexión.
- Una columna `kind` en la tabla `rules` con la migración idempotente `_ensure_rule_columns`; `add_rule`/`list_rules` con soporte de `kind`.

### Fuera de alcance (seguimiento)
- Aplicar automáticamente las reglas de IA en cada ingesta (por ahora, solo `/rule apply`).

### Pruebas
- `tests/test_ai_rules.py`: omitir las reglas de IA en `match_tags`, construcción/análisis del prompt, `classify_ai_tags` con un generate inyectado y short-circuit, persistir `kind`, y `/rule apply` con una combinación de palabra clave+IA (LLM falso) y omitir IA sin una clave.

## Línea de tiempo (2026-06-11)

Una vista temporal del archivo — el complemento temporal de la agrupación por temas.

### Comportamiento
- Un nuevo módulo `timeline.py` (Python puro, sin dependencias): `build_timeline` agrupa los elementos con fecha en cubos de calendario (mes `YYYY-MM` o día `YYYY-MM-DD`) y proporciona recuentos/fuentes/muestra por periodo; como las fechas están en ISO 8601, el cubo es simplemente un prefijo de fecha. Las fechas inválidas se descartan.
- `Repository.timeline_items` devuelve los elementos que tienen una `message_date` (limitados a propietario + fuente/etiqueta, los más nuevos primero).
- El comando del bot `/timeline [--source <url>] [--tag <tag>] [--day]` (mes de forma predeterminada) y la herramienta MCP `timeline`. Los campos del usuario en la salida se escapan con `html.escape`.
- `/help`, README y CHANGELOG se actualizaron.

### Pruebas
- `tests/test_timeline.py`: `period_key` (cubo y rechazo de una fecha incorrecta), agrupación por mes/día y ordenación descendente, alcance y ordenación de `timeline_items` y la herramienta MCP `timeline`.

## Corrección: escapar en HTML los reenvíos al archivo (2026-06-11)

- Como `send_message` envía con `parse_mode: HTML`, los campos controlados por el usuario (etiqueta de fuente, etiquetas, texto, enlace) en los reenvíos automáticos y el mensaje de confirmación de la bandeja ahora se escapan con `html.escape`. Anteriormente, la presencia de `<`, `>` o `&` causaba un error del parser de Telegram y, como resultado, el elemento fallaba silenciosamente al llegar al canal de archivo.
- Una nueva prueba en `tests/test_autoforward.py` que comprueba que estos caracteres se escapan.

## Reenvío automático a un canal de archivo (2026-06-11)

Reenvío automático de elementos etiquetados a un canal de archivo (uno de los seguimientos del motor de reglas).

### Comportamiento
- El comando `/setarchive <@channel | chat id>` establece el canal de archivo del usuario; `/setarchive off` lo desactiva, y `/setarchive` sin argumento muestra el estado actual.
- En la ruta de la bandeja de reenviados, tras un guardado exitoso, el texto reenviado se coteja con las reglas del usuario (`match_tags`); si al menos una etiqueta coincide y hay un canal de archivo establecido, el elemento se reenvía al canal de archivo con su fuente, etiquetas, texto y enlace. Un error de envío se registra silenciosamente y no rompe el flujo principal.

### Datos
- Una nueva columna `archive_chat_id` en `bot_users` con la migración idempotente `_ensure_bot_user_columns` (ALTER TABLE si falta la columna). El método `Repository.set_archive_chat`.

### Fuera de alcance (seguimiento)
- Reglas basadas en IA y reenvío automático para importaciones de canales (por ahora, solo la bandeja de reenviados).

### Pruebas
- `tests/test_autoforward.py`: la decisión/formateo de `_auto_forward` (envío cuando hay archivo+etiqueta presentes, omisión cuando falta cualquiera de los dos, absorción de errores de envío), el ciclo de `/setarchive` (set/show/clear) y la migración de columna + alcance por usuario.

## Agrupación por temas (2026-06-11)

Agrupación por temas del contenido del archivo (uno de los seguimientos de NotebookLM).

### Comportamiento
- Un nuevo módulo `clustering.py` (Python puro, sin dependencias): agrupación voraz de una sola pasada basada en la similitud del coseno con centroides móviles, y `top_terms` para construir una etiqueta de grupo a partir de los términos significativos más frecuentes (con una lista de stopwords multilingüe). Como los chunks tienen embeddings almacenados, funciona totalmente sin conexión.
- `Repository.chunks_with_embeddings` devuelve los chunks que tienen un embedding (limitados a propietario + fuente/etiqueta) y decodifica el BLOB.
- El comando del bot `/topics [--source <url>] [--tag <tag>]` y la herramienta MCP `list_topics`.
- `/help` y README se actualizaron.

### Fuera de alcance (seguimiento)
- Nombrar los grupos con un LLM y una línea de tiempo automática.

### Pruebas
- `tests/test_clustering.py`: `top_terms`, separación de grupos, el límite de grupos, rechazo de elementos sin un embedding, etiquetado/ordenación de `build_topics`, decodificación y alcance en `chunks_with_embeddings`, y la herramienta MCP `list_topics`.

## CI — pytest + ruff (2026-06-11)

Añadir una canalización de integración continua (CI) para que el código roto no llegue a `main`; anteriormente, GitHub Actions solo gestionaba los despliegues.

### CI
- Un nuevo workflow `.github/workflows/ci.yml` en cada push y pull_request: instalar dependencias, luego `ruff check` y `pytest`.
- El conjunto completo (73 pruebas) se ejecuta en CI; `test_telegram_client` también pasa sin ejecutar realmente Telethon (las importaciones son perezosas).

### Lint
- Configuración de `ruff` en `pyproject.toml` (conjunto de reglas `E,F,I,UP,B`; `line-length=140`) y adición de `ruff` a las dependencias de desarrollo.
- Corrección de todos los hallazgos de lint: eliminación de importaciones no usadas, ordenación de importaciones, `datetime.UTC`, `zip(..., strict=True)` en las rutas de crypto/coseno, `raise ... from` en los bloques except, y una anotación segura de `TelegramClient` bajo `TYPE_CHECKING`.

### Ejecutar localmente
```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest -q
```

## Fase 8 — Servidor MCP (2026-06-09)

La fase final de la hoja de ruta: un servidor MCP de solo lectura para que el archivo de Telegram del usuario pueda conectarse a otras herramientas de IA.

### Comportamiento
- Un nuevo módulo `mcp_server.py`: JSON-RPC 2.0 sobre stdio, usando solo la biblioteca estándar (sin nuevas dependencias). `handle_request` es una función pura dict→dict, y `serve_stdio` es un bucle ligero delimitado por saltos de línea sobre ella.
- Métodos de protocolo: `initialize` (protocolVersion, serverInfo, capabilities.tools), `notifications/initialized` (sin respuesta), `tools/list`, `tools/call`.
- Herramientas (todas de solo lectura): `list_sources`, `list_tags`, `search_telegram_archive` (con un filtro de fuente/etiqueta), `get_message` (el texto completo de un elemento por `media_item_id`), `ask_telegram_notebook` (RAG), `summarize_source`.
- Limitado a un único propietario de `MCP_OWNER_ID` (predeterminado `0` = el archivo web). Todas las consultas pasan por el aislamiento de `owner_id`.
- Ejecutar: `python -m telegram_notebook.mcp_server`.

### Repository
- Un nuevo método `get_media_item(owner_id, media_item_id)` para la herramienta `get_message`.

### Pruebas
- `tests/test_mcp_server.py`: initialize/tools-list, comportamiento de las notificaciones, error de método desconocido, list_sources/search/get_message, una herramienta desconocida (isError), aislamiento por propietario y un roundtrip completo de `serve_stdio`.

## Fase 7 — Resúmenes / NotebookLM (2026-06-09)

Resumen del archivo de la hoja de ruta (resumen por fuente y por etiqueta).

### Comportamiento
- `/summarize [--source <url>] [--tag <tag>]` — sin filtro, se resume todo el archivo; con `--source`, una sola fuente; y con `--tag`, una sola etiqueta (usando el mismo analizador `_split_filters`).
- El contenido (una fila por elemento, con texto y fuente) se obtiene de `Repository.summary_items` (limitado a propietario + fuente/etiqueta, con un límite predeterminado de 200 elementos).
- El resumen se construye con `SearchService.summarize`; el prompt se produce en `_build_summary_prompt` (una función pura) con las fuentes anotadas y el texto de cada elemento truncado, y se pasa a `gemini_generate_content`.

### Fuera de alcance (seguimiento)
- Agrupación por temas y una línea de tiempo automática.

### Pruebas
- `tests/test_summarize.py`: construcción del prompt (incluidas las fuentes y el alcance, truncamiento del texto), el mensaje vacío y el alcance del método `summary_items` (todo/etiqueta/fuente y aislamiento por usuario).

## Fase 6 — Trabajos de importación completa (2026-06-09)

Importación completa de canales de la hoja de ruta: una cola, un worker en segundo plano, seguimiento del progreso, reanudación tras una interrupción y cancelación.

### Modelo de datos
- La tabla `jobs` (`owner_id`, `channel_url`, `status`, `total`, `processed`, `cursor`, `limit_count`, `error`, `cancel_requested`, marcas de tiempo). status es uno de `queued|running|done|failed|cancelled`.
- Métodos de Repository: `create_job`, `get_job`, `list_jobs`, `claim_next_queued_job` (seleccionando atómicamente el trabajo más antiguo y pasándolo a running), `update_job_progress`, `finish_job`, `request_job_cancel`, `is_cancel_requested` y `requeue_running_jobs` (devolviendo a queued los trabajos running huérfanos por un worker caído).

### Worker
- Un nuevo módulo `jobs.py` con `JobWorker` (un único hilo daemon). Está desacoplado de Telegram y funciona con un `runner` inyectado para que la máquina de estados sea totalmente comprobable de forma unitaria.
- Al arrancar, vuelve a poner en cola los trabajos running huérfanos (reanudación tras una caída).

### Pipeline
- `ingest_channel` ganó los parámetros `resume_from` (min_id para la continuación), `progress_cb(processed, total, last_msg_id)` y `should_cancel()`. Para cada mensaje, se comprueba la cancelación y se actualiza el progreso/cursor. Como el almacenamiento es idempotente, la reanudación es segura.
- `iter_all_messages` ganó un parámetro `min_id`, y `limit` ahora es opcional (`None` = todos los mensajes).

### Bot
- `/import <channel_url> [limit]` (poner en cola una importación completa/reanudable), `/jobs` (estado y progreso), `/canceljob <id>`.
- `/ingest` sigue siendo la ruta inline rápida. El worker se inicia en `run_forever` y, al final de cada trabajo, envía un mensaje de done/failed/cancelled al usuario.
- `/help` se actualizó.

### Pruebas
- `tests/test_jobs.py`: el ciclo de vida del trabajo, claim atómico y ordenación, progreso/cancelación/requeue, y la máquina de estados del worker con un runner falso (done/failed/cancelled y avance del cursor para la reanudación).

## Fase 5 — Reglas + etiquetas (2026-06-09)

El motor de reglas y el sistema de etiquetas de la hoja de ruta. El usuario define una regla palabra clave→etiqueta y el contenido entrante se etiqueta automáticamente y se puede filtrar en search/ask.

### Modelo de datos
- La tabla `rules` (`owner_id`, `keyword`, `tag`, `created_at`) con un índice único en `(owner_id, keyword, tag)`.
- La tabla `content_tags` (`owner_id`, `media_item_id`, `tag`) con una clave primaria compuesta (etiquetado idempotente).
- Ambas se crean con `CREATE TABLE IF NOT EXISTS`; no se necesita una migración especial para las bases de datos existentes.

### Coincidencia y etiquetado automático
- Un nuevo módulo `rules.py` con la función pura `match_tags(text, rules)` (subcadena, sin distinguir mayúsculas/minúsculas).
- En las tres rutas de ingesta (texto de canal, transcripción de medios, bandeja de reenviados), tras almacenar el texto, el pipeline aplica las reglas del propietario y almacena las etiquetas (`_apply_rules`). Se añadió `owner_id` a los helpers internos del pipeline.

### Comandos del bot
- `/rule add <keyword> -> <tag>`, `/rule list`, `/rule remove <id>` y `/rule apply` (borrando y recalculando las etiquetas a partir de los textos almacenados).
- `/tags` — las etiquetas y el recuento distinto de elementos de cada etiqueta.
- Un filtro `--tag <tag>` para `/search` y `/ask`. El analizador `_split_source` se reemplazó por `_split_filters`, que entiende tanto `--source` (un solo token) como `--tag` (hasta el final de la línea, varias palabras).
- `/help` se actualizó.

### Búsqueda
- `SearchService.search` ganó un parámetro `tag`. La ruta de palabra clave se filtra con un join en `content_tags`; la ruta semántica (Vertex) se post-filtra con una lista de permitidos de `media_ids_for_tag`.

### Pruebas
- `tests/test_rules.py`: coincidencia pura, análisis de `/rule add`, CRUD y unicidad de reglas, almacenamiento/recuento de etiquetas, etiquetado automático en la ingesta, búsqueda filtrada por etiqueta y backfill.
- `tests/test_normalize.py`: prueba de `_split_filters` (en lugar de `_split_source`).

## Fase 4 — Bandeja de reenviados (MVP) (2026-06-09)

Implementación de la siguiente fase de la hoja de ruta: la "bandeja de entrada inteligente de Telegram". El usuario ahora puede reenviar cualquier mensaje al bot, y su texto/leyenda se almacena en una bandeja personal y buscable.

### Comportamiento
- El bot detecta los mensajes reenviados (tanto el nuevo formato `forward_origin` como los campos heredados como `forward_from`/`forward_from_chat`/`forward_sender_name`) y los enruta antes de la lógica del flujo de autenticación, para que no entre en conflicto con las respuestas de texto en el flujo de `/connect`.
- El `text` o `caption` del reenvío, junto con una etiqueta de tipo de medio (p. ej. `[Forwarded document: report.pdf]`) y la fuente (el nombre del canal/usuario de origen), se almacena.
- Cuando el origen es un canal público, se construye el enlace `https://t.me/<username>/<id>` como la fuente.
- El contenido almacenado se puede consultar a través de los mismos `/search` y `/ask` (chunk + embedding, con una alternativa de palabra clave si no hay embedding disponible).

### Modelo de datos
- La bandeja se implementa como un "canal" sintético por usuario con `channel_url = inbox://forwarded`, reutilizando el esquema existente y la ruta de búsqueda (y el aislamiento de `owner_id` de la Fase 2).
- Un nuevo método `IngestionPipeline.ingest_forwarded_message` (idempotente basado en el message_id del reenvío).

### UX del bot
- `/start` y `/help` se actualizaron para explicar la capacidad de reenvío.
- Un mensaje de orientación para los elementos solo de medios sin texto (que aún no se indexan en esta versión).
- Refactor: la configuración de Vertex relacionada con el índice se consolidó en un helper compartido (`_vertex_ingest_config`) para que `/ingest` y la bandeja la usen ambos.

### Fuera de alcance (seguimiento)
- Descarga y transcripción de medios reenviados a través de la Bot API, OCR para fotos y extracción de texto de PDF/DOCX/Excel.

### Pruebas
- `tests/test_forwarded.py`: detección de reenvíos, extracción de la fuente/enlace/etiqueta de medio e ingesta de extremo a extremo (almacenamiento y capacidad de búsqueda, idempotencia y la naturaleza por usuario de la bandeja).

## Fase 3 — Autenticación de la API web y cifrado de secretos (2026-06-09)

Los dos elementos de seguridad restantes del análisis: la autenticación de la API web y el cifrado de secretos en la base de datos.

### Autenticación de la API web
- Una nueva variable `WEB_API_TOKEN`. Cuando se establece, todos los endpoints `/api/*` (excepto `/api/health`) requieren el token; el token se envía mediante `Authorization: Bearer <token>` o el encabezado `X-API-Token` (comparación de tiempo constante).
- Cuando no se establece ningún token, la API solo acepta solicitudes de loopback (localhost), y el acceso de red no autenticado se rechaza con un 401 (seguro por defecto; anteriormente estaba completamente abierto).
- `/api/health` permanece público para el healthcheck de Docker.
- Interfaz del dashboard: todas las llamadas pasan por `fetchJson`; esta función envía el token de `localStorage` y, ante una respuesta 401, solicita al usuario un token una vez y lo almacena.

### Cifrado de secretos en reposo
- Un nuevo módulo `crypto.py`: cifrado autenticado usando solo la biblioteca estándar (separación de claves con HKDF-SHA256, un keystream con HMAC-SHA256 en modo CTR, y Encrypt-then-MAC con HMAC-SHA256; un nonce aleatorio de 128 bits para cada valor). Sin nuevas dependencias.
- Las columnas sensibles se cifran antes del almacenamiento en SQLite: en `bot_users` → `api_hash`, `session_string`, `gemini_api_key`; en `auth_flows` → `api_hash`, `session_string`, `phone_code_hash`. La lectura (`get_bot_user`/`get_auth_flow`) descifra de forma transparente.
- La clave se lee de `SECRETS_KEY`. Si no se establece, el cifrado es una operación nula (con una advertencia) y las bases de datos antiguas en texto plano siguen funcionando; los valores cifrados se distinguen del texto plano antiguo por el prefijo `enc::` para que la migración sea indolora.

### Pruebas
- `tests/test_crypto.py`: roundtrip, no determinismo, rechazo de manipulación/clave incorrecta, passthrough para None/vacío/texto plano antiguo y el comportamiento de operación nula sin una clave.
- `tests/test_web_auth.py`: aceptación de bearer/`X-API-Token`, rechazo de un token incorrecto/ausente y la restricción de loopback cuando no se establece ningún token.
- `tests/test_db.py`: nuevas pruebas para el almacenamiento cifrado de secretos y el descifrado transparente en la lectura.

### .env.example
- Adición de `WEB_API_TOKEN` y `SECRETS_KEY` junto con el comando para generar un valor.

## Fase 2 — Aislamiento de datos por usuario (2026-06-09)

El foco de esta fase es corregir la fuga de datos entre usuarios: anteriormente, `/search` y `/ask` (y la API web) operaban sobre **todos** los canales de la base de datos, y los usuarios podían ver los datos de los demás.

### Modelo de datos
- Se añadió una columna `owner_id` a la tabla `channels` y la propiedad se aplica a este nivel; como cada `message`/`media_item`/`chunk` está vinculado a un canal mediante una FK, filtrar por `channels.owner_id` en los joins aísla completamente los datos.
- La restricción global `UNIQUE(channel_url)` se reemplazó por un índice compuesto `UNIQUE(owner_id, channel_url)` para que dos usuarios puedan ingerir el mismo canal de forma independiente sin compartir una fila.
- Una migración automática (`Repository._ensure_channel_owner`) para las bases de datos antiguas: la tabla `channels` se reconstruye, se añade la columna `owner_id` y las filas heredadas se conservan con `owner_id = NULL`; es decir, en lugar de filtrarse entre usuarios, se vuelven invisibles para las consultas por usuario (y deben volver a ingerirse si es necesario).

### Aplicación del alcance
- Los métodos de Repository que devuelven o eliminan datos ahora toman `owner_id`: `upsert_channel`, `keyword_candidates`, `embedding_candidates`, `list_channels`, `delete_channel_data`, `get_chunk_by_media_and_index`.
- `SearchService.search` e `IngestionPipeline.ingest_channel` toman un parámetro `owner_id`.
- El bot de Telegram pasa el `bot_user_id` del usuario como `owner_id`; por lo tanto, `/search`, `/ask`, `/ingest`, `/sources`, `/delete`, `/status` operan solo sobre los datos de ese usuario.
- El panel web (que no tiene inicio de sesión por usuario) usa un `WEB_OWNER_ID = 0` fijo para que su archivo se mantenga separado de los archivos de los usuarios del bot.

### Endurecimiento (hardening)
- El `LIMIT` en `keyword_candidates` ahora se vincula como un parámetro en lugar de mediante interpolación de cadenas.

### Pruebas
- Las pruebas de `Repository` se actualizaron para pasar `owner_id`.
- Una nueva prueba `test_data_is_isolated_per_owner`: dos usuarios con la misma URL no ven los datos del otro, y eliminar los de uno no tiene efecto sobre los del otro.
- Una nueva prueba `test_migrates_legacy_channels_table_without_owner_id`: migración de una base de datos antigua sin `owner_id`.

## Fase 1 — Estabilizar el núcleo (2026-06-08)

Según la hoja de ruta del README, esta fase se centra en estabilizar el núcleo: seguridad, corrección de errores, comandos del bot, registro (logging) y pruebas.

### Seguridad
- El token real del bot se eliminó de `.env.example` y se vació.
  - ⚠️ Este token se incluyó previamente en el historial de git (commit `5501fda`) y es efectivamente público. Vaciar el archivo no basta; debes hacer inmediatamente `/revoke` del token en **@BotFather** y crear uno nuevo.
- Los identificadores específicos del entorno (`VERTEX_INDEX_ID`, `VERTEX_DEPLOYED_INDEX_ID`) se vaciaron en el archivo de ejemplo.

### Correcciones de errores
- `/search` y `/ask`: el usuario se lee por el `bot_user_id` real, no por `chat_id` (en los grupos estos dos difieren).
- API web: `/api/search` y `/api/ask` ahora pasan `vertex_config` (y `project_id`/`region` para ask); anteriormente siempre recurrían a la búsqueda por palabra clave.
- La respuesta de `/ask` en el bot usa `<b>` (HTML) para que se renderice correctamente con `parse_mode=HTML` (anteriormente era `**` en bruto).
- El valor predeterminado de `DB_PATH` en `.env.example` se alineó con `config.py`: `data/store.db`.

### Nuevos comandos del bot
- `/status` — estado de la conexión, clave de IA, configuración de Vertex y el número de fuentes indexadas.
- `/disconnect` — eliminar la sesión y las credenciales del usuario ("delete my data").
- `/help` — la lista de comandos.
- Los comandos ya no tienen problemas con el sufijo `@botname` ni con mayúsculas/minúsculas, y ya no entran por error en el flujo de conexión.
- Una protección para la entrada vacía en `/search`, `/ask`, `/ingest`, `/join`, `/delete`.

### Registro y manejo de errores
- Un nuevo módulo `logging_config.py` con `setup_logging()` (nivel de `LOG_LEVEL`, predeterminado INFO).
- Todos los `print()` de depuración se reemplazaron por `logging`; los valores sensibles (número de teléfono, código de inicio de sesión, `phone_code_hash`) ya no se registran.
- Una única actualización rota ya no detiene todo el bucle de polling del bot (se registra y la ejecución continúa).

### Pruebas
- Un conjunto `tests/` con pytest; 26 pruebas sin necesidad de red: `chunking`, similitud del coseno, `normalize_phone`/`normalize_code`, URL canónica, composición de texto, saneamiento de nombres de canal, `Repository` sobre un SQLite temporal y `upsert_env_values`.
- Ejecutar: `pip install -e ".[dev]"` y luego `pytest`.

### Seguimientos (para fases posteriores)
- `normalize_phone` sigue siendo ingenuo para los números con un código de país (p. ej. `09123456789` → `+09123456789`).
- `import re` en `bot.py` queda sin usar tras eliminar las regexes y se puede limpiar.
- `main.py` todavía construye el estado global en el momento de la importación; sería mejor hacerlo perezoso (lazy).
- El aislamiento de datos por usuario (Fase 2) aún no se ha hecho: `/search` y `/ask` operan sobre todos los canales, no solo sobre los datos de ese usuario.
