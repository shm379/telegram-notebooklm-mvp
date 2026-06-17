# Telegram NotebookLM MVP

**Languages / زبان‌ها:** [English](README.md) · [فارسی](README.fa.md) · [العربية](README.ar.md) · [Español](README.es.md) · [简体中文](README.zh.md)

Este es un MVP para construir un **archivo inteligente de Telegram**; un proyecto que puede recopilar el contenido de canales, chats, archivos, vídeos y mensajes reenviados, convertirlos en texto buscable y, en última instancia, responder a las preguntas del usuario como un **NotebookLM interno para Telegram**.

El objetivo final del proyecto es permitir que el usuario convierta su contenido de Telegram en una memoria buscable que pueda conectarse a herramientas de IA; desde dentro del bot de Telegram, el panel web y, en el futuro, mediante MCP para conectarse a herramientas como ChatGPT, Claude, Cursor, agentes del estilo de Codex y otros clientes de IA.

---

## Idea central

Este proyecto se enfoca en tres modos principales:

### 1. Importar canal / chat

El usuario proporciona el enlace o el ID de un canal público o de un chat al que tiene acceso, y el sistema recupera sus mensajes, leyendas y contenido multimedia.

Ejemplo:

```text
/ingest https://t.me/example_channel
```

### 2. Bandeja de reenviados

El usuario puede reenviar al bot un mensaje, publicación, archivo, foto, vídeo, PDF o cualquier contenido. El sistema lo almacena, procesa, etiqueta y lo hace buscable.

Esta parte está pensada para funcionar como una **bandeja de entrada inteligente de Telegram**.

### 3. Cuaderno de IA / RAG

Después de que el contenido se almacena e indexa, el usuario puede hacer preguntas a su archivo:

```text
/ask از بین پیام‌هایی که درباره Al Mouj ذخیره کردم، کدام‌ها درباره townhouse بودند؟
```

o:

```text
/ask ابزارهای AI که در کانال‌ها درباره ساخت ویدیو معرفی شده‌اند را دسته‌بندی کن
```

La respuesta debe venir con la fuente, el enlace del mensaje y los textos relacionados.

---

## Estado actual del MVP

En la versión actual, el proyecto tiene estas capacidades:

- Recibir un enlace de un canal de Telegram y leer los mensajes con `Telethon`
- Descargar y procesar mensajes de texto, audio y vídeo
- Extraer audio de vídeo con `ffmpeg`
- Transcribir audio/vídeo con OpenAI o Gemini
- Fragmentar textos
- Construir embeddings para la búsqueda semántica
- Búsqueda por palabra clave + semántica
- Generación inicial de respuestas basadas en RAG a partir de los resultados de búsqueda
- Un panel web ligero con `Python http.server`
- Un bot de Telegram para la orquestación y los comandos principales
- Conectar la cuenta real de Telegram del usuario mediante una cadena de sesión a través de Telethon
- **Bandeja de reenviados**: reenviar cualquier mensaje al bot almacena su texto/leyenda en la bandeja personal y buscable del usuario
- **Procesamiento de contenido multimedia reenviado**: los archivos de audio/vídeo/voz se transcriben automáticamente y las fotos/PDF se convierten en texto buscable mediante OCR (Gemini). Los archivos DOCX/XLSX también se extraen localmente (sin clave de API ni red)
- **Motor de reglas + etiquetas**: definición de reglas palabra clave→etiqueta, etiquetado automático del contenido entrante y filtrado de `/search` y `/ask` con `--tag`
- **Reglas de IA**: `/rule add-ai` con un criterio en lenguaje natural (evaluado por un LLM), y `/airules` para el etiquetado automático opcional en los nuevos reenvíos
- **Gestión y exploración de etiquetas**: `/tag rename|delete`, `/recent` para los elementos más recientes, y los endpoints web `/api/{stats,recent,timeline}` + un panel de Biblioteca en el dashboard
- **Reenvío automático a un canal de archivo**: con `/setarchive`, cualquier mensaje que reenvíes y que coincida con una regla de etiqueta también se reenvía automáticamente a tu canal de archivo
- **Trabajos de importación**: importación completa de canales en segundo plano con una cola, seguimiento del progreso, reanudación tras una interrupción y la posibilidad de cancelar
- **Resumen (NotebookLM)**: `/summarize` para construir un resumen estructurado de todo el archivo, de una fuente específica o de una etiqueta
- **Resumen periódico (digest)**: `/digest [days]` construye un resumen con IA del contenido reciente (7 días de forma predeterminada)
- **Agrupación por temas**: `/topics` agrupa el contenido del archivo por tema sin conexión (sobre los embeddings existentes); si hay una clave de Gemini presente, la etiqueta de cada grupo se construye con un LLM (de lo contrario, a partir de los términos más frecuentes)
- **Línea de tiempo**: `/timeline` agrupa el archivo por fecha (mes o día) — el complemento temporal de `/topics`
- **Exportación**: `/export` exporta todo el archivo, una sola fuente o una sola etiqueta como un archivo Markdown descargable
- **Estadísticas**: `/stats` muestra una visión general del archivo (recuento de elementos, fuentes y etiquetas, tipos de medios y rango temporal)
- **Colecciones (cuadernos)**: `/collection` agrupa varias etiquetas bajo un único nombre y muestra los elementos de la colección
- **Importación de copias de seguridad de Telegram**: toma el archivo *Machine-readable JSON* de Telegram Desktop (`result.json` o un `.zip` de la carpeta de exportación) — tanto exportaciones de un solo chat como de toda la cuenta —, hace que todos los mensajes sean buscables y devuelve una copia en **Markdown**. En el bot, basta con enviar el archivo; en la web, súbelo desde la tarjeta "Import Telegram Backup".
- **Sitio de presentación + aplicación web**: una página introductoria en `/` y un panel completo (Ingest / Search / Ask / Library / Backup import) en `/app`
- **Servidor MCP (solo lectura)**: expone el archivo a herramientas de IA con JSON-RPC sobre stdio (`python -m telegram_notebook.mcp_server`)

---

## ¿Por qué no basta con la Bot API?

Para capturar un archivo completo de un canal o chat, la Bot API por sí sola no es suficiente. La Bot API normalmente solo ve los nuevos mensajes a los que el bot tiene acceso.

Para importar el historial de canales y chats, este proyecto usa `Telethon` y MTProto; es decir, el mismo nivel de acceso que una cuenta de usuario, no solo un token de bot.

---

## Arquitectura actual

```text
Bot de Telegram
  |
  | comandos del usuario: /connect, /ingest, /search, /ask
  v
Backend de Python
  |
  +-- Cliente Telethon
  |     lectura de canales y chats
  |
  +-- Pipeline de ingesta
  |     descarga de medios, extracción de texto, transcripción
  |
  +-- Fragmentación + embedding
  |     preparación para la búsqueda semántica
  |
  +-- Servicio de búsqueda
  |     búsqueda por palabra clave + búsqueda vectorial
  |
  +-- Generador de respuestas RAG
        construcción de la respuesta a partir de las fuentes encontradas
```

---

## Tecnologías

- Python 3.11+
- Telethon
- OpenAI API
- Google Gemini / Google GenAI
- ffmpeg
- SQLite / almacén local compatible con JSON para el MVP
- Búsqueda léxica en Python + similitud del coseno
- Telegram Bot API para la interfaz de usuario
- Una interfaz web ligera con `http.server`

---

## Comandos del bot

```text
/start
Introducción al proyecto y primeros pasos

/connect
Conectar la cuenta real de Telegram del usuario

/status
Comprobar el estado de la conexión

/ingest <channel_url>
Indexación rápida e inline de un canal

/import <channel_url> [limit]
Poner en cola una importación completa y reanudable en segundo plano

/backup
Guía para importar un archivo de copia de seguridad de Telegram; basta con enviar al bot el archivo result.json o .zip

/jobs
Mostrar el estado y el progreso de los trabajos de importación

/canceljob <id>
Cancelar un trabajo en cola o en ejecución

/search <query>
Buscar en el archivo

/search <query> --source <channel_url>
Buscar solo dentro de una fuente específica

/search <query> --tag <tag>
Buscar solo dentro del contenido etiquetado

/ask <question>
Preguntar al archivo con IA

/ask <question> --source <channel_url>
Preguntar solo desde un canal o fuente específica

/ask <question> --tag <tag>
Preguntar solo desde el contenido de una etiqueta específica

/summarize [--source <url>] [--tag <tag>]
Resumir todo el archivo, una fuente o una etiqueta

/digest [days]
Resumen con IA del contenido reciente (7 días de forma predeterminada)

/topics [--source <url>] [--tag <tag>]
Agrupación por temas del contenido

/timeline [--source <url>] [--tag <tag>] [--day]
Vista temporal del archivo por mes (o por día con --day)

/export [--source <url>] [--tag <tag>]
Descargar una exportación en Markdown de todo el archivo, una fuente o una etiqueta

/recent [n]
Mostrar los n elementos más recientes (10 de forma predeterminada)

/stats
Visión general del archivo (recuento de elementos, fuentes y etiquetas, tipos de medios, rango temporal)

/sources
Mostrar las fuentes indexadas

/delete <channel_url>
Eliminar los datos de una fuente

/rule add <keyword> -> <tag>
Definir una regla de palabra clave para el etiquetado automático del contenido

/rule add-ai <criterion> -> <tag>
Definir una regla de IA (evaluada por un LLM, durante /rule apply)

/rule list
Mostrar las reglas

/rule remove <id>
Eliminar una regla

/rule apply
Volver a aplicar las reglas al contenido existente (backfill)

/airules on|off
Ejecutar automáticamente las reglas de IA en cada nuevo reenvío (opcional; desactivado de forma predeterminada)

/tags
Mostrar las etiquetas y el recuento de elementos de cada etiqueta

/tag rename <old> -> <new>
Renombrar una etiqueta (o fusionarla con una etiqueta existente)

/tag delete <tag>
Eliminar una etiqueta de todos los elementos

/collection new|add|list|remove|show <name>
Agrupar varias etiquetas bajo un "cuaderno" y mostrar sus elementos
(luego puedes resumir/exportar todo el cuaderno con /summarize --collection <name> o /export --collection <name>)

/setarchive <@channel | off>
Establecer el canal de archivo; los reenvíos etiquetados se envían a él automáticamente

/cancel
Cancelar el flujo actual
```

---

## APIs principales

### Ingest Channel

```bash
curl -X POST http://127.0.0.1:8000/api/channels/ingest \
  -H 'content-type: application/json' \
  -d '{
    "channel_url": "https://t.me/example_channel",
    "limit": 50
  }'
```

### Search

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H 'content-type: application/json' \
  -d '{
    "query": "هوش مصنوعی و تولید ویدیو",
    "channel_url": "https://t.me/example_channel",
    "top_k": 5
  }'
```

### Ask AI

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H 'content-type: application/json' \
  -d '{
    "query": "از این کانال چه ابزارهایی برای ساخت ویدیو معرفی شده؟",
    "channel_url": "https://t.me/example_channel"
  }'
```

### Stats / Recent / Timeline (solo lectura)

```bash
curl http://127.0.0.1:8000/api/stats
curl 'http://127.0.0.1:8000/api/recent?limit=10'
curl 'http://127.0.0.1:8000/api/timeline?granularity=month'
```

Estos endpoints, como el resto de la API, están protegidos con `WEB_API_TOKEN` (o solo loopback cuando no se ha establecido ningún token).

---

## Dirección del producto final

El objetivo final de este proyecto no es solo una búsqueda simple. La dirección del producto es la siguiente:

```text
Archivo de IA de Telegram
  |
  +-- Importación completa de canales y chats
  +-- Bandeja de reenviados para mensajes reenviados
  +-- Motor de reglas para separar contenido con palabra clave o IA
  +-- Etiqueta / Carpeta / Colección
  +-- Búsqueda léxica y semántica
  +-- NotebookLM interno para preguntas y respuestas
  +-- Servidor MCP para conectarse a herramientas de IA
```

---

## Motor de reglas + etiquetas

El usuario puede definir reglas palabra clave→etiqueta:

```text
/rule add Claude -> AI Tools
/rule add Al Mouj -> Real Estate
/rule add golden visa -> Oman Visa
/rule add قیمت -> Leads
```

Cualquier contenido nuevo que entre en el sistema (ingesta de canal, transcripción de medios o bandeja de reenviados) se coteja con su texto y su leyenda. Si la palabra clave de una regla (como subcadena y sin distinguir mayúsculas/minúsculas) está en el texto:

- La etiqueta correspondiente se adjunta a ese elemento
- Luego se puede filtrar con `/search ... --tag <tag>` y `/ask ... --tag <tag>`
- `/tags` muestra las etiquetas y el recuento de elementos de cada etiqueta
- `/tag rename <old> -> <new>` renombra una etiqueta (o la fusiona con una etiqueta existente), y `/tag delete <tag>` la elimina de todos los elementos
- `/rule apply` vuelve a aplicar las reglas actuales al contenido existente (backfill)

**Reglas basadas en IA:** además de las reglas de palabra clave, puedes definir una regla con un criterio en lenguaje natural cuya coincidencia decide un LLM:

```text
/rule add-ai پست‌هایی که درباره‌ی ابزارهای ساخت ویدیو با هوش مصنوعی هستند -> Video AI
/rule add-ai هر چیزی مرتبط با قیمت و فروش ملک -> Leads
```

Debido al coste del LLM, las reglas de IA solo se ejecutan durante `/rule apply` (una llamada al LLM por elemento) y requieren una clave de Gemini; si no se ha establecido ninguna clave, se ignoran y esto se informa en la salida. Las reglas de palabra clave se siguen aplicando automáticamente en cada ingesta.

**Reenvío automático:** con `/setarchive <@channel>` se establece un canal de archivo; a partir de ese momento, cualquier mensaje que reenvíes al bot cuyo texto coincida con una regla de etiqueta, además de almacenarse en la bandeja, también se reenvía a ese canal con su fuente/etiquetas/enlace (el bot debe ser administrador del canal). Para desactivarlo: `/setarchive off`.

Además de `/rule apply`, las reglas de IA también se ejecutan automáticamente en cada nuevo reenvío con `/airules on` (opcional, una llamada al LLM por elemento; las importaciones masivas de canales nunca se clasifican automáticamente).

**Aún no añadido (seguimiento):** reenvío automático para importaciones de canales (por ahora, solo la ruta de la bandeja de reenviados) y descarga/procesamiento de medios dentro de la importación completa del canal.

---

## Importación de copias de seguridad de Telegram (JSON / ZIP)

Puedes importar el historial completo de un chat o cuenta sin necesidad de `/connect`:

1. En **Telegram Desktop**, ve a `Settings → Advanced → Export Telegram data` (o, en un chat: `Export chat history`).
2. Establece el formato en **Machine-readable JSON** (no HTML).
3. La salida es un archivo `result.json` (o una carpeta que lo contiene junto con los medios); puedes comprimir la carpeta en un zip.

Luego:

- **Desde el bot:** envía el archivo `result.json` o `.zip` directamente al bot. El bot lo importa, hace que el contenido sea buscable y devuelve una copia en Markdown. (Un límite de 20 MB debido al límite de descarga de la Bot API; para un archivo más grande, usa la web.)
- **Desde la web:** en `/app`, en la tarjeta "Import Telegram Backup", sube el archivo. El contenido se vuelve buscable en el archivo web y aparece un botón de descarga en Markdown.

Cada chat se convierte en una fuente sintética `backup://<id>` y la importación es idempotente (volver a importar el mismo archivo no añade nada).

### API

```bash
curl -X POST 'http://127.0.0.1:8000/api/backup/import' \
  -H 'X-Filename: result.json' \
  -H 'content-type: application/octet-stream' \
  --data-binary @result.json
```

La respuesta incluye el número de chats/mensajes importados y el texto Markdown completo. Como el resto de la API, está protegida con `WEB_API_TOKEN` (o loopback).

---

## Servidor MCP

Se ha implementado un **servidor MCP de Telegram** (solo lectura) para que el archivo de Telegram del usuario no quede confinado al bot y pueda conectarse a otras herramientas de IA (Claude, Cursor, …). Funciona con JSON-RPC 2.0 sobre stdio y está escrito usando solo la biblioteca estándar (sin nuevas dependencias).

Ejecutar:

```bash
MCP_OWNER_ID=0 python -m telegram_notebook.mcp_server
```

`MCP_OWNER_ID` determina qué archivo de usuario se expone (predeterminado `0` = el archivo del panel web; para el archivo de un usuario del bot, proporciona su `bot_user_id`).

Las herramientas MCP actuales:

```text
list_sources              Mostrar canales/chats y la bandeja de reenviados
list_tags                 Mostrar las etiquetas y el recuento de elementos de cada etiqueta
search_telegram_archive   Buscar (con un filtro opcional de fuente/etiqueta)
get_message               El texto completo de un elemento por media_item_id
ask_telegram_notebook     Preguntas y respuestas RAG sobre el archivo
summarize_source          Resumir todo el archivo, una fuente o una etiqueta
list_topics               Agrupación por temas del contenido (sin conexión, a partir de los embeddings)
timeline                  Contar elementos por periodo de tiempo (mes/día)
archive_stats             Visión general del archivo (recuentos, tipos de medios, rango temporal)
list_recent               Lista de los elementos más recientes del archivo (los más nuevos primero)
```

Todas las herramientas son de solo lectura; las herramientas sensibles (import, forward, delete, create_rule) no se exponen deliberadamente y, si fuera necesario, deberían añadirse más adelante con permiso y confirmación.

---

## Desarrollo y pruebas

CI ejecuta lint y pruebas en cada push y PR (`.github/workflows/ci.yml`). Para ejecutarlo localmente:

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest -q
```

## Instalación

```bash
git clone https://github.com/shm379/telegram-notebooklm-mvp.git
cd telegram-notebooklm-mvp

uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

En Windows:

```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
copy .env.example .env
```

---

## Requisitos previos

- Python 3.11+
- ffmpeg
- Credenciales de la API de Telegram:
  - `TELEGRAM_API_ID`
  - `TELEGRAM_API_HASH`
  - `TELEGRAM_SESSION_STRING` es preferible para ejecuciones en producción
- `TELEGRAM_BOT_TOKEN` para ejecutar el bot
- Uno de estos proveedores:
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`

---

## Crear una sesión de Telegram

```bash
export TELEGRAM_API_ID=...
export TELEGRAM_API_HASH=...
uv run python scripts/create_telegram_session.py
```

Pon la salida en `.env` bajo `TELEGRAM_SESSION_STRING`.

Si no tienes un `TELEGRAM_SESSION_STRING`, el proyecto usa un archivo de sesión local y la primera ejecución requiere un inicio de sesión interactivo.

---

## Ejecutar la interfaz web

```bash
python -m telegram_notebook.main
```

Luego abre:

```text
http://127.0.0.1:8000        # página de presentación (introducción)
http://127.0.0.1:8000/app    # panel: Ingest / Search / Ask / Library / Backup import
```

---

## Ejecutar el bot de Telegram

```bash
python -m telegram_notebook.bot
```

---

## Limitaciones actuales

- El aislamiento de datos entre usuarios está implementado (cada usuario ve solo sus propios datos; la propiedad se aplica mediante `owner_id` en los canales).
- Se han añadido la autenticación de la API web con `WEB_API_TOKEN` y el cifrado de secretos en la base de datos con `SECRETS_KEY`; establece ambas variables para producción.
- El almacenamiento actual es adecuado para el MVP, no para un conjunto de datos grande.
- La cadena de sesión y las claves de API deberían cifrarse antes de producción.
- La importación completa de canales con cola, progreso y reanudación se admite mediante `/import` (un worker en segundo plano); todavía no hay un entorno de pruebas (sandbox) para toda la ruta de Telethon.
- Además del texto/leyenda, la bandeja de reenviados procesa los medios reenviados: audio/vídeo/voz mediante transcripción, fotos/PDF mediante OCR (Gemini multimodal) y DOCX/XLSX mediante extracción local (zipfile + XML, sin clave de API) se convierten en texto. La descarga de medios dentro de la importación completa del canal aún no se ha añadido.
- El motor de reglas se basa en la coincidencia de palabras clave (subcadena); las reglas basadas en IA y el reenvío automático a un canal de archivo aún no se han añadido.
- `/topics` realiza la agrupación por temas sobre los embeddings existentes (coseno voraz, sin conexión); requiere que el contenido esté indexado con una clave de embedding. Las etiquetas de los grupos se construyen con un LLM si hay una clave de Gemini presente y, de lo contrario, recurren a los términos más frecuentes. `/timeline` construye una vista temporal (mes/día) sobre las fechas de los mensajes.
- Para un conjunto de datos grande, es preferible migrar a PostgreSQL + pgvector o Qdrant.

---

## Notas de seguridad importantes

- No incluyas (commit) ningún token real, clave de API, cadena de sesión o credencial en el repositorio.
- Si previamente se incluyó un token real en `.env.example` o en el historial del proyecto, revoca/regenera ese token de inmediato.
- Para producción, las sesiones de los usuarios deben estar cifradas.
- Debe aplicarse un filtro de user_id para cada búsqueda o pregunta.
- El usuario debe poder hacer `disconnect` y `delete my data`.
- Las herramientas MCP deberían ser inicialmente de solo lectura.

---

## Hoja de ruta sugerida

### Fase 1 — Estabilizar el núcleo

- Limpiar los secretos del repositorio
- Corregir el README y el ejemplo de env
- Estabilizar `/connect`, `/ingest`, `/search`, `/ask`
- Corregir el manejo de errores y el registro (logging)

### Fase 2 — Modelo de datos multiusuario

- Añadir user_id a fuentes, mensajes, medios, chunks
- Aislamiento completo de los datos de los usuarios
- Permisos y control de acceso

### Fase 3 — Bandeja de reenviados

- Procesar mensajes reenviados
- Almacenar texto, leyenda, medios, documento
- OCR para fotos
- Extracción de texto de PDF/DOCX/Excel

### Fase 4 — Reglas + etiquetas

- Definir reglas de palabra clave
- Etiqueta y colección
- Reenvío automático a canales de archivo
- Reglas basadas en IA

### Fase 5 — Trabajos de importación completa

- Importación completa de canales de principio a fin
- Reanudación tras una interrupción
- Seguimiento del progreso
- Cola/worker en segundo plano

### Fase 6 — NotebookLM interno

- Mejor generación de respuestas con fuentes
- Resumen por fuente
- Resumen por etiqueta
- Línea de tiempo y agrupación por temas ✅ (`/timeline`, `/topics`)

### Fase 7 — Servidor MCP

- Herramientas MCP de solo lectura
- Conexión a clientes de IA
- Las herramientas search, ask, list_sources, get_message

---

## Resumen

Telegram NotebookLM MVP intenta convertir Telegram en una memoria inteligente; un lugar donde el usuario puede almacenar sus canales, chats y mensajes reenviados, buscar en ellos con búsqueda por palabra clave o semántica, separar el contenido con reglas y, en última instancia, hacer preguntas a su propio archivo como NotebookLM.

Este proyecto es una base para construir un producto más grande:

```text
Memoria de Telegram para asistentes de IA
```
