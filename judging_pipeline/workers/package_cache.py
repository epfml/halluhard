"""Package verification cache for coding task optimization.

Caches package existence verdicts to avoid redundant web searches.
Also includes a whitelist of well-known packages that can be skipped entirely.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Set, Optional

# Python standard library - valid for IMPORT only, NOT for install
PYTHON_STDLIB: Set[str] = {
    "os", "sys", "json", "re", "math", "random", "datetime", "time", "collections",
    "itertools", "functools", "typing", "pathlib", "subprocess", "threading",
    "multiprocessing", "asyncio", "socket", "http", "urllib", "email", "html",
    "xml", "logging", "unittest", "io", "csv", "sqlite3", "pickle", "copy",
    "hashlib", "secrets", "base64", "struct", "codecs", "locale", "gettext",
    "argparse", "configparser", "shutil", "glob", "tempfile", "gzip", "zipfile",
    "tarfile", "lzma", "bz2", "zlib", "platform", "ctypes", "abc", "contextlib",
    "dataclasses", "enum", "graphlib", "heapq", "bisect", "array", "weakref",
    "types", "pprint", "textwrap", "difflib", "string", "statistics", "fractions",
    "decimal", "numbers", "cmath", "operator", "queue", "sched", "select",
    "selectors", "signal", "mmap", "readline", "rlcompleter", "code", "codeop",
    "concurrent", "future", "fileinput", "stat", "filecmp", "fnmatch", "linecache",
    "netrc", "plistlib", "xdrlib", "wave", "colorsys", "imghdr", "sndhdr",
    "ossaudiodev", "crypt", "termios", "tty", "pty", "fcntl", "grp", "pwd",
    "spwd", "posixpath", "posix", "resource", "syslog", "aifc", "audioop",
    "chunk", "sunau", "cgi", "cgitb", "wsgiref", "ftplib", "poplib", "imaplib",
    "nntplib", "smtplib", "smtpd", "telnetlib", "uuid", "socketserver", "xmlrpc",
    "ipaddress", "ast", "symtable", "token", "keyword", "tokenize", "tabnanny",
    "pyclbr", "py_compile", "compileall", "dis", "pickletools", "formatter",
    "parser", "symbol", "test", "doctest", "pdb", "profile", "pstats", "timeit",
    "trace", "tracemalloc", "gc", "inspect", "site", "builtins", "__future__",
    "warnings", "faulthandler", "traceback", "atexit",
}

# Installable Python packages (PyPI) - valid for both import AND install
KNOWN_PYTHON_PACKAGES: Set[str] = {
    "numpy", "pandas", "scipy", "matplotlib", "seaborn", "plotly", "bokeh",
    "scikit-learn", "sklearn", "tensorflow", "torch", "pytorch", "keras",
    "transformers", "huggingface", "datasets", "tokenizers",
    "requests", "httpx", "aiohttp", "urllib3", "beautifulsoup4", "bs4", "lxml",
    "selenium", "scrapy", "playwright",
    "flask", "django", "fastapi", "starlette", "uvicorn", "gunicorn", "tornado",
    "sqlalchemy", "psycopg2", "pymysql", "redis", "pymongo", "elasticsearch",
    "celery", "dramatiq", "rq",
    "pytest", "nose", "coverage", "tox", "mypy", "pylint", "flake8", "black",
    "isort", "autopep8", "yapf", "bandit", "pre-commit",
    "pillow", "pil", "opencv-python", "cv2", "imageio", "scikit-image",
    "networkx", "igraph", "graph-tool",
    "boto3", "botocore", "google-cloud", "azure", "aws",
    "pydantic", "attrs", "dataclasses-json", "marshmallow",
    "click", "typer", "argparse", "fire", "docopt",
    "tqdm", "rich", "colorama", "termcolor",
    "pyyaml", "yaml", "toml", "tomli", "configparser", "python-dotenv", "dotenv",
    "cryptography", "pycryptodome", "paramiko", "fabric",
    "openai", "anthropic", "langchain", "llama-index",
    "streamlit", "gradio", "dash", "panel",
    "jupyter", "notebook", "ipython", "ipykernel", "ipywidgets", "nbformat",
    "sympy", "mpmath", "gmpy2",
    "arrow", "pendulum", "python-dateutil", "dateutil", "pytz",
    "regex", "chardet", "ftfy", "unidecode",
    "six", "future", "typing-extensions", "typing_extensions",
    "packaging", "setuptools", "wheel", "pip", "virtualenv", "pipenv", "poetry",
    "cython", "numba", "cffi", "pybind11",
    "jinja2", "mako", "chameleon",
    "graphviz", "pydot", "pygraphviz",
    "spacy", "nltk", "gensim", "textblob", "pattern",
    "ray", "dask", "joblib", "multiprocess",
    "orjson", "ujson", "simplejson", "rapidjson",
    "msgpack", "protobuf", "thrift", "avro",
    "websockets", "websocket-client", "python-socketio", "socketio",
    "grpcio", "grpc",
    "alembic", "flask-sqlalchemy", "django-rest-framework", "drf",
    "faker", "factory-boy", "hypothesis",
    "wrapt", "decorator", "tenacity", "backoff", "retry",
    "apscheduler", "schedule",
    "xlrd", "xlwt", "openpyxl", "xlsxwriter", "pyexcel",
    "reportlab", "pypdf2", "pdfplumber", "pymupdf", "fitz",
    "pyqt5", "pyside2", "tkinter", "wxpython", "kivy",
    "pygame", "pyglet", "arcade",
}

KNOWN_JS_PACKAGES: Set[str] = {
    # Core/built-in
    "fs", "path", "http", "https", "url", "util", "os", "crypto", "stream",
    "events", "buffer", "child_process", "cluster", "dgram", "dns", "net",
    "readline", "repl", "tls", "tty", "v8", "vm", "zlib", "assert", "console",
    "process", "timers", "querystring", "string_decoder", "perf_hooks",
    
    # Popular npm packages
    "react", "react-dom", "vue", "angular", "svelte", "next", "nuxt", "gatsby",
    "express", "koa", "fastify", "hapi", "nest", "nestjs",
    "axios", "fetch", "node-fetch", "got", "superagent", "request",
    "lodash", "underscore", "ramda", "immutable",
    "moment", "dayjs", "date-fns", "luxon",
    "webpack", "rollup", "vite", "parcel", "esbuild", "swc",
    "babel", "typescript", "ts-node", "tsx",
    "jest", "mocha", "chai", "jasmine", "vitest", "playwright", "cypress", "puppeteer",
    "eslint", "prettier", "husky", "lint-staged",
    "mongoose", "sequelize", "typeorm", "prisma", "knex", "pg", "mysql", "redis",
    "socket.io", "ws", "websocket",
    "graphql", "apollo", "urql", "relay",
    "redux", "mobx", "zustand", "recoil", "jotai", "valtio",
    "tailwindcss", "styled-components", "emotion", "sass", "less", "postcss",
    "jquery", "bootstrap", "material-ui", "mui", "antd", "chakra-ui",
    "three", "d3", "chart.js", "echarts", "highcharts",
    "sharp", "jimp", "canvas",
    "dotenv", "config", "convict",
    "uuid", "nanoid", "shortid",
    "chalk", "ora", "inquirer", "commander", "yargs", "meow",
    "debug", "winston", "pino", "bunyan", "morgan",
    "bcrypt", "jsonwebtoken", "passport", "helmet", "cors",
    "nodemailer", "aws-sdk", "firebase", "stripe",
    "cheerio", "jsdom", "htmlparser2",
    "xml2js", "fast-xml-parser",
    "csv-parse", "papaparse", "xlsx", "exceljs",
    "pdf-lib", "pdfkit", "jspdf",
    "formidable", "multer", "busboy",
    "zod", "yup", "joi", "ajv", "validator",
    "rxjs", "async", "bluebird", "p-limit", "p-queue",
}

# Elixir standard library modules (valid for import/require, not for Mix install)
ELIXIR_STDLIB: Set[str] = {
    # Kernel and core modules
    "Kernel", "Kernel.SpecialForms", "Module", "Record", "Protocol", "Behaviour",
    "Access", "Agent", "Application", "Atom", "Base", "Bitwise", "Calendar",
    "Code", "Date", "DateTime", "Duration", "Enum", "Exception", "File",
    "Float", "Function", "GenServer", "GenEvent", "HashDict", "HashSet",
    "IO", "Inspect", "Integer", "Keyword", "List", "Macro", "Map", "MapSet",
    "MatchError", "Module", "NaiveDateTime", "Node", "OptionParser", "Path",
    "Port", "Process", "Protocol", "Range", "Regex", "Registry", "Set",
    "Stream", "String", "StringIO", "Supervisor", "System", "Task", "Time",
    "Tuple", "URI", "Version",
    # ETS and DETS
    "ETS", ":ets", "DETS", ":dets",
    # Logger
    "Logger",
    # Mix (build tool)
    "Mix", "Mix.Config", "Mix.Project", "Mix.Task",
    # ExUnit (testing)
    "ExUnit", "ExUnit.Case", "ExUnit.Callbacks",
    # IEx (interactive shell)
    "IEx",
}

# Known Elixir packages from Hex.pm
KNOWN_ELIXIR_PACKAGES: Set[str] = {
    # Web frameworks
    "phoenix", "phoenix_html", "phoenix_live_view", "phoenix_live_dashboard",
    "phoenix_pubsub", "phoenix_ecto", "plug", "plug_cowboy", "cowboy",
    # Database
    "ecto", "ecto_sql", "postgrex", "myxql", "ecto_sqlite3", "mongodb",
    # HTTP clients
    "httpoison", "tesla", "finch", "hackney", "mint", "req",
    # JSON
    "jason", "poison", "jiffy", "json",
    # Authentication
    "guardian", "comeonin", "bcrypt_elixir", "argon2_elixir", "pow", "ueberauth",
    # Testing
    "ex_machina", "mox", "bypass", "mock", "faker",
    # Background jobs
    "oban", "exq", "quantum",
    # GraphQL
    "absinthe", "absinthe_plug", "dataloader",
    # Utilities
    "timex", "decimal", "uuid", "nanoid", "slugify", "earmark", "floki",
    "httpotion", "sweet_xml", "csv", "nimble_csv",
    # Crypto
    "comeonin", "pbkdf2_elixir", "cloak",
    # Geo/GIS - common packages the model might reference
    "geo", "geo_postgis", "geocalc", "distance", "topo", "geometry",
    # Config
    "dotenv", "vapor", "skogsra",
    # Logging
    "logger_file_backend", "logster",
    # Caching
    "cachex", "nebulex", "con_cache",
    # Monitoring
    "telemetry", "telemetry_metrics", "telemetry_poller",
    # Email
    "bamboo", "swoosh",
    # File uploads
    "arc", "waffle",
    # Pagination
    "scrivener", "scrivener_ecto",
    # API
    "open_api_spex", "ex_json_schema",
    # Deployment
    "distillery", "releases",
}

# R base packages (installed with R, no install needed)
R_STDLIB: Set[str] = {
    # Base R packages
    "base", "stats", "graphics", "grDevices", "utils", "datasets", "methods",
    "grid", "splines", "stats4", "tcltk", "tools", "parallel", "compiler",
    # Recommended packages (usually included)
    "MASS", "lattice", "Matrix", "nlme", "survival", "boot", "cluster",
    "codetools", "foreign", "KernSmooth", "rpart", "class", "nnet", "spatial",
}

# Known R packages from CRAN
KNOWN_R_PACKAGES: Set[str] = {
    # Tidyverse
    "tidyverse", "dplyr", "ggplot2", "tidyr", "readr", "purrr", "tibble",
    "stringr", "forcats", "lubridate", "hms", "readxl", "haven", "jsonlite",
    # Data manipulation
    "data.table", "reshape2", "plyr", "magrittr", "glue",
    # Visualization
    "ggplot2", "plotly", "shiny", "ggvis", "leaflet", "highcharter", "gganimate",
    "ggthemes", "scales", "gridExtra", "cowplot", "patchwork", "RColorBrewer",
    # Statistics & ML
    "caret", "randomForest", "xgboost", "glmnet", "e1071", "nnet", "rpart",
    "gbm", "ranger", "lightgbm", "keras", "tensorflow", "torch", "mlr3",
    # Time series
    "forecast", "zoo", "xts", "tseries", "prophet", "fable",
    # Text mining
    "tm", "tidytext", "quanteda", "text2vec", "stringi",
    # Web scraping
    "rvest", "httr", "httr2", "curl", "xml2",
    # Database
    "DBI", "RSQLite", "RMySQL", "RPostgres", "odbc", "dbplyr",
    # Reporting
    "rmarkdown", "knitr", "bookdown", "blogdown", "xaringan",
    # Development
    "devtools", "usethis", "testthat", "roxygen2", "pkgdown", "covr",
    # Spatial
    "sf", "sp", "raster", "terra", "rgdal", "rgeos", "leaflet", "tmap",
    # Bioinformatics
    "BiocManager", "Biostrings", "GenomicRanges", "DESeq2", "edgeR",
    # Other utilities
    "here", "fs", "withr", "assertthat", "checkmate", "cli", "crayon",
    "progress", "pROC", "broom", "modelr", "rsample", "recipes", "parsnip",
}

# Scala standard library (no install needed)
SCALA_STDLIB: Set[str] = {
    # Core packages
    "scala", "scala.collection", "scala.collection.mutable", "scala.collection.immutable",
    "scala.concurrent", "scala.concurrent.duration", "scala.concurrent.ExecutionContext",
    "scala.io", "scala.math", "scala.reflect", "scala.sys", "scala.util",
    "scala.util.matching", "scala.util.control", "scala.annotation",
    # Common types
    "Option", "Some", "None", "Either", "Left", "Right", "Try", "Success", "Failure",
    "Future", "Promise", "List", "Vector", "Map", "Set", "Seq", "Array",
    # Java interop (always available)
    "java.lang", "java.util", "java.io", "java.nio", "java.net", "java.time",
    "java.math", "java.text", "java.sql", "java.security",
}

# Known Scala packages from Maven/Ivy
KNOWN_SCALA_PACKAGES: Set[str] = {
    # Akka
    "akka", "akka-actor", "akka-stream", "akka-http", "akka-cluster", "akka-persistence",
    # Play Framework
    "play", "play-json", "play-ws", "play-slick",
    # Cats ecosystem
    "cats", "cats-core", "cats-effect", "cats-free", "cats-mtl",
    # ZIO
    "zio", "zio-streams", "zio-http", "zio-json", "zio-config",
    # HTTP
    "http4s", "http4s-dsl", "http4s-blaze-server", "http4s-circe",
    "sttp", "requests-scala", "scalaj-http",
    # JSON
    "circe", "circe-core", "circe-generic", "circe-parser", "spray-json",
    "json4s", "argonaut", "upickle", "play-json",
    # Database
    "slick", "doobie", "quill", "scalikejdbc", "anorm", "squeryl",
    # Testing
    "scalatest", "specs2", "munit", "scalacheck", "mockito-scala",
    # Logging
    "logback", "slf4j", "scribe", "log4s",
    # Config
    "typesafe-config", "pureconfig", "ciris",
    # Serialization
    "protobuf", "avro", "parquet",
    # Spark
    "spark", "spark-core", "spark-sql", "spark-streaming", "spark-mllib",
    # Streaming
    "fs2", "monix", "akka-stream",
    # Utilities
    "shapeless", "refined", "enumeratum", "squants", "spire", "breeze",
    "scala-async", "better-files", "os-lib", "fansi", "sourcecode",
    # Web
    "scalatra", "finatra", "cask",
    # CLI
    "scopt", "decline", "mainargs",
}

# Combined sets for different use cases
# For imports: stdlib + installable packages are all valid
ALL_IMPORTABLE_PACKAGES = (
    PYTHON_STDLIB | KNOWN_PYTHON_PACKAGES | 
    KNOWN_JS_PACKAGES |
    ELIXIR_STDLIB | KNOWN_ELIXIR_PACKAGES |
    R_STDLIB | KNOWN_R_PACKAGES |
    SCALA_STDLIB | KNOWN_SCALA_PACKAGES
)
# For installs: only packages that can actually be installed (not stdlib)
ALL_INSTALLABLE_PACKAGES = (
    KNOWN_PYTHON_PACKAGES | 
    KNOWN_JS_PACKAGES |
    KNOWN_ELIXIR_PACKAGES |
    KNOWN_R_PACKAGES |
    KNOWN_SCALA_PACKAGES
)


@dataclass
class PackageVerdict:
    """Cached verdict for a package."""
    package_name: str
    exists: bool
    reason: str
    from_whitelist: bool = False


class PackageVerdictCache:
    """Thread-safe cache for package verification results.
    
    This cache stores verdicts for packages that have been verified,
    allowing subsequent claims about the same package to skip web search.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._cache: Dict[str, PackageVerdict] = {}
        self._hits = 0
        self._misses = 0
        self._whitelist_skips = 0
    
    def _normalize_package_name(self, name: str) -> str:
        """Normalize package name for cache lookup."""
        # Lowercase, strip whitespace, handle common variations
        name = name.lower().strip()
        # Handle cv2 -> opencv-python mapping
        mappings = {
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "PIL": "pillow",
            "pil": "pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
        }
        return mappings.get(name, name)
    
    def check_whitelist(self, package_name: str, element_type: str = "import") -> Optional[PackageVerdict]:
        """Check if package is in the known-good whitelist.
        
        Args:
            package_name: The package name to check
            element_type: "import" or "install" - determines which whitelist to use
                         For imports: stdlib + installable packages are valid
                         For installs: only installable packages (not stdlib)
        
        Returns PackageVerdict if package is whitelisted, None otherwise.
        This is synchronous since whitelist is static.
        """
        normalized = self._normalize_package_name(package_name)
        
        # Choose appropriate whitelist based on element type
        if element_type == "install":
            # For installs, only check installable packages (not stdlib)
            whitelist = ALL_INSTALLABLE_PACKAGES
        else:
            # For imports, both stdlib and installable packages are valid
            whitelist = ALL_IMPORTABLE_PACKAGES
        
        if normalized in whitelist:
            self._whitelist_skips += 1
            return PackageVerdict(
                package_name=package_name,
                exists=True,
                reason=f"Well-known package '{package_name}' - skipped web search",
                from_whitelist=True,
            )
        return None
    
    async def get(self, package_name: str) -> Optional[PackageVerdict]:
        """Get cached verdict for a package."""
        normalized = self._normalize_package_name(package_name)
        
        async with self._lock:
            if normalized in self._cache:
                self._hits += 1
                return self._cache[normalized]
            self._misses += 1
            return None
    
    async def set(self, package_name: str, exists: bool, reason: str) -> None:
        """Cache a verdict for a package."""
        normalized = self._normalize_package_name(package_name)
        
        async with self._lock:
            self._cache[normalized] = PackageVerdict(
                package_name=package_name,
                exists=exists,
                reason=reason,
            )
    
    async def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        async with self._lock:
            return {
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "whitelist_skips": self._whitelist_skips,
                "cached_packages": len(self._cache),
            }

