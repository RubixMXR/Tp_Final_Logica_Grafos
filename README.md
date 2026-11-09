# Planificador Curricular — Trabajo Práctico Integrador

**Grupo:** Matias Xavier Roveri y Manuel Slepoy
**Carrera:** Ingeniería en Sistemas de Información (UTN FRRo)
**Tema:** Planificador curricular con Machine Learning y Teoría de Grafos.

## Descripción del Proyecto
Este sistema combina un modelo predictivo de Machine Learning con algoritmos de grafos para ayudar a los estudiantes a planificar su cursada. A partir del historial académico del alumno (materias aprobadas, desaprobadas y promedio), el modelo estima la probabilidad de aprobar cada materia. Luego, el algoritmo de Dijkstra encuentra la ruta de cursada con el menor riesgo acumulado, respetando estrictamente las correlatividades del plan de estudios.

## Tecnologías Utilizadas
- **Backend:** Python 3.10+, FastAPI, Uvicorn.
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib.
- **Grafos:** Implementación propia de Dijkstra con `heapq`. Validación de DAG con NetworkX.
- **Frontend:** HTML5, CSS3, Vanilla JavaScript y SVG.

## Requisitos Previos
- Python 3.10 o superior instalado.
- Git instalado.
- VS Code (recomendado).

## Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/RubixMXR/Tp_Final_Logica_Grafos.git
cd Tp_Final_Logica_Grafos
```

### 2. Crear y activar el entorno virtual
**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*(Nota: Cada vez que abras una terminal nueva, deberás volver a activar el entorno).*

### 3. Instalar dependencias
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Generar el dataset y entrenar el modelo
Antes de levantar la aplicación, es necesario generar los datos y entrenar el modelo. Ejecutá estos scripts en orden:
```bash
python backend/scripts/validar_plan.py
python backend/scripts/generar_dataset.py
python backend/scripts/entrenar_modelo.py
```

### 5. Levantar la aplicación (Requiere 2 terminales)
Ambas terminales deben estar en la raíz del proyecto y con el entorno virtual activado.

**Terminal 1 (Backend API):**
```bash
python -m uvicorn backend.api:app --reload
```
*(La API estará disponible en `http://127.0.0.1:8000`. La documentación interactiva de la API se puede ver en `http://127.0.0.1:8000/docs`).*

**Terminal 2 (Frontend):**
```bash
python -m http.server 8080 --directory frontend
```

### 6. Acceder a la aplicación
Una vez que ambos servidores estén corriendo, abrí tu navegador de preferencia e ingresá a:
[http://127.0.0.1:8080](http://127.0.0.1:8080)

*(Para detener los servidores, presioná `Ctrl + C` en cada una de las terminales).*
