import os
import sys
import warnings
import time
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. CONFIGURACIÓN DE RUTAS E IMPORTS ---
# Ajusta esto según dónde esté este archivo. Si está en la raíz, quita los '..'
# Si está en una carpeta 'tests', deja esto así:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import config

# IMPORTS REALES DE TU PROYECTO
from src.components.retriever import search_chroma
from src.components.generator import generate_response

import pandas as pd
from datasets import Dataset

# Imports de RAGAS
try:
    from ragas.metrics import (
        Faithfulness,
        AnswerSimilarity,
        ContextPrecision,
        ContextRecall,
        ResponseRelevancy
    )
    from ragas import evaluate
    from ragas.run_config import RunConfig
except ImportError as e:
    print(f"Error importando Ragas: {e}")
    sys.exit(1)

# Configuración de entorno
os.environ["OPENAI_API_KEY"] = "sk-no-key-needed"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# --- 2. CONFIGURACIÓN LLM JUEZ (RAGAS) ---
GOOGLE_API_KEY = config.GEMINI_API_KEY 

if "AIza" not in GOOGLE_API_KEY:
    print("⚠️ ¡ALERTA! No has puesto tu API Key de Google.")
    sys.exit(1)

print(f"\n🔄 Conectando con Google Gemini para Evaluación...")

# Clase para evitar Rate Limit (Plan Gratuito)
class SlowGemini(ChatGoogleGenerativeAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(5) # Espera 5 segundos entre evaluaciones
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

try:
    # Este LLM es SOLO para que RAGAS juzgue las respuestas
    ragas_llm = SlowGemini(
        model="gemini-2.5-flash-lite", # Recomiendo 1.5 Flash por estabilidad
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    print("✅ Juez LLM Conectado.")
except Exception as e:
    print(f"\n❌ Error conectando Juez: {e}")
    sys.exit(1)

# --- 3. CONFIGURACIÓN DE EMBEDDINGS (Para Ragas) ---
# Nota: Ragas necesita sus propios embeddings para calcular similitudes.
# Usamos el mismo modelo ligero que tenías para no sobrecargar.
print("🔄 Cargando Embeddings de Evaluación...")
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 4. DATASET DE PRUEBA (Ground Truth) ---
# Aquí defines las preguntas y cuál DEBERÍA ser la respuesta ideal.
# Esto es lo único manual que queda, ya que necesitas una verdad absoluta para comparar.
test_data = [
    {
        "question": "Necesito el vagón cisterna que transporta petróleo (NEFT).",
        "ground_truth": "El vagón adecuado es el que aparece en la imagen 12.jpg. Es un vagón cisterna de color rojo oscuro diseñado específicamente para el transporte de petróleo o materiales inflamables."
    },
    {
        "question": "Muéstrame el vagón de carga sellado de color azul marino profundo.",
        "ground_truth": "El vagón correspondiente es el de la imagen 08.jpg. Se trata de un vagón de carga tipo caja cerrada (boxcar) de color azul marino profundo."
    }
    # Puedes agregar más preguntas aquí si tienes las respuestas correctas en tus descripciones
]

def run_evaluation():
    print("\n--- 📊 Iniciando Evaluación RAGAS con DATOS REALES ---")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # --- BUCLE DE GENERACIÓN REAL ---
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"\nProcesando: '{q}'")
        
        # 1. RETRIEVER REAL
        # Busca en tu ChromaDB real
        retrieved_items_dicts = search_chroma(q, n_results=3)
        
        # 2. GENERADOR REAL
        # Usa tu generador (que llama a Gemini internamente)
        # Nota: Esto consumirá cuota de tu API Key también.
        generated_answer = generate_response(q, retrieved_items_dicts)
        
        # 3. PREPARAR CONTEXTO PARA RAGAS
        # Ragas espera una lista de strings ['info A', 'info B']
        # Tu retriever devuelve diccionarios, así que extraemos las descripciones.
        context_strings = [item.get('description', '') for item in retrieved_items_dicts]
        
        # Guardar en listas
        questions.append(q)
        answers.append(generated_answer)
        contexts.append(context_strings)
        ground_truths.append(gt)
        
        print("  ✅ Respuesta generada.")
        # Pausa extra de seguridad para no saturar la API (Generación + Evaluación)
        time.sleep(2)

    # Crear Dataset de HuggingFace
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # --- 5. EJECUCIÓN DE MÉTRICAS ---
    metrics_to_run = [
        Faithfulness(),      # ¿La respuesta se basa en el contexto recuperado?
        AnswerSimilarity(),  # ¿La respuesta se parece a la Ground Truth?
        ContextPrecision(),  # ¿El contexto relevante apareció primero?
        ResponseRelevancy()  # ¿La respuesta tiene sentido con la pregunta?
    ]

    print("\n🚀 Ejecutando métricas de Ragas...")
    
    run_config = RunConfig(
        max_workers=1, # Un solo hilo para evitar rate limits
        timeout=600 
    )

    results = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        llm=ragas_llm,       # El juez Gemini Lento
        embeddings=hf_embeddings,
        run_config=run_config
    )

    # --- 6. RESULTADOS ---
    print("\n================== 📈 Resultados Detallados ==================")
    df_results = results.to_pandas()
    
    # Seleccionamos columnas para mostrar limpio
    cols_to_show = ['question', 'answer', 'faithfulness', 'answer_similarity', 'context_precision', 'response_relevancy']
    final_cols = [c for c in cols_to_show if c in df_results.columns]
    
    print(df_results[final_cols])
    
    print("\n--- Promedios Globales ---")
    print(results)
    
    df_results.to_csv("resultados_real_chroma.csv", index=False)
    print("\n✅ Guardado en 'resultados_real_chroma.csv'")

if __name__ == "__main__":
    run_evaluation()