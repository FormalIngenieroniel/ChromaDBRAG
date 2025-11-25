import os
import sys
import warnings
import time
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. CONFIGURACIÓN DE RUTAS E IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import config

# IMPORTS DEL PROYECTO (CAMBIO CLAVE: Importamos el Agente de Grafos)
# Asegúrate de haber creado src/components/graph_agent.py como vimos antes
from src.components.graph_agent import graph_app 

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

# --- 2. CONFIGURACIÓN LLM JUEZ (IGUAL AL VECTORIAL) ---
GOOGLE_API_KEY = config.GEMINI_API_KEY 

if "AIza" not in GOOGLE_API_KEY:
    print("⚠️ ¡ALERTA! No has puesto tu API Key de Google.")
    sys.exit(1)

print(f"\n🔄 Conectando con Google Gemini para Evaluación (Modo Grafos)...")

# Clase para evitar Rate Limit (Plan Gratuito)
class SlowGemini(ChatGoogleGenerativeAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(5) # Espera 5 segundos entre evaluaciones
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

try:
    # Mismo Juez que en la evaluación vectorial
    ragas_llm = SlowGemini(
        model="gemini-2.5-flash-lite", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    print("✅ Juez LLM Conectado.")
except Exception as e:
    print(f"\n❌ Error conectando Juez: {e}")
    sys.exit(1)

# --- 3. CONFIGURACIÓN DE EMBEDDINGS (IGUAL AL VECTORIAL) ---
print("🔄 Cargando Embeddings de Evaluación...")
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 4. DATASET DE PRUEBA (EXACTAMENTE EL MISMO) ---
test_data = [
    {
        "question": "Necesito el vagón cisterna que transporta petróleo (NEFT).",
        "ground_truth": "El vagón adecuado es el que aparece en la imagen 12.jpg. Es un vagón cisterna de color rojo oscuro diseñado específicamente para el transporte de petróleo o materiales inflamables."
    },
    {
        "question": "Muéstrame el vagón de carga sellado de color azul marino profundo.",
        "ground_truth": "El vagón correspondiente es el de la imagen 08.jpg. Se trata de un vagón de carga tipo caja cerrada (boxcar) de color azul marino profundo."
    }
]

def run_evaluation():
    print("\n--- 🕸️ Iniciando Evaluación RAGAS - ENFOQUE GRAFOS (LangGraph) ---")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # --- BUCLE DE GENERACIÓN (ADAPTADO A GRAFOS) ---
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"\nProcesando con LangGraph: '{q}'")
        
        # 1. INVOCAR AL AGENTE DE GRAFOS
        # En lugar de llamar a retriever y generator por separado, 
        # invocamos el flujo de LangGraph que ya tiene la lógica interna.
        inputs = {"question": q, "context": [], "answer": ""}
        
        try:
            # invoke ejecuta los nodos: search_graph -> generate
            result_state = graph_app.invoke(inputs)
            
            generated_answer = result_state["answer"]
            
            # 2. EXTRAER CONTEXTO RECUPERADO
            # El estado devuelve una lista de dicts, Ragas necesita lista de strings
            retrieved_items_dicts = result_state.get("context", [])
            context_strings = [item.get('description', '') for item in retrieved_items_dicts]
            
        except Exception as e:
            print(f"❌ Error en el flujo del grafo: {e}")
            generated_answer = "Error generating response"
            context_strings = []

        # Guardar en listas
        questions.append(q)
        answers.append(generated_answer)
        contexts.append(context_strings)
        ground_truths.append(gt)
        
        print("  ✅ Respuesta generada por el Grafo.")
        time.sleep(2) 

    # Crear Dataset de HuggingFace
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # --- 5. EJECUCIÓN DE MÉTRICAS (EXACTAMENTE LAS MISMAS) ---
    metrics_to_run = [
        Faithfulness(),      
        AnswerSimilarity(),  
        ContextPrecision(),  
        ResponseRelevancy()  
    ]

    print("\n🚀 Ejecutando métricas de Ragas...")
    
    run_config = RunConfig(
        max_workers=1,
        timeout=600 
    )

    results = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        llm=ragas_llm,       
        embeddings=hf_embeddings,
        run_config=run_config
    )

    # --- 6. RESULTADOS ---
    print("\n================== 📈 Resultados Detallados (GRAFOS) ==================")
    df_results = results.to_pandas()
    
    cols_to_show = ['question', 'answer', 'faithfulness', 'answer_similarity', 'context_precision', 'response_relevancy']
    final_cols = [c for c in cols_to_show if c in df_results.columns]
    
    print(df_results[final_cols])
    
    print("\n--- Promedios Globales (GRAFOS) ---")
    print(results)
    
    # Guardamos con un nombre distinto para poder comparar los CSVs luego
    df_results.to_csv("resultados_real_graph.csv", index=False)
    print("\n✅ Guardado en 'resultados_real_graph.csv'")

if __name__ == "__main__":
    # Asegurarse de que el grafo existe antes de evaluar
    graph_file = config.CHROMA_PERSIST_DIR / "knowledge_graph.gpickle"
    if not graph_file.exists():
        print(f"❌ Error: No se encuentra el archivo del grafo en {graph_file}")
        print("   Por favor ejecuta primero: python3 src/ingestion/ingestion_graph.py")
        sys.exit(1)
        
    run_evaluation()