// const API_BASE_URL = "https://profile-xai-api.onrender.com/api";
const API_BASE_URL = "http://127.0.0.1:8000/api"

export async function downloadExplanationReport(payload: {
  profile: string
  explanation: string
  chat_history: { role: string; content: string }[]
  timestamp?: string
  usuario?: string
}): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/report/explanation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error("Error al generar el reporte")
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `reporte_explicabilidad_${Date.now()}.pdf`
  a.click()
  URL.revokeObjectURL(url)
}

export interface DatasetSchema { columns: Array<{ name: string; type: "categorical" | "numerical" | "text"; options?: string[] }>; }
export interface InstanceData { [key: string]: string | number; }
export type UserProfile = "data-scientist" | "domain-expert" | "non-expert";
export interface ChatMessage { role: "user" | "assistant"; content: string; timestamp: Date; }

export interface FeedbackPayload {
  timestamp: string
  usuario: string
  perfil: string
  q1_1: number; q1_2: number; q1_3: number
  q2_1: number; q2_2: number; q2_3: number
  q3_1: number; q3_2: number
  q_confuso: string
  q_info_adicional: string
  q_mejoras: string
  organizacion: string
  categoria: string
  comentarios: string
}

export async function submitFeedback(payload: FeedbackPayload): Promise<{ success: boolean; file: string; message: string }> {
  const res = await fetch(`${API_BASE_URL}/feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || "Error al enviar feedback")
  }
  return res.json()
}

export async function startProcessing() {
  const res = await fetch(`${API_BASE_URL}/processing/start`, { 
    method: "POST",
    headers: { "ngrok-skip-browser-warning": "true" }
  });
  const data = await res.json();
  localStorage.setItem("jobId", data.jobId);
  return data;
}

export async function uploadDataset(files: File[]) {
  const jobId = localStorage.getItem("jobId");
  const formData = new FormData();
  files.forEach(f => formData.append("files", f));
  
  const res = await fetch(`${API_BASE_URL}/upload/dataset?jobId=${jobId}`, { 
    method: "POST", 
    headers: { "ngrok-skip-browser-warning": "true" },
    body: formData 
  });
  return await res.json();
}

export async function getDatasetSchema(): Promise<DatasetSchema> {
  try {
    const jobId = localStorage.getItem("jobId");
    const res = await fetch(`${API_BASE_URL}/dataset/schema?jobId=${jobId}`, {
      headers: { "ngrok-skip-browser-warning": "true" }
    });
    
    if (!res.ok) throw new Error(`El servidor respondió con código ${res.status}`);
    return await res.json();
  } catch (error) {
    console.error("Fallo al extraer el esquema del CSV:", error);
    return { columns: [] }; 
  }
}

export async function getRandomInstance(): Promise<InstanceData> {
  try {
    const jobId = localStorage.getItem("jobId");
    const res = await fetch(`${API_BASE_URL}/dataset/random-instance?jobId=${jobId}`, {
      headers: { "ngrok-skip-browser-warning": "true" }
    });
    
    if (!res.ok) throw new Error(`El servidor respondió con código ${res.status}`);
    return await res.json();
  } catch (error) {
    console.error("Fallo al obtener la instancia aleatoria:", error);
    return {};
  }
}

// Mocks temporales
export async function uploadKnowledgeBase(files: File[]) {
  const jobId = localStorage.getItem("jobId");
  const formData = new FormData();
  files.forEach(f => formData.append("files", f));
  
  const res = await fetch(`${API_BASE_URL}/upload/knowledge-base?jobId=${jobId}`, { 
    method: "POST", 
    headers: { "ngrok-skip-browser-warning": "true" },
    body: formData 
  });
  return await res.json();
}

export async function uploadModel(files: File[]) {
  const jobId = localStorage.getItem("jobId");
  const formData = new FormData();
  files.forEach(f => formData.append("files", f));
  
  const res = await fetch(`${API_BASE_URL}/upload/model?jobId=${jobId}`, { 
    method: "POST", 
    headers: { "ngrok-skip-browser-warning": "true" },
    body: formData 
  });
  return await res.json();
}

export async function convertModel(): Promise<{ status: string; classes_detected: string[]; n_classes: number }> {
  const jobId = localStorage.getItem("jobId");
  const res = await fetch(`${API_BASE_URL}/job/${jobId}/convert`, {
    method: "POST",
    headers: { "ngrok-skip-browser-warning": "true" },
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Error al convertir el modelo");
  }
  return await res.json();
}

export async function setLabels(labelMap: Record<string, string>) {
  const jobId = localStorage.getItem("jobId");
  const res = await fetch(`${API_BASE_URL}/job/${jobId}/labels`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
    body: JSON.stringify({ labelMap }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Error al guardar etiquetas");
  }
  return await res.json();
}

// 7. Generar Explicación (El llamado a LIME en tu backend)
export async function generateExplanation(profile: string, instance: any) {
  const jobId = localStorage.getItem("jobId");
  
  const res = await fetch(`${API_BASE_URL}/explain`, {
    method: "POST",
    headers: { 
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true" 
    },
    // Enviamos el ID, el perfil de la audiencia y el diccionario con los datos del paciente/sistema
    body: JSON.stringify({ jobId, profile, instance }) 
  });
  
  if (!res.ok) {
    const errorData = await res.json();
    throw new Error(`Error en IA: ${errorData.detail || 'Fallo desconocido'}`);
  }
  
  return await res.json();
}

export async function chatRag(message: string, history: any[],explanationText: string, profile: string = "non-expert") {
  const jobId = localStorage.getItem("jobId");
  
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { 
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true"
    },
    body: JSON.stringify({ 
      jobId: jobId, 
      message: message, 
      profile: profile, 
      history: history,
      explanation_context: explanationText
    })
  });
  
  if (!res.ok) {
    throw new Error("Error al conectar con el Chat RAG");
  }
  
  // Extraemos los datos completos (el diccionario de Python)
  const data = await res.json();
  
  // ¡El truco! Devolvemos ÚNICAMENTE el texto de la respuesta para que React no se asuste
  return data.response; 
}
