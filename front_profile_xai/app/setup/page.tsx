"use client"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ArrowLeft, Loader2, FlaskConical, ExternalLink, Info, Settings2, ChevronDown, ChevronUp } from "lucide-react"
import { Navbar } from "@/components/navbar"
import { FileUploadCard } from "@/components/file-upload-card"
import { uploadKnowledgeBase, uploadModel, uploadDataset, startProcessing, convertModel } from "@/lib/mock-api"
import { useToast } from "@/hooks/use-toast"

const YOUTUBE_GUIDE = "https://www.youtube.com/watch?v=KfnhNlD8WZI"

const knowledgeTooltip = (
  <div className="space-y-3">
    <p className="font-semibold text-foreground">¿Qué subir?</p>
    <p className="text-muted-foreground leading-snug">
      Documentos que explican el dominio del problema: manuales técnicos, reportes, descripciones de variables, papers de referencia.
    </p>
    <div>
      <p className="font-medium mb-1">Formatos aceptados</p>
      <div className="flex flex-wrap gap-1">
        {["PDF", "DOCX", "TXT", "JPG", "PNG"].map(f => (
          <span key={f} className="bg-muted text-muted-foreground rounded px-1.5 py-0.5 text-xs font-mono">{f}</span>
        ))}
      </div>
    </div>
    <div>
      <p className="font-medium mb-1">Recomendaciones</p>
      <ul className="text-muted-foreground space-y-0.5 list-disc list-inside text-xs">
        <li>Usa vocabulario técnico del dominio</li>
        <li>Máx. 20 archivos por sesión</li>
        <li>PDFs con texto seleccionable funcionan mejor que escaneados</li>
      </ul>
    </div>
  </div>
)

const modelTooltip = (
  <div className="space-y-3">
    <p className="font-semibold text-foreground">¿Qué subir?</p>
    <p className="text-muted-foreground leading-snug">
      El modelo de clasificación o regresión entrenado con scikit-learn. La herramienta lo convierte automáticamente a formato ONNX.
    </p>
    <div>
      <p className="font-medium mb-1">Formatos aceptados</p>
      <div className="flex flex-wrap gap-1">
        {[".pkl", ".joblib", ".h5"].map(f => (
          <span key={f} className="bg-muted text-muted-foreground rounded px-1.5 py-0.5 text-xs font-mono">{f}</span>
        ))}
      </div>
    </div>
    <div>
      <p className="font-medium mb-1">Restricciones técnicas</p>
      <ul className="text-muted-foreground space-y-0.5 list-disc list-inside text-xs">
        <li>Solo modelos tabulares (clasificación / regresión)</li>
        <li>Compatible con scikit-learn ≥ 1.0</li>
        <li>No se soportan modelos de NLP ni visión</li>
      </ul>
    </div>
    <a href={YOUTUBE_GUIDE} target="_blank" rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs text-primary font-medium hover:underline">
      Ver guía completa <ExternalLink className="h-3 w-3" />
    </a>
  </div>
)

const datasetTooltip = (
  <div className="space-y-3">
    <p className="font-semibold text-foreground">¿Qué subir?</p>
    <p className="text-muted-foreground leading-snug">
      El dataset de entrenamiento o un subconjunto representativo (≥ 100 filas). Las columnas deben coincidir exactamente con las features del modelo.
    </p>
    <div>
      <p className="font-medium mb-1">Formatos aceptados</p>
      <div className="flex flex-wrap gap-1">
        {["CSV", "XLSX"].map(f => (
          <span key={f} className="bg-muted text-muted-foreground rounded px-1.5 py-0.5 text-xs font-mono">{f}</span>
        ))}
      </div>
    </div>
    <div>
      <p className="font-medium mb-1">Restricciones técnicas</p>
      <ul className="text-muted-foreground space-y-0.5 list-disc list-inside text-xs">
        <li>Primera fila debe ser el encabezado</li>
        <li>No incluir la columna target</li>
        <li>Sin valores nulos en features categóricas</li>
        <li>Encoding UTF-8 para CSV</li>
      </ul>
    </div>
  </div>
)

type ExplainerMethod = "auto" | "shap" | "lime" | "anchor"

const METHOD_OPTIONS: { value: ExplainerMethod; label: string; desc: string }[] = [
  { value: "auto", label: "Auto", desc: "La herramienta selecciona el método óptimo según el modelo" },
  { value: "shap", label: "SHAP", desc: "Valores de contribución por feature basados en teoría de juegos (Shapley)" },
  { value: "lime", label: "LIME", desc: "Aproximación local lineal alrededor de la instancia de interés" },
  { value: "anchor", label: "Anchor", desc: "Reglas suficientes que anclan la predicción con alta precisión" },
]

function SliderField({ label, value, min, max, step, onChange, hint }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; hint?: string
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <label className="text-sm font-medium">{label}</label>
        <span style={{
          background: "#8B1D3B", color: "#fff", fontSize: "11px",
          fontWeight: 700, padding: "1px 8px", borderRadius: "10px",
        }}>{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full accent-primary"
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{min}</span>
        {hint && <span className="text-center italic">{hint}</span>}
        <span>{max}</span>
      </div>
    </div>
  )
}

export default function SetupPage() {
  const router = useRouter()
  const { toast } = useToast()

  const [knowledgeFiles, setKnowledgeFiles] = useState<File[]>([])
  const [modelFiles, setModelFiles] = useState<File[]>([])
  const [datasetFiles, setDatasetFiles] = useState<File[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [statusMsg, setStatusMsg] = useState("")
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Explainer config
  const [method, setMethod] = useState<ExplainerMethod>("auto")
  const [shapNSamples, setShapNSamples] = useState(15)
  const [shapBackground, setShapBackground] = useState(50)
  const [limeNumFeatures, setLimeNumFeatures] = useState(10)
  const [limeNumSamples, setLimeNumSamples] = useState(1000)
  const [anchorThreshold, setAnchorThreshold] = useState(0.95)
  const [evaluateMetrics, setEvaluateMetrics] = useState(false)

  const readyCount = [knowledgeFiles.length > 0, modelFiles.length > 0, datasetFiles.length > 0].filter(Boolean).length
  const progress = (readyCount / 3) * 100
  const isReady = readyCount === 3

  const saveExplainerConfig = () => {
    const config = {
      method: method === "auto" ? null : method,
      shap: { n_samples: shapNSamples, background_size: shapBackground },
      lime: { num_features: limeNumFeatures, num_samples: limeNumSamples },
      anchor: { threshold: anchorThreshold },
      evaluate_metrics: evaluateMetrics,
    }
    localStorage.setItem("explainerConfig", JSON.stringify(config))
  }

  const handleProcess = async () => {
    saveExplainerConfig()
    setIsProcessing(true)
    try {
      setStatusMsg("Iniciando sesión...")
      await startProcessing()
      setStatusMsg("Subiendo archivos...")
      await Promise.all([
        uploadKnowledgeBase(knowledgeFiles),
        uploadDataset(datasetFiles),
        uploadModel(modelFiles),
      ])
      setStatusMsg("Convirtiendo modelo...")
      const result = await convertModel()
      localStorage.setItem("classes_detected", JSON.stringify(result.classes_detected))
      router.push("/setup/labels")
    } catch (error) {
      console.error(error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Hubo un problema de conexión con el servidor.",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
      setStatusMsg("")
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container px-4 py-8">
        <div className="max-w-5xl mx-auto space-y-6">

          {/* Tech-role banner */}
          <div style={{
            background: "#FEF3C7", border: "1px solid #F59E0B",
            borderRadius: "8px", padding: "14px 20px",
            display: "flex", alignItems: "flex-start", gap: "12px",
          }}>
            <FlaskConical style={{ color: "#B45309", width: "18px", height: "18px", flexShrink: 0, marginTop: "1px" }} />
            <div>
              <p style={{ fontSize: "13px", fontWeight: 600, color: "#92400E", marginBottom: "2px" }}>
                Configuración técnica — ML Engineer / Data Scientist
              </p>
              <p style={{ fontSize: "12px", color: "#B45309", lineHeight: 1.5 }}>
                Este paso requiere preparar tres archivos: la base de conocimientos del dominio, el modelo entrenado y el dataset de referencia.
                Haz clic en el ícono{" "}
                <Info style={{ display: "inline", width: "11px", height: "11px", verticalAlign: "middle" }} />
                {" "}de cada sección para ver los formatos aceptados y restricciones técnicas.
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <h1 className="text-3xl font-bold">Configuración del entorno de explicación</h1>
            <p className="text-muted-foreground">
              Sube tu base de conocimientos, el modelo entrenado y el conjunto de datos de referencia
            </p>
          </div>

          <div className="grid md:grid-cols-1 gap-6">
            <FileUploadCard
              title="1. Base de conocimientos multimodal"
              description="Documentación del dominio: manuales, reportes técnicos, descripciones de variables"
              acceptedTypes=".pdf,.docx,.jpg,.jpeg,.png,.txt"
              files={knowledgeFiles}
              onFilesChange={setKnowledgeFiles}
              tooltipInfo={knowledgeTooltip}
            />
            <FileUploadCard
              title="2. Modelo entrenado"
              description="Modelo de clasificación o regresión en scikit-learn (.pkl / .joblib). Se convierte automáticamente a ONNX."
              acceptedTypes=".pkl,.joblib,.h5"
              files={modelFiles}
              onFilesChange={setModelFiles}
              tooltipInfo={modelTooltip}
            />
            <FileUploadCard
              title="3. Dataset de referencia"
              description="Dataset de entrenamiento con las mismas features del modelo (sin columna target)"
              acceptedTypes=".xlsx,.csv"
              files={datasetFiles}
              onFilesChange={setDatasetFiles}
              tooltipInfo={datasetTooltip}
            />
          </div>

          {/* ── Configuración avanzada de explicadores ── */}
          <div style={{ border: "1px solid #E2DCD8", borderRadius: "10px", overflow: "hidden" }}>
            <button
              onClick={() => setShowAdvanced(v => !v)}
              style={{
                width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "14px 20px", background: showAdvanced ? "#F5F0EE" : "#FAF8F5",
                border: "none", cursor: "pointer", textAlign: "left",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <Settings2 style={{ color: "#8B1D3B", width: "17px", height: "17px" }} />
                <span style={{ fontWeight: 600, fontSize: "14px", color: "#141414" }}>
                  Configuración avanzada de explicadores
                  <span style={{ fontWeight: 400, fontSize: "12px", color: "#999", marginLeft: "6px" }}>(Opcional)</span>
                </span>
                {method !== "auto" && (
                  <span style={{
                    background: "#8B1D3B", color: "#fff", fontSize: "10px",
                    fontWeight: 700, padding: "1px 8px", borderRadius: "10px", textTransform: "uppercase",
                  }}>
                    {method}
                  </span>
                )}
              </div>
              {showAdvanced
                ? <ChevronUp style={{ color: "#777", width: "16px", height: "16px" }} />
                : <ChevronDown style={{ color: "#777", width: "16px", height: "16px" }} />}
            </button>

            {showAdvanced && (
              <div style={{ padding: "24px", background: "#fff", borderTop: "1px solid #E2DCD8" }}>
                <div className="space-y-6">

                  {/* Método */}
                  <div className="space-y-3">
                    <div>
                      <p className="text-sm font-semibold">Método de explicación</p>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Selecciona el algoritmo XAI. En modo Auto, la herramienta elige el más adecuado para el modelo.
                      </p>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {METHOD_OPTIONS.map(opt => (
                        <button
                          key={opt.value}
                          onClick={() => setMethod(opt.value)}
                          title={opt.desc}
                          style={{
                            padding: "8px 12px", borderRadius: "8px", fontSize: "13px", fontWeight: 600,
                            border: method === opt.value ? "2px solid #8B1D3B" : "2px solid #E2DCD8",
                            background: method === opt.value ? "#FDF2F4" : "#FAF8F5",
                            color: method === opt.value ? "#8B1D3B" : "#555",
                            cursor: "pointer", transition: "all 0.15s",
                          }}
                        >
                          {opt.label}
                          {opt.value === "auto" && (
                            <span style={{ display: "block", fontSize: "9px", fontWeight: 400, color: "#888", marginTop: "1px" }}>
                              recomendado
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground italic">
                      {METHOD_OPTIONS.find(o => o.value === method)?.desc}
                    </p>
                  </div>

                  {/* Parámetros SHAP */}
                  {(method === "shap" || method === "auto") && (
                    <div style={{ background: "#F8FAFF", border: "1px solid #DBEAFE", borderRadius: "8px", padding: "16px" }}>
                      <p className="text-sm font-semibold mb-4" style={{ color: "#1D4ED8" }}>Parámetros SHAP</p>
                      <div className="grid sm:grid-cols-2 gap-6">
                        <SliderField
                          label="n_samples" value={shapNSamples} min={5} max={200} step={5}
                          onChange={setShapNSamples} hint="muestras de fondo para KernelSHAP"
                        />
                        <SliderField
                          label="background_size" value={shapBackground} min={10} max={500} step={10}
                          onChange={setShapBackground} hint="filas del dataset como referencia"
                        />
                      </div>
                    </div>
                  )}

                  {/* Parámetros LIME */}
                  {(method === "lime" || method === "auto") && (
                    <div style={{ background: "#F0FDF4", border: "1px solid #BBF7D0", borderRadius: "8px", padding: "16px" }}>
                      <p className="text-sm font-semibold mb-4" style={{ color: "#15803D" }}>Parámetros LIME</p>
                      <div className="grid sm:grid-cols-2 gap-6">
                        <SliderField
                          label="num_features" value={limeNumFeatures} min={3} max={20} step={1}
                          onChange={setLimeNumFeatures} hint="features a mostrar en la explicación"
                        />
                        <SliderField
                          label="num_samples" value={limeNumSamples} min={100} max={5000} step={100}
                          onChange={setLimeNumSamples} hint="perturbaciones para ajustar el modelo local"
                        />
                      </div>
                    </div>
                  )}

                  {/* Parámetros Anchor */}
                  {(method === "anchor" || method === "auto") && (
                    <div style={{ background: "#FFF7ED", border: "1px solid #FED7AA", borderRadius: "8px", padding: "16px" }}>
                      <p className="text-sm font-semibold mb-4" style={{ color: "#C2410C" }}>Parámetros Anchor</p>
                      <SliderField
                        label="threshold" value={anchorThreshold} min={0.5} max={1.0} step={0.01}
                        onChange={setAnchorThreshold} hint="precisión mínima para aceptar una regla"
                      />
                    </div>
                  )}

                  {/* Métricas de calidad */}
                  <div style={{
                    background: evaluateMetrics ? "#FDF2F4" : "#FAF8F5",
                    border: `1px solid ${evaluateMetrics ? "#F9A8B8" : "#E2DCD8"}`,
                    borderRadius: "8px", padding: "16px 20px",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <div>
                        <p className="text-sm font-semibold">Métricas de calidad del explicador</p>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          Calcula Infidelity, Local Lipschitz y Effective Complexity para cada método
                        </p>
                      </div>
                      <button
                        onClick={() => setEvaluateMetrics(v => !v)}
                        style={{
                          width: "44px", height: "24px", borderRadius: "12px",
                          background: evaluateMetrics ? "#8B1D3B" : "#CCC",
                          border: "none", cursor: "pointer", position: "relative",
                          transition: "background 0.2s", flexShrink: 0,
                        }}
                        aria-label="Activar métricas de calidad"
                      >
                        <span style={{
                          position: "absolute", top: "3px",
                          left: evaluateMetrics ? "23px" : "3px",
                          width: "18px", height: "18px", borderRadius: "9px",
                          background: "#fff", transition: "left 0.2s",
                        }} />
                      </button>
                    </div>
                    {evaluateMetrics && (
                      <div style={{
                        marginTop: "12px", background: "#FFF7ED", border: "1px solid #FED7AA",
                        borderRadius: "6px", padding: "10px 14px", fontSize: "12px", color: "#92400E",
                      }}>
                        <strong>Advertencia:</strong> activar esta opción ejecuta los tres explicadores (SHAP, LIME y Anchor) por cada instancia,
                        lo que aumenta el tiempo de respuesta significativamente (~3×) y el consumo de RAM.
                        No recomendado en servidores con memoria limitada.
                      </div>
                    )}
                  </div>

                </div>
              </div>
            )}
          </div>

          {/* Progress & Actions */}
          <Card className="p-6 space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-medium">Progreso de la configuración</span>
                <span className="text-muted-foreground">{readyCount}/3 listos</span>
              </div>
              <Progress value={progress} />
            </div>
            <div className="flex gap-4">
              <Button variant="ghost" asChild>
                <Link href="/auth">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Atrás
                </Link>
              </Button>
              <Button className="flex-1" disabled={!isReady || isProcessing} onClick={handleProcess}>
                {isProcessing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {isProcessing ? statusMsg || "Procesando..." : "Procesar y continuar"}
              </Button>
            </div>
          </Card>

        </div>
      </main>
    </div>
  )
}
