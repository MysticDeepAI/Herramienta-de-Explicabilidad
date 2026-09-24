"use client"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ArrowLeft, Loader2, FlaskConical, ExternalLink, Info } from "lucide-react"
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
    <a
      href={YOUTUBE_GUIDE}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs text-primary font-medium hover:underline"
    >
      Ver guía completa <ExternalLink className="h-3 w-3" />
    </a>
  </div>
)

export default function SetupPage() {
  const router = useRouter()
  const { toast } = useToast()

  const [knowledgeFiles, setKnowledgeFiles] = useState<File[]>([])
  const [modelFiles, setModelFiles] = useState<File[]>([])
  const [datasetFiles, setDatasetFiles] = useState<File[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [statusMsg, setStatusMsg] = useState("")

  const readyCount = [
    knowledgeFiles.length > 0,
    modelFiles.length > 0,
    datasetFiles.length > 0,
  ].filter(Boolean).length
  const progress = (readyCount / 3) * 100
  const isReady = readyCount === 3

  const handleProcess = async () => {
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

      setStatusMsg("Convirtiendo modelo (Docker in Docker)...")
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
          <div
            style={{
              background: "#FEF3C7",
              border: "1px solid #F59E0B",
              borderRadius: "8px",
              padding: "14px 20px",
              display: "flex",
              alignItems: "flex-start",
              gap: "12px",
            }}
          >
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
