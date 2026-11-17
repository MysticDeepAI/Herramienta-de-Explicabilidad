"use client"

import { useState, useEffect, useRef } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  ArrowLeft, Shuffle, Hash, Tag, Upload,
  CheckCircle2, AlertCircle, Layers, User, FileJson,
} from "lucide-react"
import { Navbar } from "@/components/navbar"
import { getDatasetSchema, getRandomInstance, type DatasetSchema, type InstanceData } from "@/lib/mock-api"
import { useToast } from "@/hooks/use-toast"

type View = "select" | "single" | "batch"

export default function InstancePage() {
  const router = useRouter()
  const { toast } = useToast()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [view, setView] = useState<View>("select")
  const [schema, setSchema] = useState<DatasetSchema | null>(null)
  const [instance, setInstance] = useState<InstanceData>({})

  // Batch state
  const [jsonText, setJsonText] = useState("")
  const [batchInstances, setBatchInstances] = useState<InstanceData[]>([])
  const [batchError, setBatchError] = useState("")
  const [checkedIdxs, setCheckedIdxs] = useState<Set<number>>(new Set())

  useEffect(() => {
    getDatasetSchema().then(setSchema)
  }, [])

  // ── helpers ──────────────────────────────────────────────────────────────

  const handleRandomInstance = async () => {
    const randomData = await getRandomInstance()
    setInstance(randomData)
    toast({ title: "Caso aleatorio cargado", description: "Se cargó una fila aleatoria de la base de datos." })
  }

  const handleSubmitSingle = () => {
    localStorage.setItem("currentInstance", JSON.stringify(instance))
    router.push("/profile")
  }

  const parseBatchJson = (text: string) => {
    setBatchError("")
    try {
      const parsed = JSON.parse(text)
      if (!Array.isArray(parsed)) {
        setBatchError("El JSON debe ser un arreglo de objetos: [ {...}, {...} ]")
        setBatchInstances([])
        return
      }
      if (parsed.length === 0) {
        setBatchError("El arreglo está vacío.")
        setBatchInstances([])
        return
      }
      setBatchInstances(parsed as InstanceData[])
      setCheckedIdxs(new Set(parsed.map((_: InstanceData, i: number) => i)))
      toast({ title: `${parsed.length} instancias cargadas`, description: "Todas seleccionadas automáticamente." })
    } catch {
      setBatchError("JSON inválido. Verifica el formato del archivo.")
      setBatchInstances([])
    }
  }

  const handleJsonFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target?.result as string
      setJsonText(text)
      parseBatchJson(text)
    }
    reader.readAsText(file)
  }

  const toggleCheck = (e: React.MouseEvent, idx: number) => {
    e.stopPropagation()
    setCheckedIdxs(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (checkedIdxs.size === batchInstances.length) setCheckedIdxs(new Set())
    else setCheckedIdxs(new Set(batchInstances.map((_, i) => i)))
  }

  const handleExplainBatch = () => {
    const selected = [...checkedIdxs].sort().map(i => batchInstances[i])
    localStorage.setItem("batchInstances", JSON.stringify(selected))
    router.push("/profile?batch=true")
  }

  // ── derived ───────────────────────────────────────────────────────────────

  if (!schema) {
    return <div className="min-h-screen bg-background flex items-center justify-center">Cargando...</div>
  }

  const numericalCols = schema.columns.filter(c => c.type === "numerical").length
  const categoricalCols = schema.columns.filter(c => c.type === "categorical").length
  const previewCols = schema.columns.slice(0, 4)
  const allFilled = schema.columns.length > 0 &&
    schema.columns.every(col => instance[col.name] !== undefined && instance[col.name] !== "")
  const allChecked = batchInstances.length > 0 && checkedIdxs.size === batchInstances.length
  const someChecked = checkedIdxs.size > 0

  const exampleRows = [0, 1].map(() =>
    Object.fromEntries(schema.columns.map(col => [
      col.name,
      col.type === "numerical" ? Math.round(Math.random() * 100) : col.options?.[0] ?? "valor_ejemplo",
    ]))
  )
  const jsonPlaceholder = JSON.stringify(exampleRows, null, 2)

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container px-4 py-8">
        <div className="max-w-3xl mx-auto space-y-6">

          {/* ════════════════════════════════════════════
              VISTA: SELECCIÓN
          ════════════════════════════════════════════ */}
          {view === "select" && (
            <>
              <div className="space-y-1">
                <h1 className="text-3xl font-bold">Tipo de análisis</h1>
                <p className="text-muted-foreground">
                  ¿Deseas explicar un caso individual o un conjunto de instancias?
                </p>
              </div>

              {/* Chips de resumen del modelo */}
              <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                <div style={{
                  background: "#EFF6FF", border: "1px solid #BFDBFE",
                  borderRadius: "7px", padding: "8px 14px",
                  display: "flex", alignItems: "center", gap: "7px",
                }}>
                  <Hash style={{ color: "#2563EB", width: "14px", height: "14px" }} />
                  <span style={{ fontSize: "12px", color: "#1E40AF", fontWeight: 600 }}>
                    {numericalCols} variable{numericalCols !== 1 ? "s" : ""} numérica{numericalCols !== 1 ? "s" : ""}
                  </span>
                </div>
                <div style={{
                  background: "#F0FDF4", border: "1px solid #BBF7D0",
                  borderRadius: "7px", padding: "8px 14px",
                  display: "flex", alignItems: "center", gap: "7px",
                }}>
                  <Tag style={{ color: "#16A34A", width: "14px", height: "14px" }} />
                  <span style={{ fontSize: "12px", color: "#166534", fontWeight: 600 }}>
                    {categoricalCols} variable{categoricalCols !== 1 ? "s" : ""} categórica{categoricalCols !== 1 ? "s" : ""}
                  </span>
                </div>
              </div>

              {/* Las dos tarjetas de selección */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>

                {/* Tarjeta: instancia única */}
                <button
                  onClick={() => setView("single")}
                  style={{
                    textAlign: "left", background: "#fff",
                    border: "2px solid #E2DCD8", borderRadius: "14px",
                    padding: "28px 24px", cursor: "pointer",
                    transition: "border-color 0.15s, box-shadow 0.15s",
                    boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
                  }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "#8B1D3B"
                    ;(e.currentTarget as HTMLButtonElement).style.boxShadow = "0 4px 16px rgba(139,29,59,0.10)"
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "#E2DCD8"
                    ;(e.currentTarget as HTMLButtonElement).style.boxShadow = "0 1px 4px rgba(0,0,0,0.06)"
                  }}
                >
                  <div style={{
                    width: "48px", height: "48px", borderRadius: "12px",
                    background: "#FDF2F4", border: "1px solid #F9A8B8",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    marginBottom: "16px",
                  }}>
                    <User style={{ color: "#8B1D3B", width: "22px", height: "22px" }} />
                  </div>
                  <p style={{ fontSize: "17px", fontWeight: 700, color: "#141414", marginBottom: "6px" }}>
                    Por instancia
                  </p>
                  <p style={{ fontSize: "13px", color: "#666", lineHeight: 1.6, marginBottom: "14px" }}>
                    Analiza <strong>un único caso</strong>: ingresa los valores de sus variables y obtén la explicación de la predicción.
                  </p>
                  {/* Mini-ejemplo de una fila */}
                  <div style={{
                    background: "#FAF8F5", border: "1px solid #E2DCD8",
                    borderRadius: "8px", padding: "10px 12px",
                  }}>
                    <p style={{ fontSize: "10px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "#999", marginBottom: "7px" }}>
                      Ejemplo — 1 caso
                    </p>
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      {schema.columns.slice(0, 3).map(col => (
                        <div key={col.name} style={{ display: "flex", justifyContent: "space-between", fontSize: "11px" }}>
                          <span style={{ color: "#888", fontFamily: "monospace" }}>{col.name}</span>
                          <span style={{ fontWeight: 600, color: "#333", fontFamily: "monospace" }}>
                            {col.type === "numerical" ? "42" : col.options?.[0] ?? "valor"}
                          </span>
                        </div>
                      ))}
                      {schema.columns.length > 3 && (
                        <p style={{ fontSize: "10px", color: "#bbb", marginTop: "1px" }}>
                          + {schema.columns.length - 3} campo{schema.columns.length - 3 !== 1 ? "s" : ""} más
                        </p>
                      )}
                    </div>
                  </div>
                  <div style={{
                    marginTop: "16px", display: "inline-flex", alignItems: "center", gap: "6px",
                    fontSize: "13px", fontWeight: 600, color: "#8B1D3B",
                  }}>
                    Seleccionar <span style={{ fontSize: "16px" }}>→</span>
                  </div>
                </button>

                {/* Tarjeta: lote */}
                <button
                  onClick={() => setView("batch")}
                  style={{
                    textAlign: "left", background: "#fff",
                    border: "2px solid #E2DCD8", borderRadius: "14px",
                    padding: "28px 24px", cursor: "pointer",
                    transition: "border-color 0.15s, box-shadow 0.15s",
                    boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
                  }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "#8B1D3B"
                    ;(e.currentTarget as HTMLButtonElement).style.boxShadow = "0 4px 16px rgba(139,29,59,0.10)"
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "#E2DCD8"
                    ;(e.currentTarget as HTMLButtonElement).style.boxShadow = "0 1px 4px rgba(0,0,0,0.06)"
                  }}
                >
                  <div style={{
                    width: "48px", height: "48px", borderRadius: "12px",
                    background: "#F0FDF4", border: "1px solid #BBF7D0",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    marginBottom: "16px",
                  }}>
                    <Layers style={{ color: "#16A34A", width: "22px", height: "22px" }} />
                  </div>
                  <p style={{ fontSize: "17px", fontWeight: 700, color: "#141414", marginBottom: "6px" }}>
                    Por instancias
                  </p>
                  <p style={{ fontSize: "13px", color: "#666", lineHeight: 1.6, marginBottom: "14px" }}>
                    Analiza <strong>varios casos a la vez</strong>: carga un JSON con múltiples filas y obtén una explicación por cada una.
                  </p>
                  {/* Mini-ejemplo de varias filas */}
                  <div style={{
                    background: "#F0FDF4", border: "1px solid #BBF7D0",
                    borderRadius: "8px", padding: "10px 12px",
                  }}>
                    <p style={{ fontSize: "10px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "#6EE7B7", marginBottom: "7px" }}>
                      Ejemplo — 3 casos
                    </p>
                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                      {[1, 2, 3].map(n => (
                        <div key={n} style={{
                          background: "#fff", borderRadius: "5px", padding: "5px 8px",
                          border: "1px solid #D1FAE5", display: "flex", gap: "8px", alignItems: "center",
                        }}>
                          <span style={{ fontSize: "10px", fontWeight: 700, color: "#059669", minWidth: "14px" }}>
                            #{n}
                          </span>
                          <span style={{ fontSize: "10px", color: "#555", fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {schema.columns.slice(0, 2).map(col =>
                              `${col.name}: ${col.type === "numerical" ? n * 10 + 5 : col.options?.[0] ?? "val"}`
                            ).join(", ")} …
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{
                    marginTop: "16px", display: "inline-flex", alignItems: "center", gap: "6px",
                    fontSize: "13px", fontWeight: 600, color: "#16A34A",
                  }}>
                    Seleccionar <span style={{ fontSize: "16px" }}>→</span>
                  </div>
                </button>

              </div>

              <div className="flex gap-4">
                <Button variant="ghost" asChild>
                  <Link href="/setup/labels">
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Atrás
                  </Link>
                </Button>
              </div>
            </>
          )}

          {/* ════════════════════════════════════════════
              VISTA: INSTANCIA ÚNICA
          ════════════════════════════════════════════ */}
          {view === "single" && (
            <>
              <div className="space-y-1">
                <h1 className="text-3xl font-bold">Definir caso individual</h1>
                <p className="text-muted-foreground">
                  Ingresa los valores de las características del caso que deseas explicar
                </p>
              </div>

              {/* Atajo de carga aleatoria */}
              <div style={{
                background: "#FAF8F5", border: "1px solid #E2DCD8",
                borderRadius: "8px", padding: "12px 18px",
                display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <Shuffle style={{ color: "#8B1D3B", width: "15px", height: "15px" }} />
                  <span style={{ fontSize: "13px", color: "#555" }}>
                    ¿No tienes un caso específico?
                  </span>
                </div>
                <Button variant="outline" size="sm" className="bg-transparent" onClick={handleRandomInstance}>
                  <Shuffle className="mr-2 h-3.5 w-3.5" />
                  Carga un caso aleatorio de la base de datos
                </Button>
              </div>

              {/* Formulario */}
              <Card>
                <CardHeader>
                  <CardTitle>Características del caso</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Cada campo corresponde a una variable del modelo.
                  </p>
                </CardHeader>
                <CardContent className="space-y-5">
                  {schema.columns.map((column, idx) => (
                    <div key={idx} className="space-y-1.5">
                      <Label htmlFor={column.name} className="text-sm font-medium flex items-center gap-2">
                        {column.name}
                        <span style={{
                          fontSize: "10px", fontWeight: 600, textTransform: "uppercase",
                          letterSpacing: "0.05em", padding: "1px 6px", borderRadius: "4px",
                          background: column.type === "numerical" ? "#EFF6FF" : "#F0FDF4",
                          color: column.type === "numerical" ? "#2563EB" : "#16A34A",
                          border: `1px solid ${column.type === "numerical" ? "#BFDBFE" : "#BBF7D0"}`,
                        }}>
                          {column.type === "numerical" ? "Numérico" : "Categórico"}
                        </span>
                      </Label>
                      {column.type === "categorical" && column.options ? (
                        <Select
                          value={instance[column.name]?.toString() || ""}
                          onValueChange={(value) => setInstance({ ...instance, [column.name]: value })}
                        >
                          <SelectTrigger id={column.name}>
                            <SelectValue placeholder="Seleccionar categoría..." />
                          </SelectTrigger>
                          <SelectContent>
                            {column.options.map((option) => (
                              <SelectItem key={option} value={option}>{option}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      ) : (
                        <Input
                          id={column.name}
                          type={column.type === "numerical" ? "number" : "text"}
                          value={instance[column.name]?.toString() || ""}
                          onChange={(e) => {
                            const val = column.type === "numerical" ? Number(e.target.value) : e.target.value
                            setInstance({ ...instance, [column.name]: val })
                          }}
                          placeholder={column.type === "numerical" ? "Valor numérico" : "Texto"}
                        />
                      )}
                    </div>
                  ))}
                </CardContent>
              </Card>

              <div className="flex gap-4">
                <Button variant="ghost" onClick={() => setView("select")}>
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Atrás
                </Button>
                <Button className="flex-1" disabled={!allFilled} onClick={handleSubmitSingle}>
                  Seleccionar perfil de explicación
                </Button>
              </div>
            </>
          )}

          {/* ════════════════════════════════════════════
              VISTA: LOTE DE INSTANCIAS
          ════════════════════════════════════════════ */}
          {view === "batch" && (
            <>
              <div className="space-y-1">
                <h1 className="text-3xl font-bold">Cargar instancias</h1>
                <p className="text-muted-foreground">
                  Sube o pega un JSON con los casos que deseas analizar
                </p>
              </div>

              {/* Info de features */}
              <div style={{
                background: "#FFF7ED", border: "1px solid #FED7AA",
                borderRadius: "8px", padding: "14px 18px", fontSize: "12px", color: "#92400E",
              }}>
                <strong>Formato esperado:</strong> un arreglo JSON donde cada objeto tiene las features del modelo.
                <p style={{ marginTop: "8px", marginBottom: "6px", fontWeight: 600 }}>
                  Features del modelo cargado:
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                  {schema.columns.map(col => (
                    <span key={col.name} style={{
                      background: col.type === "numerical" ? "#DBEAFE" : "#D1FAE5",
                      color: col.type === "numerical" ? "#1E40AF" : "#065F46",
                      border: `1px solid ${col.type === "numerical" ? "#BFDBFE" : "#A7F3D0"}`,
                      borderRadius: "4px", padding: "1px 7px", fontSize: "11px",
                      fontFamily: "monospace", fontWeight: 600,
                    }}>
                      {col.name}
                      <span style={{ fontWeight: 400, opacity: 0.7, marginLeft: "4px" }}>
                        {col.type === "numerical" ? "num" : "cat"}
                      </span>
                    </span>
                  ))}
                </div>
              </div>

              {/* Carga de archivo */}
              <div className="space-y-3">
                <div className="flex gap-3 items-center">
                  <Button
                    variant="outline" size="sm" className="bg-transparent"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <FileJson className="mr-2 h-3.5 w-3.5" />
                    Subir archivo .json
                  </Button>
                  <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleJsonFile} />
                  <span style={{ fontSize: "12px", color: "#777" }}>o pega el JSON directamente:</span>
                </div>

                <textarea
                  value={jsonText}
                  onChange={(e) => setJsonText(e.target.value)}
                  placeholder={jsonPlaceholder}
                  style={{
                    width: "100%", minHeight: "140px", padding: "10px 14px",
                    border: "1px solid #E2DCD8", borderRadius: "8px", fontSize: "12px",
                    fontFamily: "monospace", resize: "vertical", background: "#FAF8F5",
                    outline: "none", color: "#141414",
                  }}
                />

                {batchError && (
                  <div style={{ display: "flex", gap: "8px", alignItems: "center", color: "#DC2626", fontSize: "12px" }}>
                    <AlertCircle style={{ width: "14px", height: "14px" }} />
                    {batchError}
                  </div>
                )}

                <Button size="sm" onClick={() => parseBatchJson(jsonText)} disabled={!jsonText.trim()}>
                  Cargar instancias
                </Button>
              </div>

              {/* Tabla con multi-select */}
              {batchInstances.length > 0 && (
                <div className="space-y-3">
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <p className="text-sm font-medium">
                      {batchInstances.length} instancia{batchInstances.length !== 1 ? "s" : ""} cargadas
                      {someChecked && (
                        <span style={{ color: "#8B1D3B", marginLeft: "8px" }}>
                          · {checkedIdxs.size} seleccionada{checkedIdxs.size !== 1 ? "s" : ""}
                        </span>
                      )}
                    </p>
                    <button
                      onClick={toggleSelectAll}
                      style={{ fontSize: "11px", color: "#8B1D3B", fontWeight: 600, background: "none", border: "none", cursor: "pointer", textDecoration: "underline" }}
                    >
                      {allChecked ? "Deseleccionar todas" : "Seleccionar todas"}
                    </button>
                  </div>

                  <div style={{ overflowX: "auto", borderRadius: "8px", border: "1px solid #E2DCD8" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
                      <thead>
                        <tr style={{ background: "#F5F0EE" }}>
                          <th style={{ padding: "8px 12px", textAlign: "center", fontWeight: 600, color: "#555", borderBottom: "1px solid #E2DCD8", width: "36px" }}>✓</th>
                          <th style={{ padding: "8px 12px", textAlign: "left", fontWeight: 600, color: "#555", borderBottom: "1px solid #E2DCD8" }}>#</th>
                          {previewCols.map(col => (
                            <th key={col.name} style={{ padding: "8px 12px", textAlign: "left", fontWeight: 600, color: "#555", borderBottom: "1px solid #E2DCD8", whiteSpace: "nowrap" }}>
                              {col.name}
                            </th>
                          ))}
                          {schema.columns.length > 4 && (
                            <th style={{ padding: "8px 12px", textAlign: "left", fontWeight: 600, color: "#aaa", borderBottom: "1px solid #E2DCD8" }}>
                              +{schema.columns.length - 4} más
                            </th>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {batchInstances.map((row, idx) => {
                          const isChecked = checkedIdxs.has(idx)
                          return (
                            <tr
                              key={idx}
                              style={{
                                background: isChecked ? "#FFF5F7" : idx % 2 === 0 ? "#fff" : "#FAFAFA",
                                borderLeft: isChecked ? "3px solid #8B1D3B" : "3px solid transparent",
                                cursor: "pointer", transition: "background 0.1s",
                              }}
                              onClick={(e) => toggleCheck(e, idx)}
                            >
                              <td style={{ padding: "8px 12px", borderBottom: "1px solid #F0EBE8", textAlign: "center" }}>
                                <div style={{
                                  width: "16px", height: "16px", border: `2px solid ${isChecked ? "#8B1D3B" : "#CCC"}`,
                                  borderRadius: "3px", background: isChecked ? "#8B1D3B" : "#fff",
                                  display: "flex", alignItems: "center", justifyContent: "center",
                                  margin: "0 auto", transition: "all 0.1s",
                                }}>
                                  {isChecked && <span style={{ color: "#fff", fontSize: "10px", fontWeight: 700 }}>✓</span>}
                                </div>
                              </td>
                              <td style={{ padding: "8px 12px", borderBottom: "1px solid #F0EBE8", color: "#888", fontWeight: 600 }}>
                                {isChecked
                                  ? <CheckCircle2 style={{ width: "14px", height: "14px", color: "#8B1D3B" }} />
                                  : idx + 1}
                              </td>
                              {previewCols.map(col => (
                                <td key={col.name} style={{ padding: "8px 12px", borderBottom: "1px solid #F0EBE8", color: "#333", maxWidth: "140px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                  {String(row[col.name] ?? "—")}
                                </td>
                              ))}
                              {schema.columns.length > 4 && (
                                <td style={{ padding: "8px 12px", borderBottom: "1px solid #F0EBE8", color: "#bbb" }}>…</td>
                              )}
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>

                  {someChecked && (
                    <div style={{ display: "flex", justifyContent: "flex-end" }}>
                      <Button onClick={handleExplainBatch} style={{ gap: "8px" }}>
                        <Layers className="h-4 w-4" />
                        Explicar {checkedIdxs.size} instancia{checkedIdxs.size !== 1 ? "s" : ""} seleccionada{checkedIdxs.size !== 1 ? "s" : ""}
                      </Button>
                    </div>
                  )}
                </div>
              )}

              <div className="flex gap-4">
                <Button variant="ghost" onClick={() => setView("select")}>
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Atrás
                </Button>
              </div>
            </>
          )}

        </div>
      </main>
    </div>
  )
}
