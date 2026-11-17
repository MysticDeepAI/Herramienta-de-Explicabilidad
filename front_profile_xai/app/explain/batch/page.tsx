"use client"

import React, { useState, useEffect, useRef } from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Navbar } from "@/components/navbar"
import { ChatPanel } from "@/components/chat-panel"
import {
  generateExplanation, chatRag, downloadExplanationReport,
  type UserProfile, type InstanceData, type ChatMessage,
} from "@/lib/mock-api"
import {
  ArrowLeft, ChevronDown, ChevronUp, Layers, Loader2,
  CheckCircle2, AlertCircle, FileDown, FileJson,
} from "lucide-react"

interface ExplainResult {
  instanceIdx: number
  instance: InstanceData
  status: "pending" | "loading" | "done" | "error"
  label?: string
  summary?: string
  fullText?: string
  errorMsg?: string
}

const PROFILE_LABELS: Record<string, string> = {
  "data-scientist": "Especialista en IA",
  "domain-expert":  "Experto en el Dominio",
  "non-expert":     "Usuario General",
}

export default function BatchExplainPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const profile = (searchParams.get("profile") ?? "non-expert") as UserProfile

  const [instances, setInstances] = useState<InstanceData[]>([])
  const [results, setResults] = useState<ExplainResult[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [currentIdx, setCurrentIdx] = useState<number | null>(null)
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null)
  const [done, setDone] = useState(false)

  // Chat
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isChatLoading, setIsChatLoading] = useState(false)
  const [combinedContext, setCombinedContext] = useState("")

  // Download
  const [isPdfLoading, setIsPdfLoading] = useState(false)

  useEffect(() => {
    try {
      const raw = localStorage.getItem("batchInstances")
      if (!raw) { router.push("/instance"); return }
      const parsed: InstanceData[] = JSON.parse(raw)
      if (!parsed.length) { router.push("/instance"); return }
      setInstances(parsed)
      setResults(parsed.map((inst, i) => ({
        instanceIdx: i,
        instance: inst,
        status: "pending",
      })))
    } catch {
      router.push("/instance")
    }
  }, [])

  // Rebuild combined context whenever results change
  useEffect(() => {
    const doneResults = results.filter(r => r.status === "done")
    if (!doneResults.length) return
    const ctx = [
      `=== ANÁLISIS POR LOTE — ${instances.length} INSTANCIAS ===`,
      `Perfil: ${PROFILE_LABELS[profile] ?? profile}`,
      "",
      ...doneResults.map(r =>
        `=== Instancia ${r.instanceIdx + 1} ===\nPredicción: ${r.label ?? "—"}\n${r.fullText ?? ""}`
      ),
    ].join("\n\n")
    setCombinedContext(ctx)
  }, [results])

  const runBatch = async () => {
    setIsRunning(true)
    setDone(false)
    setMessages([])
    setCombinedContext("")
    setResults(prev => prev.map(r => ({ ...r, status: "pending", label: undefined, summary: undefined, fullText: undefined, errorMsg: undefined })))

    for (let i = 0; i < instances.length; i++) {
      setCurrentIdx(i)
      setResults(prev => prev.map((r, idx) => idx === i ? { ...r, status: "loading" } : r))

      try {
        const result = await generateExplanation(profile, instances[i])
        const naturalText: string = result.natural ?? ""
        const label: string = result.label ?? result.prediction ?? "—"
        const cleanText = naturalText.replace(/\*\*(.+?)\*\*/g, '$1').replace(/\*(.+?)\*/g, '$1')
        const summary = cleanText.length > 160 ? cleanText.slice(0, 157) + "..." : cleanText

        setResults(prev => prev.map((r, idx) => idx === i ? {
          ...r, status: "done", label, summary, fullText: cleanText,
        } : r))
      } catch (err: any) {
        setResults(prev => prev.map((r, idx) => idx === i ? {
          ...r, status: "error", errorMsg: err?.message ?? "Error desconocido",
        } : r))
      }
    }

    setCurrentIdx(null)
    setIsRunning(false)
    setDone(true)
  }

  const handleSendMessage = async (message: string) => {
    const userMsg: ChatMessage = { role: "user", content: message, timestamp: new Date() }
    setMessages(prev => [...prev, userMsg])
    setIsChatLoading(true)
    try {
      const response = await chatRag(message, messages, combinedContext, profile)
      const assistantMsg: ChatMessage = { role: "assistant", content: response, timestamp: new Date() }
      setMessages(prev => [...prev, assistantMsg])
    } catch {
      const errMsg: ChatMessage = {
        role: "assistant",
        content: "Error al conectar con el asistente. Intenta de nuevo.",
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setIsChatLoading(false)
    }
  }

  const handleDownloadJson = () => {
    const doneResults = results.filter(r => r.status === "done")
    const payload = {
      profile,
      perfil_label: PROFILE_LABELS[profile] ?? profile,
      generated_at: new Date().toISOString(),
      total_instancias: instances.length,
      explicaciones: doneResults.map(r => ({
        instancia: r.instanceIdx + 1,
        datos: r.instance,
        prediccion: r.label,
        explicacion: r.fullText,
      })),
      chat_history: messages.map(m => ({ rol: m.role, mensaje: m.content })),
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `explicaciones_lote_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleDownloadPdf = async () => {
    const doneResults = results.filter(r => r.status === "done")
    if (!doneResults.length) return
    setIsPdfLoading(true)
    try {
      const explanationText = doneResults.map(r =>
        `INSTANCIA ${r.instanceIdx + 1} — Predicción: ${r.label}\n\n${r.fullText}`
      ).join("\n\n" + "─".repeat(60) + "\n\n")

      await downloadExplanationReport({
        profile,
        explanation: explanationText,
        chat_history: messages.map(m => ({ role: m.role, content: m.content })),
        timestamp: new Date().toISOString(),
        instances: doneResults.map(r => r.instance as Record<string, unknown>),
      })
    } catch {
      alert("Error al generar el PDF. Intenta de nuevo.")
    } finally {
      setIsPdfLoading(false)
    }
  }

  const completedCount = results.filter(r => r.status === "done").length
  const errorCount    = results.filter(r => r.status === "error").length
  const pendingCount  = results.filter(r => r.status === "pending").length

  if (!instances.length) return null

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container px-4 py-8">
        <div className="max-w-5xl mx-auto space-y-6">

          {/* Header */}
          <div className="space-y-2">
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <Layers style={{ color: "#8B1D3B", width: "22px", height: "22px" }} />
              <h1 className="text-3xl font-bold">Resultados del análisis por lote</h1>
            </div>
            <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
              <span style={{
                background: "#FAF8F5", border: "1px solid #E2DCD8",
                borderRadius: "6px", padding: "3px 10px", fontSize: "12px", color: "#555",
              }}>
                {instances.length} instancia{instances.length !== 1 ? "s" : ""}
              </span>
              <span style={{
                background: "#FDF2F4", border: "1px solid #F9A8B8",
                borderRadius: "6px", padding: "3px 10px", fontSize: "12px", color: "#8B1D3B", fontWeight: 600,
              }}>
                Perfil: {PROFILE_LABELS[profile] ?? profile}
              </span>
            </div>
          </div>

          {/* Iniciar / progreso */}
          {!done && !isRunning && (
            <div style={{
              background: "#EFF6FF", border: "1px solid #BFDBFE",
              borderRadius: "8px", padding: "18px 22px",
              display: "flex", alignItems: "center", justifyContent: "space-between", gap: "16px",
            }}>
              <p style={{ fontSize: "13px", color: "#1E40AF" }}>
                Se llamará al modelo <strong>{instances.length} vece{instances.length !== 1 ? "s" : "z"}</strong> de forma secuencial.
                Esto puede tomar unos minutos dependiendo del tamaño del modelo y la base de conocimientos.
              </p>
              <Button onClick={runBatch} style={{ whiteSpace: "nowrap" }}>
                <Layers className="mr-2 h-4 w-4" />
                Iniciar análisis
              </Button>
            </div>
          )}

          {isRunning && currentIdx !== null && (
            <div style={{
              background: "#FFF7ED", border: "1px solid #FED7AA",
              borderRadius: "8px", padding: "14px 18px",
              display: "flex", alignItems: "center", gap: "12px",
            }}>
              <Loader2 style={{ color: "#D97706", width: "18px", height: "18px", animation: "spin 1s linear infinite" }} />
              <p style={{ fontSize: "13px", color: "#92400E" }}>
                Analizando instancia <strong>{currentIdx + 1}</strong> de <strong>{instances.length}</strong>…
              </p>
              <div style={{ flex: 1, background: "#FEF3C7", borderRadius: "4px", height: "6px", overflow: "hidden" }}>
                <div style={{
                  background: "#D97706", height: "100%", borderRadius: "4px",
                  width: `${(currentIdx / instances.length) * 100}%`,
                  transition: "width 0.3s ease",
                }} />
              </div>
            </div>
          )}

          {done && (
            <div style={{
              background: "#F0FDF4", border: "1px solid #BBF7D0",
              borderRadius: "8px", padding: "14px 18px",
              display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap",
            }}>
              <CheckCircle2 style={{ color: "#16A34A", width: "18px", height: "18px" }} />
              <p style={{ fontSize: "13px", color: "#166534", flex: 1 }}>
                Análisis completado — <strong>{completedCount}</strong> exitosa{completedCount !== 1 ? "s" : ""}
                {errorCount > 0 && <span style={{ color: "#DC2626" }}>, {errorCount} con error</span>}.
              </p>
              <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                <Button variant="outline" size="sm" className="bg-transparent" onClick={runBatch}>
                  Repetir
                </Button>
                <Button
                  variant="outline" size="sm" className="bg-transparent"
                  onClick={handleDownloadJson}
                  disabled={completedCount === 0}
                >
                  <FileJson className="mr-2 h-3.5 w-3.5" />
                  Descargar JSON
                </Button>
                <Button
                  size="sm"
                  onClick={handleDownloadPdf}
                  disabled={completedCount === 0 || isPdfLoading}
                >
                  {isPdfLoading
                    ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                    : <FileDown className="mr-2 h-3.5 w-3.5" />
                  }
                  {isPdfLoading ? "Generando PDF…" : "Descargar PDF"}
                </Button>
              </div>
            </div>
          )}

          {/* Tabla + Chat side-by-side when done, tabla sola mientras corre */}
          <div style={{
            display: done && completedCount > 0 ? "grid" : "block",
            gridTemplateColumns: done && completedCount > 0 ? "1fr 380px" : undefined,
            gap: "20px",
            alignItems: "start",
          }}>
            {/* Tabla de resultados */}
            <div style={{ border: "1px solid #E2DCD8", borderRadius: "10px", overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
                <thead>
                  <tr style={{ background: "#F5F0EE" }}>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#555", borderBottom: "1px solid #E2DCD8", width: "44px" }}>#</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#555", borderBottom: "1px solid #E2DCD8", width: "110px" }}>Estado</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#555", borderBottom: "1px solid #E2DCD8", width: "130px" }}>Predicción</th>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#555", borderBottom: "1px solid #E2DCD8" }}>Resumen</th>
                    <th style={{ padding: "10px 16px", textAlign: "center", fontWeight: 600, color: "#555", borderBottom: "1px solid #E2DCD8", width: "90px" }}>Detalle</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, idx) => {
                    const isExpanded = expandedIdx === idx
                    return (
                      <React.Fragment key={idx}>
                        <tr
                          style={{
                            background: isExpanded ? "#FDF2F4" : idx % 2 === 0 ? "#fff" : "#FAFAFA",
                            borderLeft: isExpanded ? "3px solid #8B1D3B" : "3px solid transparent",
                          }}
                        >
                          <td style={{ padding: "12px 16px", borderBottom: isExpanded ? "none" : "1px solid #F0EBE8", color: "#888", fontWeight: 700 }}>
                            {idx + 1}
                          </td>
                          <td style={{ padding: "12px 16px", borderBottom: isExpanded ? "none" : "1px solid #F0EBE8" }}>
                            {r.status === "pending" && <span style={{ fontSize: "11px", color: "#999", fontWeight: 500 }}>En espera</span>}
                            {r.status === "loading" && (
                              <span style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", color: "#D97706", fontWeight: 600 }}>
                                <Loader2 style={{ width: "12px", height: "12px", animation: "spin 1s linear infinite" }} />
                                Analizando…
                              </span>
                            )}
                            {r.status === "done" && (
                              <span style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", color: "#16A34A", fontWeight: 600 }}>
                                <CheckCircle2 style={{ width: "12px", height: "12px" }} />
                                Listo
                              </span>
                            )}
                            {r.status === "error" && (
                              <span style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", color: "#DC2626", fontWeight: 600 }}>
                                <AlertCircle style={{ width: "12px", height: "12px" }} />
                                Error
                              </span>
                            )}
                          </td>
                          <td style={{ padding: "12px 16px", borderBottom: isExpanded ? "none" : "1px solid #F0EBE8" }}>
                            {r.label ? (
                              <span style={{
                                background: "#FDF2F4", border: "1px solid #F9A8B8",
                                borderRadius: "5px", padding: "2px 8px",
                                fontSize: "12px", fontWeight: 600, color: "#8B1D3B",
                              }}>
                                {r.label}
                              </span>
                            ) : <span style={{ color: "#ccc" }}>—</span>}
                          </td>
                          <td style={{ padding: "12px 16px", borderBottom: isExpanded ? "none" : "1px solid #F0EBE8", color: "#444", lineHeight: 1.5 }}>
                            {r.status === "error"
                              ? <span style={{ color: "#DC2626", fontSize: "12px" }}>{r.errorMsg}</span>
                              : r.summary ?? <span style={{ color: "#CCC" }}>—</span>
                            }
                          </td>
                          <td style={{ padding: "12px 16px", borderBottom: isExpanded ? "none" : "1px solid #F0EBE8", textAlign: "center" }}>
                            {r.status === "done" && (
                              <button
                                onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                                style={{
                                  display: "inline-flex", alignItems: "center", gap: "4px",
                                  fontSize: "11px", color: "#8B1D3B", fontWeight: 600,
                                  background: "none", border: "none", cursor: "pointer",
                                  padding: "4px 8px", borderRadius: "4px",
                                }}
                              >
                                {isExpanded
                                  ? <><ChevronUp style={{ width: "13px", height: "13px" }} /> Cerrar</>
                                  : <><ChevronDown style={{ width: "13px", height: "13px" }} /> Ver más</>
                                }
                              </button>
                            )}
                          </td>
                        </tr>

                        {isExpanded && r.fullText && (
                          <tr style={{ background: "#FDF2F4" }}>
                            <td colSpan={5} style={{ padding: "0 16px 20px 16px", borderBottom: "1px solid #F9A8B8" }}>
                              <div style={{
                                background: "#fff", border: "1px solid #F9A8B8",
                                borderRadius: "8px", padding: "18px 20px", marginTop: "4px",
                              }}>
                                <p style={{
                                  fontSize: "11px", fontWeight: 700, textTransform: "uppercase",
                                  letterSpacing: "0.06em", color: "#888", marginBottom: "10px",
                                }}>
                                  Explicación completa — Instancia #{idx + 1}
                                </p>
                                <p style={{ fontSize: "14px", color: "#333", lineHeight: 1.75, whiteSpace: "pre-wrap" }}>
                                  {r.fullText}
                                </p>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    )
                  })}
                </tbody>
              </table>

              {pendingCount === instances.length && !isRunning && (
                <div style={{ padding: "28px", textAlign: "center", color: "#AAA", fontSize: "13px" }}>
                  Haz clic en <strong>Iniciar análisis</strong> para comenzar.
                </div>
              )}
            </div>

            {/* Chat — columna derecha cuando hay resultados */}
            {done && completedCount > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                <div>
                  <h2 style={{ fontSize: "15px", fontWeight: 700, color: "#141414" }}>
                    Asistente Virtual
                  </h2>
                  <p style={{ fontSize: "12px", color: "#666", marginTop: "2px", lineHeight: 1.4 }}>
                    Contexto de las <strong>{completedCount}</strong> explicación{completedCount !== 1 ? "es" : ""} generadas.
                  </p>
                </div>
                <div style={{ height: "520px" }}>
                  <ChatPanel
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    isLoading={isChatLoading}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="flex gap-4">
            <Button variant="ghost" asChild>
              <Link href="/instance">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Volver a instancias
              </Link>
            </Button>
          </div>

        </div>
      </main>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
