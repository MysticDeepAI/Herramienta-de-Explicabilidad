"use client"

import { useState, useEffect } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Navbar } from "@/components/navbar"
import { ChatPanel } from "@/components/chat-panel"
import { generateExplanation, chatRag, downloadExplanationReport, type ChatMessage, type UserProfile } from "@/lib/mock-api"
import { Loader2, FileDown } from "lucide-react"

export default function NaturalExplanationPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const profile = searchParams.get("profile") as UserProfile

  const [explanation, setExplanation] = useState<string>("")
  const [fullContext, setFullContext] = useState<string>("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isChatLoading, setIsChatLoading] = useState(false)
  const [isReportLoading, setIsReportLoading] = useState(false)

useEffect(() => {
    const savedInstanceString = localStorage.getItem("currentInstance");
    const instanceData = savedInstanceString ? JSON.parse(savedInstanceString) : {};

    generateExplanation(profile || "non-expert", instanceData).then((result) => {
      const naturalText = result.natural || "No se pudo generar una explicación natural."
      setExplanation(naturalText)
      // Contexto completo para el chat: narrativa + convención SHAP + datos técnicos crudos
      const technicalText = result.technical ? JSON.stringify(result.technical) : ""
      const predictedLabel = result.label ?? result.prediction ?? "la clase predicha"
      const shapNote = `NOTA IMPORTANTE SOBRE LOS VALORES SHAP: Los valores SHAP corresponden a la clase PREDICHA ("${predictedLabel}"). Un valor SHAP positivo significa que esa variable empujó la predicción HACIA "${predictedLabel}". Un valor SHAP negativo significa que esa variable se opuso a "${predictedLabel}". El signo siempre es relativo a la clase predicha, no a una clase fija.`
      setFullContext(`${naturalText}\n\n${shapNote}\n\nDatos técnicos de la explicación:\n${technicalText}`)
      setIsLoading(false)
    }).catch(err => {
      console.error(err);
      setIsLoading(false);
    });
  }, [profile])

  const handleSendMessage = async (message: string) => {
    const userMessage: ChatMessage = {
      role: "user",
      content: message,
      timestamp: new Date(),
    }

    setMessages([...messages, userMessage])
    setIsChatLoading(true)

    const response = await chatRag(message, messages, fullContext, profile || "non-expert")

    const assistantMessage: ChatMessage = {
      role: "assistant",
      content: response,
      timestamp: new Date(),
    }

    setMessages([...messages, userMessage, assistantMessage])
    setIsChatLoading(false)
  }

  const handleDownloadReport = async () => {
    setIsReportLoading(true)
    try {
      await downloadExplanationReport({
        profile: profile || "non-expert",
        explanation,
        chat_history: messages.map((m) => ({ role: m.role, content: m.content })),
        timestamp: new Date().toLocaleString("es-CL"),
      })
    } catch (err) {
      console.error("Error descargando reporte:", err)
    } finally {
      setIsReportLoading(false)
    }
  }

  const profileLabel = profile === "domain-expert" ? "Experto en el Dominio" : "Usuario general"

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container px-4 py-8">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">Respuesta adaptada para: {profileLabel}</h1>
            </div>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Explanation */}
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle>Explicación en lenguaje natural</CardTitle>
                  <Badge variant="secondary">RAG + Contexto</Badge>
                </CardHeader>
                <CardContent>
                  {isLoading ? (
                    <div className="flex items-center justify-center py-12">
                      <Loader2 className="h-8 w-8 animate-spin text-primary" />
                    </div>
                  ) : (
                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      <div className="whitespace-pre-line leading-relaxed">{explanation}</div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Actions */}
              <Card className="p-6">
                <div className="flex flex-wrap gap-4">
                  <Button variant="outline" onClick={() => router.push("/profile")}>
                    Cambiar perfil
                  </Button>
                  <Button variant="outline" onClick={() => router.push("/instance")}>
                    Cambiar valores de la instancia
                  </Button>
                  <Button
                    variant="outline"
                    disabled={isLoading || isReportLoading}
                    onClick={handleDownloadReport}
                  >
                    {isReportLoading ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <FileDown className="mr-2 h-4 w-4" />
                    )}
                    Generar reporte PDF
                  </Button>
                  <Button
                    className="ml-auto"
                    onClick={() => {
                      router.push("/")
                    }}
                  >
                    Finalizar
                  </Button>
                </div>
              </Card>
            </div>

            {/* Chat Panel */}
            <div className="lg:col-span-1">
              <div className="lg:sticky lg:top-8 h-[600px]">
                <ChatPanel messages={messages} onSendMessage={handleSendMessage} isLoading={isChatLoading} />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
