"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Navbar } from "@/components/navbar"
import { ChatPanel } from "@/components/chat-panel"
import { MetricCard } from "@/components/metric-card"
import { generateExplanation, chatRag, type ChatMessage } from "@/lib/mock-api"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { Activity, Target, TrendingUp, Percent, Loader2 } from "lucide-react"

export default function TechnicalExplanationPage() {
  const router = useRouter()

  const [data, setData] = useState<any>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isChatLoading, setIsChatLoading] = useState(false)

  useEffect(() => {
    // 1. Recuperamos los datos del formulario guardados en el navegador
    const savedInstanceString = localStorage.getItem("currentInstance");
    const instanceData = savedInstanceString ? JSON.parse(savedInstanceString) : {};

    // 2. Enviamos la data REAL a la IA
    generateExplanation("data-scientist", instanceData)
      .then((result) => {
        setData(result.technical)
        setIsLoading(false)
      })
      .catch((error) => {
        console.error("Error al generar la explicación:", error)
        setIsLoading(false)
        // Podrías añadir un toast de error aquí si lo deseas
      })
  }, [])

  const handleSendMessage = async (message: string) => {
    const userMessage: ChatMessage = {
      role: "user",
      content: message,
      timestamp: new Date(),
    }

    setMessages([...messages, userMessage])
    setIsChatLoading(true)

    const explanationText = data ? JSON.stringify(data) : ""
    const response = await chatRag(message, messages, explanationText, "data-scientist")

    const assistantMessage: ChatMessage = {
      role: "assistant",
      content: response,
      timestamp: new Date(),
    }

    setMessages([...messages, userMessage, assistantMessage])
    setIsChatLoading(false)
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex flex-col">
        <Navbar />
        <div className="flex-1 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-background flex flex-col">
        <Navbar />
        <div className="flex-1 flex items-center justify-center">
          <p className="text-destructive">Error al cargar los datos. Verifica la conexión con el servidor.</p>
        </div>
      </div>
    )
  }

  // Manejo defensivo: Si no hay SHAP, dibujamos un gráfico de LIME para que no se vea vacío
  const chartSourceData = data.shap && data.shap.length > 0 ? data.shap : (data.lime ?? []);
  const chartData = chartSourceData.map((item: any) => ({
    name: item.feature,
    // SHAP usa contribution, LIME usa weight
    value: Math.abs(item.contribution ?? item.weight ?? 0),
  }))

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container px-4 py-8">
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold mb-2">Respuesta adaptada para: Especialista en IA</h1>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Technical Content */}
            <div className="lg:col-span-2 space-y-6">
              {/* Metrics */}
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                 {/* Manejo seguro de índices: Si el backend solo manda 1 métrica, las demás muestran "N/A" */}
                <MetricCard title={data.metrics?.[0]?.name ?? "Exactitud"} value={data.metrics?.[0]?.value ?? "N/A"} icon={Target} />
                <MetricCard title={data.metrics?.[1]?.name ?? "Precisión"} value={data.metrics?.[1]?.value ?? "N/A"} icon={Activity} />
                <MetricCard title={data.metrics?.[2]?.name ?? "Recall"} value={data.metrics?.[2]?.value ?? "N/A"} icon={TrendingUp} />
                <MetricCard title={data.metrics?.[3]?.name ?? "Puntuación F1"} value={data.metrics?.[3]?.value ?? "N/A"} icon={Percent} />
                <MetricCard title={data.metrics?.[4]?.name ?? "AUC-ROC"} value={data.metrics?.[4]?.value ?? "N/A"} icon={Activity} />
              </div>

              {/* RAW Explanations */}
              <Card>
                <CardHeader>
                  <CardTitle>Explicación Directa (RAW)</CardTitle>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="lime"> {/* Cambiamos el default a lime porque es el que tenemos conectado */}
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="shap">SHAP</TabsTrigger>
                      <TabsTrigger value="lime">LIME</TabsTrigger>
                      <TabsTrigger value="anchors">Anchors</TabsTrigger>
                    </TabsList>

                    <TabsContent value="shap" className="space-y-4 mt-4">
                      <div className="space-y-2">
                        <h4 className="font-semibold text-sm">Contribuciones de las variables</h4>
                        <div className="space-y-1">
                          {(data.shap ?? []).map((item: any, idx: number) => (
                            <div key={idx} className="flex justify-between text-sm">
                              <span>{item.feature}</span>
                              <span className="font-mono">{item.contribution.toFixed(3)}</span>
                            </div>
                          ))}
                          {(!data.shap || data.shap.length === 0) && <p className="text-sm text-muted-foreground">Datos SHAP no disponibles para este modelo.</p>}
                        </div>
                      </div>
                    </TabsContent>

                    <TabsContent value="lime" className="space-y-4 mt-4">
                      <div className="space-y-2">
                        <h4 className="font-semibold text-sm">Pesos de las variables</h4>
                        <div className="space-y-1">
                          {(data.lime ?? []).map((item: any, idx: number) => (
                            <div key={idx} className="flex justify-between text-sm">
                              <span>{item.feature}</span>
                              <span className="font-mono">{item.weight.toFixed(3)}</span>
                            </div>
                          ))}
                           {(!data.lime || data.lime.length === 0) && <p className="text-sm text-muted-foreground">Datos LIME no disponibles para este modelo.</p>}
                        </div>
                      </div>
                    </TabsContent>

                    <TabsContent value="anchors" className="space-y-4 mt-4">
                      <div className="space-y-2">
                        <h4 className="font-semibold text-sm">Explicaciones basadas en reglas</h4>
                        <div className="space-y-2">
                          {(data.anchors ?? []).map((anchor: string, idx: number) => (
                            <div key={idx} className="p-3 bg-muted rounded-md text-sm font-mono">
                              {anchor}
                            </div>
                          ))}
                          {(!data.anchors || data.anchors.length === 0) && <p className="text-sm text-muted-foreground">Datos Anchors no disponibles para este modelo.</p>}
                        </div>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>

              {/* Charts */}
              <Card>
                <CardHeader>
                  {/* Actualizamos el título del gráfico si estamos mostrando LIME */}
                  <CardTitle>Importancia de las variables {(data.shap && data.shap.length > 0) ? '(SHAP)' : '(LIME)'}</CardTitle>
                </CardHeader>
                <CardContent>
                  {chartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="name" className="text-xs" />
                        <YAxis className="text-xs" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "0.5rem",
                          }}
                        />
                        <Bar dataKey="value" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-[300px] flex items-center justify-center border border-dashed rounded-md">
                        <p className="text-muted-foreground">No hay datos para graficar</p>
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
                    className="ml-auto"
                    onClick={() => {
                      // TODO: Backend: close job, clean session, optionally persist logs/telemetry
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
