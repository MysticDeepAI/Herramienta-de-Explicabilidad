"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, Microscope, Briefcase, User, Info, Layers } from "lucide-react"
import { Navbar } from "@/components/navbar"
import { cn } from "@/lib/utils"
import type { UserProfile } from "@/lib/mock-api"

const ALL_PROFILES = [
  {
    id: "data-scientist" as UserProfile,
    icon: Microscope,
    title: "Especialista en IA",
    description: "Detalles técnicos, métricas y salidas directas de los explicadores (SHAP, LIME, Anchor)",
    example: "«La variable edad_paciente tiene un valor SHAP de +0.42, lo que indica que fue el factor de mayor peso positivo en la predicción. El modelo Anchor establece que si edad > 55 AND glucosa > 140, la probabilidad de clase 1 supera el 87%.»",
    audience: "ML Engineers, Data Scientists, investigadores",
  },
  {
    id: "domain-expert" as UserProfile,
    icon: Briefcase,
    title: "Experto en el Dominio",
    description: "Lenguaje natural con contexto especializado desde la base de conocimientos",
    example: "«El factor más determinante fue el nivel de glucosa en ayunas, que se encontraba por encima del umbral clínico establecido en la literatura para diagnóstico de riesgo. Adicionalmente, la combinación de IMC elevado y presión arterial alta refuerza esta predicción, según los protocolos de la ADA 2023.»",
    audience: "Médicos, abogados, ingenieros, especialistas del área",
  },
  {
    id: "non-expert" as UserProfile,
    icon: User,
    title: "Usuario General",
    description: "Lenguaje accesible, ejemplos claros y conclusiones simplificadas",
    example: "«El modelo indica que este caso tiene un alto riesgo de diagnóstico positivo, principalmente porque el nivel de azúcar en sangre es alto y el peso corporal está por encima del rango saludable. Estos dos factores juntos son los que más influyeron en la predicción.»",
    audience: "Pacientes, ciudadanos, público general sin formación técnica",
  },
]

export default function ProfilePage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const isBatch = searchParams.get("batch") === "true"
  const [selectedProfile, setSelectedProfile] = useState<UserProfile | null>(null)
  const [batchCount, setBatchCount] = useState(0)

  useEffect(() => {
    if (isBatch) {
      try {
        const raw = localStorage.getItem("batchInstances")
        if (raw) setBatchCount(JSON.parse(raw).length)
      } catch {}
    }
  }, [isBatch])

  const enabled: UserProfile[] = (() => {
    try {
      const stored = localStorage.getItem("enabledProfiles")
      if (stored) return JSON.parse(stored) as UserProfile[]
    } catch {}
    return ["data-scientist", "domain-expert", "non-expert"]
  })()

  const profiles = ALL_PROFILES.filter((p) => enabled.includes(p.id))

  const handleGenerate = () => {
    if (isBatch) {
      router.push(`/explain/batch?profile=${selectedProfile}`)
      return
    }
    if (selectedProfile === "data-scientist") {
      router.push(`/explain/technical?profile=${selectedProfile}`)
    } else {
      router.push(`/explain/natural?profile=${selectedProfile}`)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-6">

          <div className="space-y-2">
            <h1 className="text-3xl font-bold">Adaptar la explicación</h1>
            <p className="text-muted-foreground">
              Elige el perfil de la persona que recibirá la explicación
            </p>
          </div>

          {/* Banner de modo lote */}
          {isBatch && batchCount > 0 && (
            <div style={{
              background: "#F0FDF4", border: "1px solid #BBF7D0",
              borderRadius: "8px", padding: "14px 18px",
              display: "flex", gap: "10px", alignItems: "center",
            }}>
              <Layers style={{ color: "#16A34A", width: "16px", height: "16px", flexShrink: 0 }} />
              <p style={{ fontSize: "13px", color: "#166534", lineHeight: 1.6 }}>
                Modo lote activo — se generarán explicaciones para{" "}
                <strong>{batchCount} instancia{batchCount !== 1 ? "s" : ""}</strong>.
                El perfil seleccionado se aplicará a todas ellas.
              </p>
            </div>
          )}

          {/* Nota sobre la fuente de la explicación */}
          <div style={{
            background: "#EFF6FF", border: "1px solid #BFDBFE",
            borderRadius: "8px", padding: "14px 18px",
            display: "flex", gap: "10px", alignItems: "flex-start",
          }}>
            <Info style={{ color: "#2563EB", width: "16px", height: "16px", flexShrink: 0, marginTop: "2px" }} />
            <p style={{ fontSize: "13px", color: "#1E40AF", lineHeight: 1.6 }}>
              La explicación se genera a partir del <strong>modelo entrenado y el dataset de referencia</strong> que cargaste.
              El perfil seleccionado determina el lenguaje, nivel de detalle y contexto con que se presentará la predicción para el caso ingresado.
            </p>
          </div>

          {/* Tarjetas de perfil */}
          <div className="grid md:grid-cols-3 gap-5">
            {profiles.map((profile) => {
              const Icon = profile.icon
              const isSelected = selectedProfile === profile.id
              return (
                <Card
                  key={profile.id}
                  className={cn(
                    "cursor-pointer transition-all hover:shadow-lg flex flex-col",
                    isSelected && "ring-2 ring-primary shadow-lg",
                  )}
                  onClick={() => setSelectedProfile(profile.id)}
                >
                  <CardHeader className="text-center space-y-3 pb-3">
                    <div className="mx-auto h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center">
                      <Icon className="h-7 w-7 text-primary" />
                    </div>
                    <div className="space-y-1">
                      <CardTitle className="text-base">{profile.title}</CardTitle>
                      <CardDescription className="text-xs leading-relaxed">{profile.description}</CardDescription>
                    </div>
                  </CardHeader>

                  <div style={{ padding: "0 20px 20px", flex: 1, display: "flex", flexDirection: "column", gap: "10px" }}>
                    {/* Audiencia */}
                    <div style={{
                      background: "#F5F5F5", borderRadius: "6px",
                      padding: "8px 10px", fontSize: "11px", color: "#555",
                    }}>
                      <span style={{ fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", fontSize: "10px" }}>
                        Audiencia:
                      </span>{" "}
                      {profile.audience}
                    </div>

                    {/* Ejemplo de explicación */}
                    <div style={{
                      background: isSelected ? "#FDF2F4" : "#FAFAFA",
                      border: `1px solid ${isSelected ? "#F9A8B8" : "#E5E5E5"}`,
                      borderRadius: "6px", padding: "10px 12px",
                      flex: 1,
                    }}>
                      <p style={{
                        fontSize: "10px", fontWeight: 700, textTransform: "uppercase",
                        letterSpacing: "0.06em", color: "#888", marginBottom: "6px",
                      }}>
                        Ejemplo de salida
                      </p>
                      <p style={{ fontSize: "12px", color: "#444", lineHeight: 1.6, fontStyle: "italic" }}>
                        {profile.example}
                      </p>
                    </div>
                  </div>
                </Card>
              )
            })}
          </div>

          <div className="flex gap-4">
            <Button variant="ghost" asChild>
              <Link href="/instance">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Atrás
              </Link>
            </Button>
            <Button className="flex-1" disabled={!selectedProfile} onClick={handleGenerate}>
              {isBatch && batchCount > 0
                ? `Generar explicaciones para ${batchCount} instancia${batchCount !== 1 ? "s" : ""}`
                : "Generar explicación"
              }
            </Button>
          </div>

        </div>
      </main>
    </div>
  )
}
