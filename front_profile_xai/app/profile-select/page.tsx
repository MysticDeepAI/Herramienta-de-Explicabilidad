"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, Microscope, Briefcase, User, CheckCircle2 } from "lucide-react"
import { Navbar } from "@/components/navbar"
import { cn } from "@/lib/utils"
import type { UserProfile } from "@/lib/mock-api"

const PROFILES: { id: UserProfile; icon: React.ElementType; title: string; description: string }[] = [
  {
    id: "data-scientist",
    icon: Microscope,
    title: "Especialista en IA",
    description: "Detalles técnicos, métricas y salidas directas de los explicadores (SHAP, LIME, Anchor).",
  },
  {
    id: "domain-expert",
    icon: Briefcase,
    title: "Experto en el Dominio",
    description: "Lenguaje natural con contexto especializado desde la base de conocimientos (RAG).",
  },
  {
    id: "non-expert",
    icon: User,
    title: "Usuario General",
    description: "Lenguaje accesible, ejemplos claros y conclusiones simplificadas.",
  },
]

export default function ProfileSelectPage() {
  const router = useRouter()
  const [selected, setSelected] = useState<Set<UserProfile>>(new Set(["data-scientist", "domain-expert", "non-expert"]))

  const toggle = (id: UserProfile) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        if (next.size === 1) return prev // mínimo 1
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const handleContinue = () => {
    localStorage.setItem("enabledProfiles", JSON.stringify([...selected]))
    router.push("/setup")
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold">Perfiles de explicación</h1>
            <p className="text-muted-foreground">
              Selecciona los perfiles que estarán disponibles durante esta sesión. Puedes elegir uno o varios.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {PROFILES.map((p) => {
              const Icon = p.icon
              const active = selected.has(p.id)
              return (
                <Card
                  key={p.id}
                  onClick={() => toggle(p.id)}
                  className={cn(
                    "cursor-pointer transition-all hover:shadow-lg relative",
                    active ? "ring-2 ring-primary shadow-lg" : "opacity-60",
                  )}
                >
                  {active && (
                    <CheckCircle2 className="absolute top-3 right-3 h-5 w-5 text-primary" />
                  )}
                  <CardHeader className="text-center space-y-4">
                    <div
                      className={cn(
                        "mx-auto h-16 w-16 rounded-full flex items-center justify-center",
                        active ? "bg-primary/10" : "bg-muted",
                      )}
                    >
                      <Icon className={cn("h-8 w-8", active ? "text-primary" : "text-muted-foreground")} />
                    </div>
                    <div className="space-y-2">
                      <CardTitle>{p.title}</CardTitle>
                      <CardDescription className="leading-relaxed">{p.description}</CardDescription>
                    </div>
                  </CardHeader>
                </Card>
              )
            })}
          </div>

          <p className="text-xs text-muted-foreground text-center">
            {selected.size === 3
              ? "Todos los perfiles habilitados"
              : `${selected.size} perfil${selected.size > 1 ? "es" : ""} seleccionado${selected.size > 1 ? "s" : ""}`}
            {" · "}Se puede ajustar más adelante cambiando el perfil desde la vista de explicación.
          </p>

          <div className="flex gap-4">
            <Button variant="ghost" asChild>
              <Link href="/auth">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Atrás
              </Link>
            </Button>
            <Button className="flex-1" onClick={handleContinue}>
              Continuar a configuración
            </Button>
          </div>
        </div>
      </main>
    </div>
  )
}
