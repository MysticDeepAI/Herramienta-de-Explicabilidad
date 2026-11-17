"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tags, ArrowLeft, Info, BookOpen } from "lucide-react"
import { Navbar } from "@/components/navbar"
import { setLabels } from "@/lib/mock-api"
import { useToast } from "@/hooks/use-toast"

export default function LabelsPage() {
  const router = useRouter()
  const { toast } = useToast()

  const [classes, setClasses] = useState<string[]>([])
  const [labelMap, setLabelMap] = useState<Record<string, string>>({})
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    const raw = localStorage.getItem("classes_detected")
    if (!raw) { router.push("/setup"); return }
    try {
      const detected: string[] = JSON.parse(raw)
      setClasses(detected)
      setLabelMap(Object.fromEntries(detected.map(c => [c, ""])))
    } catch {
      router.push("/setup")
    }
  }, [])

  const allFilled = classes.length > 0 && classes.every(c => labelMap[c]?.trim())

  const handleContinue = async () => {
    setIsSaving(true)
    try {
      await setLabels(labelMap)
      router.push("/instance")
    } catch {
      toast({ title: "Error", description: "No se pudo guardar las etiquetas.", variant: "destructive" })
    } finally {
      setIsSaving(false)
    }
  }

  if (classes.length === 0) return null

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container px-4 py-8">
        <div className="max-w-2xl mx-auto space-y-6">

          <div className="space-y-2">
            <h1 className="text-3xl font-bold">Nombra las clases del modelo</h1>
            <p className="text-muted-foreground">
              Se han identificado <strong>{classes.length} clase{classes.length !== 1 ? "s" : ""}</strong> en el modelo. Asigna un nombre significativo a cada una.
            </p>
          </div>

          {/* Qué es una clase */}
          <div style={{
            background: "#EFF6FF", border: "1px solid #BFDBFE",
            borderRadius: "8px", padding: "16px 20px",
            display: "flex", gap: "12px",
          }}>
            <Info style={{ color: "#2563EB", width: "18px", height: "18px", flexShrink: 0, marginTop: "1px" }} />
            <div style={{ fontSize: "13px", color: "#1E40AF", lineHeight: 1.6 }}>
              <p style={{ fontWeight: 600, marginBottom: "4px" }}>¿Qué es una clase?</p>
              <p>
                En un modelo de clasificación, cada <strong>clase</strong> representa un posible resultado o categoría que el modelo puede predecir.
                Asignarle un nombre comprensible permite que la explicación sea más clara para cualquier audiencia.
              </p>
              <p style={{ marginTop: "6px" }}>
                Este paso es <strong>obligatorio</strong>: los nombres que ingreses aparecerán en las explicaciones generadas.
              </p>
            </div>
          </div>

          {/* Ejemplo */}
          <div style={{
            background: "#F0FDF4", border: "1px solid #BBF7D0",
            borderRadius: "8px", padding: "16px 20px",
            display: "flex", gap: "12px",
          }}>
            <BookOpen style={{ color: "#16A34A", width: "18px", height: "18px", flexShrink: 0, marginTop: "1px" }} />
            <div style={{ fontSize: "13px", color: "#166534", lineHeight: 1.6 }}>
              <p style={{ fontWeight: 600, marginBottom: "8px" }}>Ejemplo — modelo de diagnóstico oncológico</p>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <code style={{ background: "#DCFCE7", padding: "2px 8px", borderRadius: "4px", fontWeight: 700, fontSize: "12px" }}>0</code>
                  <span>→</span>
                  <span><strong>Tumor Benigno</strong> (el modelo predice que no hay malignidad)</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <code style={{ background: "#DCFCE7", padding: "2px 8px", borderRadius: "4px", fontWeight: 700, fontSize: "12px" }}>1</code>
                  <span>→</span>
                  <span><strong>Tumor Maligno</strong> (el modelo predice presencia de malignidad)</span>
                </div>
              </div>
              <p style={{ marginTop: "8px", fontStyle: "italic" }}>
                Usa términos del dominio para que la explicación sea comprensible por el público objetivo.
              </p>
            </div>
          </div>

          {/* Formulario de clases */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Tags className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">Clases detectadas</CardTitle>
              </div>
              <CardDescription>
                Escribe un nombre descriptivo para cada clase identificada en el modelo.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid sm:grid-cols-2 gap-4">
                {classes.map((cls) => (
                  <div key={cls} className="space-y-1.5">
                    <Label htmlFor={`class-${cls}`} className="text-sm font-medium">
                      Clase <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono">{cls}</code>
                    </Label>
                    <Input
                      id={`class-${cls}`}
                      placeholder={`Ej: "Resultado Negativo" o "Resultado Positivo"`}
                      value={labelMap[cls] ?? ""}
                      onChange={(e) =>
                        setLabelMap(prev => ({ ...prev, [cls]: e.target.value }))
                      }
                    />
                  </div>
                ))}
              </div>
              {!allFilled && (
                <p className="text-xs text-muted-foreground">
                  * Todos los campos son obligatorios para continuar.
                </p>
              )}
            </CardContent>
          </Card>

          <div className="flex gap-4">
            <Button variant="ghost" asChild>
              <Link href="/setup">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Atrás
              </Link>
            </Button>
            <Button className="flex-1" disabled={!allFilled || isSaving} onClick={handleContinue}>
              {isSaving ? "Guardando..." : "Continuar"}
            </Button>
          </div>

        </div>
      </main>
    </div>
  )
}
