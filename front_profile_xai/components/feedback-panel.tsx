"use client"

import { useState } from "react"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import { submitFeedback, type FeedbackPayload } from "@/lib/mock-api"
import {
  MessageSquarePlus,
  ChevronRight,
  ChevronLeft,
  CheckCircle2,
  Loader2,
} from "lucide-react"

// ── Types ─────────────────────────────────────────────────────────────────

interface FormState {
  // Paso 1
  perfil: string
  q1_1: number; q1_2: number; q1_3: number
  q2_1: number; q2_2: number; q2_3: number
  q3_1: number; q3_2: number
  // Paso 2
  q_confuso: string
  q_info_adicional: string
  q_mejoras: string
  // Paso 3
  organizacion: string
  categoria: string
  comentarios: string
}

const INITIAL_FORM: FormState = {
  perfil: "",
  q1_1: 0, q1_2: 0, q1_3: 0,
  q2_1: 0, q2_2: 0, q2_3: 0,
  q3_1: 0, q3_2: 0,
  q_confuso: "",
  q_info_adicional: "",
  q_mejoras: "",
  organizacion: "",
  categoria: "Comentario general",
  comentarios: "",
}

const TOTAL_STEPS = 3

// ── Sub-components ────────────────────────────────────────────────────────

function LikertRow({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (v: number) => void
}) {
  return (
    <div className="space-y-2 py-2 border-b border-dashed border-muted last:border-0">
      <p className="text-sm text-foreground leading-snug">{label}</p>
      <div className="flex items-center gap-2">
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            type="button"
            onClick={() => onChange(n)}
            className={cn(
              "h-9 w-9 rounded-full border-2 text-sm font-semibold transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
              value === n
                ? "border-primary bg-primary text-primary-foreground shadow-sm scale-110"
                : "border-border hover:border-primary/60 text-muted-foreground hover:text-foreground"
            )}
          >
            {n}
          </button>
        ))}
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground px-1">
        <span>Muy en desacuerdo</span>
        <span>Muy de acuerdo</span>
      </div>
    </div>
  )
}

function StepIndicator({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-1.5">
        {Array.from({ length: total }).map((_, i) => (
          <div
            key={i}
            className={cn(
              "h-2 rounded-full transition-all duration-300",
              i + 1 === current
                ? "w-6 bg-primary"
                : i + 1 < current
                  ? "w-2 bg-primary/60"
                  : "w-2 bg-muted"
            )}
          />
        ))}
      </div>
      <span className="text-xs text-muted-foreground whitespace-nowrap">
        Paso {current} de {total}
      </span>
    </div>
  )
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-wider text-primary/80 mt-4 mb-1">
      {children}
    </p>
  )
}

// ── Main component ────────────────────────────────────────────────────────

export function FeedbackPanel() {
  const [open, setOpen] = useState(false)
  const [step, setStep] = useState(1)
  const [form, setForm] = useState<FormState>(INITIAL_FORM)
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle")
  const [errorMsg, setErrorMsg] = useState("")

  const set = <K extends keyof FormState>(key: K, val: FormState[K]) =>
    setForm((prev) => ({ ...prev, [key]: val }))

  // ── Validation per step ──────────────────────────────────────────────
  const step1Valid =
    form.perfil !== "" &&
    [form.q1_1, form.q1_2, form.q1_3, form.q2_1, form.q2_2, form.q2_3, form.q3_1, form.q3_2].every(
      (v) => v > 0
    )

  const canNext = step === 1 ? step1Valid : true

  // ── Handlers ──────────────────────────────────────────────────────────
  const handleNext = () => setStep((s) => Math.min(s + 1, TOTAL_STEPS))
  const handleBack = () => setStep((s) => Math.max(s - 1, 1))

  const handleOpen = (v: boolean) => {
    setOpen(v)
    if (!v) {
      // Reset after close animation
      setTimeout(() => {
        setStep(1)
        setForm(INITIAL_FORM)
        setStatus("idle")
        setErrorMsg("")
      }, 400)
    }
  }

  const handleSubmit = async () => {
    setStatus("loading")
    setErrorMsg("")
    try {
      const now = new Date()
      const payload: FeedbackPayload = {
        timestamp: now.toLocaleString("es-CL", {
          year: "numeric", month: "2-digit", day: "2-digit",
          hour: "2-digit", minute: "2-digit", second: "2-digit",
        }),
        usuario: "Anónimo",
        ...form,
      }
      await submitFeedback(payload)
      setStatus("success")
    } catch (e: unknown) {
      setStatus("error")
      setErrorMsg(e instanceof Error ? e.message : "Error desconocido")
    }
  }

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <>
      {/* Floating tab */}
      <button
        onClick={() => handleOpen(true)}
        aria-label="Abrir formulario de feedback"
        className={cn(
          "fixed right-0 top-1/2 z-40 flex items-center gap-1.5 -translate-y-1/2",
          "bg-primary text-primary-foreground",
          "py-3 px-2 rounded-l-lg shadow-lg",
          "transition-transform duration-200 hover:-translate-x-0.5 hover:scale-105 hover:shadow-xl",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2"
        )}
        style={{ writingMode: "vertical-lr", textOrientation: "mixed" }}
      >
        <MessageSquarePlus className="h-4 w-4 rotate-180" />
        <span className="text-xs font-semibold tracking-wide rotate-180">Feedback</span>
      </button>

      {/* Slide-out panel */}
      <Sheet open={open} onOpenChange={handleOpen}>
        <SheetContent
          side="right"
          className="w-full sm:max-w-[520px] flex flex-col p-0 gap-0 overflow-hidden"
        >
          {/* Static header */}
          <SheetHeader className="px-6 pt-6 pb-4 border-b bg-muted/30 shrink-0">
            <SheetTitle className="text-base leading-tight">
              Evaluación de Usuario — Herramienta de Explicabilidad
              <span className="block text-xs font-normal text-muted-foreground mt-0.5">
                Piloto SUSESO
              </span>
            </SheetTitle>
            <SheetDescription className="text-xs leading-relaxed mt-1">
              Este formulario busca realizar pruebas de validación para la herramienta de
              Explicabilidad. Califique las afirmaciones en una escala del{" "}
              <strong>1 al 5</strong>{" "}
              (1&nbsp;=&nbsp;Muy en desacuerdo, 3&nbsp;=&nbsp;Neutral, 5&nbsp;=&nbsp;Muy
              de acuerdo).
            </SheetDescription>
          </SheetHeader>

          {/* Progress bar */}
          <div className="px-6 pt-3 pb-0 shrink-0">
            <StepIndicator current={step} total={TOTAL_STEPS} />
          </div>

          {/* Scrollable form body — min-h-0 is required so flex-1 bounds height */}
          <ScrollArea className="flex-1 min-h-0 px-6 py-3">
            {/* ── STEP 1 ── */}
            {step === 1 && (
              <div className="space-y-4 pb-4">
                {/* Perfil */}
                <div className="space-y-1.5">
                  <Label htmlFor="perfil" className="font-semibold">
                    Seleccione su perfil <span className="text-destructive">*</span>
                  </Label>
                  <Select
                    value={form.perfil}
                    onValueChange={(v) => set("perfil", v)}
                  >
                    <SelectTrigger id="perfil" className="w-full">
                      <SelectValue placeholder="Elige tu perfil…" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="data-scientist">
                        Especialista en IA / Data Scientist
                      </SelectItem>
                      <SelectItem value="domain-expert">
                        Experto en el Dominio (médico / funcionario)
                      </SelectItem>
                      <SelectItem value="non-expert">Usuario General</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Dimensión 1 */}
                <SectionHeading>Dimensión 1 — Calidad de la Explicación</SectionHeading>
                <LikertRow
                  label="La explicación me ayudó a entender cómo funciona el modelo de licencias médicas."
                  value={form.q1_1}
                  onChange={(v) => set("q1_1", v)}
                />
                <LikertRow
                  label="La explicación contiene suficiente detalle para realizar mi trabajo."
                  value={form.q1_2}
                  onChange={(v) => set("q1_2", v)}
                />
                <LikertRow
                  label="La explicación fue precisa y coherente con la normativa."
                  value={form.q1_3}
                  onChange={(v) => set("q1_3", v)}
                />

                {/* Dimensión 2 */}
                <SectionHeading>Dimensión 2 — Satisfacción y Confianza</SectionHeading>
                <LikertRow
                  label="Me siento satisfecho con la Herramienta de Explicabilidad."
                  value={form.q2_1}
                  onChange={(v) => set("q2_1", v)}
                />
                <LikertRow
                  label="Confío en la evaluación que hizo el sistema sobre la licencia."
                  value={form.q2_2}
                  onChange={(v) => set("q2_2", v)}
                />
                <LikertRow
                  label="Usaría esta herramienta para apoyar mis decisiones futuras."
                  value={form.q2_3}
                  onChange={(v) => set("q2_3", v)}
                />

                {/* Dimensión 3 */}
                <SectionHeading>Dimensión 3 — Adaptabilidad</SectionHeading>
                <LikertRow
                  label="El lenguaje utilizado fue apropiado para mi nivel de conocimiento técnico."
                  value={form.q3_1}
                  onChange={(v) => set("q3_1", v)}
                />
                <LikertRow
                  label="La cantidad de información mostrada fue adecuada."
                  value={form.q3_2}
                  onChange={(v) => set("q3_2", v)}
                />

                {/* Progreso visible para saber cuánto falta */}
                {(() => {
                  const answered = [
                    form.q1_1, form.q1_2, form.q1_3,
                    form.q2_1, form.q2_2, form.q2_3,
                    form.q3_1, form.q3_2,
                  ].filter((v) => v > 0).length
                  const missingProfile = form.perfil === ""
                  if (step1Valid) return null
                  return (
                    <div className="rounded-md bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 px-3 py-2 space-y-1">
                      {missingProfile && (
                        <p className="text-xs text-amber-700 dark:text-amber-400">
                          • Selecciona tu perfil.
                        </p>
                      )}
                      {answered < 8 && (
                        <p className="text-xs text-amber-700 dark:text-amber-400">
                          • Responde todas las preguntas ({answered}/8 respondidas). Desplázate hacia abajo para ver las que faltan.
                        </p>
                      )}
                    </div>
                  )
                })()}
              </div>
            )}

            {/* ── STEP 2 ── */}
            {step === 2 && (
              <div className="space-y-5 pb-4">
                <p className="text-sm text-muted-foreground">
                  Responde con tus propias palabras. Todos los campos son opcionales.
                </p>

                <div className="space-y-1.5">
                  <Label htmlFor="q_confuso" className="text-sm font-medium">
                    ¿Hubo algún término técnico o gráfico que le resultara confuso?
                  </Label>
                  <Textarea
                    id="q_confuso"
                    rows={3}
                    placeholder="Describe aquí…"
                    value={form.q_confuso}
                    onChange={(e) => set("q_confuso", e.target.value)}
                  />
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="q_info_adicional" className="text-sm font-medium">
                    ¿Qué información adicional le hubiera gustado ver para tomar una
                    decisión más segura?
                  </Label>
                  <Textarea
                    id="q_info_adicional"
                    rows={3}
                    placeholder="Describe aquí…"
                    value={form.q_info_adicional}
                    onChange={(e) => set("q_info_adicional", e.target.value)}
                  />
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="q_mejoras" className="text-sm font-medium">
                    ¿Hubo alguna parte de la herramienta que no le resultara intuitiva?
                    ¿Cómo cree que podría mejorar?
                  </Label>
                  <Textarea
                    id="q_mejoras"
                    rows={3}
                    placeholder="Describe aquí…"
                    value={form.q_mejoras}
                    onChange={(e) => set("q_mejoras", e.target.value)}
                  />
                </div>
              </div>
            )}

            {/* ── STEP 3 ── */}
            {step === 3 && (
              <div className="space-y-5 pb-4">
                {status === "success" ? (
                  <div className="flex flex-col items-center justify-center gap-4 py-12 text-center">
                    <CheckCircle2 className="h-14 w-14 text-green-500" />
                    <p className="text-lg font-semibold text-foreground">
                      ¡Gracias por tu evaluación!
                    </p>
                    <p className="text-sm text-muted-foreground max-w-xs">
                      Tu feedback ha sido registrado correctamente y ayudará a mejorar
                      la Herramienta de Explicabilidad.
                    </p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-2"
                      onClick={() => handleOpen(false)}
                    >
                      Cerrar
                    </Button>
                  </div>
                ) : (
                  <>
                    <div className="space-y-1.5">
                      <Label htmlFor="organizacion" className="text-sm font-medium">
                        Organización{" "}
                        <span className="text-muted-foreground font-normal">(opcional)</span>
                      </Label>
                      <Input
                        id="organizacion"
                        placeholder="Ej. SUSESO, Hospital San José…"
                        value={form.organizacion}
                        onChange={(e) => set("organizacion", e.target.value)}
                      />
                    </div>

                    <div className="space-y-1.5">
                      <Label htmlFor="categoria" className="text-sm font-medium">
                        Categoría
                      </Label>
                      <Select
                        value={form.categoria}
                        onValueChange={(v) => set("categoria", v)}
                      >
                        <SelectTrigger id="categoria" className="w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="Comentario general">
                            Comentario general
                          </SelectItem>
                          <SelectItem value="Sugerencia de mejora">
                            Sugerencia de mejora
                          </SelectItem>
                          <SelectItem value="Reporte de problema">
                            Reporte de problema
                          </SelectItem>
                          <SelectItem value="Consulta técnica">
                            Consulta técnica
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-1.5">
                      <Label htmlFor="comentarios" className="text-sm font-medium">
                        Comparte tus comentarios aquí
                      </Label>
                      <Textarea
                        id="comentarios"
                        rows={4}
                        placeholder="Cualquier observación adicional…"
                        value={form.comentarios}
                        onChange={(e) => set("comentarios", e.target.value)}
                      />
                    </div>

                    {status === "error" && (
                      <p className="text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">
                        {errorMsg || "Ocurrió un error al enviar. Intenta de nuevo."}
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
          </ScrollArea>

          {/* Navigation footer — oculto en pantalla de éxito */}
          {status !== "success" && (
            <div className="px-6 py-4 border-t bg-background shrink-0 flex items-center justify-between gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleBack}
                disabled={step === 1}
                className="gap-1"
              >
                <ChevronLeft className="h-4 w-4" />
                Atrás
              </Button>

              {step < TOTAL_STEPS ? (
                <Button
                  size="sm"
                  onClick={handleNext}
                  disabled={!canNext}
                  className="gap-1"
                >
                  Siguiente
                  <ChevronRight className="h-4 w-4" />
                </Button>
              ) : (
                <Button
                  size="sm"
                  onClick={handleSubmit}
                  disabled={status === "loading"}
                  className="gap-2 min-w-[140px]"
                >
                  {status === "loading" ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Enviando…
                    </>
                  ) : (
                    "Enviar Feedback"
                  )}
                </Button>
              )}
            </div>
          )}
        </SheetContent>
      </Sheet>
    </>
  )
}
