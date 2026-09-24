import Link from "next/link"
import { ArrowRight, Table2, GitBranch, BarChart3, Sparkles } from "lucide-react"
import { Navbar } from "@/components/navbar"

export default function HomePage() {
  return (
    <div style={{ background: "#FAF8F5", minHeight: "100vh" }}>
      <Navbar />

      <main style={{ maxWidth: "860px", margin: "0 auto", padding: "0 24px" }}>

        {/* ── Chip ────────────────────────────────────────────────── */}
        <div style={{ paddingTop: "56px", paddingBottom: "32px" }}>
          <span style={{
            display: "inline-flex", alignItems: "center", gap: "8px",
            background: "#F0E8EC", color: "#8B1D3B",
            fontSize: "12px", fontWeight: 500, letterSpacing: "0.01em",
            padding: "5px 12px", borderRadius: "999px",
          }}>
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#8B1D3B", flexShrink: 0 }} />
            Herramienta · Datos Tabulares · Clasificación y Regresión
          </span>
        </div>

        {/* ── Hero ────────────────────────────────────────────────── */}
        <h1 style={{
          fontSize: "clamp(2.4rem, 5vw, 3.6rem)",
          fontWeight: 700,
          lineHeight: 1.1,
          color: "#141414",
          margin: 0,
          letterSpacing: "-0.02em",
          maxWidth: "680px",
        }}>
          Herramienta de explicabilidad{" "}
          <em style={{ fontStyle: "italic", color: "#8B1D3B" }}>adaptativa</em>.
        </h1>

        <p style={{
          marginTop: "20px",
          fontSize: "1.15rem",
          lineHeight: 1.65,
          color: "#4A4A4A",
          maxWidth: "560px",
        }}>
          Genera explicaciones interpretables para modelos que trabajan con
          <strong style={{ color: "#141414" }}> datos tabulares</strong> de
          <strong style={{ color: "#141414" }}> clasificación</strong> y
          <strong style={{ color: "#141414" }}> regresión</strong>, adaptadas al
          perfil de cada audiencia.
        </p>

        <div style={{ marginTop: "32px" }}>
          <Link href="/auth" style={{
            display: "inline-flex", alignItems: "center", gap: "8px",
            background: "#141414", color: "#fff",
            padding: "12px 24px", borderRadius: "8px",
            fontSize: "15px", fontWeight: 500,
            textDecoration: "none",
          }}>
            Iniciar análisis
            <ArrowRight size={16} />
          </Link>
        </div>

        {/* ── Scope tags ──────────────────────────────────────────── */}
        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginTop: "40px" }}>
          {[
            { icon: <Table2 size={14} />, label: "Datos tabulares" },
            { icon: <GitBranch size={14} />, label: "Modelos de clasificación" },
            { icon: <BarChart3 size={14} />, label: "Modelos de regresión" },
            { icon: <Sparkles size={14} />, label: "SHAP · LIME · Anchor" },
          ].map(({ icon, label }) => (
            <span key={label} style={{
              display: "inline-flex", alignItems: "center", gap: "6px",
              background: "#fff", border: "1px solid #E2DCD8",
              padding: "5px 12px", borderRadius: "6px",
              fontSize: "12px", color: "#5A5A5A", fontWeight: 500,
            }}>
              {icon} {label}
            </span>
          ))}
        </div>

        {/* ── Divider ─────────────────────────────────────────────── */}
        <div style={{ borderTop: "1px solid #E2DCD8", margin: "56px 0 48px" }} />

        {/* ── Info cards ──────────────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "16px" }}>
          {[
            {
              label: "¿Por qué usarla?",
              body: [
                "Los modelos de caja negra no explican sus decisiones por defecto.",
                "Genera explicaciones locales por instancia usando métodos agnósticos al modelo.",
                "Permite cumplir estándares de transparencia requeridos por organismos reguladores.",
              ],
            },
            {
              label: "¿En qué consiste?",
              body: [
                "Carga tu modelo ONNX, dataset y base de conocimiento.",
                "Selecciona una instancia y el perfil de la audiencia (técnico, experto o general).",
                "Obtén la explicación con SHAP, LIME y Anchor más una narrativa con RAG + Gemini.",
              ],
            },
            {
              label: "¿Qué obtienes?",
              body: [
                "Explicación en lenguaje natural adaptada al perfil seleccionado.",
                "Visualizaciones técnicas: valores SHAP, pesos LIME, reglas Anchor.",
                "Reporte PDF descargable con la explicación y el historial del chat.",
              ],
            },
          ].map(({ label, body }) => (
            <div key={label} style={{
              background: "#fff", border: "1px solid #E2DCD8",
              borderRadius: "10px", padding: "24px",
            }}>
              <p style={{
                fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em",
                color: "#8B1D3B", textTransform: "uppercase", margin: "0 0 14px",
              }}>
                {label}
              </p>
              <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: "8px" }}>
                {body.map((item) => (
                  <li key={item} style={{ fontSize: "14px", lineHeight: 1.55, color: "#3A3A3A" }}>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* ── Divider ─────────────────────────────────────────────── */}
        <div style={{ borderTop: "1px solid #E2DCD8", margin: "56px 0 48px" }} />

        {/* ── Steps ───────────────────────────────────────────────── */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0", alignItems: "flex-start" }}>
          {[
            { n: "01", title: "Configura el entorno", desc: "Sube modelo ONNX, dataset y base de conocimiento (opcional)." },
            { n: "02", title: "Define las etiquetas", desc: "Mapea las clases de salida del modelo a nombres comprensibles." },
            { n: "03", title: "Selecciona la instancia", desc: "Elige o genera aleatoriamente una fila del dataset para explicar." },
            { n: "04", title: "Obtén la explicación", desc: "Elige el perfil de audiencia y genera la explicación adaptativa." },
          ].map(({ n, title, desc }, i, arr) => (
            <div key={n} style={{ display: "flex", alignItems: "flex-start", flex: "1 1 180px", minWidth: "160px" }}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div style={{
                  width: "32px", height: "32px", borderRadius: "50%",
                  background: "#141414", color: "#fff",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: "12px", fontWeight: 700, flexShrink: 0,
                }}>
                  {n}
                </div>
                {i < arr.length - 1 && (
                  <div style={{ width: "1px", height: "100%", background: "#D8D2CE", marginTop: "4px", flex: 1, minHeight: "20px" }} />
                )}
              </div>
              <div style={{ marginLeft: "14px", paddingBottom: "32px" }}>
                <p style={{ margin: "0 0 4px", fontSize: "14px", fontWeight: 600, color: "#141414", lineHeight: 1.3 }}>{title}</p>
                <p style={{ margin: 0, fontSize: "13px", color: "#6A6A6A", lineHeight: 1.5 }}>{desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* ── Divider ─────────────────────────────────────────────── */}
        <div style={{ borderTop: "1px solid #E2DCD8", margin: "8px 0 48px" }} />

        {/* ── Perfiles ────────────────────────────────────────────── */}
        <p style={{
          fontSize: "10px", fontWeight: 700, letterSpacing: "0.1em",
          color: "#8B8080", textTransform: "uppercase", margin: "0 0 20px",
        }}>
          ¿Para quién es?
        </p>

        <h2 style={{ fontSize: "1.75rem", fontWeight: 700, color: "#141414", margin: "0 0 12px", letterSpacing: "-0.01em" }}>
          La explicación se adapta a tu perfil.
        </h2>
        <p style={{ fontSize: "1rem", lineHeight: 1.65, color: "#4A4A4A", maxWidth: "580px", margin: "0 0 32px" }}>
          No todas las personas que interactúan con un modelo de IA necesitan el mismo
          nivel de detalle. La Herramienta de Explicabilidad genera explicaciones distintas según quién
          las recibe, sin cambiar el modelo ni el análisis de fondo.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: "16px", marginBottom: "16px" }}>
          {[
            {
              tag: "Perfil técnico",
              tagColor: "#1D3A8B",
              tagBg: "#E8EDFB",
              title: "Especialista en IA",
              audience: "Data Scientists · Ingenieros de ML",
              desc: "Accede a los valores brutos de SHAP, pesos LIME y reglas Anchor. Visualizaciones técnicas, métricas de calidad y datos directos del explicador.",
              gets: ["Valores SHAP por feature", "Gráfico de barras LIME", "Reglas Anchor con precisión y cobertura", "Métricas del modelo"],
            },
            {
              tag: "Perfil experto",
              tagColor: "#1D6B3A",
              tagBg: "#E8F5EC",
              title: "Experto en el Dominio",
              audience: "Médicos · Funcionarios · Analistas",
              desc: "Narrativa en el lenguaje del dominio, sin jerga de ML. Usa la base de conocimiento (RAG) para contextualizar los factores relevantes del caso.",
              gets: ["Explicación en lenguaje del dominio", "Factores principales sin tecnicismos", "Contexto desde la base de conocimiento", "Chat con el asistente RAG"],
            },
            {
              tag: "Perfil general",
              tagColor: "#6B4A1D",
              tagBg: "#F5EEE8",
              title: "Usuario General",
              audience: "Ciudadanos · Usuarios finales · Directivos",
              desc: "Explicación simple, clara y sin tecnicismos. El sistema traduce la predicción del modelo a lenguaje cotidiano con ejemplos comprensibles.",
              gets: ["Resumen en lenguaje simple", "Razones concretas de la decisión", "Sin jerga técnica ni médica", "Chat de preguntas y respuestas"],
            },
          ].map(({ tag, tagColor, tagBg, title, audience, desc, gets }) => (
            <div key={title} style={{
              background: "#fff", border: "1px solid #E2DCD8",
              borderRadius: "10px", padding: "24px",
              display: "flex", flexDirection: "column", gap: "14px",
            }}>
              <div>
                <span style={{
                  display: "inline-block",
                  background: tagBg, color: tagColor,
                  fontSize: "10px", fontWeight: 700, letterSpacing: "0.07em",
                  textTransform: "uppercase", padding: "3px 9px", borderRadius: "4px",
                  marginBottom: "10px",
                }}>
                  {tag}
                </span>
                <p style={{ margin: "0 0 2px", fontSize: "15px", fontWeight: 700, color: "#141414" }}>{title}</p>
                <p style={{ margin: 0, fontSize: "11px", color: "#8A8080", fontWeight: 500 }}>{audience}</p>
              </div>

              <p style={{ margin: 0, fontSize: "13px", lineHeight: 1.6, color: "#4A4A4A" }}>{desc}</p>

              <div>
                <p style={{
                  margin: "0 0 8px", fontSize: "10px", fontWeight: 700,
                  letterSpacing: "0.07em", color: "#8B8080", textTransform: "uppercase",
                }}>
                  Qué obtiene
                </p>
                <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: "5px" }}>
                  {gets.map((item) => (
                    <li key={item} style={{ display: "flex", alignItems: "flex-start", gap: "8px", fontSize: "12.5px", color: "#3A3A3A" }}>
                      <span style={{ color: tagColor, fontWeight: 700, marginTop: "1px", flexShrink: 0 }}>→</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>

        <div style={{ marginBottom: "56px" }}>
          <Link href="/auth" style={{
            display: "inline-flex", alignItems: "center", gap: "8px",
            background: "#8B1D3B", color: "#fff",
            padding: "11px 22px", borderRadius: "8px",
            fontSize: "14px", fontWeight: 500,
            textDecoration: "none",
          }}>
            Comenzar — elige tu perfil al ingresar
            <ArrowRight size={15} />
          </Link>
        </div>

        {/* ── Divider ─────────────────────────────────────────────── */}
        <div style={{ borderTop: "1px solid #E2DCD8", margin: "0 0 48px" }} />

        {/* ── About section ───────────────────────────────────────── */}
        <p style={{
          fontSize: "10px", fontWeight: 700, letterSpacing: "0.1em",
          color: "#8B8080", textTransform: "uppercase", margin: "0 0 20px",
        }}>
          Sobre la herramienta
        </p>

        <h2 style={{ fontSize: "1.75rem", fontWeight: 700, color: "#141414", margin: "0 0 12px", letterSpacing: "-0.01em" }}>
          ¿Qué es la Herramienta de Explicabilidad?
        </h2>
        <p style={{ fontSize: "1rem", lineHeight: 1.65, color: "#4A4A4A", maxWidth: "600px", margin: "0 0 32px" }}>
          Un sistema de explicabilidad local que combina métodos agnósticos al modelo
          (SHAP, LIME, Anchor) con generación de lenguaje natural vía RAG + Gemini
          para producir narrativas adaptadas al perfil del receptor de la decisión.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px", marginBottom: "64px" }}>
          {[
            {
              label: "Tipo de modelo compatible",
              body: "Cualquier modelo exportado a formato ONNX que realice clasificación binaria, multiclase o regresión sobre datos tabulares.",
            },
            {
              label: "Cuándo usarla",
              body: "Con modelos ya entrenados que requieran transparencia ante usuarios finales, auditores o reguladores.",
            },
            {
              label: "En qué se basa",
              body: "En métodos de XAI agnósticos al modelo: SHAP, LIME y Anchor, combinados con LLM para generar narrativas adaptadas.",
            },
          ].map(({ label, body }) => (
            <div key={label} style={{
              background: "#fff", border: "1px solid #E2DCD8",
              borderRadius: "10px", padding: "20px",
            }}>
              <p style={{
                fontSize: "10px", fontWeight: 700, letterSpacing: "0.07em",
                color: "#8B8080", textTransform: "uppercase", margin: "0 0 10px",
              }}>
                {label}
              </p>
              <p style={{ margin: 0, fontSize: "13.5px", lineHeight: 1.6, color: "#3A3A3A" }}>{body}</p>
            </div>
          ))}
        </div>

        {/* ── Agradecimientos ─────────────────────────────────────── */}
        <div style={{ borderTop: "1px solid #E2DCD8", margin: "0 0 40px" }} />

        <div style={{
          display: "flex", alignItems: "flex-start", gap: "28px",
          flexWrap: "wrap", paddingBottom: "64px",
        }}>
          {/* ANID logo recreado */}
          <div style={{ flexShrink: 0 }}>
            <div style={{ width: "120px" }}>
              {/* Franjas de color */}
              <div style={{ display: "flex", height: "10px", borderRadius: "2px 2px 0 0", overflow: "hidden" }}>
                <div style={{ flex: "0 0 62%", background: "#005DAA" }} />
                <div style={{ flex: "0 0 38%", background: "#E5002B" }} />
              </div>
              {/* Texto del logo */}
              <div style={{
                background: "#fff", border: "1px solid #E2DCD8", borderTop: "none",
                borderRadius: "0 0 2px 2px", padding: "8px 8px 6px",
              }}>
                <p style={{
                  margin: 0, fontSize: "10px", fontWeight: 600,
                  color: "#6B6B6B", lineHeight: 1.35,
                  fontFamily: "system-ui, sans-serif",
                }}>
                  Agencia Nacional<br />de Investigación<br />y Desarrollo
                </p>
              </div>
            </div>
          </div>

          {/* Texto de agradecimiento */}
          <div style={{ flex: 1, minWidth: "200px" }}>
            <p style={{
              fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em",
              color: "#8B8080", textTransform: "uppercase", margin: "0 0 8px",
            }}>
              Agradecimientos
            </p>
            <p style={{ margin: 0, fontSize: "13px", lineHeight: 1.65, color: "#5A5A5A" }}>
              Esta herramienta es desarrollada por <strong style={{ color: "#141414" }}>GobLab UAI</strong> con
              el apoyo de la <strong style={{ color: "#141414" }}>Agencia Nacional de Investigación y
              Desarrollo (ANID)</strong> — Subdirección de Investigación Aplicada / Concurso IDeA I+D 2023,
              proyecto <strong style={{ color: "#141414" }}>ID23I10357</strong>.
            </p>
          </div>
        </div>

      </main>
    </div>
  )
}
