import Link from "next/link"
import Image from "next/image"

export function Navbar() {
  return (
    <nav style={{ background: "#141414", borderBottom: "none" }}>
      <div style={{
        maxWidth: "1200px", margin: "0 auto",
        padding: "0 24px", height: "60px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>

        <Link href="/" style={{ display: "flex", alignItems: "center", textDecoration: "none", height: "44px" }}>
          <Image
            src="/goblab-logo.webp"
            alt="GobLab UAI"
            height={44}
            width={220}
            style={{ height: "44px", width: "auto", objectFit: "contain" }}
            priority
          />
        </Link>

        <span style={{
          color: "#777", fontSize: "11px", fontWeight: 500,
          letterSpacing: "0.08em", textTransform: "uppercase",
        }}>
          Herramienta de Explicabilidad
        </span>
      </div>
    </nav>
  )
}
