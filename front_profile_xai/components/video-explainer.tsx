"use client"

import { useEffect, useRef, useState } from "react"

export function VideoExplainer() {
  const frameRef = useRef<HTMLIFrameElement>(null)
  const [height, setHeight] = useState(640)

  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      if (e.source !== frameRef.current?.contentWindow) return
      if (e.data?.type === "xai-video-height" && typeof e.data.h === "number") {
        setHeight(e.data.h)
      }
    }
    window.addEventListener("message", onMessage)
    return () => window.removeEventListener("message", onMessage)
  }, [])

  return (
    <iframe
      ref={frameRef}
      src="/video/index.html?embed=1"
      title="Video: cómo funciona la herramienta de explicabilidad"
      allow="autoplay; fullscreen"
      allowFullScreen
      loading="lazy"
      style={{ width: "100%", height, border: 0, display: "block" }}
    />
  )
}
