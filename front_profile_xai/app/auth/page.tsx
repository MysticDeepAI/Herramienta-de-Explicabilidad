import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ArrowLeft } from "lucide-react"
import { Navbar } from "@/components/navbar"

export default function AuthPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container px-4 py-16">
        <div className="max-w-md mx-auto space-y-8">
          <div className="text-center space-y-2">
            <h1 className="text-3xl font-bold">Acceso</h1>
            <p className="text-muted-foreground">Inicie sesión o continúe como invitado para comenzar su análisis</p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Autenticación</CardTitle>
              <CardDescription>La autenticación es opcional por ahora; la autenticación real se integrará más adelante</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button variant="outline" className="w-full bg-transparent" asChild>
                <Link href="/profile-select">Continue con Google</Link>
              </Button>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-card px-2 text-muted-foreground">or</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Correo</Label>
                <Input id="email" type="email" placeholder="you@example.com" />
              </div>

              <Button className="w-full" asChild>
                <Link href="/profile-select">Continue</Link>
              </Button>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-card px-2 text-muted-foreground">or</span>
                </div>
              </div>

              <Button variant="secondary" className="w-full" asChild>
                <Link href="/profile-select">Continuar como invitado</Link>
              </Button>
            </CardContent>
          </Card>

          <div className="text-center">
            <Button variant="ghost" asChild>
              <Link href="/">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Retornar a inicio
              </Link>
            </Button>
          </div>
        </div>
      </main>
    </div>
  )
}
