// Middleware for session management and auth protection
import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'

export async function updateSession(request: NextRequest) {

  let supabaseResponse = NextResponse.next({
    request,
  })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          const cookies = request.cookies.getAll()
          return cookies
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value))
          supabaseResponse = NextResponse.next({
            request,
          })
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  // Refresh session and get user
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession()

  if (sessionError) {
    console.error('Session error:', sessionError)
  }

  // Refresh session if expiring soon
  if (session?.expires_at) {
    const expiresAt = new Date(session.expires_at * 1000)
    const now = new Date()
    const timeUntilExpiry = expiresAt.getTime() - now.getTime()
    
    // Preemptive refresh before expiry
    if (timeUntilExpiry < 10 * 60 * 1000) {
      const { data: refreshData, error: refreshError } = await supabase.auth.refreshSession()
      if (refreshError) {
        console.error('Token refresh error:', refreshError)
      }
    }
  }

  const {
    data: { user },
  } = await supabase.auth.getUser()

  // Protect authenticated API endpoints
  if (request.nextUrl.pathname.startsWith('/api/train/history') || 
      (request.nextUrl.pathname.startsWith('/api/train/') && 
       request.method === 'DELETE')) {
    if (!user && !session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }
  }

  return supabaseResponse
}