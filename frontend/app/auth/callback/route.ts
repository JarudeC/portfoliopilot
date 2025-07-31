import { createClient } from '@/lib/supabase/server'
import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  const { searchParams, origin } = new URL(request.url)
  const code = searchParams.get('code')
  const next = searchParams.get('next') ?? '/'
  

  if (code) {
    const supabase = await createClient()
    
    const { data, error } = await supabase.auth.exchangeCodeForSession(code)
    
    
    if (!error && data.session) {
      return NextResponse.redirect(`${origin}${next}`)
    }
    
    if (error) {
      console.error('OAuth callback error:', error)
      return NextResponse.redirect(`${origin}/?message=Authentication failed: ${error.message}`)
    }
  } else {
  }

  return NextResponse.redirect(`${origin}/`)
}