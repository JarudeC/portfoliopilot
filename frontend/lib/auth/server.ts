// Server-side authentication utilities
import { createClient } from '@/lib/supabase/server'
import { NextRequest } from 'next/server'

export async function getAuthenticatedUser(req?: NextRequest) {
  const supabase = await createClient()
  
  try {
    const { data: { user }, error } = await supabase.auth.getUser()
    
    
    if (error) {
      console.error('Auth error:', error)
      return null
    }
    
    return user
  } catch (error) {
    console.error('Failed to get authenticated user:', error)
    return null
  }
}

export async function requireAuth(req?: NextRequest) {
  const user = await getAuthenticatedUser(req)
  
  
  if (!user) {
    throw new Error('Authentication required')
  }
  
  return user
}

export function createAuthError(message: string = 'Unauthorized', status = 401) {
  return Response.json(
    { error: message },
    { status }
  )
}