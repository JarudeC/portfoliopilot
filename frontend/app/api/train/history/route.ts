import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { requireAuth, createAuthError } from "@/lib/auth/server";

export async function GET(req: NextRequest) {
  try {
    
    const user = await requireAuth();

    // Rate limiting check (simple in-memory implementation)
    const userAgent = req.headers.get('user-agent') || 'unknown';
    const clientId = req.ip || userAgent;
    
    if (!checkRateLimit(clientId, 'history', 60, 100)) {
      return NextResponse.json(
        { error: 'Rate limit exceeded' },
        { status: 429 }
      );
    }

    const { searchParams } = new URL(req.url);
    const type = searchParams.get('type') as 'forecast' | 'backtest' | null;
    const limit = parseInt(searchParams.get('limit') || '50');
    const offset = parseInt(searchParams.get('offset') || '0');

    const logService = new TrainingLogService(true);
    
    let logs;
    if (type) {
      logs = await logService.getLogsByType(type, limit);
    } else {
      logs = await logService.getUserLogs(user.id, limit, offset);
    }

    return NextResponse.json({ logs: logs || [] });
  } catch (error) {
    console.error('Failed to fetch training history:', error);
    
    if (error instanceof Error && error.message === 'Authentication required') {
      return createAuthError();
    }
    
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Simple in-memory rate limiting
const rateLimitMap = new Map<string, { count: number; resetTime: number }>();

function checkRateLimit(clientId: string, endpoint: string, windowSeconds: number, maxRequests: number): boolean {
  const key = `${clientId}:${endpoint}`;
  const now = Date.now();
  const windowMs = windowSeconds * 1000;
  
  const current = rateLimitMap.get(key);
  
  if (!current || now > current.resetTime) {
    rateLimitMap.set(key, { count: 1, resetTime: now + windowMs });
    return true;
  }
  
  if (current.count >= maxRequests) {
    return false;
  }
  
  current.count++;
  return true;
}