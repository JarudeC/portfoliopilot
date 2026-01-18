import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { getAuthenticatedUser } from "@/lib/auth/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(
  req: NextRequest,
  { params }: { params: { algo: string } }
) {
  try {
    console.log('Forecast API: Starting request processing');
    
    // Require authentication for forecast requests
    console.log('Forecast API: Checking authentication');
    const user = await getAuthenticatedUser();
    if (!user) {
      console.error('Forecast API: Authentication failed - no user');
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }
    console.log('Forecast API: Authentication successful for user:', user.id);

    const { algo } = await params;
    const body = await req.json();
    
    console.log('Forecast API: Processing request', { 
      algo, 
      bodyKeys: Object.keys(body),
      backend: BACKEND
    });
    
    // Forward request to backend
    const backendUrl = `${BACKEND}/forecast/${algo}`;
    console.log('Forecast API: Calling backend at:', backendUrl);
    
    const r = await fetch(backendUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    console.log('Forecast API: Backend response status:', r.status);
    
    if (!r.ok) {
      console.error('Forecast API: Backend request failed:', r.status, r.statusText);
      const errorText = await r.text();
      console.error('Forecast API: Backend error details:', errorText);
    }

    const data = await r.json();
    console.log('Forecast API: Backend response data keys:', Object.keys(data));

    // Note: Individual forecast logging is now handled by the dashboard 
    // to create combined forecast sessions with multiple tickers
    // instead of separate logs per ticker

    return NextResponse.json(data, { status: r.status });
  } catch (error) {
    console.error('Forecast API: Request failed with error:', error);
    if (error instanceof Error) {
      console.error('Forecast API: Error details:', {
        name: error.name,
        message: error.message,
        stack: error.stack
      });
    }
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}