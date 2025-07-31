import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { getAuthenticatedUser } from "@/lib/auth/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(
  req: NextRequest,
  { params }: { params: { algo: string } }
) {
  try {
    // Require authentication for forecast requests
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const { algo } = await params;
    const body = await req.json();
    
    // Forward request to backend
    const r = await fetch(`${BACKEND}/forecast/${algo}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await r.json();

    // Note: Individual forecast logging is now handled by the dashboard 
    // to create combined forecast sessions with multiple tickers
    // instead of separate logs per ticker

    return NextResponse.json(data, { status: r.status });
  } catch (error) {
    console.error('Forecast request failed:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}