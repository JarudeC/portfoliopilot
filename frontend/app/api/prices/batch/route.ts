import { NextRequest, NextResponse } from "next/server";
import { getAuthenticatedUser } from "@/lib/auth/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

/**
 * POST /api/prices/batch
 *
 * Fetch historical stock prices for multiple tickers in parallel.
 * Uses ThreadPoolExecutor on backend for concurrent yfinance calls.
 *
 * Request body:
 *   - tickers: string[] (e.g., ["AAPL", "GOOGL", "MSFT"])
 *   - start: string (YYYY-MM-DD)
 *   - end: string (YYYY-MM-DD)
 *
 * Response:
 *   - { [ticker]: { dates: string[], prices: number[] } | { error: string } }
 */
export async function POST(req: NextRequest) {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const body = await req.json();

    if (!body.tickers || !Array.isArray(body.tickers) || !body.start || !body.end) {
      return NextResponse.json(
        { error: 'Missing required fields: tickers (array), start, end' },
        { status: 400 }
      );
    }

    const response = await fetch(`${BACKEND}/prices/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Batch Prices API: Backend error:', response.status, errorText);
      return NextResponse.json(
        { error: `Backend error: ${errorText}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Batch Prices API: Request failed:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
