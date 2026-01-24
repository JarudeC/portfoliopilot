// NOTE: Single-ticker endpoint replaced by /api/prices/batch - kept for potential future use
//
// import { NextRequest, NextResponse } from "next/server";
// import { getAuthenticatedUser } from "@/lib/auth/server";
//
// const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";
//
// /**
//  * POST /api/prices
//  *
//  * Fetch historical stock prices from Yahoo Finance via the backend.
//  * This is a dedicated endpoint for fetching raw price data without
//  * running any forecasting models.
//  *
//  * Request body:
//  *   - ticker: string (e.g., "AAPL")
//  *   - start: string (YYYY-MM-DD)
//  *   - end: string (YYYY-MM-DD)
//  *
//  * Response:
//  *   - ticker: string
//  *   - dates: string[]
//  *   - prices: number[]
//  */
// export async function POST(req: NextRequest) {
//   try {
//     // Require authentication
//     const user = await getAuthenticatedUser();
//     if (!user) {
//       return NextResponse.json(
//         { error: 'Authentication required' },
//         { status: 401 }
//       );
//     }
//
//     const body = await req.json();
//
//     // Validate required fields
//     if (!body.ticker || !body.start || !body.end) {
//       return NextResponse.json(
//         { error: 'Missing required fields: ticker, start, end' },
//         { status: 400 }
//       );
//     }
//
//     // Forward request to backend
//     const backendUrl = `${BACKEND}/prices`;
//     const response = await fetch(backendUrl, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify(body),
//     });
//
//     if (!response.ok) {
//       const errorText = await response.text();
//       console.error('Prices API: Backend error:', response.status, errorText);
//       return NextResponse.json(
//         { error: `Backend error: ${errorText}` },
//         { status: response.status }
//       );
//     }
//
//     const data = await response.json();
//     return NextResponse.json(data);
//   } catch (error) {
//     console.error('Prices API: Request failed:', error);
//     return NextResponse.json(
//       { error: 'Internal server error' },
//       { status: 500 }
//     );
//   }
// }
