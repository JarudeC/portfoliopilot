import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { getAuthenticatedUser } from "@/lib/auth/server";

export async function POST(req: NextRequest) {
  try {
    // Require authentication for logging forecast sessions
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const body = await req.json();

    const logService = new TrainingLogService(true);
    
    const logData = {
      type: body.type,
      stocks: body.stocks,
      model: body.model,
      parameters: body.parameters,
      results: body.results,
      charts: body.charts,
      metrics: null // Forecasts don't have performance metrics like backtests
    };
    
    await logService.createLog(logData, user.id);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to log forecast session:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}