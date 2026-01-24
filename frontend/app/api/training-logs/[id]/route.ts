// API endpoint for training log operations (GET, POST for Claude logging, DELETE)
import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { requireAuth, createAuthError, getAuthenticatedUser } from "@/lib/auth/server";

export async function POST(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = await params;

    // Handle Claude backtest logging
    if (id.startsWith('claude-')) {
      const { data, originalParams } = await req.json();

      // Check user authentication for logging
      let user;
      try {
        user = await getAuthenticatedUser();
      } catch (authError) {
        console.error('Authentication failed:', authError);
        return createAuthError();
      }

      const logService = new TrainingLogService(true);

      // Use the same logging logic as traditional algorithms
      const isBacktestComplete = data.status === 'done' && data.nav && data.metrics;

      if (isBacktestComplete) {
        // Get stock list from response
        const stocks = data.tickers || Object.keys(data.weights || {}) || [];

        // Ensure we have a valid model name
        const modelName = data.algo || originalParams?.algo || 'CUSTOM_AI_STRATEGY';

        const navValues = Object.values(data.nav || {}) as number[];
        const logData = {
          type: 'backtest' as const,
          stocks: stocks as string[],
          model: modelName as string,
          parameters: {
            job_id: id,
            ...(originalParams || {}),
            weights: data.weights
          },
          results: {
            returns: (data.returns || []) as number[],
            cumulative_returns: navValues,
            dates: Object.keys(data.nav || {}),
            trades: data.trades || [],
            weights: data.weights
          },
          charts: {
            nav: data.nav || {},
            equity_curve: Object.keys(data.nav || {}).map((date, i) => ({
              date,
              value: navValues[i]
            }))
          },
          metrics: data.metrics || null
        };

        if (!user) {
          return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
        }
        const result = await logService.createLog(logData, user.id);

        console.log('Successfully logged Claude backtest result:', {
          type: logData.type,
          model: modelName,
          job_id: id
        });

        return NextResponse.json({ success: true, id: result.id });
      } else {
        return NextResponse.json({
          error: 'Invalid backtest data - missing required fields'
        }, { status: 400 });
      }
    }

    return NextResponse.json({ error: 'Invalid request' }, { status: 400 });
  } catch (error) {
    console.error('Failed to log Claude backtest session:', error);

    if (error instanceof Error && error.message === 'Authentication required') {
      return createAuthError();
    }

    return NextResponse.json(
      { error: 'Failed to log backtest session' },
      { status: 500 }
    );
  }
}

export async function GET(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = await params;
    const user = await requireAuth();

    const logService = new TrainingLogService(true);
    const log = await logService.getLogById(id);

    if (!log) {
      return NextResponse.json(
        { error: 'Training log not found' },
        { status: 404 }
      );
    }

    // Verify log ownership
    if (log.user_id !== user.id) {
      return NextResponse.json(
        { error: 'Forbidden' },
        { status: 403 }
      );
    }

    // Hydrate log to load results and charts from storage
    const hydratedLog = await logService.hydrateLog(log);
    return NextResponse.json(hydratedLog);
  } catch (error) {
    console.error('Failed to fetch training log:', error);

    if (error instanceof Error && error.message === 'Authentication required') {
      return createAuthError();
    }

    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = await params;
    const user = await requireAuth();

    const logService = new TrainingLogService(true);

    // Check log ownership before delete
    const log = await logService.getLogById(id);
    if (!log) {
      return NextResponse.json(
        { error: 'Training log not found' },
        { status: 404 }
      );
    }

    if (log.user_id !== user.id) {
      return NextResponse.json(
        { error: 'Forbidden' },
        { status: 403 }
      );
    }

    await logService.deleteLog(id);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to delete training log:', error);

    if (error instanceof Error && error.message === 'Authentication required') {
      return createAuthError();
    }

    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
