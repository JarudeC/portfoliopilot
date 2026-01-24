// API endpoint for polling backtest job status from Python backend
import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { getAuthenticatedUser } from "@/lib/auth/server";
import { jobParameters, loggedJobs } from "../route";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function GET(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = await params;

    // Proxy to Python backend for job status
    const r = await fetch(`${BACKEND}/backtest/${id}`);
    const data = await r.json();

    // Log successful training completion to database
    const isBacktestComplete = data.status === 'done' && data.nav && data.metrics;
    const isForecastComplete = data.status === 'done' && data.predictions;

    if (r.ok && (isBacktestComplete || isForecastComplete)) {
      try {
        // Check if this job has already been logged to prevent duplicates
        if (loggedJobs.has(id)) {
          return NextResponse.json(data, { status: r.status });
        }

        // Retrieve cached job parameters
        const originalParams = jobParameters.get(id);

        // Check user authentication for logging
        let user;
        try {
          user = await getAuthenticatedUser();
        } catch (authError) {
          console.error('Authentication failed:', authError);
          user = null;
        }

        // Skip logging if no authenticated user
        if (!user) {
          return NextResponse.json(data, { status: r.status });
        }

        const logService = new TrainingLogService(true);

        // Identify job type from response data
        const isForecast = !!data.predictions;

        // Get stock list from response
        const stocks = data.tickers || Object.keys(data.weights || {}) || [];

        // Ensure we have a valid model name - backend always provides algo field
        const modelName = data.algo || originalParams?.algo || data.model || data.algorithm;

        // Skip logging if we somehow don't have a model name (shouldn't happen with current backend)
        if (!modelName) {
          console.error('No model name found for job:', id, 'data:', data, 'params:', originalParams);
          return NextResponse.json(data, { status: r.status });
        }

        const backtestNavValues = Object.values(data.nav || {}) as number[];
        const logData = {
          type: (isForecast ? 'forecast' : 'backtest') as 'forecast' | 'backtest',
          stocks: stocks as string[],
          model: modelName as string,
          parameters: {
            job_id: id,
            ...(originalParams || {}),
            // Also include backend response parameters
            weights: data.weights,
            ...(isForecast && { forecast_days: data.forecast_days })
          },
          results: isForecast ? {
            predictions: data.predictions || []
          } : {
            returns: (data.returns || []) as number[],
            cumulative_returns: backtestNavValues,
            dates: Object.keys(data.nav || {}),
            trades: data.trades || [],
            weights: data.weights
          },
          charts: isForecast ? null : {
            nav: data.nav || {},
            equity_curve: Object.keys(data.nav || {}).map((date, i) => ({
              date,
              value: backtestNavValues[i]
            }))
          },
          metrics: data.metrics || null
        };

        await logService.createLog(logData, user.id);

        // Remove cached parameters and cleanup
        jobParameters.delete(id);

        // Mark job as logged only after successful database insert
        loggedJobs.add(id);

        console.log('Successfully logged training result:', { type: logData.type, model: modelName, job_id: id });
      } catch (logError) {
        console.error('Failed to log completed training:', logError);
        if (logError instanceof Error) {
          console.error('Error stack:', logError.stack);
        }
        // Continue on logging failure
      }
    }

    return NextResponse.json(data, { status: r.status });
  } catch (error) {
    console.error('Failed to fetch backtest status:', error);

    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
