// API endpoint for training job status and results
import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { requireAuth, createAuthError, getAuthenticatedUser } from "@/lib/auth/server";
import { jobParameters, loggedJobs } from "../route";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

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
    
    // Handle database log requests (UUID format)
    if (id.includes('-')) {
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
    }

    // Handle Claude backtest logging (special case for claude-* IDs)
    if (id.startsWith('claude-')) {
      // This is a Claude backtest completion, not a backend job
      // The request body should contain the completion data
      return NextResponse.json({ error: 'Use POST for Claude logging' }, { status: 405 });
    }

    // Proxy to training backend
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
    } else {
    }
    
    return NextResponse.json(data, { status: r.status });
  } catch (error) {
    console.error('Failed to fetch training data:', error);
    
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
