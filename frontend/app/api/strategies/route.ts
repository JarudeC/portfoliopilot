/**
 * API routes for listing and creating strategies.
 * GET /api/strategies - List user's strategies
 * POST /api/strategies - Create new strategy
 */

import { NextRequest, NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/server';
import { getStrategyService } from '@/lib/services/strategies';

export async function GET(req: NextRequest) {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const { searchParams } = new URL(req.url);
    const mode = searchParams.get('mode') as 'backtest' | 'forecast' | null;

    const strategyService = getStrategyService();
    const strategies = await strategyService.listStrategies(
      user.id,
      mode || undefined
    );

    return NextResponse.json({ strategies });
  } catch (error) {
    console.error('Error listing strategies:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to list strategies' },
      { status: 500 }
    );
  }
}

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
    const { name, description, code, mode } = body;

    // Validate required fields
    if (!name || typeof name !== 'string' || name.trim().length === 0) {
      return NextResponse.json(
        { error: 'Strategy name is required' },
        { status: 400 }
      );
    }

    if (!code || typeof code !== 'string' || code.trim().length === 0) {
      return NextResponse.json(
        { error: 'Strategy code is required' },
        { status: 400 }
      );
    }

    if (!mode || !['backtest', 'forecast'].includes(mode)) {
      return NextResponse.json(
        { error: 'Valid mode (backtest or forecast) is required' },
        { status: 400 }
      );
    }

    const strategyService = getStrategyService();

    // Check if name already exists
    const nameExists = await strategyService.nameExists(user.id, name.trim());
    if (nameExists) {
      return NextResponse.json(
        { error: 'A strategy with this name already exists' },
        { status: 409 }
      );
    }

    const strategy = await strategyService.saveStrategy(user.id, {
      name: name.trim(),
      description: description?.trim() || undefined,
      code: code.trim(),
      mode,
    });

    return NextResponse.json({ strategy }, { status: 201 });
  } catch (error) {
    console.error('Error creating strategy:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to create strategy' },
      { status: 500 }
    );
  }
}
