/**
 * API routes for single strategy operations.
 * GET /api/strategies/[id] - Get single strategy
 * PUT /api/strategies/[id] - Update strategy
 * DELETE /api/strategies/[id] - Delete strategy
 */

import { NextRequest, NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/server';
import { getStrategyService } from '@/lib/services/strategies';

interface RouteParams {
  params: Promise<{ id: string }>;
}

export async function GET(req: NextRequest, { params }: RouteParams) {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const { id } = await params;
    const strategyService = getStrategyService();
    const strategy = await strategyService.getStrategy(id);

    if (!strategy) {
      return NextResponse.json(
        { error: 'Strategy not found' },
        { status: 404 }
      );
    }

    // Verify ownership
    if (strategy.user_id !== user.id) {
      return NextResponse.json(
        { error: 'Access denied' },
        { status: 403 }
      );
    }

    return NextResponse.json({ strategy });
  } catch (error) {
    console.error('Error getting strategy:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to get strategy' },
      { status: 500 }
    );
  }
}

export async function PUT(req: NextRequest, { params }: RouteParams) {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const { id } = await params;
    const body = await req.json();
    const { name, description, code } = body;

    // Validate name if provided
    if (name !== undefined && (typeof name !== 'string' || name.trim().length === 0)) {
      return NextResponse.json(
        { error: 'Strategy name cannot be empty' },
        { status: 400 }
      );
    }

    const strategyService = getStrategyService();

    // Check if new name conflicts with existing
    if (name) {
      const nameExists = await strategyService.nameExists(user.id, name.trim(), id);
      if (nameExists) {
        return NextResponse.json(
          { error: 'A strategy with this name already exists' },
          { status: 409 }
        );
      }
    }

    const strategy = await strategyService.updateStrategy(id, user.id, {
      name: name?.trim(),
      description: description?.trim(),
      code: code?.trim(),
    });

    return NextResponse.json({ strategy });
  } catch (error) {
    console.error('Error updating strategy:', error);
    const message = error instanceof Error ? error.message : 'Failed to update strategy';
    const status = message.includes('not found') || message.includes('access denied') ? 404 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}

export async function DELETE(req: NextRequest, { params }: RouteParams) {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const { id } = await params;
    const strategyService = getStrategyService();
    await strategyService.deleteStrategy(id, user.id);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting strategy:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to delete strategy' },
      { status: 500 }
    );
  }
}
