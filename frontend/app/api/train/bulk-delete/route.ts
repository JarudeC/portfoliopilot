import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { requireAuth, createAuthError } from "@/lib/auth/server";

export async function POST(req: NextRequest) {
  try {
    const user = await requireAuth();
    const { logIds } = await req.json();

    if (!Array.isArray(logIds) || logIds.length === 0) {
      return NextResponse.json(
        { error: 'Invalid request: logIds array is required' },
        { status: 400 }
      );
    }

    const logService = new TrainingLogService(true);
    
    // Verify ownership of all logs before deleting any
    const verificationPromises = logIds.map(async (id: string) => {
      const log = await logService.getLogById(id);
      if (!log) {
        throw new Error(`Training log ${id} not found`);
      }
      if (log.user_id !== user.id) {
        throw new Error(`Access denied to training log ${id}`);
      }
      return log;
    });

    try {
      await Promise.all(verificationPromises);
    } catch (error) {
      return NextResponse.json(
        { error: error instanceof Error ? error.message : 'Access denied' },
        { status: 403 }
      );
    }

    // Delete all logs - this will perform database deletion
    const deletePromises = logIds.map((id: string) => 
      logService.deleteLog(id)
    );

    await Promise.all(deletePromises);

    return NextResponse.json({ 
      success: true, 
      deleted: logIds.length 
    });
  } catch (error) {
    console.error('Bulk delete failed:', error);
    
    if (error instanceof Error && error.message === 'Authentication required') {
      return createAuthError();
    }
    
    return NextResponse.json(
      { error: 'Failed to delete training logs' },
      { status: 500 }
    );
  }
}