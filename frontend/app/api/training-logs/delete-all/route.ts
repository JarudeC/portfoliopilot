import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { requireAuth, createAuthError } from "@/lib/auth/server";

export async function POST(req: NextRequest) {
  try {
    const user = await requireAuth();
    const logService = new TrainingLogService(true);
    
    // Get all logs for the user first
    const allLogs = await logService.getUserLogs(user.id, 1000, 0); // Get up to 1000 logs
    
    if (allLogs.length === 0) {
      return NextResponse.json({ 
        success: true, 
        deleted: 0,
        message: 'No logs to delete'
      });
    }

    // Delete all logs - this will perform database deletion for each one
    const deletePromises = allLogs.map(log => 
      logService.deleteLog(log.id)
    );

    await Promise.all(deletePromises);

    return NextResponse.json({ 
      success: true, 
      deleted: allLogs.length 
    });
  } catch (error) {
    console.error('Delete all failed:', error);
    
    if (error instanceof Error && error.message === 'Authentication required') {
      return createAuthError();
    }
    
    return NextResponse.json(
      { error: 'Failed to delete all training logs' },
      { status: 500 }
    );
  }
}