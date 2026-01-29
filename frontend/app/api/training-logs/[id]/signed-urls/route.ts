import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { requireAuth, createAuthError } from "@/lib/auth/server";

export async function GET(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = await params;
    const user = await requireAuth();

    const logService = new TrainingLogService(true);
    const log = await logService.getLogById(id);

    if (!log || log.user_id !== user.id) {
      return NextResponse.json({ error: 'Not found' }, { status: 404 });
    }

    const lazyLog = await logService.toLazyLog(log);

    return NextResponse.json({
      results_signed_url: lazyLog.results_signed_url,
      charts_signed_url: lazyLog.charts_signed_url,
    });
  } catch (error) {
    if (error instanceof Error && error.message === 'Authentication required') {
      return createAuthError();
    }
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
